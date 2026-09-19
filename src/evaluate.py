"""C2 regime map + C3 honest policy benchmark.

Pipeline:
  1. Per-exit cost table: FLOPs (hook-measured) + wall latency + measured Joules
     (GPU box only; None on CPU) across batches [1,8,16,32,64].
  2. Collect per-sample head statistics on calib/test splits (confs, entropy,
     correctness) — one model pass, all policies evaluated as cheap numpy ops.
  3. Policies: confidence(tau), entropy(eta), eefp-style(tau,beta),
     marginal-utility(lambda) with MEASURED or FLOP costs.
  4. Accuracy-cost Pareto frontier + FLOP-vs-measured divergence report.

Novelty honesty: marginal-utility is rational metareasoning with a measured
cost term (cite verdict). This file benchmarks it; it does not claim it.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F

try:
    from utils import (cuda_sync, ensure_dir, get_git_hash, load_config, resolve_device,
                       save_json, set_seed, torch_info, vram_info)
except ImportError:
    from src.utils import (cuda_sync, ensure_dir, get_git_hash, load_config, resolve_device,
                           save_json, set_seed, torch_info, vram_info)

try:
    from models import build_model, default_io, flops_per_exit, model_from_checkpoint
except ImportError:
    from src.models import build_model, default_io, flops_per_exit, model_from_checkpoint

try:
    from measure import EnergyMeter
except ImportError:
    from src.measure import EnergyMeter

try:
    from train import get_dataloaders, is_synthetic
except ImportError:
    from src.train import get_dataloaders, is_synthetic


# ----------------------------------------------------------------------------
# Stage 1: per-exit cost table
# ----------------------------------------------------------------------------
def measure_latency_per_exit(model, input_shape: tuple, device: torch.device,
                             iters: int = 200, warmup: int = 20) -> list[float]:
    """Wall seconds per forward_exit call (mean). CPU-safe."""
    model.eval()
    n = int(getattr(model, "num_heads", 1))
    lat = []
    with torch.no_grad():
        for idx in range(n):
            x = torch.randn(*input_shape, device=device)
            for _ in range(warmup):
                _ = model.forward_exit(x, idx) if hasattr(model, "forward_exit") else model(x)
            cuda_sync()
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = model.forward_exit(x, idx) if hasattr(model, "forward_exit") else model(x)
            cuda_sync()
            t1 = time.perf_counter()
            lat.append((t1 - t0) / max(1, iters))
    return lat


def make_exit_fn(model, x: torch.Tensor, idx: int, mode: str = "cascade"):
    """No-arg callable that executes exit `idx` once.

    prefix : forward_exit only (the network prefix + head idx).
    cascade: deployed early-exit path (iter_heads): heads 0..idx-1 are also
             evaluated and each exit decision is made on the host
             (softmax-max + .item() sync), exactly what a real run pays.
    """
    if mode == "cascade" and hasattr(model, "iter_heads"):
        def fn():
            for h, lg in enumerate(model.iter_heads(x)):
                if h == idx:
                    break
                F.softmax(lg, dim=1).max(1).values.min().item()  # exit decision
        return fn
    if hasattr(model, "forward_exit"):
        return lambda: model.forward_exit(x, idx)
    return lambda: model(x)


def _iters_for_window(fn, iters: int, min_window_s: float, probe: int = 20,
                      cap: int = 200_000) -> int:
    """>= iters, and enough iterations that the window lasts >= min_window_s.

    Short windows are dominated by NVML counter/sample update granularity
    (tens of ms), so a fixed 1000 iters of a 0.3 ms kernel is NOT enough.
    """
    cuda_sync()
    t0 = time.perf_counter()
    for _ in range(probe):
        fn()
    cuda_sync()
    per = max(1e-7, (time.perf_counter() - t0) / probe)
    return int(min(cap, max(int(iters), math.ceil(min_window_s / per))))


def gpu_warmup(seconds: float, size: int = 2048) -> dict:
    """Sustained matmul so clocks/temperature reach steady state before timed windows."""
    if seconds <= 0 or not torch.cuda.is_available():
        return {"seconds": 0.0, "skipped": True}
    a = torch.randn(size, size, device="cuda")
    b = torch.randn(size, size, device="cuda")
    t_end = time.perf_counter() + float(seconds)
    n = 0
    with torch.no_grad():
        while time.perf_counter() < t_end:
            _ = (a @ b).sum()
            torch.cuda.synchronize()
            n += 1
    del a, b
    torch.cuda.empty_cache()
    return {"seconds": float(seconds), "iters": n, "skipped": False}


def measure_energy_per_exit(meter, model, input_shape: tuple, device: torch.device,
                            idle_power_w: float | None, iters: int = 1000,
                            repeats: int = 3, min_window_s: float = 3.0,
                            warmup: int = 10, mode: str = "cascade") -> dict:
    """Per-exit marginal Joules PER SAMPLE (mean +- std over `repeats` windows).

    Exits are measured round-robin with a rotated start each repeat so slow
    thermal/clock drift does not line up with exit index (that would fake a
    FLOP-vs-Joule ordering effect). Returns None entries when no energy
    source exists — never fake data.
    """
    n_heads = int(getattr(model, "num_heads", 1))
    batch = int(input_shape[0])
    empty = {"mode": mode, "batch": batch, "mean_j": [None] * n_heads,
             "std_j": [None] * n_heads, "iters": [None] * n_heads, "windows": [],
             "adjacent_resolvable_3sigma": [None] * max(0, n_heads - 1)}
    if meter is None or meter.source == "none":
        return {**empty, "skipped": True, "reason": "no NVML/CUDA on this machine"}
    if idle_power_w is None:
        return {**empty, "skipped": True, "reason": "no idle baseline: marginal energy undefined"}
    model.eval()
    x = torch.randn(*input_shape, device=device)
    samples: list[list[float]] = [[] for _ in range(n_heads)]
    iters_used = [0] * n_heads
    windows = []
    with torch.no_grad():
        fns = [make_exit_fn(model, x, i, mode) for i in range(n_heads)]
        for i, fn in enumerate(fns):
            for _ in range(warmup):
                fn()
            iters_used[i] = _iters_for_window(fn, iters, min_window_s)
        for r in range(max(1, int(repeats))):
            for k in range(n_heads):
                i = (k + r) % n_heads
                res = meter.measure_fn(fns[i], iters=iters_used[i], warmup=2,
                                       idle_power_w=idle_power_w)
                per_iter = res.per_iter_marginal_j
                if per_iter is None:
                    return {**empty, "skipped": True,
                            "reason": f"energy source returned no data (source={res.source})"}
                samples[i].append(per_iter / batch)
                windows.append({"exit": i, "repeat": r, "duration_s": res.duration_s,
                                "iters": res.iters, "source": res.source,
                                "per_sample_marginal_j": per_iter / batch,
                                "per_sample_total_j": (res.per_iter_j or 0.0) / batch,
                                "temp_c": res.device_after.get("temperature_c"),
                                "clock_mhz": res.device_after.get("clock_graphics_mhz"),
                                "effective_hz": res.effective_hz})
    mean = [float(np.mean(s)) for s in samples]
    std = [float(np.std(s, ddof=1)) if len(s) > 1 else None for s in samples]
    resolvable: list[bool | None] = []
    for h in range(n_heads - 1):
        if std[h] is None or std[h + 1] is None:
            resolvable.append(None)
        else:
            noise = math.sqrt(std[h] ** 2 + std[h + 1] ** 2)
            resolvable.append(bool(abs(mean[h + 1] - mean[h]) > 3.0 * noise))
    return {"mode": mode, "batch": batch, "mean_j": mean, "std_j": std,
            "iters": iters_used, "windows": windows,
            "adjacent_resolvable_3sigma": resolvable, "skipped": False}


def divergence_report(flops: list, mean_j: list, std_j: list) -> dict:
    """FLOP order vs measured-Joule order (KILL-3), with a 3-sigma significance test.

    A pair (i, j) with flops[i] < flops[j] is a SIGNIFICANT disagreement only if
    joules[i] exceeds joules[j] by more than 3 * combined std.
    """
    if any(v is None for v in mean_j):
        return {"available": False}
    fo = np.array(flops, dtype=float)
    jo = np.array(mean_j, dtype=float)
    fo_n = (fo - fo.min()) / max(1e-12, fo.max() - fo.min())
    jo_n = (jo - jo.min()) / max(1e-12, jo.max() - jo.min())
    sig = []
    for i in range(len(fo)):
        for j in range(len(fo)):
            if fo[i] < fo[j]:
                si = std_j[i] or 0.0
                sj = std_j[j] or 0.0
                if jo[i] - jo[j] > 3.0 * math.sqrt(si ** 2 + sj ** 2):
                    sig.append([i, j])
    fo_ord, jo_ord = np.argsort(fo, kind="stable"), np.argsort(jo, kind="stable")
    return {"available": True,
            "flops_order": list(map(int, fo_ord)),
            "joule_order": list(map(int, jo_ord)),
            "orders_agree": bool((fo_ord == jo_ord).all()),
            "significant_disagreements_3sigma": sig,
            "max_abs_gap_normalized": float(np.abs(fo_n - jo_n).max())}


# ----------------------------------------------------------------------------
# Stage 2: collect per-sample head stats (single pass)
# ----------------------------------------------------------------------------
@torch.no_grad()
def collect_stats(model, loader, device: torch.device) -> dict:
    """Returns numpy arrays: conf [N,H], entropy [N,H], pred [N,H], correct [N,H], label [N]."""
    model.eval()
    confs, ents, preds, labels = [], [], [], []
    for x, y in loader:
        x = x.to(device)
        heads = model.forward_all(x) if hasattr(model, "forward_all") else [model(x)]
        b = x.size(0)
        c = np.zeros((b, len(heads)))
        e = np.zeros((b, len(heads)))
        p = np.zeros((b, len(heads)), dtype=np.int64)
        for h, lg in enumerate(heads):
            pr = F.softmax(lg, dim=1)
            c[:, h] = pr.max(1).values.cpu().numpy()
            e[:, h] = -(pr * (pr + 1e-12).log()).sum(1).cpu().numpy()
            p[:, h] = lg.argmax(1).cpu().numpy()
        confs.append(c)
        ents.append(e)
        preds.append(p)
        labels.append(y.numpy())
    conf = np.concatenate(confs, axis=0)
    ent = np.concatenate(ents, axis=0)
    pred = np.concatenate(preds, axis=0)
    lab = np.concatenate(labels, axis=0)
    correct = (pred == lab[:, None])
    return {"conf": conf, "entropy": ent, "pred": pred, "label": lab,
            "correct": correct, "n": int(lab.shape[0]), "heads": int(pred.shape[1])}


def _summarize(chosen: np.ndarray, correct: np.ndarray, costs: list[float] | None) -> dict:
    n = len(chosen)
    acc = float(correct[np.arange(n), chosen].mean()) if n else 0.0
    avg_exit = float(chosen.mean()) if n else 0.0
    exit_rate = float((chosen < correct.shape[1] - 1).mean()) if n else 0.0
    avg_cost = None
    if costs is not None:
        c = np.array(costs, dtype=float)
        avg_cost = float(c[chosen].mean())
    return {"acc": acc, "avg_exit": avg_exit, "exit_rate": exit_rate, "avg_cost": avg_cost, "n": n}


def policy_confidence(stats: dict, tau: float, costs=None) -> dict:
    conf, correct = stats["conf"], stats["correct"]
    n, H = conf.shape
    chosen = np.full(n, H - 1)
    for i in range(n):
        for h in range(H - 1):
            if conf[i, h] >= tau:
                chosen[i] = h
                break
    out = _summarize(chosen, correct, costs)
    out.update({"policy": "confidence", "tau": tau})
    return out


def policy_entropy(stats: dict, eta: float, costs=None) -> dict:
    ent, correct = stats["entropy"], stats["correct"]
    n, H = ent.shape
    chosen = np.full(n, H - 1)
    for i in range(n):
        for h in range(H - 1):
            if ent[i, h] <= eta:
                chosen[i] = h
                break
    out = _summarize(chosen, correct, costs)
    out.update({"policy": "entropy", "eta": eta})
    return out


def policy_eefp(stats: dict, tau: float, beta: float, costs_norm: list[float],
               costs_per_exit=None) -> dict:
    """EEFP-style: exit if conf[h] - beta*cost_norm[h] >= tau (correctness + cost)."""
    conf, correct = stats["conf"], stats["correct"]
    n, H = conf.shape
    cn = np.array(costs_norm, dtype=float)
    chosen = np.full(n, H - 1)
    for i in range(n):
        for h in range(H - 1):
            if conf[i, h] - beta * cn[h] >= tau:
                chosen[i] = h
                break
    out = _summarize(chosen, correct, costs_per_exit)
    out.update({"policy": "eefp", "tau": tau, "beta": beta})
    return out


def estimate_gains(calib_stats: dict) -> list[float]:
    """G[h] = acc[h+1] - acc[h] on calibration, clipped >= 0. Length H-1."""
    correct = calib_stats["correct"]
    accs = correct.mean(axis=0)
    return [max(0.0, float(accs[h + 1] - accs[h])) for h in range(len(accs) - 1)]


def policy_marginal_utility(stats: dict, lam: float, gains: list[float],
                            marginal_costs: list[float], costs_per_exit=None) -> dict:
    """Continue iff expected-gain / marginal-cost > lambda (metareasoning rule).

    gain_proxy(sample at h) = (1 - conf[h]) * G[h]; C[h] = marginal cost of
    running head h+1. Pass C normalized by the full-model cost so lambda is
    unit-free (see lambda_grid). Myopic one-step lookahead, as in the classic
    value-of-computation rule. Costs are floored at 1e-12 (never divide by zero).
    """
    conf, correct = stats["conf"], stats["correct"]
    n, H = conf.shape
    C = [max(1e-12, float(c)) for c in marginal_costs]
    chosen = np.full(n, H - 1)
    for i in range(n):
        h = 0
        while h < H - 1:
            gain = (1.0 - float(conf[i, h])) * float(gains[h])
            if gain / C[h] > lam:
                h += 1  # pay cost, continue
            else:
                break
        chosen[i] = h
    out = _summarize(chosen, correct, costs_per_exit)
    out.update({"policy": "marginal-utility", "lambda": lam})
    return out


def pareto_frontier(points: list[dict], acc_key="acc", cost_key="avg_cost") -> list[int]:
    """Indices of non-dominated points (higher acc, lower cost). NaN costs excluded."""
    valid = [(i, p) for i, p in enumerate(points)
             if p.get(acc_key) is not None and p.get(cost_key) is not None
             and math.isfinite(p[cost_key])]
    front = []
    for i, p in valid:
        dominated = False
        for j, q in valid:
            if j == i:
                continue
            if q[acc_key] >= p[acc_key] and q[cost_key] <= p[cost_key] and \
               (q[acc_key] > p[acc_key] or q[cost_key] < p[cost_key]):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return sorted(front)


def lambda_grid(calib_stats: dict, gains: list[float], marg_norm: list[float],
                base: list[float], qs=(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)) -> list[float]:
    """Config lambdas plus quantiles of the calibration gain/cost ratios.

    Units: expected accuracy gain per FRACTION of full-model cost. Without the
    data-driven quantiles a fixed grid collapses to always-first-exit or
    always-final-exit depending on the cost unit (FLOPs ~1e8 vs Joules ~1e-3).
    """
    conf = calib_stats["conf"]
    ratios = []
    for h in range(conf.shape[1] - 1):
        r = (1.0 - conf[:, h]) * float(gains[h]) / float(marg_norm[h])
        ratios.append(r[r > 0])
    allr = np.concatenate(ratios) if ratios else np.array([])
    extra = [float(np.quantile(allr, q)) for q in qs] if allr.size else []
    return sorted({round(float(v), 8) for v in list(base) + extra if v > 0})


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="C2+C3 benchmark (verdict audit)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--arch", default=None, help="default: arch stored in --ckpt")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--batches", nargs="*", type=int, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--results", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--lat-iters", type=int, default=200)
    ap.add_argument("--energy-iters", type=int, default=None,
                    help="min iters per energy window (default: config eval.iters_per_exit)")
    ap.add_argument("--repeats", type=int, default=None,
                    help="energy windows per exit (default: config eval.repeats)")
    ap.add_argument("--min-window-s", type=float, default=None,
                    help="min seconds per energy window (default: config measure.min_window_seconds)")
    ap.add_argument("--warmup-seconds", type=float, default=None,
                    help="GPU warmup before timed windows (default: config measure.warmup_seconds)")
    ap.add_argument("--idle-seconds", type=float, default=None,
                    help="idle baseline length (default: config measure.idle_seconds)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    mc, ec = dict(cfg.get("measure", {})), dict(cfg.get("eval", {}))
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(args.device or cfg.get("device", "auto"))
    dataset = (args.dataset or cfg.get("dataset", "cifar10")).lower()
    batches = args.batches or ec.get("batches", [1, 8, 16, 32, 64])
    taus = [float(t) for t in ec.get("thresholds", [0.6, 0.7, 0.8, 0.9])]
    base_lams = [float(v) for v in ec.get("lambdas", [0.001, 0.01, 0.05])]
    energy_iters = int(args.energy_iters or ec.get("iters_per_exit", 1000))
    repeats = int(args.repeats or ec.get("repeats", 3))
    min_window_s = float(args.min_window_s if args.min_window_s is not None
                         else mc.get("min_window_seconds", 3.0))
    warmup_s = float(args.warmup_seconds if args.warmup_seconds is not None
                     else mc.get("warmup_seconds", 120))
    idle_s = float(args.idle_seconds if args.idle_seconds is not None
                   else mc.get("idle_seconds", 60))
    min_iters = int(mc.get("min_iters", 1000))
    if energy_iters < min_iters:
        print(f"[evaluate] WARNING: --energy-iters {energy_iters} < measure.min_iters {min_iters}; "
              f"numbers are NOT publishable under the C1 protocol.", flush=True)
    results = Path(args.results or cfg.get("results_root", "./results"))
    ensure_dir(results)

    nc, ch, (Hh, Ww) = default_io(dataset)
    if args.ckpt:
        model, arch = model_from_checkpoint(args.ckpt, dataset, device, arch=args.arch,
                                            fallback_arch=cfg.get("student", "resnet14"))
    else:
        arch = args.arch or cfg.get("student", "resnet14")
        model = build_model(dataset, arch, nc, ch).to(device)
        print("[evaluate] no --ckpt; random-init net (plumbing only).", flush=True)
    model.eval()

    _, test_loader, _, _ = get_dataloaders(
        dataset, cfg.get("data_root", "./data"), 128, 0,
        int(cfg.get("seed", 42)), force_synthetic=args.synthetic or args.ckpt is None)
    synthetic = is_synthetic(test_loader)
    if synthetic:
        print("[evaluate] WARNING: SYNTHETIC data (random labels): accuracy/policy "
              "numbers are plumbing only, not results.", flush=True)
    # calib/test split for gains (same seed/split as calibrate.py)
    from torch.utils.data import Subset
    from torch.utils.data import DataLoader as DL
    n = len(test_loader.dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(int(cfg.get("seed", 42)))).tolist()
    calib_loader = DL(Subset(test_loader.dataset, idx[:n // 2]), batch_size=128)
    deploy_loader = DL(Subset(test_loader.dataset, idx[n // 2:]), batch_size=128)

    # --- FLOPs (prefix and deployed cascade), batch-1 latency ---
    flops = flops_per_exit(model, (1, ch, Hh, Ww), device="cpu")
    flops_cascade = flops_per_exit(model, (1, ch, Hh, Ww), device="cpu", cascade=True)
    lat1 = measure_latency_per_exit(model, (1, ch, Hh, Ww), device, iters=args.lat_iters)

    # --- energy setup: settled idle baseline, then warmup to steady state ---
    meter = EnergyMeter(poll_hz=float(mc.get("poll_hz", 150.0))) if device.type == "cuda" else None
    idle_info: dict = {"skipped": True, "reason": "no NVML/CUDA on this machine"}
    idle_p = None
    warm_info: dict = {"skipped": True}
    batch_rows: list[dict] = []
    try:
        if meter is not None and meter.source != "none":
            cuda_sync()
            time.sleep(5.0)  # let clocks drop after the latency loops
            print(f"[evaluate] idle baseline {idle_s}s ...", flush=True)
            idle = meter.measure_idle(seconds=idle_s)
            idle_p = idle.get("idle_power_w")
            idle_info = {"idle_power_w": idle_p, "seconds": idle_s, "source": meter.source,
                         "device": idle.get("device_after", {})}
            print(f"[evaluate] idle={idle_p} W ({meter.source}); warmup {warmup_s}s ...", flush=True)
            warm_info = gpu_warmup(warmup_s)

        # --- batch sweep: latency + energy (prefix & cascade) per exit ---
        for b in batches:
            shape = (b, ch, Hh, Ww)
            try:
                lat = measure_latency_per_exit(model, shape, device,
                                               iters=max(20, args.lat_iters // 4))
                row = {"batch": b, "lat_per_exit_s": lat,
                       "lat_per_sample_ms": [v * 1000 / b for v in lat],
                       "flops_per_exit": flops, "flops_per_exit_cascade": flops_cascade}
                for mode in ("prefix", "cascade"):
                    e = measure_energy_per_exit(meter, model, shape, device, idle_p,
                                                iters=energy_iters, repeats=repeats,
                                                min_window_s=min_window_s, mode=mode)
                    row[f"energy_{mode}"] = e
                    row[f"divergence_{mode}"] = divergence_report(
                        flops if mode == "prefix" else flops_cascade, e["mean_j"], e["std_j"])
                    if not e.get("skipped"):
                        print(f"[evaluate] b={b} {mode:7s} J/sample={e['mean_j']} std={e['std_j']} "
                              f"resolvable={e['adjacent_resolvable_3sigma']}", flush=True)
                row["vram"] = vram_info()
                batch_rows.append(row)
            except RuntimeError as err:
                if "out of memory" in str(err).lower():
                    print(f"[evaluate] OOM at batch {b}; recorded, sweep stopped (not shrunk).",
                          flush=True)
                    batch_rows.append({"batch": b, "oom": True})
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    break
                raise
    finally:
        if meter is not None:
            meter.close()

    # --- policies on deploy split; cost = batch-1 CASCADE (the deployed path) ---
    calib_stats = collect_stats(model, calib_loader, device)
    deploy_stats = collect_stats(model, deploy_loader, device)
    gains = estimate_gains(calib_stats)
    b1 = next((r for r in batch_rows if r.get("batch") == 1 and not r.get("oom")), None)
    e1 = b1.get("energy_cascade") if b1 else None
    has_joules = bool(e1 and not e1.get("skipped") and all(v is not None for v in e1["mean_j"]))
    if has_joules:
        base = [float(v) for v in e1["mean_j"]]
        cost_kind = "measured-joules-per-sample (batch 1, cascade, idle-subtracted)"
    else:
        base = [float(v) for v in flops_cascade]
        cost_kind = "flops-cascade (energy unavailable on this machine or batch 1 not swept)"
    full = abs(base[-1]) if abs(base[-1]) > 0 else 1.0
    marg_raw = [base[h + 1] - base[h] for h in range(len(base) - 1)]
    marg_norm = [max(1e-9, v / full) for v in marg_raw]
    cnorm = [v / full for v in base]
    lams = lambda_grid(calib_stats, gains, marg_norm, base_lams)

    points: list[dict] = []
    for t in taus:
        points.append(policy_confidence(deploy_stats, t, base))
    for eta in [0.3, 0.6, 1.0]:
        points.append(policy_entropy(deploy_stats, eta, base))
    for beta in [0.1, 0.3]:
        for t in taus[::2]:
            points.append(policy_eefp(deploy_stats, t, beta, cnorm, base))
    for lam in lams:
        r = policy_marginal_utility(deploy_stats, lam, gains, marg_norm, base)
        r["gains"] = gains
        points.append(r)
    for p in points:
        p["cost_kind"] = cost_kind
    front = pareto_frontier(points)

    out = {
        "dataset": dataset, "arch": arch, "device": str(device), "ckpt": args.ckpt,
        "synthetic_data": synthetic,
        "git": get_git_hash("."), "vram": vram_info(), "torch": torch_info(),
        "precision": {"dtype": "float32",
                      "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                      "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                      "cudnn_benchmark": bool(torch.backends.cudnn.benchmark)},
        "protocol": {"energy_iters_min": energy_iters, "repeats": repeats,
                     "min_window_s": min_window_s, "warmup": warm_info, "idle_seconds": idle_s},
        "num_heads": int(getattr(model, "num_heads", 1)),
        "flops_per_exit": flops, "flops_per_exit_cascade": flops_cascade,
        "lat_per_exit_s_b1": lat1, "idle_info": idle_info,
        "joules_per_exit_marginal": e1["mean_j"] if e1 else None,
        "cost_kind": cost_kind, "cost_table": base, "marginal_cost_norm": marg_norm,
        "nonpositive_marginal_cost": [bool(v <= 0) for v in marg_raw],
        "gains_calib": gains, "lambdas": lams,
        "lambda_units": "expected accuracy gain per fraction of full-model cost",
        "batch_rows": batch_rows, "points": points, "pareto_idx": front,
        "divergence": (b1 or {}).get("divergence_prefix", {"available": False}),
    }
    save_json(out, results / f"eval_{dataset}_{arch}.json")
    print(f"[evaluate] heads={out['num_heads']} flops={flops} cascade_flops={flops_cascade}", flush=True)
    print(f"[evaluate] cost={base} ({cost_kind})", flush=True)
    print(f"[evaluate] pareto={front} divergence(b1,prefix)={out['divergence']}", flush=True)
    for p in points:
        if "tau" in p:
            extra = f"tau={p['tau']}"
        elif "lambda" in p:
            extra = f"lambda={p['lambda']:.4g}"
        else:
            extra = f"eta={p.get('eta')}"
        print(f"  {p['policy']:16s} {extra:16s} acc={p['acc']:.4f} "
              f"exit_rate={p['exit_rate']:.3f} avg_cost={p['avg_cost']:.6g}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
