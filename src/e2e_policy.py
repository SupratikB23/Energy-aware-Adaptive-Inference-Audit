"""End-to-end measured energy of early-exit POLICIES on real inputs (C2/C3 core).

Papers price an early-exit policy with an additive table:
    E_policy ~= sum_h P(exit at h) * E_exit[h]          (per-exit table, or FLOPs)
This script MEASURES E_policy directly: the real policy runs over the real deploy
split at each batch size, and the measured Joules/sample are compared with
  (a) the per-exit table prediction (from evaluate.py eval JSONs, same batch), and
  (b) the FLOP prediction (cascade-FLOP fraction x measured full-model energy).

At batch > 1 two runtimes are measured:
  compact   - exited samples are dropped between blocks (index_select; DREX-style rebatching)
  batchwait - the whole batch keeps running until every sample is confident
At batch 1 they are identical, so only `compact` is run.

Policies: full model (no exit heads), global confidence tau, per-head thresholds.
Predictions of the timed runtime are checked against the offline policy (agreement).
Novelty honesty: the exit rules are standard; the contribution is the measured
audit of how they are priced (verdict C1/C2).
"""
from __future__ import annotations

import argparse
import json
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
    from models import default_io, flops_per_exit, model_from_checkpoint
    from measure import EnergyMeter
    from train import get_dataloaders, is_synthetic
    from evaluate import _chosen_perhead, collect_stats, gpu_warmup
except ImportError:  # pragma: no cover
    from src.utils import (cuda_sync, ensure_dir, get_git_hash, load_config, resolve_device,
                           save_json, set_seed, torch_info, vram_info)
    from src.models import default_io, flops_per_exit, model_from_checkpoint
    from src.measure import EnergyMeter
    from src.train import get_dataloaders, is_synthetic
    from src.evaluate import _chosen_perhead, collect_stats, gpu_warmup


# ----------------------------------------------------------------------------
# Runtimes
# ----------------------------------------------------------------------------
@torch.no_grad()
def run_policy_batch(model, x: torch.Tensor, taus_h, mode: str) -> torch.Tensor:
    """Run one batch through the early-exit runtime. Returns per-sample predictions
    (in input order). taus_h=None => full model, no exit heads evaluated."""
    n = x.shape[0]
    H = int(model.num_heads)
    if taus_h is None:
        return model.forward_exit(x, H - 1).argmax(1)
    pred = torch.empty(n, dtype=torch.long, device=x.device)
    alive = torch.arange(n, device=x.device)          # original indices still running
    decided = torch.zeros(n, dtype=torch.bool, device=x.device)
    h = x
    for k, (blk, head) in enumerate(model.exit_blocks()):
        h = blk(h)
        logits = head(h)
        if k == H - 1:
            if mode == "compact":
                pred[alive] = logits.argmax(1)
            else:
                pred[~decided] = logits.argmax(1)[~decided]
            break
        conf, p = F.softmax(logits, dim=1).max(1)
        exit_now = conf >= float(taus_h[k])
        if mode == "compact":
            pred[alive[exit_now]] = p[exit_now]
            keep = (~exit_now).nonzero().squeeze(1)      # host sync: the exit decision
            if keep.numel() == 0:
                break
            alive = alive.index_select(0, keep)
            h = h.index_select(0, keep)
        elif mode == "batchwait":
            newly = exit_now & ~decided
            pred[newly] = p[newly]
            decided |= exit_now
            if bool(decided.all()):                     # host sync: stop only when ALL exited
                break
        else:
            raise ValueError(f"unknown mode {mode!r}")
    return pred


def make_pass(model, X: torch.Tensor, batch: int, taus_h, mode: str):
    """No-arg callable: one pass of the runtime over all rows of X in chunks of `batch`."""
    N = X.shape[0]

    def one_pass():
        for s in range(0, N, batch):
            run_policy_batch(model, X[s:s + batch], taus_h, mode)
    return one_pass


def _passes_for_window(fn, min_window_s: float) -> tuple[int, float]:
    cuda_sync()
    t0 = time.perf_counter()
    fn()
    cuda_sync()
    per = max(1e-6, time.perf_counter() - t0)
    return max(1, math.ceil(min_window_s / per)), per


# ----------------------------------------------------------------------------
# Predictions (the additive models under audit)
# ----------------------------------------------------------------------------
def pooled_tables(paths: list[str] | None) -> dict:
    """{batch: mean per-exit cascade J/sample} pooled over eval JSONs."""
    out: dict[int, list] = {}
    for p in paths or []:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        for r in d.get("batch_rows", []):
            e = r.get("energy_cascade") or {}
            if r.get("oom") or e.get("skipped") or any(v is None for v in e.get("mean_j", [None])):
                continue
            out.setdefault(int(r["batch"]), []).append(e["mean_j"])
    return {b: [float(v) for v in np.mean(np.array(t, dtype=float), 0)] for b, t in out.items()}


def perhead_from_eval(path: str, drops_pts=(0.3, 1.0, 2.5)) -> list[list[float]]:
    """Cheapest calib-tuned per-head settings in an evaluate.py JSON whose deploy
    accuracy is within `drops_pts` of the full model (one per target, de-duplicated)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    full = d.get("deploy_full_acc")
    pts = [p for p in d.get("points", []) if p.get("policy") == "confidence-perhead"]
    if full is None or not pts:
        raise ValueError(f"{path}: no confidence-perhead points / deploy_full_acc (re-run evaluate.py v2)")
    out: list[list[float]] = []
    for dp in drops_pts:
        ok = [p for p in pts if p["acc"] >= full - dp / 100.0]
        if ok:
            best = min(ok, key=lambda p: p["avg_cost"])["taus_h"]
            if best not in out:
                out.append([float(t) for t in best])
    return out


def exit_fractions(chosen: np.ndarray, H: int) -> list[float]:
    return [float((chosen == h).mean()) for h in range(H)]


def summarize_prediction(measured: float | None, predicted: float | None) -> float | None:
    if measured is None or predicted is None or measured == 0:
        return None
    return float(100.0 * (predicted - measured) / measured)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="End-to-end measured energy of early-exit policies")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--arch", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batches", nargs="*", type=int, default=None)
    ap.add_argument("--taus", nargs="*", type=float, default=[0.9, 0.97, 0.99],
                    help="global confidence thresholds (default: LTT-selected alpha=0.05/0.03/0.02)")
    ap.add_argument("--perhead", nargs="*", default=[],
                    help='per-head thresholds, one string per policy, e.g. "0.99,0.97,0.9"')
    ap.add_argument("--perhead-from", default=None,
                    help="evaluate.py JSON: add its cheapest per-head settings within 0.3/1.0/2.5 pts of full acc")
    ap.add_argument("--modes", nargs="*", default=["compact", "batchwait"])
    ap.add_argument("--eval-json", nargs="*", default=None,
                    help="evaluate.py outputs: their per-exit cascade tables give the additive prediction")
    ap.add_argument("--max-samples", type=int, default=2500, help="deploy rows used for energy passes")
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--min-window-s", type=float, default=None)
    ap.add_argument("--warmup-seconds", type=float, default=None)
    ap.add_argument("--idle-seconds", type=float, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--results", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    mc, ec = dict(cfg.get("measure", {})), dict(cfg.get("eval", {}))
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(args.device or cfg.get("device", "auto"))
    dataset = (args.dataset or cfg.get("dataset", "cifar10")).lower()
    batches = args.batches or ec.get("batches", [1, 8, 16, 32, 64])
    repeats = int(args.repeats or ec.get("repeats", 3))
    min_window_s = float(args.min_window_s if args.min_window_s is not None else mc.get("min_window_seconds", 3.0))
    warmup_s = float(args.warmup_seconds if args.warmup_seconds is not None else mc.get("warmup_seconds", 120))
    idle_s = float(args.idle_seconds if args.idle_seconds is not None else mc.get("idle_seconds", 60))
    results = Path(args.results or cfg.get("results_root", "./results"))
    ensure_dir(results)

    model, arch = model_from_checkpoint(args.ckpt, dataset, device, arch=args.arch,
                                        fallback_arch=cfg.get("student", "resnet14"))
    model.eval()
    H = int(model.num_heads)
    nc, ch, (Hh, Ww) = default_io(dataset)

    _, test_loader, _, _ = get_dataloaders(dataset, cfg.get("data_root", "./data"), 128, 0,
                                           int(cfg.get("seed", 42)), force_synthetic=args.synthetic)
    synthetic = is_synthetic(test_loader)
    from torch.utils.data import DataLoader, Subset
    n = len(test_loader.dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(int(cfg.get("seed", 42)))).tolist()
    deploy_idx = idx[n // 2:]                              # same deploy split as evaluate/calibrate
    deploy_loader = DataLoader(Subset(test_loader.dataset, deploy_idx), batch_size=128)
    stats = collect_stats(model, deploy_loader, device)   # offline policy truth (full deploy split)

    m = min(int(args.max_samples), len(deploy_idx))
    xs, ys = [], []
    for x, y in DataLoader(Subset(test_loader.dataset, deploy_idx[:m]), batch_size=256):
        xs.append(x)
        ys.append(y)
    X = torch.cat(xs).to(device)
    Y = torch.cat(ys).to(device)
    sub = {k: stats[k][:m] for k in ("conf", "correct")}  # same first m rows, same order

    policies: list[dict] = [{"name": "full", "taus_h": None}]
    for t in args.taus:
        policies.append({"name": f"conf-{t:g}", "taus_h": [float(t)] * (H - 1)})
    perhead = [[float(v) for v in s.split(",")] for s in args.perhead]
    if args.perhead_from:
        perhead += [t for t in perhead_from_eval(args.perhead_from) if t not in perhead]
    for th in perhead:
        if len(th) != H - 1:
            raise ValueError(f"per-head thresholds {th} need {H - 1} values for {H} heads")
        policies.append({"name": "perhead-" + "/".join(f"{v:g}" for v in th), "taus_h": th})
    for p in policies:
        ch_full = (np.full(stats["conf"].shape[0], H - 1) if p["taus_h"] is None
                   else _chosen_perhead(stats["conf"], p["taus_h"]))
        ch_sub = ch_full[:m]
        p["acc_deploy"] = float(stats["correct"][np.arange(len(ch_full)), ch_full].mean())
        p["exit_frac_deploy"] = exit_fractions(ch_full, H)
        p["exit_frac_energy_subset"] = exit_fractions(ch_sub, H)
        p["acc_energy_subset"] = float(sub["correct"][np.arange(m), ch_sub].mean())

    # correctness: the timed runtime must implement the offline policy
    for p in policies:
        for mode in (["compact"] if p["taus_h"] is None else args.modes):
            pr = torch.cat([run_policy_batch(model, X[s:s + 64], p["taus_h"], mode)
                            for s in range(0, m, 64)])
            p.setdefault("runtime_acc", {})[mode] = float((pr == Y).float().mean())
    flops_c = flops_per_exit(model, (1, ch, Hh, Ww), device="cpu", cascade=True)
    tables = pooled_tables(args.eval_json)

    meter = EnergyMeter(poll_hz=float(mc.get("poll_hz", 150.0))) if device.type == "cuda" else None
    idle_p, idle_info, warm = None, {"skipped": True}, {"skipped": True}
    rows: list[dict] = []
    try:
        if meter is not None and meter.source != "none":
            cuda_sync()
            time.sleep(5.0)
            print(f"[e2e] idle baseline {idle_s}s ...", flush=True)
            idle = meter.measure_idle(seconds=idle_s)
            idle_p = idle.get("idle_power_w")
            idle_info = {"idle_power_w": idle_p, "seconds": idle_s, "source": meter.source}
            print(f"[e2e] idle={idle_p} W; warmup {warmup_s}s ...", flush=True)
            warm = gpu_warmup(warmup_s)
        for b in batches:
            cells = [(p, mode) for p in policies
                     for mode in (["compact"] if (p["taus_h"] is None or b == 1) else args.modes)]
            fns, passes = [], []
            for p, mode in cells:
                fn = make_pass(model, X, b, p["taus_h"], mode)
                fn()  # warm (cudnn autotune for every shape the runtime produces)
                k, _ = _passes_for_window(fn, min_window_s)
                fns.append(fn)
                passes.append(k)
            samples = {i: [] for i in range(len(cells))}
            lat = {i: [] for i in range(len(cells))}
            clocks = {i: [] for i in range(len(cells))}
            for r in range(repeats):
                for j in range(len(cells)):
                    i = (j + r) % len(cells)
                    if meter is not None and meter.source != "none" and idle_p is not None:
                        res = meter.measure_fn(fns[i], iters=passes[i], warmup=0, idle_power_w=idle_p)
                        if res.per_iter_marginal_j is not None:
                            samples[i].append(res.per_iter_marginal_j / m)
                        lat[i].append(1e3 * res.duration_s / (passes[i] * m))
                        clocks[i].append(res.device_after.get("clock_graphics_mhz"))
                    else:
                        cuda_sync()
                        t0 = time.perf_counter()
                        for _ in range(passes[i]):
                            fns[i]()
                        cuda_sync()
                        lat[i].append(1e3 * (time.perf_counter() - t0) / (passes[i] * m))
            full_i = next(i for i, (p, _) in enumerate(cells) if p["taus_h"] is None)
            e_full = float(np.mean(samples[full_i])) if samples[full_i] else None
            for i, (p, mode) in enumerate(cells):
                e = float(np.mean(samples[i])) if samples[i] else None
                sd = float(np.std(samples[i], ddof=1)) if len(samples[i]) > 1 else None
                f = p["exit_frac_energy_subset"]
                tab = tables.get(b)
                pred_table = (None if (p["taus_h"] is None or tab is None)
                              else float(sum(fh * eh for fh, eh in zip(f, tab))))
                if p["taus_h"] is None:
                    pred_flops = e_full
                elif e_full is not None:
                    pred_flops = float(sum(fh * fl for fh, fl in zip(f, flops_c)) / flops_c[-1] * e_full)
                else:
                    pred_flops = None
                row = {"batch": b, "policy": p["name"], "mode": mode, "taus_h": p["taus_h"],
                       "acc_deploy": p["acc_deploy"], "exit_frac": f, "passes": passes[i],
                       "j_per_sample": e, "j_std": sd,
                       "cv_pct": (100 * sd / e) if (sd is not None and e) else None,
                       "ms_per_sample": float(np.mean(lat[i])) if lat[i] else None,
                       "clock_mhz": [c for c in clocks[i] if c is not None],
                       "saving_vs_full_pct": (100 * (e_full - e) / e_full) if (e is not None and e_full) else None,
                       "pred_table_j": pred_table, "pred_table_err_pct": summarize_prediction(e, pred_table),
                       "pred_flops_j": pred_flops, "pred_flops_err_pct": summarize_prediction(e, pred_flops)}
                rows.append(row)
                ej = "n/a" if e is None else f"{1e3 * e:.3f} mJ"
                print(f"[e2e] b={b:<3d} {p['name']:24s} {mode:9s} acc={p['acc_deploy']:.4f} E={ej} "
                      f"save={row['saving_vs_full_pct'] if row['saving_vs_full_pct'] is None else round(row['saving_vs_full_pct'], 1)}% "
                      f"tableErr={row['pred_table_err_pct'] if row['pred_table_err_pct'] is None else round(row['pred_table_err_pct'], 1)}% "
                      f"flopErr={row['pred_flops_err_pct'] if row['pred_flops_err_pct'] is None else round(row['pred_flops_err_pct'], 1)}% "
                      f"lat={row['ms_per_sample']:.4f} ms/sample", flush=True)
    finally:
        if meter is not None:
            meter.close()

    # regime summary: cheapest per-sample energy of each policy at batch 1 vs full model at any batch
    def _e(b, name, mode="compact"):
        r = next((r for r in rows if r["batch"] == b and r["policy"] == name and r["mode"] == mode), None)
        return r["j_per_sample"] if r else None
    regime = {"full_by_batch": {b: _e(b, "full") for b in batches},
              "policies_at_b1": {p["name"]: _e(1, p["name"]) for p in policies} if 1 in batches else {}}
    out = {"dataset": dataset, "arch": arch, "ckpt": args.ckpt, "device": str(device),
           "synthetic_data": synthetic, "git": get_git_hash("."), "torch": torch_info(), "vram": vram_info(),
           "energy_subset_n": m, "num_heads": H, "flops_cascade": flops_c,
           "protocol": {"repeats": repeats, "min_window_s": min_window_s, "warmup": warm,
                        "idle": idle_info, "modes": args.modes},
           "eval_json": args.eval_json, "table_batches": sorted(tables),
           "policies": policies, "rows": rows, "regime": regime}
    save_json(out, results / f"e2e_{dataset}_{arch}.json")
    bad = [(p["name"], md, a) for p in policies for md, a in p["runtime_acc"].items()
           if abs(a - p["acc_energy_subset"]) > 0.005]
    if bad:
        print(f"[e2e] WARNING runtime != offline policy accuracy (>0.5 pt): {bad}", flush=True)
    print(f"[e2e] saved {results / f'e2e_{dataset}_{arch}.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
