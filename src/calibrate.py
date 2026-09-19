"""Threshold sweep + Learn-Then-Test (LTT) fixed-sequence calibration.

Accuracy-risk LTT is implemented now (C3 support). Energy-risk LTT is a thin
wrapper over the same machinery and is marked EXPERIMENTAL / C4-deferred: it
requires validated per-exit Joules from src/measure.py on the GPU box.

Statistics (Angelopoulos et al., LTT):
  Per-sample loss L_i = 1[policy wrong AND full model right] in {0,1}.
  R(lam) = E[L] upper-bounds the accuracy drop vs the full model and is
  bounded in [0,1], which Hoeffding requires. (The naive per-sample
  acc-drop loss 1[pol wrong] - 1[full wrong] lives in {-1,0,1}; plugging it
  into the [0,1] Hoeffding bound is anti-conservative.)
  H0(lam): R(lam) > alpha. Hoeffding p-value:
      p = exp(-2 n max(0, alpha - R_hat)^2)
  Fixed-sequence testing from most-conservative to most-aggressive at level
  delta controls FWER without further correction. We return the certified set
  and pick the most aggressive (cheapest) certified candidate.

Exchangeability warning: the guarantee is void if calibration and deployment
device states differ (cool/idle vs hot/loaded). That violation IS the C4
experiment — never hide it.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from utils import ensure_dir, load_config, resolve_device, save_json, set_seed
except ImportError:
    from src.utils import ensure_dir, load_config, resolve_device, save_json, set_seed

try:
    from models import build_model, default_io, model_from_checkpoint
except ImportError:
    from src.models import build_model, default_io, model_from_checkpoint

try:
    from train import get_dataloaders, is_synthetic
except ImportError:
    from src.train import get_dataloaders, is_synthetic


# ----------------------------------------------------------------------------
# LTT core (pure statistics, GPU-free, unit-tested)
# ----------------------------------------------------------------------------
def hoeffding_pvalue(n: int, emp_risk: float, alpha: float) -> float:
    """Super-uniform p-value for H0: true risk > alpha given empirical risk."""
    if n <= 0:
        raise ValueError("n must be > 0")
    gap = alpha - float(emp_risk)
    if gap <= 0:
        return 1.0
    return float(math.exp(-2.0 * n * gap * gap))


def fixed_sequence_select(ordered_lambdas: list, emp_risks: list[float],
                          n: int, alpha: float, delta: float) -> dict:
    """Test most-conservative -> most-aggressive; stop at first non-rejection.

    Returns {certified: [...], selected: lam|None, pvalues: [...]} where
    selected is the most aggressive certified candidate (or None).
    """
    if len(ordered_lambdas) != len(emp_risks):
        raise ValueError("lambdas and risks must align")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    certified, pvals = [], []
    for lam, r in zip(ordered_lambdas, emp_risks):
        p = hoeffding_pvalue(n, r, alpha)
        pvals.append(p)
        if p <= delta:
            certified.append(lam)
        else:
            break
    return {"certified": certified,
            "selected": certified[-1] if certified else None,
            "pvalues": pvals}


# ----------------------------------------------------------------------------
# Policy simulation (confidence exits) — shared with evaluate.py logic
# ----------------------------------------------------------------------------
@torch.no_grad()
def simulate_confidence(model, loader: DataLoader, device: torch.device,
                        tau: float) -> dict:
    """Run confidence policy: exit at first head with max-softmax >= tau.

    Returns {n, acc, full_acc, acc_drop, extra_err, exit_rate, avg_exit,
    per_head_counts}. extra_err = P(policy wrong AND full right), the bounded
    LTT loss. All computed vs the full-model head on the SAME loader.
    """
    model.eval()
    n_heads = int(getattr(model, "num_heads", 1))
    correct_pol, correct_full, extra, total = 0, 0, 0, 0
    counts = [0] * n_heads
    for x, y in loader:
        x = x.to(device)
        heads = model.forward_all(x) if hasattr(model, "forward_all") else [model(x)]
        probs = torch.stack([F.softmax(h.float(), dim=1) for h in heads], 1).cpu()  # [B,H,C]
        confs, preds = probs.max(2)                                               # [B,H]
        y = y.cpu()
        # first head (excluding final) with conf >= tau, else final head
        chosen = torch.full_like(y, n_heads - 1)
        if n_heads > 1:
            hit = confs[:, :-1] >= tau
            chosen = torch.where(hit.any(1), hit.int().argmax(1), chosen)
        pol_ok = preds.gather(1, chosen[:, None]).squeeze(1) == y
        full_ok = preds[:, -1] == y
        correct_pol += int(pol_ok.sum())
        correct_full += int(full_ok.sum())
        extra += int((~pol_ok & full_ok).sum())
        counts = [c + int((chosen == h).sum()) for h, c in enumerate(counts)]
        total += int(y.numel())
    acc = correct_pol / max(1, total)
    full_acc = correct_full / max(1, total)
    return {"n": total, "acc": acc, "full_acc": full_acc,
            "acc_drop": full_acc - acc, "extra_err": extra / max(1, total),
            "exit_rate": sum(counts[:-1]) / max(1, total),
            "avg_exit": sum(c * i for i, c in enumerate(counts)) / max(1, total),
            "per_head_counts": counts}


def sweep_thresholds(model, loader, device, thresholds: list[float]) -> list[dict]:
    rows = []
    for tau in thresholds:
        r = simulate_confidence(model, loader, device, float(tau))
        r["tau"] = float(tau)
        # LTT risk: mean of the bounded {0,1} loss 1[policy wrong AND full right]
        # (>= acc_drop), so the Hoeffding p-value in fixed_sequence_select is valid.
        r["emp_risk"] = r["extra_err"]
        rows.append(r)
    return rows


def calibrate_accuracy_risk(rows: list[dict], alpha: float, delta: float) -> dict:
    """rows from sweep on the CALIBRATION split. Most-conservative = largest tau."""
    ordered = sorted(rows, key=lambda r: -r["tau"])
    lams = [r["tau"] for r in ordered]
    risks = [r["emp_risk"] for r in ordered]
    n = int(ordered[0]["n"]) if ordered else 0
    sel = fixed_sequence_select(lams, risks, n, alpha, delta)
    return {"alpha": alpha, "delta": delta, "n": n,
            "ordered_taus": lams, "emp_risks": risks, **sel, "rows": rows}


def calibrate_energy_risk(per_tau_joules: list[float], taus: list[float],
                          budget_j: float, alpha: float, delta: float, n: int) -> dict:
    """EXPERIMENTAL (C4). Risk = P(per-sample energy > budget).

    per_tau_joules here must be per-sample measured Joules (list per tau) or
    caller passes empirical violation rates directly via emp_risks. This helper
    accepts mean energies and converts with a Markov bound placeholder ONLY for
    plumbing tests — DO NOT publish numbers from this path without real
    per-sample energy distributions from measure.py.
    """
    # Plumbing: empirical violation rate unknown from means alone; caller should
    # pass real rates. We return the selection machinery with a clear flag.
    return {"experimental": True,
            "warning": "C4-deferred: requires per-sample measured Joules; do not publish from means.",
            "budget_j": budget_j, "alpha": alpha, "delta": delta, "n": n,
            "taus": list(taus)}


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _split_loader(loader: DataLoader, frac: float, seed: int):
    """Deterministic 2-way split of a loader's underlying dataset (calib/test)."""
    import torch as _t
    ds = loader.dataset
    n = len(ds)
    n1 = int(n * frac)
    g = _t.Generator().manual_seed(seed)
    idx = _t.randperm(n, generator=g).tolist()
    from torch.utils.data import Subset
    l1 = DataLoader(Subset(ds, idx[:n1]), batch_size=loader.batch_size, shuffle=False)
    l2 = DataLoader(Subset(ds, idx[n1:]), batch_size=loader.batch_size, shuffle=False)
    return l1, l2


def main() -> int:
    ap = argparse.ArgumentParser(description="LTT calibration for exit thresholds")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ckpt", default=None, help="trained exit model checkpoint")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--results", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(args.device or cfg.get("device", "auto"))
    dataset = (args.dataset or cfg.get("dataset", "cifar10")).lower()
    alpha = float(args.alpha if args.alpha is not None else cfg.get("ltt", {}).get("alpha", 0.01))
    delta = float(args.delta if args.delta is not None else cfg.get("ltt", {}).get("delta", 0.10))
    taus = [float(t) for t in cfg.get("eval", {}).get("thresholds", [0.6, 0.7, 0.8, 0.9])]
    results = Path(args.results or cfg.get("results_root", "./results"))
    ensure_dir(results)

    if args.ckpt:
        model, arch = model_from_checkpoint(args.ckpt, dataset, device, arch=args.arch,
                                            fallback_arch=cfg.get("student", "resnet14"))
    else:
        # No checkpoint: calibration plumbing demo on a random-init net.
        print("[calibrate] no --ckpt given; using random-init net (plumbing only).", flush=True)
        arch = args.arch or cfg.get("student", "resnet14")
        nc, ch, _ = default_io(dataset)
        model = build_model(dataset, arch, nc, ch).to(device)
    model.eval()

    _, test_loader, _, _ = get_dataloaders(
        dataset, cfg.get("data_root", "./data"), 128, 0,
        int(cfg.get("seed", 42)), force_synthetic=args.synthetic or args.ckpt is None)
    calib_loader, deploy_loader = _split_loader(
        test_loader, float(cfg.get("ltt", {}).get("calib_split", 0.5)), int(cfg.get("seed", 42)))

    rows = sweep_thresholds(model, calib_loader, device, taus)
    sel = calibrate_accuracy_risk(rows, alpha, delta)
    # Honest check: evaluate selected tau on the held-out deploy split.
    deploy_check = None
    if sel["selected"] is not None:
        deploy_check = simulate_confidence(model, deploy_loader, device, float(sel["selected"]))
    out = {"dataset": dataset, "arch": arch, "device": str(device), "ckpt": args.ckpt,
           "synthetic_data": is_synthetic(test_loader),
           "risk": "P(policy wrong AND full right) (bounded {0,1} loss, Hoeffding)",
           "calibration": sel, "deploy_check": deploy_check}
    save_json(out, results / f"ltt_{dataset}_{arch}.json")
    print(f"[calibrate] alpha={alpha} delta={delta} selected={sel['selected']} "
          f"certified={sel['certified']}", flush=True)
    if deploy_check:
        print(f"[calibrate] deploy acc={deploy_check['acc']:.4f} "
              f"drop={deploy_check['acc_drop']:.4f} exit_rate={deploy_check['exit_rate']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
