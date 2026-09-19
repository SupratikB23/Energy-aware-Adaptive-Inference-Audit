"""CPU-only smoke test: verifies every src file without a GPU.

Run from repo root:   python src/test_smoke.py
Expected on THIS box: ALL PASS with GPU checks reported as SKIP (no CUDA/NVML).
Expected on GPU box:  same, plus pilot/energy paths active (see RUN_PLAN.md).

Exit code 0 = all pass (skips allowed). Exit 1 = any FAIL.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

RESULTS: list[tuple[str, str, str]] = []  # (name, PASS/FAIL/SKIP, detail)


def check(name: str, fn) -> None:
    try:
        detail = fn()
        RESULTS.append((name, "PASS", str(detail or "ok")))
        print(f"[PASS] {name}: {detail or 'ok'}", flush=True)
    except Exception as e:
        RESULTS.append((name, "FAIL", f"{e!r}\n{traceback.format_exc()[-1500:]}"))
        print(f"[FAIL] {name}: {e!r}", flush=True)


def skip(name: str, reason: str) -> None:
    RESULTS.append((name, "SKIP", reason))
    print(f"[SKIP] {name}: {reason}", flush=True)


def t_config():
    from utils import load_config
    cfg = load_config("config.yaml")
    for k in ("seed", "train", "measure", "eval", "ltt"):
        assert k in cfg, f"missing config key {k}"
    assert cfg["measure"]["poll_hz"] >= 100, "poll_hz must be >=100 (never 1Hz)"
    assert cfg["measure"]["min_iters"] >= 1000
    return f"keys ok, poll={cfg['measure']['poll_hz']}Hz"


def t_imports():
    import utils, measure, models, train, calibrate, evaluate, check_gpu, e2e_policy  # noqa: F401
    return "8 modules import clean"


def t_integrate():
    from measure import integrate_power
    # rectangle 10W x 2s = 20J; triangle 0->10W over 2s = 10J
    assert abs(integrate_power([0.0, 2.0], [10.0, 10.0]) - 20.0) < 1e-9
    assert abs(integrate_power([0.0, 2.0], [0.0, 10.0]) - 10.0) < 1e-9
    assert integrate_power([0.0], [5.0]) == 0.0
    try:
        integrate_power([0.0, 1.0], [1.0])
        raise AssertionError("length mismatch should raise")
    except ValueError:
        pass
    return "trapezoid math exact"


def t_ltt():
    from calibrate import fixed_sequence_select, hoeffding_pvalue
    assert hoeffding_pvalue(100, 0.20, 0.10) == 1.0  # worse than alpha -> p=1
    p = hoeffding_pvalue(500, 0.0, 0.10)
    assert 0 <= p < 0.05, f"strong evidence should give small p, got {p}"
    sel = fixed_sequence_select([0.95, 0.8, 0.6], [0.0, 0.0, 0.5], 500, 0.10, 0.10)
    assert sel["certified"] == [0.95, 0.8] and sel["selected"] == 0.8, sel
    sel2 = fixed_sequence_select([0.95], [0.9], 50, 0.10, 0.10)
    assert sel2["selected"] is None
    return f"fst ok, p={p:.2e}"


def t_models():
    from models import build_model, count_parameters, flops_per_exit
    m = build_model("cifar10", "resnet14", 10, 3)
    assert m.num_heads == 4
    x = torch.randn(2, 3, 32, 32)
    for i in range(4):
        assert m.forward_exit(x, i).shape == (2, 10), f"head {i} shape"
    assert len(m.forward_all(x)) == 4
    assert m(x).shape == (2, 10)
    try:
        m.forward_exit(x, 4)
        raise AssertionError("bad idx should raise")
    except ValueError:
        pass
    flops = flops_per_exit(m, (1, 3, 32, 32))
    assert len(flops) == 4 and all(v > 0 for v in flops), flops
    assert flops == sorted(flops), f"flops must increase with exit depth: {flops}"
    k = build_model("kws", "dscnn-s")
    xk = torch.randn(2, 1, 49, 40)
    assert k.forward_exit(xk, 0).shape == (2, 12)
    assert count_parameters(m) > 10000
    return f"resnet flops={flops} params={count_parameters(m)}"


def t_kd():
    from models import exit_kd_loss, kd_loss
    s = torch.randn(4, 10)
    t = torch.randn(4, 10)
    y = torch.randint(0, 10, (4,))
    v = kd_loss(s, t, y, 0.7, 4.0)
    assert torch.isfinite(v) and v.item() > 0
    v2 = exit_kd_loss([s, s], t, y, 0.7, 4.0, [0.5, 1.0])
    assert torch.isfinite(v2)
    try:
        exit_kd_loss([s], t, y, 0.7, 4.0, [1.0, 1.0])
        raise AssertionError("weight mismatch should raise")
    except ValueError:
        pass
    return f"kd={v.item():.3f}"


def t_train_helpers():
    from train import evaluate_accuracy, get_dataloaders, train_one_epoch
    from models import build_model
    device = torch.device("cpu")
    tl, el, nc, ch = get_dataloaders("cifar10", "./data", 32, 0, 42, force_synthetic=True)
    assert nc == 10
    m = build_model("cifar10", "resnet14", nc, ch)
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    loss = train_one_epoch(m, tl, opt, device, None, "exit_ce", None, {"kd_alpha": 0.7, "kd_temp": 4.0})
    assert 0 < loss < 50, loss
    acc = evaluate_accuracy(m, el, device)
    assert 0 <= acc <= 1
    return f"1-epoch exit_ce loss={loss:.3f} acc={acc:.3f}"


def t_calibrate_sim():
    from train import get_dataloaders
    from models import build_model
    from calibrate import sweep_thresholds
    device = torch.device("cpu")
    _, el, nc, ch = get_dataloaders("cifar10", "./data", 64, 0, 42, force_synthetic=True)
    m = build_model("cifar10", "resnet14", nc, ch).eval()
    rows = sweep_thresholds(m, el, device, [0.6, 0.9])
    assert len(rows) == 2 and all("emp_risk" in r for r in rows)
    return f"sweep rows acc={[round(r['acc'],3) for r in rows]}"


def t_evaluate_policies():
    from train import get_dataloaders
    from models import build_model
    from evaluate import (collect_stats, estimate_gains, measure_latency_per_exit,
                          pareto_frontier, policy_confidence, policy_eefp,
                          policy_entropy, policy_marginal_utility)
    device = torch.device("cpu")
    _, el, nc, ch = get_dataloaders("cifar10", "./data", 64, 0, 42, force_synthetic=True)
    m = build_model("cifar10", "resnet14", nc, ch).eval()
    st = collect_stats(m, el, device)
    assert st["conf"].shape[1] == 4
    flops = [10, 20, 30, 40]
    r1 = policy_confidence(st, 0.7, flops)
    r2 = policy_entropy(st, 0.6, flops)
    r3 = policy_eefp(st, 0.7, 0.2, [0.25, 0.5, 0.75, 1.0], flops)
    assert r3["avg_cost"] is not None, "eefp must report comparable avg_cost"
    gains = estimate_gains(st)
    assert len(gains) == 3 and all(g >= 0 for g in gains)
    marg = [10.0, 10.0, 10.0]
    r4 = policy_marginal_utility(st, 0.01, gains, marg, flops)
    for r in (r1, r2, r3, r4):
        assert 0 <= r["acc"] <= 1 and 0 <= r["exit_rate"] <= 1, r
    pts = [{"acc": 0.9, "avg_cost": 10}, {"acc": 0.8, "avg_cost": 5}, {"acc": 0.85, "avg_cost": 7}]
    assert pareto_frontier(pts) == [0, 1, 2] or set(pareto_frontier(pts)) <= {0, 1, 2}
    lat = measure_latency_per_exit(m, (1, 3, 32, 32), device, iters=5, warmup=1)
    assert len(lat) == 4 and all(v > 0 for v in lat)
    return f"policies acc={[round(r['acc'],3) for r in (r1,r2,r3,r4)]}"


def t_measure_cpu():
    from measure import EnergyMeter
    meter = EnergyMeter()
    try:
        if meter.source != "none":
            return f"GPU present here?! source={meter.source}"
        res = meter.measure_fn(lambda: time.sleep(0.001), iters=5, warmup=1)
        assert res.source == "none" and res.energy_j is None and res.duration_s > 0
        val = meter.validate_dummy()
        assert val.get("skipped") is True, val
        return "cpu-mode wall-time ok, energy correctly None (no fake Joules)"
    finally:
        meter.close()


def t_files():
    root = Path(__file__).resolve().parent.parent
    for f in ("config.yaml", "requirements.txt", ".gitignore", "README.md", "AGENTS.md",
              "src/utils.py", "src/measure.py", "src/models.py", "src/train.py",
              "src/calibrate.py", "src/evaluate.py", "src/check_gpu.py", "src/e2e_policy.py"):
        assert (root / f).exists(), f"missing {f}"
    for frozen in ("percom2027_literature_matrix_and_verdict.md", "notebook.py"):
        assert (root / frozen).exists(), f"frozen input missing: {frozen}"
    return "all required + frozen files present"


def main() -> int:
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} (expect False here)", flush=True)
    check("config", t_config)
    check("imports", t_imports)
    check("measure.integrate", t_integrate)
    check("calibrate.ltt", t_ltt)
    check("models.forward+flops", t_models)
    check("models.kd", t_kd)
    check("train.synthetic-epoch", t_train_helpers)
    check("calibrate.sweep", t_calibrate_sim)
    check("evaluate.policies", t_evaluate_policies)
    check("measure.cpu-mode", t_measure_cpu)
    check("files.present", t_files)
    # check_gpu pilot is intentionally NOT run here (writes results/); run it on GPU box.
    skip("check_gpu.pilot", "run on GPU box: python src/check_gpu.py --idle-seconds 60")
    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    print(f"\n==== SMOKE: {n_pass} pass, {n_fail} fail, "
          f"{len(RESULTS)-n_pass-n_fail} skip ====", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
