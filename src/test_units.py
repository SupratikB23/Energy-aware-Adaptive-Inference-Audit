"""pytest unit/integration suite (CPU-only). Run from repo root:

    python -m pytest src/test_units.py -q

GPU code paths are exercised with a physics-simulated fake NVML device
(`SimGpu`): true power steps between idle/load, a hardware energy counter
integrates it exactly, and the power readout can be a moving average (the
Ampere+ nvmlDeviceGetPowerUsage behaviour). No GPU or pynvml needed.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

import calibrate  # noqa: E402
import evaluate  # noqa: E402
import measure  # noqa: E402
import models  # noqa: E402
import train  # noqa: E402


# ----------------------------------------------------------------------------
# Fake NVML device
# ----------------------------------------------------------------------------
class SimGpu:
    """Piecewise-constant true power; exact energy counter; averaged power readout."""

    def __init__(self, p_idle=20.0, p_load=200.0, avg_window_s=0.0, poll_noise_w=0.0,
                 has_counter=True):
        self.p_idle, self.p_load = p_idle, p_load
        self.avg_window_s = avg_window_s
        self.poll_noise_w = poll_noise_w
        self.has_counter = has_counter
        self._lock = threading.Lock()
        self._segs = [(time.perf_counter(), p_idle)]  # (t_start, power)
        self._rng = np.random.default_rng(0)

    def set_load(self, on: bool) -> None:
        p = self.p_load if on else self.p_idle
        with self._lock:
            if self._segs[-1][1] != p:
                self._segs.append((time.perf_counter(), p))

    def _energy_between(self, t0: float, t1: float) -> float:
        e = 0.0
        segs = self._segs + [(float("inf"), 0.0)]
        for (ts, p), (tn, _) in zip(segs[:-1], segs[1:]):
            a, b = max(ts, t0), min(tn, t1)
            if b > a:
                e += p * (b - a)
        return e

    # --- Nvml-compatible surface used by EnergyMeter ---
    def read_energy_mj(self) -> int:
        if not self.has_counter:
            raise RuntimeError("not supported")
        with self._lock:
            return int(1000.0 * self._energy_between(self._segs[0][0], time.perf_counter()))

    def read_power_w(self) -> float:
        now = time.perf_counter()
        with self._lock:
            if self.avg_window_s > 0:
                p = self._energy_between(now - self.avg_window_s, now) / self.avg_window_s
            else:  # current segment directly (a 1 ns window loses float precision at high uptime)
                p = self._segs[-1][1]
        if self.poll_noise_w:
            p += float(self._rng.uniform(-self.poll_noise_w, self.poll_noise_w))
        return p

    def device_state(self):
        return measure.DeviceState(gpu_name="SimGPU", power_w=self.read_power_w())

    def has_energy_counter(self) -> bool:
        return self.has_counter

    def close(self) -> None:
        pass


def sim_meter(sim: SimGpu, poll_hz: float = 200.0) -> measure.EnergyMeter:
    m = measure.EnergyMeter(poll_hz=poll_hz)  # CPU box: comes up with source 'none'
    m.nvml, m.has_nvml, m.has_counter = sim, True, sim.has_counter
    return m


def busy(sim: SimGpu, dt: float = 0.002):
    def fn():
        sim.set_load(True)
        time.sleep(dt)
    return fn


# ----------------------------------------------------------------------------
# measure.py
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("t,p,expected", [
    ([0.0, 2.0], [10.0, 10.0], 20.0),
    ([0.0, 2.0], [0.0, 10.0], 10.0),
    ([0.0, 1.0, 3.0], [5.0, 5.0, 5.0], 15.0),
    ([0.0], [5.0], 0.0),
    ([], [], 0.0),
])
def test_integrate_power(t, p, expected):
    assert measure.integrate_power(t, p) == pytest.approx(expected)


@pytest.mark.parametrize("t,p", [([0.0, 1.0], [1.0]), ([1.0, 0.5], [1.0, 1.0])])
def test_integrate_power_rejects_bad_input(t, p):
    with pytest.raises(ValueError):
        measure.integrate_power(t, p)


def test_energy_meter_rejects_1hz_polling():
    with pytest.raises(ValueError):
        measure.EnergyMeter(poll_hz=1.0)


def test_cpu_meter_never_fakes_joules():
    m = measure.EnergyMeter()
    try:
        assert m.source == "none"
        r = m.measure_fn(lambda: None, iters=3, warmup=0, idle_power_w=10.0)
        assert r.energy_j is None and r.marginal_energy_j is None and r.per_iter_j is None
        assert m.validate_dummy()["skipped"] is True
    finally:
        m.close()


def test_meter_start_stop_misuse():
    m = measure.EnergyMeter()
    with pytest.raises(RuntimeError):
        m.stop()
    m.start()
    with pytest.raises(RuntimeError):
        m.start()
    m.stop()


def test_counter_energy_and_idle_subtraction_are_correct():
    sim = SimGpu(p_idle=20.0, p_load=200.0)
    m = sim_meter(sim)
    r = m.measure_fn(busy(sim), iters=200, warmup=1, idle_power_w=20.0)
    sim.set_load(False)
    assert r.source == "counter"
    assert r.energy_j == pytest.approx(200.0 * r.duration_s, rel=0.02)
    assert r.marginal_energy_j == pytest.approx(180.0 * r.duration_s, rel=0.02)
    assert r.per_iter_marginal_j == pytest.approx(r.marginal_energy_j / 200)
    assert r.samples > 10 and r.energy_poll_j == pytest.approx(r.energy_j, rel=0.05)


def test_validation_passes_on_good_telemetry():
    sim = SimGpu(avg_window_s=0.3)
    v = sim_meter(sim).validate_dummy(seconds=1.5, prewarm_seconds=0.6, workload=busy(sim))
    assert v["basis"] == "counter-vs-poll"
    assert v["passed"] is True and v["residual"] < 0.05, v


def test_validation_fails_on_wrong_poll_telemetry():
    sim = SimGpu()
    sim.read_power_w = lambda: 500.0  # stuck/wrong sensor vs 200 W truth
    v = sim_meter(sim).validate_dummy(seconds=1.0, prewarm_seconds=0.1, workload=busy(sim))
    assert v["passed"] is False and v["residual"] > 0.05


def test_poll_only_validation_is_not_a_vacuous_pass():
    """Regression: used to return passed=True, residual=0.0 for ANY telemetry."""
    sim = SimGpu(has_counter=False)
    sim.read_power_w = lambda: float(np.random.uniform(1, 500))  # garbage
    v = sim_meter(sim).validate_dummy(seconds=0.5, prewarm_seconds=0.0, workload=busy(sim))
    assert v["passed"] is False
    assert v["basis"] == "poll-only-unverifiable"


def test_prewarm_removes_averaging_lag_bias():
    """Ampere+ power readout is a moving average: a cold-start window biases poll low."""
    kw = dict(p_idle=20.0, p_load=200.0, avg_window_s=0.5)
    sim_cold = SimGpu(**kw)
    cold = sim_meter(sim_cold).validate_dummy(seconds=1.5, prewarm_seconds=0.0,
                                               workload=busy(sim_cold))
    sim_warm = SimGpu(**kw)
    warm = sim_meter(sim_warm).validate_dummy(seconds=1.5, prewarm_seconds=0.8,
                                               workload=busy(sim_warm))
    assert cold["residual"] > 0.05, cold["residual"]
    assert warm["residual"] < 0.03, warm["residual"]


def test_validation_threshold_is_configurable():
    sim = SimGpu()
    sim.read_power_w = lambda: 210.0  # ~5% high vs 200W truth
    v = sim_meter(sim).validate_dummy(seconds=0.8, prewarm_seconds=0.1, workload=busy(sim),
                                      max_residual=0.20)
    assert v["passed"] is True and v["max_residual"] == 0.20


# ----------------------------------------------------------------------------
# models.py
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("ds,arch,shape", [
    ("cifar10", "resnet14", (2, 3, 32, 32)),
    ("cifar10", "resnet18", (2, 3, 32, 32)),
    ("kws", "dscnn-s", (2, 1, 49, 40)),
    ("har", "harcnn", (2, 9, 32, 32)),
])
def test_iter_heads_matches_forward_all_and_forward_exit(ds, arch, shape):
    torch.manual_seed(0)
    nc, ch, _ = models.default_io(ds)
    m = models.build_model(ds, arch, nc, ch).eval()
    x = torch.randn(*shape)
    with torch.no_grad():
        a = m.forward_all(x)
        b = list(m.iter_heads(x))
        assert len(a) == len(b) == m.num_heads
        for i in range(m.num_heads):
            assert torch.allclose(a[i], b[i], atol=1e-5)
            assert torch.allclose(a[i], m.forward_exit(x, i), atol=1e-5)
            assert a[i].shape == (shape[0], nc)


def test_flops_cascade_includes_earlier_heads_and_restores_mode():
    m = models.build_model("cifar10", "resnet14", 10, 3).train()
    pre = models.flops_per_exit(m, (1, 3, 32, 32))
    cas = models.flops_per_exit(m, (1, 3, 32, 32), cascade=True)
    assert pre == sorted(pre) and cas == sorted(cas)
    assert cas[0] == pre[0]
    assert all(c > p for c, p in zip(cas[1:], pre[1:]))  # + earlier FC heads
    assert m.training, "flops_per_exit must not leave the model in eval mode"


def test_kd_loss_alpha_semantics():
    torch.manual_seed(0)
    s, t, y = torch.randn(8, 10), torch.randn(8, 10), torch.randint(0, 10, (8,))
    ce = F.cross_entropy(s, y)
    kl = F.kl_div(F.log_softmax(s / 4, 1), F.softmax(t / 4, 1), reduction="batchmean") * 16
    assert torch.allclose(models.kd_loss(s, t, y, 0.0, 4.0), ce)
    assert torch.allclose(models.kd_loss(s, t, y, 1.0, 4.0), kl)  # was CE (inverted)
    assert torch.allclose(models.kd_loss(s, t, y, 0.7, 4.0), 0.7 * kl + 0.3 * ce)
    with pytest.raises(ValueError):
        models.kd_loss(s, t, y, 1.5, 4.0)


def test_kd_loss_gradient_flows_under_fp16_teacher():
    s = torch.randn(4, 10, requires_grad=True)
    t = torch.randn(4, 10).half()
    loss = models.exit_kd_loss([s, s * 2], t, torch.randint(0, 10, (4,)), 0.7, 4.0, [0.5, 1.0])
    loss.backward()
    assert torch.isfinite(loss) and s.grad is not None


def _save_ckpt(path, ds, arch):
    nc, ch, _ = models.default_io(ds)
    m = models.build_model(ds, arch, nc, ch)
    torch.save({"state_dict": m.state_dict(), "arch": arch, "dataset": ds}, path)
    return m


def test_model_from_checkpoint_uses_stored_arch(tmp_path):
    """Regression: evaluate --ckpt kws_*_dscnn-s.pt without --arch crashed (built resnet14)."""
    p = tmp_path / "kws.pt"
    ref = _save_ckpt(p, "kws", "dscnn-s")
    m, arch = models.model_from_checkpoint(str(p), "kws", "cpu")
    assert arch == "dscnn-s" and isinstance(m, models.SmallExitCNN)
    x = torch.randn(1, 1, 49, 40)
    assert torch.allclose(m.eval()(x), ref.eval()(x))


def test_model_from_checkpoint_rejects_dataset_mismatch(tmp_path):
    p = tmp_path / "c.pt"
    _save_ckpt(p, "cifar10", "resnet14")
    with pytest.raises(ValueError, match="trained on"):
        models.model_from_checkpoint(str(p), "kws", "cpu")


class _Evil:
    def __reduce__(self):
        return (print, ("checkpoint code execution!",))


def test_checkpoint_loading_refuses_pickled_code(tmp_path):
    """Security: a .pt from an untrusted source must not execute code on load."""
    p = tmp_path / "evil.pt"
    torch.save({"state_dict": {}, "payload": _Evil()}, p)
    with pytest.raises(Exception):
        models.load_checkpoint(str(p), "cpu")


# ----------------------------------------------------------------------------
# train.py
# ----------------------------------------------------------------------------
def test_cifar_download_failure_is_an_error_not_silent_noise(monkeypatch):
    def boom(*a, **k):
        raise OSError("no internet")
    monkeypatch.setattr(train.datasets, "CIFAR10", boom)
    with pytest.raises(RuntimeError, match="--synthetic"):
        train.get_dataloaders("cifar10", "./data", 8, 0, 0)


def test_synthetic_loaders_are_flagged():
    tl, el, nc, ch = train.get_dataloaders("kws", "./data", 16, 0, 0)
    assert train.is_synthetic(tl) and train.is_synthetic(el) and (nc, ch) == (12, 1)


@pytest.mark.parametrize("mode", ["teacher", "exit_ce", "exit_kd", "student"])
def test_train_one_epoch_all_modes(mode):
    torch.manual_seed(0)
    tl, _, nc, ch = train.get_dataloaders("kws", "./data", 64, 0, 0)
    m = models.build_model("kws", "dscnn-s", nc, ch)
    teacher = models.build_model("kws", "dscnn-s", nc, ch).eval() if mode in ("exit_kd", "student") else None
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    w0 = [p.clone() for p in m.parameters()]
    loss = train.train_one_epoch(m, tl, opt, torch.device("cpu"), None, mode, teacher,
                                 {"grad_accum": 3})  # 32 batches % 3 != 0 -> remainder flush
    assert math.isfinite(loss) and loss > 0
    assert any(not torch.equal(a, b) for a, b in zip(w0, m.parameters()))


# ----------------------------------------------------------------------------
# calibrate.py (LTT)
# ----------------------------------------------------------------------------
def test_hoeffding_pvalue_properties():
    assert calibrate.hoeffding_pvalue(100, 0.2, 0.1) == 1.0
    assert calibrate.hoeffding_pvalue(100, 0.1, 0.1) == 1.0
    assert calibrate.hoeffding_pvalue(1000, 0.0, 0.1) == pytest.approx(math.exp(-20))
    assert calibrate.hoeffding_pvalue(2000, 0.0, 0.1) < calibrate.hoeffding_pvalue(1000, 0.0, 0.1)
    with pytest.raises(ValueError):
        calibrate.hoeffding_pvalue(0, 0.0, 0.1)


def test_fixed_sequence_stops_at_first_failure():
    s = calibrate.fixed_sequence_select([0.9, 0.8, 0.7, 0.6], [0.0, 0.5, 0.0, 0.0], 500, 0.1, 0.1)
    assert s["certified"] == [0.9] and s["selected"] == 0.9  # 0.7/0.6 never tested


def _reference_confidence(model, loader, tau):
    """Original per-sample loop, kept as an oracle for the vectorized version."""
    H = model.num_heads
    counts, cp, cf, extra, n = [0] * H, 0, 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            probs = [F.softmax(h, 1) for h in model.forward_all(x)]
            confs = [p.max(1).values for p in probs]
            preds = [p.argmax(1) for p in probs]
            for i in range(x.size(0)):
                c = H - 1
                for h in range(H - 1):
                    if float(confs[h][i]) >= tau:
                        c = h
                        break
                counts[c] += 1
                ok, fok = int(preds[c][i]) == int(y[i]), int(preds[-1][i]) == int(y[i])
                cp += ok
                cf += fok
                extra += (not ok) and fok
                n += 1
    return counts, cp / n, cf / n, extra / n


@pytest.mark.parametrize("tau", [0.0, 0.11, 0.2, 1.01])
def test_vectorized_confidence_matches_reference(tau):
    torch.manual_seed(0)
    _, el, nc, ch = train.get_dataloaders("kws", "./data", 64, 0, 0)
    m = models.build_model("kws", "dscnn-s", nc, ch).eval()
    r = calibrate.simulate_confidence(m, el, torch.device("cpu"), tau)
    counts, acc, facc, extra = _reference_confidence(m, el, tau)
    assert r["per_head_counts"] == counts
    assert r["acc"] == pytest.approx(acc) and r["full_acc"] == pytest.approx(facc)
    assert r["extra_err"] == pytest.approx(extra)


def test_ltt_risk_is_bounded_and_dominates_acc_drop():
    _, el, nc, ch = train.get_dataloaders("kws", "./data", 64, 0, 0)
    m = models.build_model("kws", "dscnn-s", nc, ch).eval()
    for r in calibrate.sweep_thresholds(m, el, torch.device("cpu"), [0.0, 0.1, 0.5]):
        assert 0.0 <= r["emp_risk"] <= 1.0
        assert r["emp_risk"] >= r["acc_drop"] - 1e-12


# ----------------------------------------------------------------------------
# evaluate.py (policies, energy table, divergence)
# ----------------------------------------------------------------------------
def _fake_stats(n=2000, H=4, seed=0):
    rng = np.random.default_rng(seed)
    conf = rng.uniform(0.2, 1.0, (n, H))
    correct = rng.uniform(0, 1, (n, H)) < np.array([0.6, 0.7, 0.8, 0.9])[:H]
    ent = -np.log(conf)
    return {"conf": conf, "correct": correct, "entropy": ent}


@pytest.mark.parametrize("costs", [
    [152765056, 211485952, 270207488, 328930304],  # FLOPs
    [0.020, 0.026, 0.032, 0.038],                  # Joules
])
def test_marginal_utility_grid_is_not_degenerate(costs):
    """Regression: fixed lambdas gave always-exit-0 (FLOPs) / always-final (Joules)."""
    st = _fake_stats()
    gains = evaluate.estimate_gains(st)
    full = costs[-1]
    mn = [max(1e-9, (costs[h + 1] - costs[h]) / full) for h in range(3)]
    lams = evaluate.lambda_grid(st, gains, mn, [0.001, 0.01, 0.05])
    exits = [evaluate.policy_marginal_utility(st, l, gains, mn, costs)["avg_exit"] for l in lams]
    assert min(exits) < 1.0 and max(exits) > 2.0, exits
    assert all(a >= b - 1e-12 for a, b in zip(exits, exits[1:])), "avg_exit must fall with lambda"


def test_confidence_policy_extremes():
    st = _fake_stats()
    assert evaluate.policy_confidence(st, 0.0, [1, 2, 3, 4])["avg_exit"] == 0.0
    r = evaluate.policy_confidence(st, 1.01, [1, 2, 3, 4])
    assert r["avg_exit"] == 3.0 and r["exit_rate"] == 0.0 and r["avg_cost"] == 4.0


def test_pareto_frontier():
    pts = [{"acc": 0.9, "avg_cost": 10}, {"acc": 0.8, "avg_cost": 5},
           {"acc": 0.7, "avg_cost": 7}, {"acc": 0.9, "avg_cost": 12},
           {"acc": 0.95, "avg_cost": None}]
    assert evaluate.pareto_frontier(pts) == [0, 1]


def test_divergence_detects_only_significant_disagreements():
    flops = [1, 2, 3, 4]
    ok = evaluate.divergence_report(flops, [1.0, 2.0, 3.0, 4.0], [0.1] * 4)
    assert ok["orders_agree"] and ok["significant_disagreements_3sigma"] == []
    noisy = evaluate.divergence_report(flops, [1.0, 2.05, 2.0, 4.0], [0.1] * 4)
    assert not noisy["orders_agree"] and noisy["significant_disagreements_3sigma"] == []
    real = evaluate.divergence_report(flops, [1.0, 3.0, 2.0, 4.0], [0.01] * 4)
    assert real["significant_disagreements_3sigma"] == [[1, 2]]
    assert evaluate.divergence_report(flops, [None] * 4, [None] * 4) == {"available": False}


class _FakeMeter:
    """Returns per-iter marginal Joules from a table keyed by exit index."""
    source = "counter"

    def __init__(self, true_j, noise=0.0, seed=0):
        self.true_j, self.noise = true_j, noise
        self.order = []
        self.rng = np.random.default_rng(seed)

    def measure_fn(self, fn, iters, warmup, idle_power_w):
        fn()
        i = fn.exit_idx
        self.order.append(i)
        per = self.true_j[i] + self.rng.normal(0, self.noise)
        return measure.EnergyResult(duration_s=1.0, iters=iters, per_iter_j=per + 0.01,
                                    per_iter_marginal_j=per, source="counter")


@pytest.fixture
def tagged_exit_fns(monkeypatch):
    orig = evaluate.make_exit_fn

    def tagged(model, x, idx, mode="cascade"):
        f = orig(model, x, idx, mode)
        g = lambda: f()  # noqa: E731
        g.exit_idx = idx
        return g
    monkeypatch.setattr(evaluate, "make_exit_fn", tagged)


def test_energy_table_repeats_round_robin_and_resolvability(tagged_exit_fns):
    m = models.build_model("kws", "dscnn-s", 12, 1).eval()
    meter = _FakeMeter([0.4, 0.8, 1.6], noise=0.001)
    e = evaluate.measure_energy_per_exit(meter, m, (4, 1, 49, 40), torch.device("cpu"),
                                         idle_power_w=20.0, iters=2, repeats=3, min_window_s=0.0)
    assert not e["skipped"]
    assert meter.order == [0, 1, 2, 1, 2, 0, 2, 0, 1]  # rotated each repeat
    assert e["mean_j"] == pytest.approx([0.1, 0.2, 0.4], abs=1e-3)  # per SAMPLE (batch 4)
    assert all(s is not None and s > 0 for s in e["std_j"])
    assert e["adjacent_resolvable_3sigma"] == [True, True]
    flat = evaluate.measure_energy_per_exit(_FakeMeter([1.0, 1.0, 1.0], noise=0.05), m,
                                            (1, 1, 49, 40), torch.device("cpu"), 20.0,
                                            iters=2, repeats=3, min_window_s=0.0)
    assert flat["adjacent_resolvable_3sigma"] == [False, False]  # KILL-2 would fire


def test_energy_table_skips_without_meter_or_idle():
    m = models.build_model("kws", "dscnn-s", 12, 1)
    for meter, idle in [(None, 20.0), (_FakeMeter([1, 1, 1]), None)]:
        e = evaluate.measure_energy_per_exit(meter, m, (1, 1, 49, 40), torch.device("cpu"), idle)
        assert e["skipped"] and e["mean_j"] == [None] * 3


@pytest.mark.parametrize("mode", ["prefix", "cascade"])
def test_exit_fns_run(mode):
    m = models.build_model("cifar10", "resnet14", 10, 3).eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        for i in range(m.num_heads):
            evaluate.make_exit_fn(m, x, i, mode)()


# ----------------------------------------------------------------------------
# CLI end-to-end (CPU, synthetic, tmp results dir)
# ----------------------------------------------------------------------------
def _run(args, tmp_path):
    return subprocess.run([sys.executable, *args, "--results", str(tmp_path)],
                          cwd=str(ROOT), capture_output=True, text=True, timeout=900)


def test_cli_check_gpu_cpu_box(tmp_path):
    r = _run(["src/check_gpu.py", "--idle-seconds", "0.1", "--stress-seconds", "0.1"], tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads((tmp_path / "pilot.json").read_text())["skipped"] is True


def test_cli_train_calibrate_evaluate_pipeline(tmp_path):
    r = _run(["src/train.py", "--dataset", "kws", "--mode", "teacher", "--arch", "dscnn-s",
              "--epochs", "1", "--batch-size", "128", "--device", "cpu"], tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    tck = tmp_path / "kws_teacher_dscnn-s_e1_b128.pt"
    r = _run(["src/train.py", "--dataset", "kws", "--mode", "exit_kd", "--arch", "dscnn-s",
              "--teacher-ckpt", str(tck), "--epochs", "1", "--batch-size", "128",
              "--device", "cpu"], tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    ck = tmp_path / "kws_exit_kd_dscnn-s_e1_b128.pt"
    meta = json.loads((tmp_path / "kws_exit_kd_dscnn-s_e1_b128.json").read_text())
    assert meta["synthetic_data"] is True

    r = _run(["src/calibrate.py", "--ckpt", str(ck), "--dataset", "kws", "--synthetic"], tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    ltt = json.loads((tmp_path / "ltt_kws_dscnn-s.json").read_text())
    assert ltt["synthetic_data"] is True and ltt["arch"] == "dscnn-s"

    # no --arch: must be read from the checkpoint (RUN_PLAN step 5)
    r = _run(["src/evaluate.py", "--ckpt", str(ck), "--dataset", "kws", "--synthetic",
              "--batches", "1", "4", "--lat-iters", "8"], tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    ev = json.loads((tmp_path / "eval_kws_dscnn-s.json").read_text())
    assert ev["synthetic_data"] is True and ev["arch"] == "dscnn-s"
    assert ev["cost_kind"].startswith("flops-cascade")
    assert [b["batch"] for b in ev["batch_rows"]] == [1, 4]
    assert ev["batch_rows"][0]["energy_cascade"]["skipped"] is True  # no fake Joules on CPU
    assert ev["pareto_idx"] and all(0 <= i < len(ev["points"]) for i in ev["pareto_idx"])
    mu = [p["avg_exit"] for p in ev["points"] if p["policy"] == "marginal-utility"]
    if all(g == 0 for g in ev["gains_calib"]):  # random labels: later heads buy nothing
        assert set(mu) == {0.0}, mu
    else:
        assert len(set(mu)) > 1, f"marginal-utility grid degenerate: {mu}"


# ----------------------------------------------------------------------------
# Pen tests: every externally-controlled input channel
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("evil_arch", ["../../../../Windows/evil", r"..\..\evil", "resnet14; rm -rf /"])
def test_pen_ckpt_arch_field_cannot_traverse_paths(tmp_path, evil_arch):
    """Ckpt 'arch' ends up in output filenames: it must be rejected by the whitelist first."""
    p = tmp_path / "x.pt"
    m = models.build_model("cifar10", "resnet14", 10, 3)
    torch.save({"state_dict": m.state_dict(), "arch": evil_arch, "dataset": "cifar10"}, p)
    with pytest.raises(ValueError, match="unknown arch"):
        models.model_from_checkpoint(str(p), "cifar10", "cpu")


def test_pen_ckpt_wrong_shapes_fail_loudly(tmp_path):
    p = tmp_path / "x.pt"
    sd = models.build_model("cifar10", "resnet18", 10, 3).state_dict()
    torch.save({"state_dict": sd, "arch": "resnet14", "dataset": "cifar10"}, p)  # lies about arch
    with pytest.raises(RuntimeError):
        models.model_from_checkpoint(str(p), "cifar10", "cpu")


def test_pen_tampered_cifar_pickle_is_never_unpickled(tmp_path):
    """torchvision unpickles CIFAR batches; a planted file must fail the MD5 check first."""
    import pickle
    from torchvision import datasets
    d = tmp_path / "cifar-10-batches-py"
    d.mkdir()
    marker = tmp_path / "PWNED"

    class Payload:
        def __reduce__(self):
            return (open, (str(marker), "w"))
    blob = pickle.dumps(Payload())
    for f in ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4",
              "data_batch_5", "test_batch", "batches.meta"]:
        (d / f).write_bytes(blob)
    with pytest.raises(RuntimeError):
        datasets.CIFAR10(root=str(tmp_path), train=True, download=False)
    assert not marker.exists(), "tampered pickle was executed"


def test_pen_config_yaml_rejects_python_tags(tmp_path):
    from utils import load_config
    marker = tmp_path / "PWNED"
    p = tmp_path / "evil.yaml"
    p.write_text(f"seed: !!python/object/apply:builtins.open ['{marker.as_posix()}', 'w']\n")
    with pytest.raises(Exception):
        load_config(p)
    assert not marker.exists()
    (tmp_path / "list.yaml").write_text("- 1\n- 2\n")
    with pytest.raises(ValueError):
        load_config(tmp_path / "list.yaml")


@pytest.mark.parametrize("ds", ["../../etc", "cifar10; rm -rf /", "kws$(whoami)"])
def test_pen_dataset_arg_is_whitelisted(ds):
    with pytest.raises(ValueError):
        models.default_io(ds)


def test_pen_no_shell_execution_in_src():
    """No shell=True / os.system / eval / exec / raw pickle / unsafe yaml anywhere in src."""
    import re
    bad = re.compile(r"shell\s*=\s*True|os\.system|os\.popen|(?<![\w.])eval\(|(?<![\w.])exec\(|pickle\.load|"
                     r"yaml\.load\(|yaml\.unsafe_load|weights_only\s*=\s*False")
    for f in SRC.glob("*.py"):
        if f.name.startswith("test_"):
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            assert not bad.search(line), f"{f.name}:{i}: {line.strip()}"
    for f in SRC.glob("*.py"):
        if f.name.startswith("test_"):
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if "torch.load(" in line:
                assert "weights_only=True" in line, f"{f.name}:{i} torch.load without weights_only"


def test_cli_train_accepts_documented_flags(tmp_path):
    """RUN_PLAN documents --grad-accum / --num-workers: they must parse."""
    r = _run(["src/train.py", "--dataset", "kws", "--mode", "exit_ce", "--arch", "dscnn-s",
              "--epochs", "1", "--batch-size", "256", "--device", "cpu", "--synthetic",
              "--grad-accum", "2", "--num-workers", "0"], tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads((tmp_path / "kws_exit_ce_dscnn-s_e1_b256.json").read_text())["grad_accum"] == 2


def test_cli_train_exit_kd_without_teacher_fails_cleanly(tmp_path):
    r = _run(["src/train.py", "--dataset", "kws", "--mode", "exit_kd", "--epochs", "1",
              "--device", "cpu", "--synthetic"], tmp_path)
    assert r.returncode == 2 and "--teacher-ckpt is required" in r.stdout
