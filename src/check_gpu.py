"""Phase 0 pilot (RUN_PLAN Step 0): can this box do the science?

Checks, in order:
  1. torch/CUDA info + nvidia-smi (if present)
  2. NVML caps: power/temp/util/clocks + energy-counter presence
  3. Idle baseline (short here; full 60s before real experiments)
  4. Controlled load: sustained matmul raises power/temp (repeatable states?)
  5. Dummy-kernel validation via EnergyMeter (kill criterion Day 1-2)

CPU-only machines: prints a clear SKIP message and writes pilot.json with
skipped=True, exit 0 — so `test_smoke.py` passes off-box.
GPU box: exit 1 with FAILED validation if residual > threshold (do not proceed).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

try:
    from utils import ensure_dir, get_git_hash, load_config, save_json, set_seed, torch_info, vram_info
except ImportError:
    from src.utils import ensure_dir, get_git_hash, load_config, save_json, set_seed, torch_info, vram_info

try:
    from measure import EnergyMeter, Nvml, HAS_NVML
except ImportError:
    from src.measure import EnergyMeter, Nvml, HAS_NVML


def _nvidia_smi() -> dict:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,"
                              "power.limit,clocks.gr,clocks.mem",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15)
        if out.returncode == 0 and out.stdout.strip():
            return {"present": True, "raw": out.stdout.strip()}
        return {"present": False, "raw": out.stderr.strip()[-500:]}
    except FileNotFoundError:
        return {"present": False, "raw": "nvidia-smi not found"}
    except Exception as e:
        return {"present": False, "raw": repr(e)[:500]}


def _stress_matmul(seconds: float, size: int = 2048) -> dict:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.randn(size, size, device=dev)
    b = torch.randn(size, size, device=dev)
    t_end = time.perf_counter() + seconds
    iters = 0
    with torch.no_grad():
        while time.perf_counter() < t_end:
            _ = (a @ b).sum()
            if dev == "cuda":
                torch.cuda.synchronize()
            iters += 1
    return {"device": dev, "seconds": seconds, "iters": iters}


def _cuda_kernel_check() -> dict:
    """Run a real conv+matmul on the GPU. Catches torch wheels built without
    kernels for this GPU's arch (e.g. RTX 50 / sm_120 needs a cu128 build),
    which otherwise only fail later, mid-training."""
    try:
        cap = torch.cuda.get_device_capability(0)
        archs = torch.cuda.get_arch_list()
        x = torch.randn(2, 3, 32, 32, device="cuda")
        w = torch.randn(8, 3, 3, 3, device="cuda")
        y = torch.nn.functional.conv2d(x, w, padding=1)
        z = (y.flatten(1) @ y.flatten(1).T).sum().item()
        torch.cuda.synchronize()
        return {"ok": bool(z == z), "capability": f"sm_{cap[0]}{cap[1]}", "torch_arch_list": archs}
    except Exception as e:
        return {"ok": False, "error": repr(e)[:500],
                "hint": "install a torch build matching this GPU (RTX 50xx: --index-url "
                        "https://download.pytorch.org/whl/cu128)"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 0 GPU pilot")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--idle-seconds", type=float, default=10.0)
    ap.add_argument("--stress-seconds", type=float, default=10.0)
    ap.add_argument("--results", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    results = Path(args.results or cfg.get("results_root", "./results"))
    ensure_dir(results)

    info = {"torch": torch_info(), "vram": vram_info(), "smi": _nvidia_smi(),
            "has_nvml": bool(HAS_NVML), "git": get_git_hash(".")}
    print(f"[pilot] torch={info['torch']}", flush=True)
    print(f"[pilot] nvidia-smi present={info['smi']['present']}", flush=True)

    if not torch.cuda.is_available() or not HAS_NVML:
        reason = []
        if not torch.cuda.is_available():
            reason.append("torch.cuda unavailable")
        if not HAS_NVML:
            reason.append("pynvml not installed")
        info.update({"skipped": True, "reason": "; ".join(reason),
                     "next": "run this file on the RTX GPU box after pip install -r requirements.txt"})
        print(f"[pilot] SKIP ({info['reason']}). No GPU checks possible here — OK for CPU box.", flush=True)
        save_json(info, results / "pilot.json")
        return 0

    kc = _cuda_kernel_check()
    info["cuda_kernel_check"] = kc
    print(f"[pilot] cuda kernel check: {kc}", flush=True)
    if not kc.get("ok"):
        save_json(info, results / "pilot.json")
        print("[pilot] FAIL: CUDA kernels do not run on this GPU with this torch build.", flush=True)
        return 1

    # --- NVML caps ---
    try:
        nv = Nvml(index=0)
    except Exception as e:
        info.update({"skipped": False, "nvml_error": repr(e)})
        print(f"[pilot] NVML init FAILED: {e!r}", flush=True)
        save_json(info, results / "pilot.json")
        return 1
    try:
        before = nv.device_state().to_dict()
        has_counter = nv.has_energy_counter()
        info.update({"device_before": before, "has_energy_counter": has_counter})
        print(f"[pilot] GPU={before['gpu_name']} driver={before['driver']} "
              f"power={before['power_w']}W temp={before['temperature_c']}C "
              f"counter={has_counter}", flush=True)
    finally:
        nv.close()

    # --- idle + stress + validation via EnergyMeter ---
    mc = dict(cfg.get("measure", {}))
    thr = float(mc.get("dummy_max_residual", 0.05))
    meter = EnergyMeter(poll_hz=float(mc.get("poll_hz", 150.0)))
    try:
        print(f"[pilot] idle baseline {args.idle_seconds}s ...", flush=True)
        idle = meter.measure_idle(seconds=float(args.idle_seconds))
        print(f"[pilot] idle power={idle.get('idle_power_w')}W source={meter.source}", flush=True)
        print(f"[pilot] stress matmul {args.stress_seconds}s ...", flush=True)
        s0 = meter.nvml.device_state().to_dict() if meter.nvml else {}
        stress = _stress_matmul(float(args.stress_seconds))
        s1 = meter.nvml.device_state().to_dict() if meter.nvml else {}
        print(f"[pilot] stress iters={stress['iters']} temp {s0.get('temperature_c')}->{s1.get('temperature_c')}C "
              f"power {s0.get('power_w')}->{s1.get('power_w')}W", flush=True)
        val = meter.validate_dummy(seconds=float(mc.get("dummy_seconds", 10.0)),
                                   max_residual=thr)
        print(f"[pilot] dummy validation: {val}", flush=True)
        info.update({"idle": idle, "stress": stress, "state_before_stress": s0,
                     "state_after_stress": s1, "validation": val,
                     "meter_source": meter.source})
    finally:
        meter.close()

    save_json(info, results / "pilot.json")
    v = info.get("validation", {})
    if v.get("skipped"):
        # CUDA + pynvml are present here, so a skip means NVML is unusable: that is a FAIL.
        print("[pilot] FAIL: validation skipped although CUDA+pynvml are present "
              "(EnergyMeter could not open NVML). STOP.", flush=True)
        return 1
    if v.get("passed") is True:
        print(f"[pilot] PASS: residual {v.get('residual')} <= {thr}. Proceed to C1.", flush=True)
        return 0
    print(f"[pilot] FAIL: residual {v.get('residual')} > {thr} "
          f"(basis={v.get('basis')}; {v.get('reason', '')}). "
          f"STOP — do not run policy benchmarks on bad telemetry.", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
