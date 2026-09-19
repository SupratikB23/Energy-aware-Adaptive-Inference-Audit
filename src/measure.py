"""C1 measurement protocol (verdict S4).

Preferred path: hardware-integrated millijoule counter
  nvmlDeviceGetTotalEnergyConsumption (mJ since driver load).
Fallback: background power polling at 100-200 Hz + trapezoidal integration.
Never 1 Hz. Never single-shot. Always torch.cuda.synchronize() around windows.

CPU-safe: imports and math work without a GPU or pynvml. GPU methods raise a
clear RuntimeError (or return skipped) instead of crashing at import time.
"""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field

try:
    import pynvml  # type: ignore
    HAS_NVML = True
except Exception:
    pynvml = None  # type: ignore
    HAS_NVML = False

import torch

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
try:
    from utils import cuda_sync
except ImportError:  # allow `python -m src.measure` from repo root
    from src.utils import cuda_sync

# Re-export for callers that do `from measure import HAS_NVML`.
__all__ = [
    "HAS_NVML",
    "DeviceState",
    "EnergyResult",
    "Nvml",
    "EnergyMeter",
    "integrate_power",
    "cuda_sync_if_needed",
]


def cuda_sync_if_needed() -> None:
    cuda_sync()


def integrate_power(times: list[float], powers_w: list[float]) -> float:
    """Trapezoidal integration of power (W) over time (s) -> energy (J)."""
    if len(times) != len(powers_w):
        raise ValueError("times and powers_w must have equal length")
    if len(times) < 2:
        return 0.0
    e = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        if dt < 0:
            raise ValueError("non-monotonic timestamps in power trace")
        e += 0.5 * (powers_w[i] + powers_w[i - 1]) * dt
    return float(e)


@dataclass
class DeviceState:
    gpu_name: str = "unknown"
    driver: str = "unknown"
    temperature_c: float = -1.0
    power_w: float = -1.0
    util_gpu_pct: float = -1.0
    util_mem_pct: float = -1.0
    clock_graphics_mhz: float = -1.0
    clock_mem_mhz: float = -1.0
    mem_used_mb: float = -1.0
    mem_total_mb: float = -1.0
    power_limit_w: float = -1.0
    process_count: int = -1
    vram_allocated_mb: float = -1.0  # torch-side allocation (shifts idle power)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EnergyResult:
    duration_s: float = 0.0
    iters: int = 0
    energy_counter_j: float | None = None  # from hardware mJ counter
    energy_poll_j: float | None = None     # from trapezoidal integration
    energy_j: float | None = None          # preferred estimate (counter if present else poll)
    avg_power_w: float | None = None
    idle_power_w: float | None = None
    marginal_energy_j: float | None = None  # energy_j - idle_power*duration
    per_iter_j: float | None = None
    per_iter_marginal_j: float | None = None
    samples: int = 0
    effective_hz: float = 0.0
    source: str = "none"  # counter | poll | none
    device_before: dict = field(default_factory=dict)
    device_after: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Nvml:
    """Thin, honest NVML wrapper. No shims, no fake data."""

    def __init__(self, index: int = 0):
        if not HAS_NVML:
            raise RuntimeError(
                "pynvml (nvidia-ml-py) is not installed. "
                "On the GPU box run: pip install -r requirements.txt"
            )
        assert pynvml is not None
        try:
            pynvml.nvmlInit()
        except Exception as e:
            raise RuntimeError(f"nvmlInit() failed: {e!r}") from e
        self._inited = True
        try:
            count = pynvml.nvmlDeviceGetCount()
        except Exception as e:
            self.close()
            raise RuntimeError(f"nvmlDeviceGetCount() failed: {e!r}") from e
        if count < 1:
            self.close()
            raise RuntimeError("NVML reports 0 GPUs")
        if index >= count:
            self.close()
            raise RuntimeError(f"gpu_index {index} out of range (count={count})")
        self.index = index
        try:
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception as e:
            self.close()
            raise RuntimeError(f"nvmlDeviceGetHandleByIndex({index}) failed: {e!r}") from e
        self._counter_ok: bool | None = None

    def close(self) -> None:
        if getattr(self, "_inited", False):
            try:
                assert pynvml is not None
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._inited = False

    def __enter__(self) -> "Nvml":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # -- basic info --
    def gpu_name(self) -> str:
        assert pynvml is not None
        try:
            n = pynvml.nvmlDeviceGetName(self.handle)
            return n.decode() if isinstance(n, bytes) else str(n)
        except Exception:
            return "unknown"

    def driver(self) -> str:
        assert pynvml is not None
        try:
            v = pynvml.nvmlSystemGetDriverVersion()
            return v.decode() if isinstance(v, bytes) else str(v)
        except Exception:
            return "unknown"

    # -- energy counter --
    def has_energy_counter(self) -> bool:
        if self._counter_ok is not None:
            return self._counter_ok
        assert pynvml is not None
        if not hasattr(pynvml, "nvmlDeviceGetTotalEnergyConsumption"):
            self._counter_ok = False
            return False
        try:
            v = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            self._counter_ok = isinstance(v, int) and v > 0
        except Exception:
            self._counter_ok = False
        return bool(self._counter_ok)

    def read_energy_mj(self) -> int:
        assert pynvml is not None
        return int(pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle))

    def read_power_w(self) -> float:
        assert pynvml is not None
        return float(pynvml.nvmlDeviceGetPowerUsage(self.handle)) / 1000.0

    def device_state(self) -> DeviceState:
        assert pynvml is not None
        s = DeviceState(gpu_name=self.gpu_name(), driver=self.driver())
        try:
            s.power_w = self.read_power_w()
        except Exception:
            pass
        try:
            s.temperature_c = float(
                pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            )
        except Exception:
            pass
        try:
            u = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            s.util_gpu_pct = float(u.gpu)
            s.util_mem_pct = float(u.memory)
        except Exception:
            pass
        try:
            s.clock_graphics_mhz = float(
                pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_GRAPHICS)
            )
        except Exception:
            pass
        try:
            s.clock_mem_mhz = float(
                pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
            )
        except Exception:
            pass
        try:
            m = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            s.mem_used_mb = float(m.used) / (1024 ** 2)
            s.mem_total_mb = float(m.total) / (1024 ** 2)
        except Exception:
            pass
        try:
            # milliwatts -> watts; may be unsupported on some parts
            s.power_limit_w = float(pynvml.nvmlDeviceGetPowerManagementLimit(self.handle)) / 1000.0
        except Exception:
            pass
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
            s.process_count = len(procs)
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                s.vram_allocated_mb = float(torch.cuda.memory_allocated(0)) / (1024 ** 2)
        except Exception:
            pass
        return s


class EnergyMeter:
    """Windowed energy measurement with counter-preferred, poll-fallback design.

    Usage:
        meter = EnergyMeter(gpu_index=0, poll_hz=150.0)
        meter.start()
        ... run workload (looped, >=1000 iters for per-exit windows) ...
        res = meter.stop(iters=n, idle_power_w=baseline)
    or:
        res = meter.measure_fn(fn, iters=1000, warmup=10)
    """

    def __init__(self, gpu_index: int = 0, poll_hz: float = 150.0, use_counter: bool = True):
        if poll_hz < 20 or poll_hz > 1000:
            raise ValueError(f"poll_hz {poll_hz} outside sane range [20, 1000]")
        self.gpu_index = gpu_index
        self.poll_hz = float(poll_hz)
        self.use_counter = bool(use_counter)
        self.nvml: Nvml | None = None
        self.has_nvml = False
        self.has_counter = False
        if HAS_NVML and torch.cuda.is_available():
            try:
                self.nvml = Nvml(index=gpu_index)
                self.has_nvml = True
                self.has_counter = self.nvml.has_energy_counter() if use_counter else False
            except Exception:
                self.nvml = None
                self.has_nvml = False
                self.has_counter = False
        # poll thread state
        self._times: list[float] = []
        self._powers: list[float] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        self._e0_mj: int | None = None
        self._dev_before: DeviceState | None = None
        self._running = False

    def close(self) -> None:
        if self.nvml is not None:
            try:
                self.nvml.close()
            except Exception:
                pass
            self.nvml = None

    @property
    def source(self) -> str:
        if self.has_counter:
            return "counter"
        if self.has_nvml:
            return "poll"
        return "none"

    # -- polling --
    def _poll_loop(self) -> None:
        assert self.nvml is not None
        interval = 1.0 / self.poll_hz
        t_start = time.perf_counter()
        while not self._stop_event.is_set():
            loop_t = time.perf_counter()
            try:
                p = self.nvml.read_power_w()
                self._times.append(loop_t - t_start)
                self._powers.append(p)
            except Exception:
                pass
            elapsed = time.perf_counter() - loop_t
            time.sleep(max(0.0, interval - elapsed))

    def start(self) -> None:
        if self._running:
            raise RuntimeError("EnergyMeter.start() called while already running")
        cuda_sync_if_needed()
        self._times = []
        self._powers = []
        self._dev_before = self.nvml.device_state() if self.nvml else DeviceState()
        self._e0_mj = None
        if self.has_counter and self.nvml is not None:
            try:
                self._e0_mj = self.nvml.read_energy_mj()
            except Exception:
                self._e0_mj = None
        self._t0 = time.perf_counter()
        if self.nvml is not None:
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()
        self._running = True

    def stop(self, iters: int = 1, idle_power_w: float | None = None) -> EnergyResult:
        if not self._running:
            raise RuntimeError("EnergyMeter.stop() called without start()")
        cuda_sync_if_needed()
        t1 = time.perf_counter()
        e1_mj: int | None = None
        if self.has_counter and self.nvml is not None and self._e0_mj is not None:
            try:
                e1_mj = self.nvml.read_energy_mj()
            except Exception:
                e1_mj = None
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=5.0)
            self._thread = None
        duration = max(1e-9, t1 - self._t0)
        dev_after = self.nvml.device_state() if self.nvml else DeviceState()

        e_counter = None
        if e1_mj is not None and self._e0_mj is not None and e1_mj >= self._e0_mj:
            e_counter = (e1_mj - self._e0_mj) / 1000.0  # mJ -> J
        e_poll = None
        if len(self._times) >= 2:
            try:
                e_poll = integrate_power(self._times, self._powers)
            except Exception:
                e_poll = None
        if self.has_counter and e_counter is not None:
            e_best: float | None = e_counter
            src = "counter"
        elif e_poll is not None:
            e_best = e_poll
            src = "poll"
        else:
            e_best = None
            src = "none"
        avg_p = (e_best / duration) if (e_best is not None and duration > 0) else None
        marginal = None
        if e_best is not None and idle_power_w is not None:
            marginal = e_best - idle_power_w * duration
        n = max(1, int(iters))
        res = EnergyResult(
            duration_s=float(duration),
            iters=n,
            energy_counter_j=e_counter,
            energy_poll_j=e_poll,
            energy_j=e_best,
            avg_power_w=avg_p,
            idle_power_w=idle_power_w,
            marginal_energy_j=marginal,
            per_iter_j=(e_best / n) if e_best is not None else None,
            per_iter_marginal_j=(marginal / n) if marginal is not None else None,
            samples=len(self._times),
            effective_hz=(len(self._times) / duration) if duration > 0 else 0.0,
            source=src,
            device_before=self._dev_before.to_dict() if self._dev_before else {},
            device_after=dev_after.to_dict(),
        )
        self._running = False
        return res

    def measure_fn(self, fn, iters: int = 1000, warmup: int = 10,
                   idle_power_w: float | None = None):
        """Loop fn() iters times inside one window. fn must take no args."""
        if iters < 1:
            raise ValueError("iters must be >= 1")
        for _ in range(max(0, int(warmup))):
            fn()
        cuda_sync_if_needed()
        self.start()
        try:
            for _ in range(int(iters)):
                fn()
        finally:
            res = self.stop(iters=iters, idle_power_w=idle_power_w)
        return res

    def measure_idle(self, seconds: float = 60.0) -> dict:
        """Idle baseline: sleep `seconds` inside a window. Returns dict with idle_power_w."""
        if seconds <= 0:
            raise ValueError("seconds must be > 0")
        self.start()
        try:
            time.sleep(float(seconds))
        finally:
            res = self.stop(iters=1, idle_power_w=None)
        out = res.to_dict()
        out["idle_power_w"] = res.avg_power_w
        return out

    def validate_dummy(self, seconds: float = 5.0, size: int = 2048,
                       max_residual: float = 0.05, prewarm_seconds: float = 2.0,
                       workload=None) -> dict:
        """Fixed synthetic load; cross-checks two INDEPENDENT energy paths.

        Residual = |E_counter - E_poll| / E_counter over the same window.
        The load is started `prewarm_seconds` BEFORE the window opens: on
        Ampere+ (RTX 30/40/50) nvmlDeviceGetPowerUsage is a ~1 s moving
        average, so a cold start would bias the poll path low and fail
        validation for a reason unrelated to the pipeline.

        Poll-only devices (no hardware counter) have NO independent reference:
        comparing E_poll with P_avg*t is circular (P_avg is derived from
        E_poll), so that case returns passed=False with basis
        'poll-only-unverifiable' instead of a vacuous pass. Validate such a
        device against an external meter.

        `workload` (no-arg callable, optional) replaces the default CUDA
        matmul; used by the CPU unit tests with a fake NVML.
        On CPU-only machines returns {passed: None, skipped: True}.
        """
        if not self.has_nvml or self.nvml is None:
            return {"passed": None, "skipped": True,
                    "reason": "no NVML/CUDA; run on GPU box", "residual": None}
        if workload is None:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            a = torch.randn(size, size, device=dev)
            b = torch.randn(size, size, device=dev)

            def workload():
                with torch.no_grad():
                    _ = (a @ b).sum()
                cuda_sync_if_needed()

        t_pre = time.perf_counter() + max(0.0, float(prewarm_seconds))
        while time.perf_counter() < t_pre:
            workload()
        t_end = time.perf_counter() + float(seconds)
        self.start()
        try:
            while time.perf_counter() < t_end:
                workload()
        finally:
            res = self.stop(iters=1)
        ec, ep = res.energy_counter_j, res.energy_poll_j
        out = {"skipped": False, "max_residual": float(max_residual),
               "prewarm_seconds": float(prewarm_seconds), "result": res.to_dict()}
        if ec is not None and ep is not None and ec > 0:
            resid = abs(ec - ep) / ec
            out.update({"passed": bool(resid <= max_residual), "residual": float(resid),
                        "basis": "counter-vs-poll"})
            return out
        if ep is not None and ec is None:
            out.update({"passed": False, "residual": None, "basis": "poll-only-unverifiable",
                        "reason": "no hardware energy counter: poll integration has no "
                                  "independent reference here; validate with an external meter"})
            return out
        out.update({"passed": False, "residual": None, "basis": "none",
                    "reason": "no energy samples (counter delta and poll trace both empty)"})
        return out
