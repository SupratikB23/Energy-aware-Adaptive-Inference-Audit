"""Shared helpers: seeds, config, device, logging, provenance.

CPU-safe. No NVML import here. All GPU queries live in measure.py / check_gpu.py.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str | Path = "config.yaml") -> dict:
    path = Path(path)
    if not path.exists():
        # Allow running from src/ as CWD: fall back to sibling ../config.yaml
        alt = Path(__file__).resolve().parent.parent / "config.yaml"
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"config not found: {path} (also tried {alt})")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} did not parse to a dict")
    return cfg


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic where affordable; keep benchmark on for speed on GPU box.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def resolve_device(want: str = "auto") -> torch.device:
    want = (want or "auto").lower()
    if want == "cpu":
        return torch.device("cpu")
    if want == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("config device=cuda but torch.cuda.is_available() is False")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def refuse_silent_cpu(device: torch.device, requested: str | None, synthetic: bool, who: str) -> None:
    """Real runs must not silently fall back to the CPU (GPU idle, hours wasted, no energy).

    Allowed: an explicit --device cpu, or --synthetic (smoke/CI runs).
    """
    if device.type == "cpu" and (requested or "auto").lower() != "cpu" and not synthetic:
        print(f"[{who}] FAILED: no CUDA GPU is visible to torch (torch {torch.__version__}, "
              f"cuda available = {torch.cuda.is_available()}). This is usually a CPU-only torch "
              "wheel in the active venv. Install the CUDA build first (RTX 50: cu128, see "
              "RUN_PLAN.md Step 0a), then rerun. Pass --device cpu only if a CPU run is intended.",
              flush=True)
        raise SystemExit(2)


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_session_id() -> str:
    """Short, sortable, collision-resistant id for one measurement session."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{os.urandom(3).hex()}"


def resolve_session(explicit: str | None = None) -> str:
    """Explicit --session > PERCOM_SESSION env var > freshly generated id."""
    s = (explicit or os.environ.get("PERCOM_SESSION") or "").strip()
    return s or new_session_id()


def result_stem(dataset: str, arch: str, ckpt_meta: dict | None = None,
                tag: str | None = None) -> str:
    """Stable output stem; a checkpoint's training seed prevents seed collisions.

    `tag` separates repeated sessions of the same checkpoint (e.g. run A/B), so
    cross-session repeats never overwrite each other.
    """
    seed = (ckpt_meta or {}).get("seed")
    stem = f"{dataset}_{arch}" + (f"_s{seed}" if seed is not None else "")
    if tag:
        safe = "".join(c if (c.isalnum() or c in "-_") else "-" for c in str(tag))[:40]
        stem += f"_{safe}"
    return stem


# Two-sided 95% Student-t critical values (df -> t). df > 30 uses the normal limit.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080,
    22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
    29: 2.045, 30: 2.042,
}


def t_critical_95(df: int) -> float:
    if df <= 0:
        raise ValueError("df must be > 0")
    return _T95.get(int(df), 1.960)


def mean_ci95(values) -> dict:
    """Mean, sample SD, SEM and two-sided 95% CI (Student-t). None-safe."""
    vals = [float(v) for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "sem": None,
                "ci95_low": None, "ci95_high": None, "ci95_halfwidth": None,
                "cv_pct": None}
    mean = float(np.mean(vals))
    if n == 1:
        return {"n": 1, "mean": mean, "sd": None, "sem": None,
                "ci95_low": None, "ci95_high": None, "ci95_halfwidth": None,
                "cv_pct": None}
    sd = float(np.std(vals, ddof=1))
    sem = sd / float(np.sqrt(n))
    half = t_critical_95(n - 1) * sem
    return {"n": n, "mean": mean, "sd": sd, "sem": sem,
            "ci95_low": mean - half, "ci95_high": mean + half,
            "ci95_halfwidth": half,
            "cv_pct": (100.0 * sd / abs(mean)) if mean else None}


def get_git_hash(repo: str | Path = ".") -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "nogit"


def torch_info() -> dict:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def vram_info() -> dict:
    """VRAM allocation snapshot. CPU-safe (zeros when no CUDA)."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "total_mb": 0.0}
    try:
        alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)
        res = torch.cuda.memory_reserved(0) / (1024 ** 2)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        return {"allocated_mb": float(alloc), "reserved_mb": float(res), "total_mb": float(total)}
    except Exception:
        return {"allocated_mb": -1.0, "reserved_mb": -1.0, "total_mb": -1.0}


def save_json(obj: dict, path: str | Path) -> Path:
    path = Path(path)
    if path.parent.as_posix() not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    return path


def cuda_sync() -> None:
    """Synchronize CUDA if available. No-op on CPU. Call before/after every timed window."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
