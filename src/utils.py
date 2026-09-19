"""Shared helpers: seeds, config, device, logging, provenance.

CPU-safe. No NVML import here. All GPU queries live in measure.py / check_gpu.py.
"""
from __future__ import annotations

import json
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


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
