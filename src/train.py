"""8GB-safe training: teacher | student | exit_ce | exit_kd.

Modes:
  teacher  - plain CE on final head (backbone trained as single-exit net)
  student  - smaller net, plain CE (KD baseline without teacher)
  exit_ce  - early-exit net, CE on every head (no teacher)
  exit_kd  - early-exit student distilled from teacher final (exit-aware KD)

Datasets: cifar10 (torchvision) + kws/har (synthetic windows sized like the
real features so the pervasive track runs end-to-end on any box; replace with
real loaders when you stage SpeechCommands / UCI-HAR on the GPU box).

Checkpoints + metrics go to results/. OOM exits with code 2 and a clear message.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

try:
    from utils import ensure_dir, get_git_hash, load_config, resolve_device, save_json, set_seed, vram_info
except ImportError:
    from src.utils import ensure_dir, get_git_hash, load_config, resolve_device, save_json, set_seed, vram_info

try:
    from models import build_model, default_io, exit_kd_loss, kd_loss, model_from_checkpoint
except ImportError:
    from src.models import build_model, default_io, exit_kd_loss, kd_loss, model_from_checkpoint


def is_synthetic(loader: DataLoader) -> bool:
    """True when the loader serves the random-tensor fallback (not real data)."""
    return isinstance(loader.dataset, TensorDataset)


HAR_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
HAR_SHA256 = "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"
HAR_CHANNELS = ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y",
                "body_gyro_z", "total_acc_x", "total_acc_y", "total_acc_z"]


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_uci_har(data_root: str, download: bool = True):
    """Real UCI-HAR raw inertial windows -> (Xtr, ytr, Xte, yte) tensors.

    X: float32 [N, 9, 1, 128] (9 channels x 2.56 s @ 50 Hz), z-scored per channel
    with TRAIN statistics. y: int64 in [0, 6). Official subject-disjoint split.
    Security: the archive is SHA-256 pinned; members are read by exact name and
    parsed as text with numpy (no pickle, nothing extracted to disk). Parsed
    arrays are cached as .npz and re-loaded with allow_pickle=False.
    """
    import io
    import urllib.request
    import zipfile

    import numpy as np
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "uci_har_inertial.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        arrs = [d["Xtr"], d["ytr"], d["Xte"], d["yte"]]
    else:
        zpath = root / "uci_har.zip"
        if not zpath.exists():
            if not download:
                raise FileNotFoundError(f"{zpath} missing")
            print(f"[data] downloading UCI-HAR (~58 MB) -> {zpath}", flush=True)
            tmp = zpath.with_suffix(".part")
            with urllib.request.urlopen(HAR_URL, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.replace(zpath)
        got = _sha256(zpath)
        if got != HAR_SHA256:
            raise RuntimeError(f"UCI-HAR archive sha256 mismatch ({got}); refusing to use it. "
                               f"Delete {zpath} and retry.")
        inner = zipfile.ZipFile(zpath).read("UCI HAR Dataset.zip")
        zi = zipfile.ZipFile(io.BytesIO(inner))

        def split(name: str):
            x = np.stack([np.loadtxt(io.BytesIO(zi.read(
                f"UCI HAR Dataset/{name}/Inertial Signals/{c}_{name}.txt")), dtype=np.float32)
                for c in HAR_CHANNELS], axis=1)                       # [N, 9, 128]
            y = np.loadtxt(io.BytesIO(zi.read(f"UCI HAR Dataset/{name}/y_{name}.txt"))).astype(np.int64) - 1
            return x, y
        Xtr, ytr = split("train")
        Xte, yte = split("test")
        np.savez(cache, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
        arrs = [Xtr, ytr, Xte, yte]
    Xtr, ytr, Xte, yte = arrs
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))  # noqa: E731
    return (t(Xtr).float().unsqueeze(2), t(ytr).long(), t(Xte).float().unsqueeze(2), t(yte).long())


class _RealTensorDataset(torch.utils.data.Dataset):
    """TensorDataset twin that is NOT flagged as synthetic by is_synthetic()."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x, self.y = x, y

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def get_dataloaders(dataset: str, data_root: str, batch_size: int, num_workers: int,
                    seed: int, force_synthetic: bool = False):
    """Returns (train_loader, test_loader, num_classes, in_channels)."""
    g = torch.Generator().manual_seed(seed)
    ds = dataset.lower()
    if ds == "cifar10" and not force_synthetic:
        try:
            tfm_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            tfm_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            train = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tfm_train)
            test = datasets.CIFAR10(root=data_root, train=False, download=True, transform=tfm_test)
            nc, ch, _ = default_io("cifar10")
            tl = DataLoader(train, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available())
            el = DataLoader(test, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available())
            return tl, el, nc, ch
        except Exception as e:
            # Never silently train/evaluate on noise: real runs must have real data.
            raise RuntimeError(
                f"CIFAR10 load/download failed ({e!r}). Stage it under {data_root!r} "
                f"or pass --synthetic for a plumbing-only run.") from e
    if ds == "har" and not force_synthetic:
        try:
            Xtr, ytr, Xte, yte = load_uci_har(data_root)
        except Exception as e:
            raise RuntimeError(f"UCI-HAR load/download failed ({e!r}). Put the official zip at "
                               f"{data_root}/uci_har.zip or pass --synthetic.") from e
        nc, ch, _ = default_io("har")
        tl = DataLoader(_RealTensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
        el = DataLoader(_RealTensorDataset(Xte, yte), batch_size=batch_size, shuffle=False)
        return tl, el, nc, ch
    # Synthetic fallback (kws until real features are staged; any dataset with --synthetic)
    nc, ch, (H, W) = default_io(ds if ds in ("kws", "har") else "cifar10")
    ntr, nte = 2048, 512
    Xtr = torch.randn(ntr, ch, H, W, generator=g)
    ytr = torch.randint(0, nc, (ntr,), generator=g)
    Xte = torch.randn(nte, ch, H, W, generator=g)
    yte = torch.randint(0, nc, (nte,), generator=g)
    tl = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    el = DataLoader(TensorDataset(Xte, yte), batch_size=batch_size, shuffle=False)
    return tl, el, nc, ch


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device,
                      head: int = -1) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if hasattr(model, "forward_exit"):
            idx = model.num_heads - 1 if head < 0 else head  # type: ignore[attr-defined]
            logits = model.forward_exit(x, idx)  # type: ignore[attr-defined]
        else:
            logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def _scaler(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except Exception:
            return None
    return None


def train_one_epoch(model, loader, optimizer, device, scaler, mode: str,
                    teacher=None, cfg_train: dict | None = None) -> float:
    model.train()
    if teacher is not None:
        teacher.eval()
    cfg_train = cfg_train or {}
    alpha = float(cfg_train.get("kd_alpha", 0.7))
    temp = float(cfg_train.get("kd_temp", 4.0))
    ew = list(cfg_train.get("exit_weights", [0.3, 0.3, 0.4, 1.0]))
    accum = max(1, int(cfg_train.get("grad_accum", 1)))
    ce = nn.CrossEntropyLoss()
    running, n = 0.0, 0
    nbatches = 0
    optimizer.zero_grad(set_to_none=True)
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        use_amp = scaler is not None
        try:
            with torch.amp.autocast("cuda", enabled=use_amp):
                if mode in ("teacher", "student"):
                    logits = model(x) if not hasattr(model, "forward_exit") else model.forward_exit(x, model.num_heads - 1)
                    if mode == "student" and teacher is not None:
                        with torch.no_grad():
                            t = teacher(x) if not hasattr(teacher, "forward_exit") else teacher.forward_exit(x, teacher.num_heads - 1)
                        loss = kd_loss(logits, t, y, alpha, temp)
                    else:
                        loss = ce(logits, y)
                elif mode == "exit_ce":
                    heads = model.forward_all(x)
                    loss = sum(ce(h, y) for h in heads) / len(heads)
                elif mode == "exit_kd":
                    assert teacher is not None, "exit_kd needs --teacher-ckpt"
                    with torch.no_grad():
                        t = teacher(x) if not hasattr(teacher, "forward_exit") else teacher.forward_exit(x, teacher.num_heads - 1)
                    heads = model.forward_all(x)
                    w = (ew + [1.0] * len(heads))[:len(heads)]
                    loss = exit_kd_loss(heads, t, y, alpha, temp, w)
                else:
                    raise ValueError(f"unknown mode {mode}")
            loss = loss / accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (step + 1) % accum == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise MemoryError(
                    "CUDA OOM: lower --batch-size (try 32), keep --grad-accum 2, "
                    "ensure amp is on. Nothing was silently shrunk."
                ) from e
            raise
        running += float(loss.item()) * x.size(0) * accum
        n += x.size(0)
        nbatches += 1
    # flush remainder grads (when batches don't divide evenly into accum steps)
    if nbatches % accum != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return running / max(1, n)


def main() -> int:
    ap = argparse.ArgumentParser(description="8GB-safe training (verdict C2/C3 baselines)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dataset", default=None, help="overrides config dataset")
    ap.add_argument("--mode", default="exit_kd", choices=["teacher", "student", "exit_ce", "exit_kd"])
    ap.add_argument("--arch", default=None, help="overrides student arch")
    ap.add_argument("--teacher-arch", default=None)
    ap.add_argument("--teacher-ckpt", default=None, help="required for exit_kd / student+KD")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--synthetic", action="store_true", help="force synthetic data (smoke test)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--grad-accum", type=int, default=None, help="overrides train.grad_accum")
    ap.add_argument("--num-workers", type=int, default=None, help="overrides train.num_workers (0 on Windows)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = (args.dataset or cfg.get("dataset", "cifar10")).lower()
    tc = dict(cfg.get("train", {}))
    if args.grad_accum is not None:
        tc["grad_accum"] = max(1, int(args.grad_accum))
    if args.num_workers is not None:
        tc["num_workers"] = max(0, int(args.num_workers))
    batch_size = int(args.batch_size or tc.get("batch_size", 64))
    epochs_key = {"teacher": "epochs_teacher", "student": "epochs_student"}.get(args.mode, "epochs_exit")
    epochs = int(args.epochs or tc.get(epochs_key, 20))
    lr = float(args.lr or tc.get("lr", 0.05))
    device = resolve_device(args.device or cfg.get("device", "auto"))
    set_seed(int(cfg.get("seed", 42)))
    results = Path(args.results or cfg.get("results_root", "./results"))
    ensure_dir(results)

    train_loader, test_loader, nc, ch = get_dataloaders(
        dataset, cfg.get("data_root", "./data"), batch_size,
        int(tc.get("num_workers", 0)), int(cfg.get("seed", 42)),
        force_synthetic=args.synthetic)

    if args.mode == "teacher":
        arch = args.arch or args.teacher_arch or cfg.get("teacher", "resnet18")
    else:
        arch = args.arch or cfg.get("student", "resnet14")
    model = build_model(dataset, arch, nc, ch).to(device)

    teacher = None
    if args.mode in ("exit_kd",) or (args.mode == "student" and args.teacher_ckpt):
        if not args.teacher_ckpt:
            print("ERROR: --teacher-ckpt is required for exit_kd (train teacher first).", flush=True)
            return 2
        teacher, tarch = model_from_checkpoint(
            args.teacher_ckpt, dataset, device, arch=args.teacher_arch,
            fallback_arch=cfg.get("teacher", "resnet18"))
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    opt = torch.optim.SGD(model.parameters(), lr=lr,
                          momentum=float(tc.get("momentum", 0.9)),
                          weight_decay=float(tc.get("weight_decay", 5e-4)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    scaler = _scaler(device, bool(tc.get("amp", True)))

    synthetic = is_synthetic(train_loader)
    if synthetic:
        print("[train] WARNING: SYNTHETIC data (random labels): checkpoint is plumbing only.",
              flush=True)
    print(f"[train] mode={args.mode} arch={arch} dataset={dataset} "
          f"device={device} epochs={epochs} batch={batch_size} vram={vram_info()}", flush=True)
    t0 = time.time()
    hist = []
    try:
        for ep in range(epochs):
            loss = train_one_epoch(model, train_loader, opt, device, scaler, args.mode, teacher, tc)
            sched.step()
            acc = evaluate_accuracy(model, test_loader, device)
            per_head = []
            if hasattr(model, "forward_exit"):
                for h in range(model.num_heads):  # type: ignore[attr-defined]
                    per_head.append(round(evaluate_accuracy(model, test_loader, device, h), 4))
            print(f"[train] epoch {ep+1}/{epochs} loss={loss:.4f} acc={acc:.4f} heads={per_head}", flush=True)
            hist.append({"epoch": ep + 1, "loss": loss, "acc": acc, "per_head_acc": per_head})
    except MemoryError as e:
        print(f"[train] {e}", flush=True)
        return 2
    except Exception:
        traceback.print_exc()
        return 1

    tag = f"{dataset}_{args.mode}_{arch}_e{epochs}_b{batch_size}"
    ckpt = results / f"{tag}.pt"
    torch.save({"state_dict": model.state_dict(), "arch": arch, "dataset": dataset,
                "mode": args.mode, "epochs": epochs, "synthetic_data": synthetic}, ckpt)
    save_json({"tag": tag, "mode": args.mode, "arch": arch, "dataset": dataset,
               "epochs": epochs, "batch_size": batch_size, "lr": lr,
               "grad_accum": int(tc.get("grad_accum", 1)),
               "synthetic_data": synthetic, "seed": int(cfg.get("seed", 42)),
               "history": hist, "seconds": time.time() - t0,
               "git": get_git_hash("."), "vram": vram_info()}, results / f"{tag}.json")
    print(f"[train] saved {ckpt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
