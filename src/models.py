"""Models: teacher / student / early-exit student + KD losses.

All models expose a common early-exit interface:
    model.num_heads  -> int (early heads + final head)
    model.forward_exit(x, idx) -> logits for head idx only (efficient prefix)
    model.forward_all(x) -> list[logits] for every head

CIFAR track uses EarlyExitResNet. Pervasive track (KWS/HAR) uses SmallExitCNN
with the same interface but fewer stages. Both run on CPU and fit in 8GB.

FLOP estimates are measured with forward hooks (MACs), not guessed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# ResNet blocks (CIFAR-style: 3x3 stride-1 stem, no maxpool)
# ----------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ExitHead(nn.Module):
    """Tiny classifier head: GAP + FC. Keeps exit overhead small and measurable."""

    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(x).flatten(1))


class EarlyExitResNet(nn.Module):
    """CIFAR ResNet with 3 early exits + final head (num_heads=4).

    exit 0 -> after stage1 (64ch), 1 -> after stage2 (128ch),
    2 -> after stage3 (256ch), 3 -> final (512ch).
    forward_exit(x, idx) executes ONLY the prefix needed for head idx.
    """

    def __init__(self, depths=(2, 2, 2, 2), in_channels: int = 3, num_classes: int = 10,
                 base: int = 64):
        super().__init__()
        d1, d2, d3, d4 = depths
        self.conv1 = nn.Conv2d(in_channels, base, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(base)
        self.stage1 = self._stage(base, base, d1, stride=1)
        self.stage2 = self._stage(base, base * 2, d2, stride=2)
        self.stage3 = self._stage(base * 2, base * 4, d3, stride=2)
        self.stage4 = self._stage(base * 4, base * 8, d4, stride=2)
        self.exit1 = ExitHead(base, num_classes)
        self.exit2 = ExitHead(base * 2, num_classes)
        self.exit3 = ExitHead(base * 4, num_classes)
        self.final = ExitHead(base * 8, num_classes)
        self.num_heads = 4
        self.head_dims = [base, base * 2, base * 4, base * 8]

    @staticmethod
    def _stage(in_ch: int, out_ch: int, n: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, max(1, n)):
            layers.append(BasicBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def _stem(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn1(self.conv1(x)))

    def forward_exit(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= self.num_heads:
            raise ValueError(f"exit idx {idx} out of range [0, {self.num_heads})")
        h = self._stem(x)
        h = self.stage1(h)
        if idx == 0:
            return self.exit1(h)
        h = self.stage2(h)
        if idx == 1:
            return self.exit2(h)
        h = self.stage3(h)
        if idx == 2:
            return self.exit3(h)
        h = self.stage4(h)
        return self.final(h)

    def iter_heads(self, x: torch.Tensor):
        """Lazily yield head logits in order; stages after a `break` never run.

        This is the deployed early-exit execution path (a cascade): exiting at
        head k costs the prefix PLUS heads 0..k-1 and their exit decisions.
        """
        h = self.stage1(self._stem(x))
        yield self.exit1(h)
        h = self.stage2(h)
        yield self.exit2(h)
        h = self.stage3(h)
        yield self.exit3(h)
        yield self.final(self.stage4(h))

    def exit_blocks(self) -> list[tuple]:
        """[(block, head), ...]: block k maps the previous feature map to the
        input of head k. Lets a runtime drop exited samples between blocks
        (per-sample early exit inside a batch = compaction / rebatching)."""
        return [(lambda x: self.stage1(self._stem(x)), self.exit1),
                (self.stage2, self.exit2), (self.stage3, self.exit3),
                (self.stage4, self.final)]

    def forward_all(self, x: torch.Tensor) -> list[torch.Tensor]:
        h = self._stem(x)
        h1 = self.stage1(h)
        o1 = self.exit1(h1)
        h2 = self.stage2(h1)
        o2 = self.exit2(h2)
        h3 = self.stage3(h2)
        o3 = self.exit3(h3)
        h4 = self.stage4(h3)
        o4 = self.final(h4)
        return [o1, o2, o3, o4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_exit(x, self.num_heads - 1)


class SmallExitCNN(nn.Module):
    """Lightweight CNN for KWS (1ch melspec) / HAR (sensor windows).

    3 conv stages + 3 heads (2 early + final). Same interface as ResNet above.
    Input: (N, in_channels, H, W). Any H/W works (GAP heads).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 12,
                 widths=(32, 64, 128)):
        super().__init__()
        w1, w2, w3 = widths
        self.conv1 = nn.Conv2d(in_channels, w1, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(w1)
        self.conv2 = nn.Conv2d(w1, w2, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(w2)
        self.conv3 = nn.Conv2d(w2, w3, 3, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(w3)
        self.exit1 = ExitHead(w1, num_classes)
        self.exit2 = ExitHead(w2, num_classes)
        self.final = ExitHead(w3, num_classes)
        self.num_heads = 3

    def forward_exit(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= self.num_heads:
            raise ValueError(f"exit idx {idx} out of range [0, {self.num_heads})")
        h = F.relu(self.bn1(self.conv1(x)))
        if idx == 0:
            return self.exit1(h)
        h = F.relu(self.bn2(self.conv2(h)))
        if idx == 1:
            return self.exit2(h)
        h = F.relu(self.bn3(self.conv3(h)))
        return self.final(h)

    def iter_heads(self, x: torch.Tensor):
        """Lazily yield head logits in order (see EarlyExitResNet.iter_heads)."""
        h = F.relu(self.bn1(self.conv1(x)))
        yield self.exit1(h)
        h = F.relu(self.bn2(self.conv2(h)))
        yield self.exit2(h)
        yield self.final(F.relu(self.bn3(self.conv3(h))))

    def exit_blocks(self) -> list[tuple]:
        """[(block, head), ...] (see EarlyExitResNet.exit_blocks)."""
        return [(lambda x: F.relu(self.bn1(self.conv1(x))), self.exit1),
                (lambda h: F.relu(self.bn2(self.conv2(h))), self.exit2),
                (lambda h: F.relu(self.bn3(self.conv3(h))), self.final)]

    def forward_all(self, x: torch.Tensor) -> list[torch.Tensor]:
        h1 = F.relu(self.bn1(self.conv1(x)))
        o1 = self.exit1(h1)
        h2 = F.relu(self.bn2(self.conv2(h1)))
        o2 = self.exit2(h2)
        h3 = F.relu(self.bn3(self.conv3(h2)))
        o3 = self.final(h3)
        return [o1, o2, o3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_exit(x, self.num_heads - 1)


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------
def build_model(dataset: str = "cifar10", arch: str = "resnet14",
                num_classes: int = 10, in_channels: int = 3) -> nn.Module:
    """arch: resnet34|resnet18|resnet14 (CIFAR) or dscnn-s|harcnn|tinycnn (pervasive).

    resnet34/resnet18 are teachers; resnet14 is the default early-exit student.
    """
    arch = arch.lower()
    dataset = dataset.lower()
    if arch in ("resnet18", "teacher"):
        return EarlyExitResNet(depths=(2, 2, 2, 2), in_channels=in_channels,
                               num_classes=num_classes)
    if arch in ("resnet34",):
        return EarlyExitResNet(depths=(3, 4, 6, 3), in_channels=in_channels,
                               num_classes=num_classes)
    if arch in ("resnet14", "student", "resnet18small"):
        return EarlyExitResNet(depths=(2, 1, 1, 1), in_channels=in_channels,
                               num_classes=num_classes)
    if arch in ("dscnn-s", "harcnn", "tinycnn", "small"):
        ch = 1 if dataset == "kws" else in_channels
        nc = 12 if dataset == "kws" else num_classes
        return SmallExitCNN(in_channels=ch, num_classes=nc)
    raise ValueError(f"unknown arch {arch!r} (try resnet18/resnet14/dscnn-s)")


def load_checkpoint(path: str, device) -> dict:
    """Load a train.py checkpoint safely (weights_only: no pickle code execution).

    Returns a dict with at least 'state_dict'; 'arch'/'dataset' when saved by train.py.
    """
    sd = torch.load(path, map_location=device, weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        return sd
    return {"state_dict": sd}


def model_from_checkpoint(path: str, dataset: str, device, arch: str | None = None,
                          fallback_arch: str = "resnet14") -> tuple[nn.Module, str]:
    """Build the model a checkpoint was trained as, then load its weights.

    Arch precedence: explicit `arch` > arch stored in ckpt > fallback_arch.
    Raises ValueError if the ckpt was trained on a different dataset.
    """
    ck = load_checkpoint(path, device)
    ck_ds = ck.get("dataset")
    if ck_ds is not None and str(ck_ds).lower() != dataset.lower():
        raise ValueError(f"checkpoint {path} was trained on {ck_ds!r}, not {dataset!r}")
    arch = arch or ck.get("arch") or fallback_arch
    nc, ch, _ = default_io(dataset)
    model = build_model(dataset, arch, nc, ch).to(device)
    model.load_state_dict(ck["state_dict"])
    return model, arch


def default_io(dataset: str) -> tuple[int, int, tuple[int, int]]:
    """(num_classes, in_channels, (H, W)) per dataset track."""
    d = dataset.lower()
    if d == "cifar10":
        return 10, 3, (32, 32)
    if d == "kws":
        return 12, 1, (49, 40)   # SpeechCommands mel window placeholder
    if d == "har":
        return 6, 9, (1, 128)    # UCI-HAR: 9 inertial channels x 128 samples (2.56 s @ 50 Hz)
    raise ValueError(f"unknown dataset {dataset!r} (cifar10|kws|har)")


# ----------------------------------------------------------------------------
# KD losses
# ----------------------------------------------------------------------------
def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
            targets: torch.Tensor, alpha: float = 0.7, temp: float = 4.0) -> torch.Tensor:
    """Hinton KD: alpha * KL(T) + (1-alpha) * CE. alpha=1 -> pure KD, alpha=0 -> pure CE."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"kd alpha must be in [0,1], got {alpha}")
    ce = F.cross_entropy(student_logits, targets)
    if alpha <= 0.0:
        return ce
    log_p = F.log_softmax(student_logits.float() / temp, dim=1)
    q = F.softmax(teacher_logits.float() / temp, dim=1)
    kl = F.kl_div(log_p, q, reduction="batchmean") * (temp * temp)
    return alpha * kl + (1.0 - alpha) * ce


def exit_kd_loss(student_heads: list[torch.Tensor], teacher_final: torch.Tensor,
                 targets: torch.Tensor, alpha: float = 0.7, temp: float = 4.0,
                 exit_weights: list[float] | None = None) -> torch.Tensor:
    """Exit-aware KD: every head is distilled from the teacher final logits.

    NOTE (LEAP arxiv:2605.01058 warning): layer-aligned distillation can suppress
    the representational convergence early exits rely on. Monitor per-exit
    accuracy during training; if early heads collapse, lower alpha for heads 0-1.
    """
    if exit_weights is None:
        exit_weights = [1.0] * len(student_heads)
    if len(exit_weights) != len(student_heads):
        raise ValueError("exit_weights length must match number of heads")
    total = 0.0
    for w, s in zip(exit_weights, student_heads):
        total = total + w * kd_loss(s, teacher_final, targets, alpha, temp)
    return total / max(1e-9, sum(exit_weights))


# ----------------------------------------------------------------------------
# Params + FLOPs (hook-measured MACs per exit)
# ----------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def flops_per_exit(model: nn.Module, input_shape=(1, 3, 32, 32),
                   device: str = "cpu", cascade: bool = False) -> list[int]:
    """Measure MACs for each exit via forward hooks. Increasing with idx.

    cascade=False: prefix only (forward_exit). cascade=True: prefix + every
    earlier exit head (iter_heads), i.e. what a deployed early-exit run pays.
    """
    try:
        orig_device = next(model.parameters()).device
    except StopIteration:
        orig_device = torch.device(device)
    was_training = model.training
    model = model.to(device).eval()
    dummy = torch.zeros(*input_shape, device=device)
    n = int(getattr(model, "num_heads", 1))
    out: list[int] = []
    for idx in range(n):
        counter = {"macs": 0}

        def _conv_hook(mod, inp, outp):
            # inp[0]: (N,Cin,Hin,Win); outp: (N,Cout,Hout,Wout)
            try:
                x, y = inp[0], outp
                macs = (y.shape[0] * y.shape[1] * y.shape[2] * y.shape[3]
                        * x.shape[1] * mod.kernel_size[0] * mod.kernel_size[1]
                        // mod.groups)
                counter["macs"] += int(macs)
            except Exception:
                pass

        def _lin_hook(mod, inp, outp):
            try:
                counter["macs"] += int(inp[0].shape[0] * mod.in_features * mod.out_features)
            except Exception:
                pass

        handles = []
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                handles.append(m.register_forward_hook(_conv_hook))
            elif isinstance(m, nn.Linear):
                handles.append(m.register_forward_hook(_lin_hook))
        try:
            with torch.no_grad():
                if cascade and hasattr(model, "iter_heads"):
                    for h, _ in enumerate(model.iter_heads(dummy)):  # type: ignore[attr-defined]
                        if h == idx:
                            break
                elif hasattr(model, "forward_exit"):
                    model.forward_exit(dummy, idx)  # type: ignore[attr-defined]
                else:
                    model(dummy)
        finally:
            for h in handles:
                h.remove()
        out.append(int(counter["macs"]))
    try:
        model.to(orig_device)
    except Exception:
        pass
    model.train(was_training)
    # Guard against degenerate zero counts (e.g. mocked modules in tests)
    return [max(1, v) for v in out]
