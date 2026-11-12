"""Fig.: CIFAR-10 end-to-end change vs. the full model, forest plot (sessions A and B)."""
import matplotlib.pyplot as plt
import numpy as np

from . import style
from .data import CIFAR_POLICIES

BATCHES = (1, 8, 16)


def make(res, out_dir):
    sess = {s: res.e2e("cifar10", s) for s in ("A", "B")}
    fig, axes = plt.subplots(1, 3, figsize=(3.28, 1.85), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    y = np.arange(len(CIFAR_POLICIES))[::-1]
    for ax, b in zip(axes, BATCHES):
        ax.axvline(0, color=style.GREY, lw=0.7, ls="--", zorder=1)
        for s, color, marker, dy in (("A", style.TEAL, "o", 0.14), ("B", style.ORANGE, "D", -0.14)):
            rows = sess[s]
            f = res.full(rows, b)
            if f is None:
                continue
            for yi, (pid, _) in zip(y, CIFAR_POLICIES):
                r = res.row(rows, pid, b)
                if r is None:
                    continue
                q, h = res.ratio_ci(r, f)
                ax.errorbar(100 * (q - 1), yi + dy, xerr=100 * h, fmt=marker, ms=3.0,
                            color=color, mfc=color, mec="white", mew=0.3, elinewidth=0.8,
                            zorder=3)
        ax.set_title(r"$b$ = %d" % b, pad=2)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, len(CIFAR_POLICIES) - 0.4)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([n for _, n in CIFAR_POLICIES])
    for ax in axes:
        ax.tick_params(axis="y", length=0)
    axes[0].set_xlim(-40, 30)
    axes[1].set_xlim(-18, 31)
    axes[2].set_xlim(-18, 31)
    axes[1].set_xlabel("Energy change vs. full model at the same batch size (%)")
    h1 = plt.Line2D([], [], color=style.TEAL, marker="o", ls="", ms=3.5, label="Session A")
    h2 = plt.Line2D([], [], color=style.ORANGE, marker="D", ls="", ms=3.2, label="Session B")
    fig.legend(handles=[h1, h2], loc="upper center", ncol=2, bbox_to_anchor=(0.6, 1.0),
               handletextpad=0.2, columnspacing=1.2)
    fig.subplots_adjust(top=0.8)
    return style.save(fig, out_dir, "fig_cifar_e2e")
