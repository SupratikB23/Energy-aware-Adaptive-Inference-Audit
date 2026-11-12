"""Fig.: priced by the table, paid at the meter (batch 1, session A), grouped bars with CIs."""
import matplotlib.pyplot as plt
import numpy as np

from . import style
from .data import CIFAR_POLICIES, HAR_POLICIES, SHORT


def _panel(ax, res, dataset, policies, title):
    rows = res.e2e(dataset, "A")
    f = res.full(rows, 1)
    x = np.arange(len(policies))
    w = 0.26
    flops, table, meter, err = [], [], [], []
    for pid, _ in policies:
        r = res.row(rows, pid, 1)
        flops.append(r["pred_flops_j"] / f["j_per_sample"])
        table.append(r["pred_table_j"] / f["j_per_sample"])
        q, h = res.ratio_ci(r, f)
        meter.append(q)
        err.append(h)
    ax.bar(x - w, flops, w, color="white", ec=style.GREEN, hatch="////", lw=0.6, label="Op. count")
    ax.bar(x, table, w, color="white", ec=style.INDIGO, hatch="\\\\\\\\", lw=0.6, label="Table")
    ax.bar(x + w, meter, w, color=style.TEAL, ec=style.TEAL, lw=0.6, label="Meter (95% CI)")
    ax.errorbar(x + w, meter, yerr=err, fmt="none", ecolor=style.INK, elinewidth=0.7)
    ax.axhline(1.0, color=style.GREY, lw=0.7, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[p] for p, _ in policies])
    ax.set_title(title, pad=2)
    ax.grid(axis="x", visible=False)


def make(res, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(style.COL_W, 1.75),
                             gridspec_kw={"width_ratios": [5, 3], "wspace": 0.28})
    _panel(axes[0], res, "cifar10", CIFAR_POLICIES, "CIFAR-10, ResNet-14")
    _panel(axes[1], res, "har", HAR_POLICIES, "UCI-HAR, 97k CNN")
    axes[0].set_ylim(0, 1.5)
    axes[1].set_ylim(0, 3.8)
    axes[0].set_ylabel("Energy / full model")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0),
               handlelength=1.4, handletextpad=0.4, columnspacing=1.0)
    fig.subplots_adjust(top=0.8)
    return style.save(fig, out_dir, "fig_table_vs_meter")
