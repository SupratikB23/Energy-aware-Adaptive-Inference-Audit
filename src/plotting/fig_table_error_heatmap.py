"""Fig.: table error delta for every policy cell (session A), annotated heatmaps."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from . import style
from .data import CIFAR_POLICIES, HAR_POLICIES, SHORT

CIFAR_COLS = [(1, "compact"), (8, "compact"), (8, "batchwait"), (16, "compact"), (16, "batchwait")]
HAR_COLS = [(1, "compact"), (64, "compact"), (64, "batchwait")]
CMAP = LinearSegmentedColormap.from_list("under", ["#FDF1E7", style.ORANGE, "#7A3410"])
VMIN, VMAX = 20, 90  # |delta| in percent

# Fixed cell geometry (inches) so both panels have identical cells.
CW, CH = 0.25, 0.19


def _grid(res, dataset, policies, cols):
    rows = res.e2e(dataset, "A")
    m = np.full((len(policies), len(cols)), np.nan)
    for i, (pid, _) in enumerate(policies):
        for j, (b, mode) in enumerate(cols):
            r = res.row(rows, pid, b, mode)
            if r is not None:
                m[i, j] = r["pred_table_err_pct"]
    return m


def _draw(ax, m, policies, cols, title):
    ax.imshow(-m, cmap=CMAP, vmin=VMIN, vmax=VMAX, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, ("%d" % round(v)).replace("-", "−"), ha="center", va="center", fontsize=6.5,
                    color="white" if -v > 60 else style.INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["%d%s" % (b, "C" if mo == "compact" else "W") for b, mo in cols])
    ax.set_yticks(range(len(policies)))
    ax.set_yticklabels([SHORT[p] for p, _ in policies])
    ax.tick_params(which="both", length=0)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(policies)), minor=True)
    ax.grid(which="minor", color="white", lw=1.0)
    ax.grid(which="major", visible=False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, pad=3)


def make(res, out_dir):
    mc = _grid(res, "cifar10", CIFAR_POLICIES, CIFAR_COLS)
    mh = _grid(res, "har", HAR_POLICIES, HAR_COLS)
    fw, fh = style.COL_W, 1.42
    fig = plt.figure(figsize=(fw, fh))
    top = 0.22
    left_c, left_h = 0.42, 2.45
    axc = fig.add_axes([left_c / fw, (fh - top - CH * 5) / fh, CW * 5 / fw, CH * 5 / fh])
    axh = fig.add_axes([left_h / fw, (fh - top - CH * 3) / fh, CW * 3 / fw, CH * 3 / fh])
    _draw(axc, mc, CIFAR_POLICIES, CIFAR_COLS, "CIFAR-10")
    _draw(axh, mh, HAR_POLICIES, HAR_COLS, "UCI-HAR")
    axc.set_xlabel("Batch size, runtime")
    fig.text((left_h + CW * 1.5) / fw, (fh - top - CH * 3 - 0.36) / fh,
             "C = compaction\nW = batch-wait", ha="center", va="top", fontsize=6.5,
             color=style.GREY)
    return style.save(fig, out_dir, "fig_table_error")
