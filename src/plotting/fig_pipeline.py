"""Fig. 1: audit pipeline (full text width). White boxes, black outlines, black text.

The dashed container marks the controlled measurement environment (the GPU under a
locked clock); everything inside it is metered, everything outside is offline.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from . import style

W, H = 100.0, 40.0
K = style.INK


def _box(ax, cx, cy, w, h, title, lines, lw=0.8, fs=6.6):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.3,rounding_size=0.8",
                                fc="white", ec=K, lw=lw, zorder=2))
    if not title:
        return
    top = cy + (len(lines) * 2.25) / 2
    ax.text(cx, top + 0.3, title, ha="center", va="center", fontsize=7.2,
            fontweight="bold", color=K, zorder=3)
    for i, s in enumerate(lines):
        ax.text(cx, top - 2.45 * (i + 1) + 0.3, s, ha="center", va="center",
                fontsize=fs, color=K, zorder=3)


def _arrow(ax, p, q, label, lpos, ha="center", va="bottom"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.75,
                                 color=K, shrinkA=0, shrinkB=0, zorder=1))
    ax.text(lpos[0], lpos[1], label, ha=ha, va=va, fontsize=6.4, color=K, style="italic",
            zorder=3)


def make(res, out_dir):
    fig, ax = plt.subplots(figsize=(style.PAGE_W - 0.06, 2.75))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # columns: offline (left), metered (middle, inside the container), offline (right)
    xl, xm, xr = 13.0, 50.0, 87.0
    yt, yb = 29.0, 12.5
    bw_side, bw_mid, bh = 24.0, 26.0, 9.6
    y3, y4 = 26.5, 12.6

    # dashed container: controlled measurement environment
    cx0, cx1, cy0, cy1 = 34.0, 66.0, 0.8, 39.2
    ax.add_patch(FancyBboxPatch((cx0, cy0), cx1 - cx0, cy1 - cy0,
                                boxstyle="round,pad=0,rounding_size=1.2",
                                fc="none", ec=K, lw=0.8, ls=(0, (4, 2.5)), zorder=0))
    ax.text((cx0 + cx1) / 2, cy1 - 1.6, "Controlled measurement environment",
            ha="center", va="center", fontsize=7.0, fontweight="bold", color=K)
    ax.text((cx0 + cx1) / 2, cy1 - 3.9, "RTX 5060 Laptop GPU, clock locked at 1500 MHz",
            ha="center", va="center", fontsize=6.3, color=K)

    _box(ax, xl, yt, bw_side, bh, "(1) Train",
         ["ResNet-14 (CIFAR-10), 1-D CNN (HAR)", "exit-aware KD, seeds 42/43/44"])
    _box(ax, xl, yb, bw_side, bh, "(2) Calibrate policies",
         [r"LTT on calibration half: $\tau$", "per-head thresholds P1 to P3"])

    _box(ax, xm, y3, bw_mid, bh, "(3) Per-exit measurement",
         ["prefix and cascade runs per exit", r"$\Rightarrow$ per-exit energy $\varepsilon_h$"])
    _box(ax, xm, y4, bw_mid, bh, "(4) Real early-exit runtime",
         ["compaction / batch-wait, deployment half",
          r"$\Rightarrow$ metered $E_{\rm meas}$, exit fractions $p_h$"])

    _box(ax, xr, yt, bw_side, bh, "(5) Additive price",
         [r"$\hat{E}_{\rm tab}=\Sigma_h\,p_h\,\varepsilon_h$", "and operation-count price"])
    _box(ax, xr, yb, bw_side, bh, "(6) Audit",
         [r"table error $\delta=(\hat{E}-E_{\rm meas})/E_{\rm meas}$",
          "gate check, session A vs. B"])

    # protocol strip inside the container
    _box(ax, xm, 3.9, 30.0, 3.8, "", [], lw=0.6)
    ax.text(xm, 4.75, "Protocol: NVML energy counter, idle subtracted, 10 windows",
            ha="center", va="center", fontsize=5.9, color=K, zorder=3)
    ax.text(xm, 3.05, r"($\geq$1000 iterations and $\geq$3 s each), Student-$t$ 95% CI",
            ha="center", va="center", fontsize=5.9, color=K, zorder=3)

    hs, hm = bw_side / 2 + 0.3, bw_mid / 2 + 0.3
    vt, vb = yt - bh / 2 - 0.3, yb + bh / 2 + 0.3
    _arrow(ax, (xl + hs, y3), (xm - hm, y3), "checkpoints", (30.0, y3 + 0.5))
    _arrow(ax, (xl, vt), (xl, vb), "checkpoints", (xl + 0.8, (vt + vb) / 2), "left", "center")
    _arrow(ax, (xl + hs, y4), (xm - hm, y4), r"$\tau$, P1-P3", (30.0, y4 + 0.5))
    _arrow(ax, (xm + hm, y3), (xr - hs, y3), r"$\varepsilon_h$", (70.0, y3 + 0.5))
    _arrow(ax, (xm + hm, y4 + 2.5), (xr - hs, yt - 4.0), r"$p_h$", (70.4, 19.6), "right", "center")
    _arrow(ax, (xm + hm, y4 - 2.0), (xr - hs, y4 - 2.0), r"$E_{\rm meas}$", (70.0, y4 - 1.5))
    _arrow(ax, (xr, vt), (xr, vb), r"$\hat{E}_{\rm tab}$", (xr + 0.8, (vt + vb) / 2),
           "left", "center")
    return style.save(fig, out_dir, "fig_pipeline", max_w=style.PAGE_W)
