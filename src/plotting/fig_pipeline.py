"""Fig.: audit pipeline (boxes and arrows), drawn with matplotlib patches only."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from . import style

W, H = 100.0, 46.0
BW, BH = 27.0, 11.5


def _box(ax, x, y, title, sub, face, edge):
    ax.add_patch(FancyBboxPatch((x - BW / 2, y - BH / 2), BW, BH,
                                boxstyle="round,pad=0.4,rounding_size=1.6",
                                fc=face, ec=edge, lw=0.8))
    ax.text(x, y + 2.3, title, ha="center", va="center", fontsize=7.5,
            fontweight="bold", color=style.INK)
    ax.text(x, y - 2.6, sub, ha="center", va="center", fontsize=6.6, color=style.INK)


def _arrow(ax, p, q, color=style.INK, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6.5, lw=0.7,
                                 color=color, connectionstyle="arc3,rad=%g" % rad,
                                 shrinkA=0, shrinkB=0))


def make(res, out_dir):
    fig, ax = plt.subplots(figsize=(style.COL_W, 1.6))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    xs = (15.0, 50.0, 85.0)
    yt, yb = 38.5, 19.5
    g, gi = "#EAF3E3", style.GREEN
    p, pi = "#ECEEF7", style.INDIGO
    m, mi = "#E3F2F0", style.TEAL
    a, ai = "#FBEBDD", style.ORANGE

    _box(ax, xs[0], yt, "Train", "KD, 3 seeds", g, gi)
    _box(ax, xs[1], yt, "Per-exit energy", r"prefix, cascade: $\varepsilon_h$", p, pi)
    _box(ax, xs[2], yt, "Additive table", r"$\hat{E}_{\rm tab}=\Sigma_h\, p_h\varepsilon_h$", p, pi)
    _box(ax, xs[0], yb, "Calibrate", r"LTT, policies: $p_h$", g, gi)
    _box(ax, xs[1], yb, "Real runtime", r"meter: $E_{\rm meas}$", m, mi)
    _box(ax, xs[2], yb, "Audit", r"$\delta$, gate, A/B", a, ai)

    hw, hh = BW / 2 + 0.6, BH / 2 + 0.6
    _arrow(ax, (xs[0] + hw, yt), (xs[1] - hw, yt))
    _arrow(ax, (xs[1] + hw, yt), (xs[2] - hw, yt))
    _arrow(ax, (xs[0], yt - hh), (xs[0], yb + hh))
    _arrow(ax, (xs[0] + hw, yb), (xs[1] - hw, yb))
    _arrow(ax, (xs[1] + hw, yb), (xs[2] - hw, yb))
    _arrow(ax, (xs[2], yt - hh), (xs[2], yb + hh))
    # exit fractions p_h feed the table
    _arrow(ax, (xs[0] + hw, yb + 3.5), (xs[2] - hw, yt - 3.5), color=style.GREY, rad=0.0)

    ax.text(W / 2, 4.2, "NVML energy counter  |  idle subtracted  |  clock locked at 1500 MHz  |"
            "  10 randomized windows,\n" r"each $\geq$1000 iterations and $\geq$3 s  |"
            "  Student-$t$ 95% CI", ha="center", va="center", fontsize=6.3, color=style.GREY,
            linespacing=1.25)
    return style.save(fig, out_dir, "fig_pipeline")
