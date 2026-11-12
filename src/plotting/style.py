"""Shared figure style for the paper: Times New Roman, IEEE column sizes, fixed palette.

Every figure is drawn at its final printed size (IEEE single column = 3.5 in), so the
8 pt text here is 8 pt in the PDF. Fonts are embedded as TrueType (pdf.fonttype 42),
which the PeRConAI call requires ("all fonts embedded").
"""
import itertools
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

COL_W = 3.5    # IEEE single-column width, inches
PAGE_W = 7.16  # IEEE two-column text width, inches

# Palette: teal / orange / indigo / green, plus neutrals.
TEAL = "#2A9D8F"
ORANGE = "#E07A3F"
INDIGO = "#4F5DA8"
GREEN = "#6A994E"
GREY = "#6B6B6B"
LIGHT = "#E9ECEF"
INK = "#222222"

_TIMES_FILES = ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf")


def _register_times():
    """Register Times New Roman explicitly; fail loudly if it is missing."""
    font_dirs = [os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
                 "/usr/share/fonts/truetype/msttcorefonts", "/Library/Fonts"]
    found = False
    for d in font_dirs:
        for f in _TIMES_FILES:
            p = os.path.join(d, f)
            if os.path.exists(p):
                font_manager.fontManager.addfont(p)
                found = True
    if not found:
        raise RuntimeError("Times New Roman not found; install it before plotting "
                           "(the paper requires Times text in figures).")


def apply():
    _register_times()
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.4,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "errorbar.capsize": 1.8,
        "lines.linewidth": 1.0,
        "hatch.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "figure.dpi": 150,
    })


def text_overlaps(fig):
    """Return pairs of visible, non-empty text artists whose boxes intersect.

    Used as a hard check after every figure: the paper must have no overlapping text.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    # Tick labels that matplotlib does not draw (ticks outside the view) are excluded.
    all_ticklabels, drawn = set(), set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                all_ticklabels.update((id(tk.label1), id(tk.label2)))
            if not ax.axison:
                continue
            for tk in axis._update_ticks():
                drawn.update((id(tk.label1), id(tk.label2)))
    texts = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if id(t) in all_ticklabels and id(t) not in drawn:
            continue
        try:
            bb = t.get_window_extent(r)
        except Exception:
            continue
        if bb.width > 0 and bb.height > 0:
            texts.append((t, bb.shrunk(0.98, 0.9)))
    bad = []
    for (a, ba), (b, bb) in itertools.combinations(texts, 2):
        if ba.overlaps(bb):
            bad.append((a.get_text(), b.get_text()))
    return bad


def save(fig, out_dir, name):
    """Save PDF (for LaTeX) and PNG (for checking); raise if any text overlaps."""
    os.makedirs(out_dir, exist_ok=True)
    bad = text_overlaps(fig)
    if bad:
        raise RuntimeError("%s: overlapping text: %s" % (name, bad[:6]))
    pdf = os.path.join(out_dir, name + ".pdf")
    w_in = fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * 0.02
    if w_in > COL_W + 1e-3:
        raise RuntimeError("%s: %.2f in wide, exceeds the %.1f in column" % (name, w_in, COL_W))
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(out_dir, name + ".png"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return pdf
