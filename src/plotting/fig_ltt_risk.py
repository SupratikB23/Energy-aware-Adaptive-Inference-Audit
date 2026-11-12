"""Fig.: Learn-then-Test calibration risk vs. threshold, three seeds, with the alpha target."""
import matplotlib.pyplot as plt

from . import style

SEEDS = ((42, style.TEAL, "o"), (43, style.ORANGE, "s"), (44, style.INDIGO, "^"))


def _panel(ax, res, dataset, title):
    alpha = None
    for seed, color, marker in SEEDS:
        d = res.ltt(dataset, seed)
        cal = d["calibration"]
        alpha = cal["alpha"]
        rows = sorted(cal["rows"], key=lambda r: r["tau"])
        taus = [r["tau"] for r in rows]
        risk = [100 * r["emp_risk"] for r in rows]
        ax.plot(taus, risk, color=color, lw=0.9, marker=marker, ms=2.2, label="Seed %d" % seed)
        sel = cal["selected"]
        rs = [100 * r["emp_risk"] for r in rows if r["tau"] == sel]
        if rs:
            ax.plot([sel], rs, marker=marker, ms=5.5, mfc="none", mec=color, mew=0.9, ls="")
    ax.axhline(100 * alpha, color=style.GREY, ls="--", lw=0.7)
    ax.text(0.03, 100 * alpha - 0.15, r"$\alpha$ = %g%%" % (100 * alpha), transform=ax.get_yaxis_transform(),
            ha="left", va="top", fontsize=6.5, color=style.GREY)
    ax.set_title(title, pad=2)
    ax.set_xlabel(r"Confidence threshold $\tau$")


def make(res, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(style.COL_W, 1.55), gridspec_kw={"wspace": 0.3})
    _panel(axes[0], res, "cifar10", "CIFAR-10")
    _panel(axes[1], res, "har", "UCI-HAR")
    axes[0].set_ylabel("Calibration risk (%)")
    for ax in axes:
        ax.set_ylim(bottom=0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), columnspacing=1.2)
    fig.subplots_adjust(top=0.8)
    return style.save(fig, out_dir, "fig_ltt")
