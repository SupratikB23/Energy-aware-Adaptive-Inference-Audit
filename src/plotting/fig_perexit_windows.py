"""Fig.: per-exit energy, prefix vs. cascade, at batch 1: bars (mean, 95% CI) plus every window.

Ten windows per cell are too few for a violin or density estimate, so each window is
drawn as a point instead of inventing a distribution shape.
"""
import matplotlib.pyplot as plt
import numpy as np

from . import style


def _panel(ax, res, dataset, title, rng):
    br = res.batch_row(res.eval(dataset), 1)
    w = 0.36
    names = None
    for k, (mode, color, dx) in enumerate((("prefix", style.INDIGO, -w / 2),
                                            ("cascade", style.TEAL, w / 2))):
        e = br["energy_" + mode]
        mean = np.array(e["mean_j"]) * 1e3
        ci = np.array(e["ci95_j"]) * 1e3
        x = np.arange(len(mean))
        ax.bar(x + dx, mean, w * 0.92, color=color, alpha=0.28, ec=color, lw=0.6,
               label="Prefix only" if mode == "prefix" else "Deployed cascade")
        ax.errorbar(x + dx, mean, yerr=ci, fmt="none", ecolor=style.INK, elinewidth=0.7, zorder=4)
        win = res.windows_by_exit(e)
        for h in sorted(win):
            v = np.array(win[h]) * 1e3
            jit = rng.uniform(-w * 0.28, w * 0.28, size=len(v))
            ax.scatter(h + dx + jit, v, s=2.2, color=color, lw=0, zorder=3)
        names = ["Exit %d" % i for i in range(len(mean) - 1)] + ["Final"]
        if mode == "cascade":
            pref = np.array(br["energy_prefix"]["mean_j"]) * 1e3
            toll = mean[-1] / pref[-1]
            top = max(np.max(np.array(win[len(mean) - 1]) * 1e3), mean[-1] + ci[-1])
            ax.text(x[-1], top * 1.04, r"%.2f$\times$" % toll, ha="center", va="bottom",
                    fontsize=6.5, color=style.INK)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    ax.set_title(title, pad=2)
    ax.grid(axis="x", visible=False)
    ymax = max(max(np.array(br["energy_" + m]["mean_j"]) * 1e3 + np.array(br["energy_" + m]["ci95_j"]) * 1e3)
               for m in ("prefix", "cascade"))
    allw = [v * 1e3 for m in ("prefix", "cascade") for v in
            sum(res.windows_by_exit(br["energy_" + m]).values(), [])]
    ax.set_ylim(0, 1.22 * max(ymax, max(allw)))


def make(res, out_dir):
    rng = np.random.default_rng(0)  # jitter only; does not touch the data
    fig, axes = plt.subplots(1, 2, figsize=(style.COL_W, 1.7),
                             gridspec_kw={"width_ratios": [4, 3], "wspace": 0.3})
    _panel(axes[0], res, "cifar10", "CIFAR-10, $b$ = 1", rng)
    _panel(axes[1], res, "har", "UCI-HAR, $b$ = 1", rng)
    axes[0].set_ylabel("mJ per sample")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), columnspacing=1.2)
    fig.subplots_adjust(top=0.8)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=6.5)
    return style.save(fig, out_dir, "fig_perexit")
