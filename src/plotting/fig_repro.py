"""Fig.: (a) cross-session spread of identical cells; (b) unlocked / locked per-exit energy."""
import matplotlib.pyplot as plt
import numpy as np

from . import style


def _sessions(ax, res):
    a = res.audit_compare()
    groups = [("cifar10", 1, "compact", "1C"), ("cifar10", 8, "compact", "8C"),
              ("cifar10", 8, "batchwait", "8W"), ("har", 64, "compact", "64C"),
              ("har", 64, "batchwait", "64W")]
    rng = np.random.default_rng(1)  # jitter only
    for gi, (ds, b, mode, _) in enumerate(groups):
        v = [p["spread_pct"] for c in a["cells"] for p in c["per_index"]
             if c["cell"]["dataset"] == ds and c["cell"]["batch"] == b and c["cell"]["mode"] == mode]
        color = style.TEAL if ds == "cifar10" else style.ORANGE
        ax.scatter(gi + rng.uniform(-0.18, 0.18, len(v)), v, s=7, color=color, lw=0, zorder=3)
    ax.axhline(a.get("within_pct", 10.0), color=style.GREY, ls="--", lw=0.7)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[3] for g in groups], fontsize=6.5)
    ax.text(1.0, 13.2, "CIFAR-10", color=style.TEAL, ha="center", va="top", fontsize=6.5)
    ax.text(3.5, 13.2, "HAR", color=style.ORANGE, ha="center", va="top", fontsize=6.5)
    ax.set_xlabel("Batch, runtime")
    ax.set_ylabel("A/B spread (%)")
    ax.set_ylim(0, 14)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.grid(axis="x", visible=False)
    ax.set_title("(a) Two sessions, same cell", pad=2)


def _clock(ax, res):
    groups = [("cifar10", 1, "CIFAR, 1"), ("cifar10", 8, "CIFAR, 8"), ("har", 1, "HAR, 1")]
    x0, ticks, labels = 0.0, [], []
    w = 0.38
    for ds, b, name in groups:
        lk = res.batch_row(res.eval(ds), b)
        ul = res.batch_row(res.eval(ds, unlocked=True), b)
        n = len(lk["energy_cascade"]["mean_j"])
        for mode, color, dx in (("prefix", style.INDIGO, -w / 2), ("cascade", style.TEAL, w / 2)):
            r = np.array(ul["energy_" + mode]["mean_j"]) / np.array(lk["energy_" + mode]["mean_j"])
            ax.bar(x0 + np.arange(n) + dx, r, w * 0.92, color=color, alpha=0.85, lw=0,
                   label="Prefix" if mode == "prefix" else "Cascade")
        ticks.append(x0 + (n - 1) / 2)
        labels.append(name)
        x0 += n + 0.8
    ax.axhline(1.0, color=style.GREY, lw=0.7, ls="--")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("Unlocked / locked")
    ax.set_ylim(0, 4.7)
    ax.grid(axis="x", visible=False)
    ax.set_title("(b) Default vs. locked clock, per exit", pad=2)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[:2], l[:2], loc="upper left", ncol=2, handlelength=1.0, columnspacing=0.8,
              borderaxespad=0.1)


def make(res, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(style.COL_W, 1.6),
                             gridspec_kw={"width_ratios": [2.1, 3], "wspace": 0.42})
    _sessions(axes[0], res)
    _clock(axes[1], res)
    return style.save(fig, out_dir, "fig_repro")
