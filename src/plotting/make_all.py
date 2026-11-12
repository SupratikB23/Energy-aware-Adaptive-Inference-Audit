"""Regenerate every paper figure from the measured result JSONs.

Usage (from the repo root):
    python -m src.plotting.make_all --results new_results/results \
        --out IEEE-conference-template-062824/figs

Writes <name>.pdf (used by LaTeX, fonts embedded) and <name>.png (for visual checks).
Fails if a result file is missing or any figure has overlapping text.
"""
import argparse
import os
import sys

if __package__ in (None, ""):  # allow `python src/plotting/make_all.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "src.plotting"

from . import style  # noqa: E402
from . import (fig_cifar_e2e, fig_ltt_risk, fig_perexit_windows, fig_pipeline,  # noqa: E402
               fig_repro, fig_table_error_heatmap, fig_table_vs_meter)
from .data import Results  # noqa: E402

FIGURES = [fig_pipeline, fig_perexit_windows, fig_ltt_risk, fig_cifar_e2e,
           fig_table_vs_meter, fig_table_error_heatmap, fig_repro]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=os.path.join("new_results", "results"),
                    help="folder that contains p3/ and p3_unlocked/")
    ap.add_argument("--out", default=os.path.join("IEEE-conference-template-062824", "figs"))
    args = ap.parse_args(argv)
    style.apply()
    res = Results(args.results)
    for mod in FIGURES:
        path = mod.make(res, args.out)
        print("wrote", path)


if __name__ == "__main__":
    main()
