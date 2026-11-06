"""Publishability audit for measured-energy results (C1 protocol gate).

Two subcommands, both CPU-only and safe to run anywhere:

  verify   check every eval_*/e2e_* JSON against the C1 protocol: >=10 windows
           per cell, every window >=1000 iters and >=3 s, verifiably pinned
           clock, no co-tenant GPU process, real (non-synthetic) data, and a
           session id. Exit 1 if any file fails a hard gate.

  compare  cross-session agreement: group the same (dataset, arch, batch, mode)
           cell across independent sessions and report the between-session
           spread of the mean. The paper's reproducibility claim needs this
           within `--within` percent.

Usage:
    python src/audit.py verify  --results results
    python src/audit.py compare --results results --within 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load(paths: list[Path]) -> list[tuple[Path, dict]]:
    out = []
    for p in paths:
        try:
            out.append((p, json.loads(p.read_text(encoding="utf-8"))))
        except Exception as e:  # corrupt/partial file is itself a finding
            out.append((p, {"_load_error": repr(e)[:300]}))
    return out


def _is_eval(d: dict) -> bool:
    return "batch_rows" in d


def _is_e2e(d: dict) -> bool:
    return "rows" in d and "policies" in d


def _check_windows(windows: list[dict], min_iters: int, min_window_s: float) -> list[str]:
    bad = []
    for i, w in enumerate(windows):
        it = w.get("iters")
        du = w.get("duration_s")
        if it is None or int(it) < min_iters:
            bad.append(f"window {i} iters={it} < {min_iters}")
        if du is None or float(du) < 0.98 * min_window_s:
            bad.append(f"window {i} duration={du}s < {min_window_s}s")
    return bad


def verify_one(doc: dict, min_iters: int = 1000, min_repeats: int = 10,
               min_window_s: float = 3.0, max_clock_spread: float = 30.0) -> dict:
    """Hard-gate one result file. Returns {file-agnostic verdict, failures, warnings}."""
    failures: list[str] = []
    warnings: list[str] = []

    if "_load_error" in doc:
        return {"kind": "unreadable", "failures": [doc["_load_error"]], "warnings": []}

    if doc.get("synthetic_data") is not False:
        failures.append("synthetic_data is not False (accuracy/policy numbers are plumbing)")
    if not doc.get("session_id"):
        failures.append("missing session_id (cannot attribute the measurement session)")

    clocks = doc.get("clock_check") or {}
    pinned = clocks.get("pinned")
    spread = clocks.get("spread_mhz")
    if pinned is not True:
        failures.append(f"clock not verifiably pinned (verdict={clocks.get('verdict')}, "
                        f"spread={spread} MHz)")
    elif spread is not None and float(spread) > max_clock_spread:
        failures.append(f"clock spread {spread} MHz > {max_clock_spread}")

    coexist = doc.get("co_tenancy") or {}
    nprocs = coexist.get("compute_process_count")
    if nprocs is not None and int(nprocs) > 1:
        failures.append(f"{nprocs} compute processes on the GPU at snapshot "
                        f"(close background GPU work)")

    proto = doc.get("protocol") or {}
    reps = proto.get("repeats")
    if reps is None or int(reps) < min_repeats:
        failures.append(f"protocol.repeats={reps} < {min_repeats}")

    if _is_eval(doc):
        kind = "eval"
        rows = doc.get("batch_rows") or []
        n_measured = 0
        for r in rows:
            if r.get("oom"):
                warnings.append(f"batch {r.get('batch')} OOM (sweep stopped, not shrunk)")
                continue
            for mode in ("prefix", "cascade"):
                e = r.get(f"energy_{mode}") or {}
                if e.get("skipped"):
                    continue
                n_measured += 1
                failures += [f"b{r.get('batch')} {mode}: {m}"
                             for m in _check_windows(e.get("windows") or [], min_iters, min_window_s)]
                if e.get("clock_stable") is False:
                    failures.append(f"b{r.get('batch')} {mode}: clock unstable within run")
                if e.get("negative_marginal"):
                    warnings.append(f"b{r.get('batch')} {mode}: negative idle-subtracted "
                                    f"marginal at exits {e['negative_marginal']}")
        if n_measured == 0:
            failures.append("no measured energy rows (latency/FLOPs only)")

    elif _is_e2e(doc):
        kind = "e2e"
        rows = doc.get("rows") or []
        measured = [r for r in rows if r.get("j_per_sample") is not None]
        if not measured:
            failures.append("no measured Joules (latency-only output)")
        for r in measured:
            tag = f"b{r.get('batch')} {r.get('policy')} {r.get('mode')}"
            if int(r.get("n_windows") or 0) < min_repeats:
                failures.append(f"{tag}: n_windows={r.get('n_windows')} < {min_repeats}")
            if r.get("j_ci95_halfwidth") is None:
                failures.append(f"{tag}: missing 95% CI (single-window or old schema)")
            mw = r.get("min_window_s")
            if mw is not None and float(mw) < 0.98 * min_window_s:
                failures.append(f"{tag}: shortest window {mw:.2f}s < {min_window_s}s")
            if r.get("cv_pct") is not None and float(r["cv_pct"]) > 20.0:
                warnings.append(f"{tag}: within-cell CV {r['cv_pct']:.1f}%")
        if proto.get("latency_only"):
            failures.append("protocol.latency_only is True")
    else:
        kind = "unknown"
        failures.append("unrecognised result schema (expected eval_* or e2e_*)")

    return {"kind": kind, "failures": failures, "warnings": warnings,
            "passed": not failures}


def _eval_cells(doc: dict) -> dict[tuple, list[float]]:
    cells = {}
    seed = doc.get("train_seed")
    for r in doc.get("batch_rows", []):
        if r.get("oom"):
            continue
        for mode in ("prefix", "cascade"):
            e = r.get(f"energy_{mode}") or {}
            if e.get("skipped") or any(v is None for v in (e.get("mean_j") or [None])):
                continue
            cells[(doc.get("dataset"), doc.get("arch"), seed, int(r["batch"]), mode,
                   "exit_table")] = [float(v) for v in e["mean_j"]]
    return cells


def _e2e_cells(doc: dict) -> dict[tuple, list[float]]:
    cells = {}
    seed = doc.get("train_seed")
    for r in doc.get("rows", []):
        if r.get("j_per_sample") is None:
            continue
        cells[(doc.get("dataset"), doc.get("arch"), seed, int(r["batch"]),
               r.get("mode"), r.get("policy"))] = [float(r["j_per_sample"])]
    return cells


def compare_docs(docs: list[tuple[Path, dict]], within_pct: float = 10.0) -> dict:
    """Group identical cells across sessions and report between-session spread."""
    groups: dict[tuple, list[tuple[Path, list[float]]]] = {}
    for path, d in docs:
        if "_load_error" in d:
            continue
        cells = _eval_cells(d) if _is_eval(d) else (_e2e_cells(d) if _is_e2e(d) else {})
        for k, v in cells.items():
            groups.setdefault(k, []).append((path, v))

    cells = []
    for k, runs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if len(runs) < 2:
            continue
        n = min(len(v) for _, v in runs)
        vals = [[v[i] for _, v in runs] for i in range(n)]
        per_exit = []
        for i, x in enumerate(vals):
            m = sum(x) / len(x)
            spread = 100.0 * (max(x) - min(x)) / m if m else None
            per_exit.append({"exit_or_index": i, "sessions": x, "mean": m,
                             "spread_pct": spread,
                             "agree": (spread is not None and spread <= within_pct)})
        ok = all(e["agree"] for e in per_exit) if per_exit else False
        cells.append({"cell": {"dataset": k[0], "arch": k[1], "train_seed": k[2],
                               "batch": k[3], "mode": k[4], "policy": k[5]},
                      "n_sessions": len(runs),
                      "files": [str(p) for p, _ in runs],
                      "per_index": per_exit, "agree_within_pct": within_pct,
                      "agrees": ok})
    return {"within_pct": within_pct, "n_cells_compared": len(cells),
            "n_disagreeing": sum(1 for c in cells if not c["agrees"]), "cells": cells}


def main() -> int:
    ap = argparse.ArgumentParser(description="C1 publishability audit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="hard-gate result files against the C1 protocol")
    v.add_argument("--results", default="results")
    v.add_argument("--glob", nargs="*", default=["eval_*.json", "e2e_*.json"])
    v.add_argument("--min-iters", type=int, default=1000)
    v.add_argument("--min-repeats", type=int, default=10)
    v.add_argument("--min-window-s", type=float, default=3.0)
    v.add_argument("--max-clock-spread", type=float, default=30.0)
    v.add_argument("--out", default=None)
    v.add_argument("--recursive", action="store_true",
                   help="search subdirectories too (Phase 1/2 results live in runs/<name>/)")

    c = sub.add_parser("compare", help="cross-session agreement of repeated cells")
    c.add_argument("--results", default="results")
    c.add_argument("--glob", nargs="*", default=["eval_*.json", "e2e_*.json"])
    c.add_argument("--within", type=float, default=10.0)
    c.add_argument("--out", default=None)
    c.add_argument("--recursive", action="store_true")

    args = ap.parse_args()
    root = Path(args.results)
    paths: list[Path] = []
    for pat in args.glob:
        paths += sorted(root.rglob(pat) if args.recursive else root.glob(pat))

    if args.cmd == "verify":
        docs = _load(paths)
        if not docs:
            print(f"[audit] FAIL: no result files matched {args.glob} under {root}", flush=True)
            return 2
        rows = []
        for p, d in docs:
            r = verify_one(d, min_iters=args.min_iters, min_repeats=args.min_repeats,
                           min_window_s=args.min_window_s,
                           max_clock_spread=args.max_clock_spread)
            rows.append({"file": str(p), **r})
            status = "PASS" if r["passed"] else "FAIL"
            print(f"[audit] {status} {p.name} ({r['kind']})", flush=True)
            for f in r["failures"]:
                print(f"         fail: {f}", flush=True)
            for w in r["warnings"]:
                print(f"         warn: {w}", flush=True)
        n_fail = sum(1 for r in rows if not r["passed"])
        out = {"command": "verify", "results_root": str(root), "n_files": len(rows),
               "n_failed": n_fail, "thresholds": {"min_iters": args.min_iters,
                                                  "min_repeats": args.min_repeats,
                                                  "min_window_s": args.min_window_s,
                                                  "max_clock_spread_mhz": args.max_clock_spread},
               "files": rows}
        dest = Path(args.out) if args.out else root / "audit_verify.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[audit] {len(rows) - n_fail}/{len(rows)} files pass; wrote {dest}", flush=True)
        return 1 if n_fail else 0

    docs = _load(paths)
    out = {"command": "compare", "results_root": str(root), "n_files": len(docs),
           **compare_docs(docs, within_pct=args.within)}
    dest = Path(args.out) if args.out else root / "audit_compare.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    for c_ in out["cells"]:
        cc = c_["cell"]
        print(f"[audit] {cc['dataset']}/{cc['arch']} s={cc['train_seed']} b={cc['batch']} "
              f"{cc['mode']} {cc['policy']}: {c_['n_sessions']} sessions, "
              f"agrees={c_['agrees']}", flush=True)
    print(f"[audit] compared {out['n_cells_compared']} repeated cells, "
          f"{out['n_disagreeing']} disagree > {args.within}%; wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
