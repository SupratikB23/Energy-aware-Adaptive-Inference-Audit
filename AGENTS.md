# AGENTS.md — hard rules for any AI agent working in this repo

## 0. Never push
- NEVER run `git push`, `git commit`, `gh pr create`, or any publish/release command.
- Do not amend commits. Stage nothing unless the user explicitly says `commit`.
- Remote = read-only. Local edits are fine; pushing is forbidden.

## 1. Do not touch the three source documents
- `percom2027_literature_matrix_and_verdict.md`, `ChatGPT-Explain LTT Threshold Selection-20260914-1326.md`, `notebook.py` are frozen inputs. Never edit them.

## 2. Novelty honesty (from literature verdict)
- The rule `continue iff E[acc gain]/marginal cost > lambda` is rational metareasoning / value-of-computation (Russell & Wefald, 1990s). NEVER present it as a new controller.
- `Accuracy-per-Joule` name is taken (PMC12899382). LTT-on-exits is taken (SAFE-KD). LTT-on-budget is taken (Calibrate-Then-Delegate). KD+exits alone is not a claim.
- Our contribution is the MEASURED audit: C1 protocol, C2 regime map, C3 honest benchmark. Cite Green MLOps (2601.04250), EEFP (2508.21495), Predictive Exit (2206.04685), 2312.02741 (NVML pathologies) wherever the controller/cost is discussed.

## 3. Measurement discipline
- No energy claim without `torch.cuda.synchronize()` around the window, looped integration (>=1000 iters), idle subtraction, device-state log, and dummy-kernel residual <= 5%.
- Prefer `nvmlDeviceGetTotalEnergyConsumption` (hardware-integrated mJ). Polling at 100-200 Hz + trapezoidal rule is the fallback, never 1 Hz.
- Never synthesize Joules. The old `nvmlDeviceGetC2cModeInfoV -> return None` shim is allowed ONLY for optional metadata, never for power/energy.
- On validation failure: STOP and report. Do not proceed to policy benchmarks on bad telemetry (kill criteria in RUN_PLAN.md).

## 4. 8GB VRAM ceiling
- Default train batch 64, eval sweep [1,8,16,32,64]. AMP on. `num_workers=0` on Windows.
- Every GPU script must log `memory_allocated/total` (VRAM allocation shifts idle power).
- Handle OOM gracefully with a clear message; never silently shrink the experiment.

## 5. Minimal repo
- One code folder: `src/`. Two gitignored folders: `data/`, `results/`. Root: `.gitignore`, `requirements.txt`, `config.yaml`, `README.md`, `RUN_PLAN.md`, `AGENTS.md`.
- Do not create new top-level folders without asking. Do not commit `data/`, `results/`, `*.pt`, `*.csv`.
- Keep diffs small and file-scoped. Fix the file asked for; don't refactor the world.

## 6. Reproducibility
- Fixed seeds, pinned requirements, per-run JSON with GPU name, driver, clocks, temp, power limit, batch, and git hash (or `nogit`).
- `python src/test_smoke.py` must pass on CPU before asking the user to run anything on the GPU box.
