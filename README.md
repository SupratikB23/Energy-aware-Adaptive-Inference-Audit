# PerCom 2027 — When is adaptive inference actually worth its energy?

Measured-energy audit of early-exit policies (C1 protocol + C2 regime map + C3 honest benchmark).
See `RUN_PLAN.md` for the exact GPU-box run order. See `AGENTS.md` for AI rules.

Research statement and controlling verdicts (C1-C4, kill criteria):
[percom2027_literature_matrix_and_verdict.md](percom2027_literature_matrix_and_verdict.md)

Current framing (v2): [RESEARCH_STATEMENT.md](RESEARCH_STATEMENT.md) ·
first RTX 3070 results: [RESULTS.md](RESULTS.md)

## Repo layout (minimal)
```text
.gitignore  requirements.txt  config.yaml  README.md  RUN_PLAN.md  AGENTS.md
src/
  check_gpu.py   # Phase 0 pilot
  measure.py     # C1 EnergyMeter
  models.py      # teacher/student/early-exit + KD losses
  train.py       # 8GB-safe training
  calibrate.py   # threshold sweep + LTT (C4 deferred)
  evaluate.py    # C2+C3 per-exit energy table, policies, per-head baseline, KILL-4
  e2e_policy.py  # end-to-end measured policy energy vs table/FLOP predictions
  utils.py       # seeds, config, logging
  test_smoke.py  # CPU-only full check: python src/test_smoke.py
  test_units.py  # pytest suite incl. fake-NVML GPU paths: python -m pytest src/test_units.py -q
data/      (gitignored)
results/   (gitignored)
```

## Quickstart (GPU box)
```bash
# install CUDA torch first (see RUN_PLAN Step 0a; RTX 50 needs the cu128 index)
pip install -r requirements.txt
python src/test_smoke.py
python -m pytest src/test_units.py -q
python src/check_gpu.py --idle-seconds 60 --stress-seconds 30
python src/train.py --dataset cifar10 --mode teacher
python src/train.py --dataset cifar10 --mode exit_kd --teacher-ckpt results/cifar10_teacher_resnet18_e20_b64.pt
python src/evaluate.py --ckpt results/cifar10_exit_kd_resnet14_e20_b64.pt --dataset cifar10 --batches 1 8 16 32 64
```

## Novelty statement (copy into paper, do not reword into a false claim)
Our controller is an instantiation of rational metareasoning / value-of-computation
with a *measured* per-exit Joule cost term. Contribution is the validated protocol
(C1), regime map (C2), and honest benchmark (C3) — not the decision rule.
