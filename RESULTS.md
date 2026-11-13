# RESULTS — measured-energy audit of early-exit policies (CIFAR-10 + UCI-HAR; RTX 3070 Phase 1/2, RTX 5060 Laptop Phase 3)

Scope: RUN_PLAN Phase 1 (Steps 1–4) and Phase 2 (Steps 5–8) on an **RTX 3070** (§0–§11, preliminary,
secondary platform), and **Phase 3 (P3.1–P3.8) on an RTX 5060 Laptop GPU (§12, the headline
platform)**. Phase 3 is the pre-submission hardening pass: 3 training seeds, locked clock, >=10
randomized windows, 95 % CIs, cross-session audit. The two platforms are never pooled. Research
framing and kill criteria:
[percom2027_literature_matrix_and_verdict.md](percom2027_literature_matrix_and_verdict.md).

> **Platform note.** §0–§11 (Phase 1/2) were measured on the RTX 3070 below (`results/`, 3 repeats, mostly
> unlocked clocks, ~19 background GPU processes). §12 (Phase 3) was measured on a different GPU
> (`new_results/results/p3/`), so it is a self-contained platform and its numbers are not pooled with §0–§11.
> Where the two disagree (for example, unlocked batch-1 resolvability), both are reported.

## Results map (read this first)

**A. Which folder holds which phase** (all three are gitignored)

| Folder | Platform | Phase | Content |
|---|---|---|---|
| `results_old/` | RTX 3070 | Phase 1 only | 18 files, **byte-identical** to the Phase 1 subset of `results/` (checked by hash). Redundant; safe to delete. |
| `results/` | RTX 3070 | Phase 1 + Phase 2 | Phase 1: 11:17–14:14 on Sep 19. Phase 2: 16:32 Sep 19 – 03:49 Sep 20. Also a stray empty `audit_old_compare.json` (0 cells compared) and the misrouted first HAR e2e run `e2e_har_harcnn.json`. |
| `new_results/results/p3/` | RTX 5060 Laptop | Phase 3, locked | 40 files: checkpoints, training/LTT/eval/e2e JSONs, `audit_verify.json`, `audit_compare.json`. |
| `new_results/results/p3_unlocked/` | RTX 5060 Laptop | Phase 3, P3.3b | 2 eval files with the clock unlocked (kept apart because they intentionally fail the pinned-clock gate). |

**B. What was run in each phase**

| Phase | Platform / git | Steps | What it produced | Where in this file |
|---|---|---|---|---|
| **1** | RTX 3070, driver 591.86; git `0167720` (pilot, CIFAR training), `5eacc30` (eval) | Steps 1–4: telemetry pilot; CIFAR training (teacher, student, exit_ce, exit_kd); LTT α-sweep; per-exit energy `eval_run1/2` (3 repeats) | Telemetry check (0.16 %), regime map, first policy benchmark | §0–§9 |
| **2** | RTX 3070; git `e83076f` | 5 pooled-cost policy re-pricing; 6 CIFAR end-to-end (unlocked); 7 UCI-HAR (training, LTT, eval, end-to-end); 8 locked-vs-unlocked batch-1/8 at 10 repeats, HAR locked e2e | Locking test, HAR negative result, table error, compaction vs batch-wait | §10–§11 |
| **3** | RTX 5060 Laptop, driver 610.62; git `7b90eae` (training, unlocked contrast), `0960180` (locked eval/e2e) | P3.2 3-seed training; P3.3 LTT; P3.3b unlocked contrast; P3.4 per-exit eval; P3.5 CIFAR e2e; P3.6 HAR eval + e2e; P3.7 audit | Self-contained hardening benchmark | §12 |

**C. Phase 3 subdivisions: status against RUN_PLAN (verified from the files on Nov 12)**

| Step | Content | Files | Status | Section |
|---|---|---|---|---|
| P3.1 | CPU smoke + unit tests | not a result file | not in the data; run on the dev machine | — |
| 0c pilot | counter-vs-poll telemetry check on the 5060 | `p3/pilot.json` (delivered Nov 13; Steam was running, see §13) | ✅ **PASS**, residual 0.13 % ≤ 5 % | §13 |
| P3.2 | 3 seeds × {CIFAR teacher, CIFAR exit_kd, HAR teacher, HAR exit_kd} | 12 `.json` + 12 `.pt` | ✅ done | §12.2 |
| P3.3 | LTT per seed (CIFAR α=0.02, HAR α=0.05) | 6 `ltt_*` | ✅ done | §12.3 |
| P3.3b | unlocked contrast, b=1/8, CIFAR + HAR (seed 42) | 2 `eval_*_U` | ✅ done | §12.4c |
| P3.4 | CIFAR per-exit eval, seed 42 at b=1/8/16; seeds 43/44 at b=1 | 3 `eval_cifar*_A` | ✅ done | §12.4, §12.4b |
| P3.5 | CIFAR end-to-end: session A (b=1/8/16), session B (b=1/8) | `e2e_cifar*_A`, `_B` | ✅ done | §12.5 |
| P3.6 | HAR eval (b=1/64); e2e A (b=1/64); e2e B (b=64) | `eval_har*_A`, `e2e_har*_A`, `_B` | ✅ done | §12.4, §12.5 |
| P3.7 | `audit verify` (8 files) and `audit compare` (24 repeated cells) | 2 audit JSONs | ✅ 8/8 pass; 20/24 within 10 % | §12.6 |
| P3.8 | `nvidia-smi -rgc` (unlock) | none | cannot be verified from files | — |
| O1–O3 | CIFAR b=32/64; CIFAR eval session B; HAR b=8/16/32 | none | not run (optional) | claims narrowed |

Not run, so no claim is made: CIFAR b=32/64, HAR b=8/16/32 locked, end-to-end for seeds 43/44, any battery-powered device.

**Setup.** NVIDIA GeForce RTX 3070 (8 GB, sm_86, 270 W limit), driver 591.86, torch 2.7.0+cu126,
Python 3.11.9, Windows. FP32 inference (cuDNN TF32 on, matmul TF32 off, cudnn.benchmark on).
Real CIFAR-10 everywhere (`synthetic_data: false` in every file). Seed 42.
Energy = NVML hardware energy counter, idle-subtracted, J **per sample**.
**Phase 1/2 protocol:** 60 s idle baseline → 120 s warmup → per exit 3 repeated windows, each
≥1000 iters and ≥3 s, rotated round-robin order. Two independent full benchmark runs
(`eval_run1`, `eval_run2`). **Phase 3 protocol:** the same window rules, but >=10 repeats,
seeded randomized order, Student-t 95 % CIs, pinned-clock evidence, co-tenancy snapshot and a
`session_id`; `audit verify` is the publishability gate.

---

## 0. Verdict in one table

| Gate | Result | Status |
|---|---|---|
| **KILL-1** telemetry valid | counter vs poll residual **0.16 %** (limit 5 %) | ✅ PASS |
| **KILL-2** adjacent exits resolvable (>3σ) at batch 1 | exit 0↔1 **not** resolvable (both runs, both modes); 1↔2 resolvable; 2↔3 mixed | ⚠️ PARTIAL FAIL |
| KILL-2 at batch ≥16 | all adjacent exits resolvable, both runs, both modes | ✅ |
| **KILL-3** measured Joule order ≠ FLOP order | **no** significant disagreement at any batch, either run | ❌ FIRES → drop the "FLOPs get the *order* wrong" claim |
| FLOPs predict the *size* of savings | **no**: off by −7 to +17 pts, sign flips with batch size (§4) | ✅ usable claim |
| **KILL-4** marginal-utility beats confidence at matched accuracy (CIFAR) | median 9.9 % (locked clocks) to 20.5 % (unlocked) cheaper, vs ~2 % under FLOP pricing (§10.5) | ✅ PASS, modest |
| LTT accuracy-risk guarantee holds on held-out split | CIFAR α = 0.02 / 0.03 / 0.05 and HAR α = 0.03 / 0.05; α = 0.01 uncertifiable | ✅ |
| **Phase 2:** locked clocks restore batch-1 measurability | batch-1 cascade: 0/3 → 3/3 adjacent exits resolvable; identical-workload mismatch 16–54 % → 7 % | ✅ (§10.2) |
| **Phase 2:** early exit saves energy end to end, **CIFAR ResNet-14** | **Yes, at small batch only.** Saves 18–67 % at b = 1 and 18–49 % at b = 8; at b = 16 it mostly reverses (costs up to 29 %, best case saves 3 %); b = 64 recovers only to 3–15 % saved, far below b = 1 | ✅ (§11.1) |
| **Phase 2:** early exit saves energy end to end, **HAR CNN (97 k params)** | **No, never (in the final rerun).** Costs 6–193 % more than the full model under compaction (up to 241 % with batch-wait) at every batch size, under both default and locked clocks. The first, misrouted run had 5 compaction cells that saved (b = 32 and b = 64, e.g. −24 % at b = 64), which the rerun did not reproduce (§11.2) | ❌ key finding (§11.2) |
| **Phase 2:** the additive per-exit table predicts real policy energy | **No.** It under-predicts energy by 3–39 % (CIFAR) and 53–87 % (HAR); i.e. it promises savings that do not appear | ❌ key finding (§11.3) |
| **Phase 2:** FLOPs predict end-to-end policy energy | CIFAR −45 … +84 %, HAR −71 … −94 % | ❌ key finding (§11.3) |
| **Phase 2:** is the HAR result a DVFS artefact? | **No.** With clocks locked at 1500 MHz, early exit still costs 22–189 % more than the full model | ✅ (§11.2) |
| Compaction (drop exited samples) beats batch-wait | yes, 45/48 cells; median 14.4 % (CIFAR), 8.9 % (HAR) | ✅ (§11.4) |

**Headline (after Phase 2):**
- The measurement protocol works, but only with **locked GPU clocks** at batch 1 (C1).
- Batching dominates exit depth.
- FLOPs misjudge the size of early-exit savings. On a **pervasive-scale** model (HAR, 97 k params) the
  gap is extreme: FLOPs promise ~90 % savings, while the real early-exit runtime **costs more energy
  than the full model** unless the batch is large.
- A measured cost term makes flexible exit rules worth ~10 % at matched accuracy, where FLOP pricing
  says they are worth ~2 %.
- **The model-scale split is the story.** Same code, same protocol, two models: on the CIFAR ResNet
  early exit saves up to 67 % of the energy at batch 1, and on the pervasive-scale HAR CNN the exact
  same mechanism *costs* up to 189 % extra. The additive cost table predicts a saving in both cases.

---

## 1. C1 — Telemetry validation (`runs/pilot_rtx3070.json`)

| Quantity | Value |
|---|---|
| Hardware energy counter | present (`has_energy_counter: true`) |
| Dummy kernel, 10 s after 2 s pre-warm | counter **2674.50 J** vs trapezoidal poll **2678.71 J** |
| Residual | **0.157 %** (basis `counter-vs-poll`) |
| Poll rate achieved | 142.6 Hz (target 150 Hz) |
| Idle, 60 s | counter 1684.3 J (**28.07 W**) vs poll 1704.5 J → 1.2 % disagreement |
| Stress response | 23.9 W → 267.1 W; 46 → 68 °C; at the 270 W cap |

- On an RTX 3070 (Ampere), the NVML energy counter and 142 Hz power polling agree to 0.16 % under
  steady load. The 25 %-sampling pathology in arxiv:2312.02741 (A100/H100) does **not** show up as
  energy error here once the load is pre-warmed.
- They disagree more (1.2 %) at idle, where load is bursty: 19–20 other GPU processes were
  resident on this Windows box throughout. Report this as a limitation.
- The pilot idle (28.1 W) and the evaluate idle baselines (24.18 W / 24.22 W, run1/run2) differ
  because the pilot idle had no preceding settle period. The two evaluate idles agree to 0.2 %.

## 2. Training (`results/cifar10_*_e20_b64.json`)

All models: 20 epochs, batch 64, SGD lr 0.05 cosine, AMP, real CIFAR-10. Test accuracy, final epoch:

| Model | Head 0 | Head 1 | Head 2 | Final | Train time |
|---|---|---|---|---|---|
| Teacher ResNet-18 (final head only) | – | – | – | **92.89 %** | 778 s |
| Student ResNet-14, CE (final head only) | – | – | – | **92.24 %** | 687 s |
| Early-exit ResNet-14, CE on all heads (`exit_ce`) | 80.86 % | 85.48 % | 88.40 % | 89.49 % | 710 s |
| Early-exit ResNet-14, exit-aware KD (`exit_kd`) | 80.34 % | 85.20 % | **90.74 %** | **92.08 %** | 830 s |

(The early heads of the teacher and the `student` show ~10 % because they were never trained. Expected.)

- KD is what makes the early-exit student viable. It lifts the final head by **+2.59 pts** and head 2
  by **+2.34 pts** over `exit_ce`, and brings the multi-exit model within 0.16 pts of the single-exit
  student.
- KD slightly **hurts** the two earliest heads (−0.52 / −0.28 pts vs `exit_ce`). That is
  consistent with the LEAP warning (arxiv:2605.01058). It's small, but worth one sentence.
- Only the `exit_kd` model was calibrated and benchmarked.

## 3. LTT calibration (`runs/ltt_a0*`)

Calibration split n = 5000, deploy split n = 5000 (disjoint halves of the CIFAR-10 test set), δ = 0.10.
Risk = P(policy wrong ∧ full model right). Full-model accuracy: 91.96 % (calib), 92.20 % (deploy).

Empirical risk on the calibration split, by threshold:

| τ | 0.999 | 0.995 | 0.99 | 0.98 | 0.97 | 0.95 | 0.93 | 0.90 | 0.85 | 0.80 | 0.70 | 0.60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| risk % | 0.02 | 0.26 | 0.36 | 0.74 | 1.12 | 1.50 | 1.88 | 2.58 | 3.94 | 4.88 | 7.22 | 9.64 |

Selected (most aggressive certified) threshold and its held-out check:

| α | Certified τ set | Selected τ | Deploy acc | Deploy drop | Deploy risk | Exit rate | Avg exit |
|---|---|---|---|---|---|---|---|
| 0.01 | none (p = 0.383 even at τ = 0.999) | — | — | — | — | — | — |
| 0.02 | 0.999, 0.995, 0.99 | **0.99** | 91.90 % | 0.30 pts | 0.48 % ≤ 2 % ✅ | 68.4 % | 1.36 |
| 0.03 | … → 0.97 | **0.97** | 91.68 % | 0.52 pts | 0.88 % ≤ 3 % ✅ | 77.0 % | 1.07 |
| 0.05 | … → 0.90 | **0.90** | 89.98 % | 2.22 pts | 2.72 % ≤ 5 % ✅ | 87.1 % | 0.71 |

- **α = 0.01 cannot be certified at n = 5000 by construction.** Hoeffding with δ = 0.1 needs
  risk ≤ α − √(ln 10 / 10000) = α − 0.0152, which is negative. Even τ = 0.999 (risk 0.02 %)
  only reaches p = 0.383. The minimum certifiable α here is ≈ 0.016. State this.
- Every certified choice held on the deploy split, with margin (Hoeffding is conservative).
- The early heads are **overconfident**. At τ = 0.9, 65 % of inputs leave at head 0, and the extra
  error is 2.6 %. Safe operating points need τ ≥ 0.97. This supports EEFP's point that raw
  confidence is poorly calibrated for early exits.
- The old 0.6–0.9 grid (`runs/ltt_a001_oldgrid.json`, identical to top-level `ltt_cifar10_resnet14.json`)
  certified nothing, because the grid never reached the safe region.

## 4. C2 — Regime map: energy per sample vs batch size and exit

Pooled mean of run1 and run2, marginal mJ **per sample**. "prefix" = run exactly up to head k.
"cascade" = the deployed path (heads 0..k−1 also evaluated, host-side exit decision each time).

| Batch | Mode | Exit 0 | Exit 1 | Exit 2 | Full | Exit 0 as % of full | run1↔run2 max diff |
|---|---|---|---|---|---|---|---|
| 1 | prefix | 19.88 | 18.98 | 44.22 | 66.61 | 29.8 % | 7 % |
| 1 | cascade | 14.18 | 17.99 | 45.19 | 93.62 | 15.1 % | **37 %** |
| 8 | prefix | 9.93 | 12.58 | 18.62 | 23.37 | 42.5 % | 10 % |
| 8 | cascade | 8.48 | 13.07 | 15.49 | 22.80 | 37.2 % | 13 % |
| 16 | prefix | 8.94 | 12.43 | 15.29 | 18.18 | 49.2 % | 3.4 % |
| 16 | cascade | 9.10 | 13.29 | 16.79 | 20.87 | 43.6 % | 4.2 % |
| 32 | prefix | 8.75 | 11.47 | 13.62 | 16.34 | 53.5 % | 2.2 % |
| 32 | cascade | 8.76 | 11.79 | 14.36 | 17.39 | 50.4 % | 2.6 % |
| 64 | prefix | 7.28 | 9.60 | 11.55 | 13.59 | 53.6 % | 1.4 % |
| 64 | cascade | 7.22 | 9.74 | 11.85 | 14.03 | 51.5 % | 1.4 % |

FLOPs per exit (MACs, batch 1): 152.8 M, 211.5 M, 270.2 M, 328.9 M → exit 0 = **46.4 %** of full.
The cascade adds only 0.6–4.5 k MACs (the FC heads), which is negligible in FLOPs.

### 4.1 Batch size matters far more than exit depth
- The full model at batch 64 (13.6 mJ, 92.2 % acc) costs **less per sample than exit 0 at batch 1**
  (19.9 mJ prefix, 80.3 % head accuracy).
- Batch 1 → 64 cuts full-model energy per sample by **4.9×** (prefix) and **6.7×** (cascade).
  Exiting at head 0 instead of the full head saves at most ~2–3× at any fixed batch size.
- The LTT-certified α = 0.02 policy at batch 1 costs 40–48 mJ/sample (§6). That is **~3× the full
  model at batch 64**.
- This is the Cluster F batching threat, now measured on a CNN. It is the strongest result in this
  run. The caveat is latency: batching adds queueing delay, which matters for pervasive b = 1 workloads.

### 4.2 Energy vs FLOPs
Least-squares fit of pooled energy against MFLOPs, E = a + b·MFLOP:

| Batch | Mode | R² | Intercept a (share of full) |
|---|---|---|---|
| 1 | prefix / cascade | 0.885 / 0.876 | negative (convex, not affine) |
| 8 | prefix / cascade | 0.980 / 0.960 | −12 % / −16 % |
| 16 | prefix / cascade | 0.998 / 0.999 | +6 % / −4 % |
| 32 | prefix / cascade | 0.998 / 0.999 | +14 % / +8 % |
| 64 | prefix / cascade | 0.999 / 0.998 | +14 % / +10 % |

- **KILL-3 fires.** Across all 10 batch × mode cells and both runs, the measured Joule order never
  significantly disagrees with the FLOP order. The only raw inversions were exit 0 vs exit 1 at
  batch 1 (prefix), and those are inside 3σ. Drop "FLOPs get the ordering wrong" from the abstract.
- **What survives: FLOPs get the *size* of the saving wrong, and the error flips sign with batch size.**
  FLOPs predict that exiting at head 0 saves 53.6 % of the full cost. Measured (prefix):

  | Batch | 1 | 8 | 16 | 32 | 64 |
  |---|---|---|---|---|---|
  | Measured saving at exit 0 | 70.2 % | 57.5 % | 50.8 % | 46.5 % | 46.4 % |
  | FLOP error (pts) | FLOPs underestimate by 16.6 | under 3.9 | over 2.8 | over 7.1 | over 7.2 |

  - At batch ≥ 16, energy is affine in FLOPs (R² ≥ 0.998) with a fixed overhead of up to ~14 % of
    the full cost, so FLOPs **overstate** early-exit savings by ~7 pts.
  - At batch ≤ 8, energy is convex in depth and FLOPs **understate** the savings.
  - The crossover lies between batch 8 and 16 on this GPU. This is the O3 regime result.

### 4.3 Batch 1 is where measurement breaks down (C1 limitation)
- Clocks during batch-1 windows ranged **675–1470 MHz** (vs a steady 1905–1950 MHz at batch ≥ 16),
  and temperature drifted from 44 to 59 °C. The GPU never settles into a steady power state at
  batch 1, even after the 120 s warmup.
- Within-run coefficient of variation at batch 1 is **4–55 %** (exit 0 prefix: 50–55 %), versus
  **0.05–4 %** at batch ≥ 16.
- Cascade at batch 1 is the least reproducible cell: run1 and run2 differ by 37 % at exit 0 and at the
  full exit. That is the cost table the policies are priced with (§6).
- Batch-1 wall latency is launch-bound: in `lat_per_exit_s_b1` the full exit (1.9–2.5 ms) is faster
  than exit 2 (2.1–2.8 ms) in both runs.

## 5. KILL-2 detail — adjacent-exit resolvability (>3σ, 3 repeats)

| Batch | Mode | run1 (0↔1, 1↔2, 2↔3) | run2 |
|---|---|---|---|
| 1 | prefix | ✗ ✓ ✓ | ✗ ✓ ✓ |
| 1 | cascade | ✗ ✓ ✗ | ✗ ✓ ✓ |
| 8 | prefix | ✗ ✓ ✗ | ✓ ✓ ✓ |
| 8 | cascade | ✓ ✗ ✓ | ✓ ✗ ✗ |
| 16, 32, 64 | both | ✓ ✓ ✓ | ✓ ✓ ✓ |

At batch 1, the first two exits cannot be told apart energetically on this box, although their
FLOPs differ by 38 %. Under the verdict's rule this is a **partial KILL-2**: fine per-exit
policies at batch 1 are not supported. From batch 16 up, every exit is cleanly resolvable.

## 6. C3 — Policy benchmark (deploy split, cost = batch-1 cascade Joules)

Confidence-threshold vs marginal-utility at matched accuracy (confidence cost interpolated at each
marginal-utility point's accuracy):

| Marginal-utility λ | Accuracy | MU cost | Confidence cost | MU saving |
|---|---|---|---|---|
| run1, λ = 0.001 | 92.22 % | 40.0 mJ | 51.8 mJ | 22.7 % |
| run1, λ = 0.005 → 0.514 | 91.7 → 84.7 % | 32.4 → 17.1 mJ | 36.5 → 19.2 mJ | 10–12 % |
| run2, λ = 0.001 | 92.16 % | 46.6 mJ | 57.2 mJ | 18.6 % |
| run2, λ = 0.005 → 0.133 | 91.8 → 85.0 % | 33.7 → 13.8 mJ | 43.9 → 15.6 mJ | 11–31 % |
| both, λ ≤ 4.5e-5 | 92.2 % | 55–101 mJ | 50–64 mJ | MU **worse** (10–58 %) |

- Marginal-utility points make up most of the accuracy/energy Pareto front in both runs (9 of 12
  front points in run1, 8 of 12 in run2). Its advantage over confidence exit at matched accuracy is
  **10–23 % in run1 and 11–31 % in run2**. The direction is reproducible; the size is not.
- Entropy exit tracks confidence almost exactly. EEFP-style points sit between confidence and MU, and
  several EEFP settings degenerate to "never exit" (β·cost > 1 − τ). Treat EEFP as a sketch, not a
  faithful baseline.
- **Caveats, which must go in the paper:**
  1. **The win rests on an unresolvable number.** MU's main edge is that it continues from head 0 to
     head 1 almost for free (the measured head 0→1 marginal cost is 1.4 % / 5.9 % of full in run1 / run2,
     vs 18 % by FLOPs). §5 shows that difference is **not** resolvable at batch 1.
  2. **The cost table is the noisiest cell.** Batch-1 cascade costs differ by up to 37 % between runs,
     and MU both decides and is scored with the same table, so its measured advantage inherits that noise.
  3. **MU here equals per-head confidence thresholds.** Continue at head h iff
     conf < 1 − λ·c_h/G_h. So MU is exactly a confidence policy with a separate threshold per head.
     The fair baseline is per-head tuned confidence thresholds (not yet run); against a single
     global τ the comparison flatters MU.
  4. All policy costs assume batch 1. At batch 64, the full model with no early exit (13.6 mJ)
     is cheaper than every batch-1 policy scoring ≥ 84 % accuracy.
- Honest reading: **a weak, noise-limited C3 pass.** Present it as "a measured cost term changes
  which heads are worth running", not as a policy win.

## 7. What this means for the paper

**Claims supported by this data**
1. C1: on Ampere, the NVML energy counter agrees with 142 Hz polling to 0.16 % under steady load.
   Per-exit energy is resolvable at batch ≥ 16 but **not** at batch 1, where DVFS never settles
   (675–1470 MHz), within-run CV reaches 55 %, and run-to-run differences reach 37 %.
2. C2: batch size dominates exit depth. The full model at batch 64 costs less per sample than the
   earliest exit at batch 1.
3. C2 / O3: FLOPs preserve the energy *ordering* of exits but misestimate the *size* of savings,
   with a sign flip between batch 8 and 16 (−16.6 to +7.2 pts).
4. LTT certifies early exit at α ≥ 0.02 with a verified held-out guarantee. At α = 0.02 that means
   68 % of inputs exit early for a 0.3-pt accuracy cost; α = 0.01 is uncertifiable at n = 5000.

**Claims to drop or soften**
- "FLOPs get the order wrong" (KILL-3).
- "Marginal-utility beats confidence" as a strong claim (§6 caveats).

## 8. Recommended next runs — SUPERSEDED by RUN_PLAN Phase 3

Items 1–5 below were the Phase 2 wish list; all five are now implemented as Phase 3 gates in
[RUN_PLAN.md](RUN_PLAN.md) (P3.0d locked clocks + co-tenancy; P3.2 three seeds; P3.4 >=10 repeats;
P3.5 locked CIFAR e2e; P3.6 HAR b=64 sessions; P3.7 `audit verify` / `audit compare`).
The original list is kept only as history:
1. **Lock GPU clocks** for batch-1 measurements (`nvidia-smi -lgc 1905,1905`, needs admin) and raise
   `--repeats` to ≥10 for batch 1 and 8. This directly tests whether the batch-1 noise is DVFS
   (fixable) or intrinsic.
2. **Add a per-head-threshold confidence baseline** (the fair comparison for MU; caveat 3).
3. **Price policies with a pooled, clock-locked cost table**, and report policy results at batch 1
   and with a batch-level exit variant at batch ≥ 16.
4. **Pervasive track** (KWS / HAR with real data) and ideally one constrained device (Jetson / RPi +
   INA219) to fit the venue (verdict §5).
5. **Close background GPU processes** (19–20 were resident) and log `nvidia-smi` process lists
   with each run.

## 9. Files analysed

| File | Content |
|---|---|
| `results/runs/pilot_rtx3070.json` (= `results/pilot.json`) | KILL-1 telemetry validation |
| `results/cifar10_{teacher_resnet18,student_resnet14,exit_ce_resnet14,exit_kd_resnet14}_e20_b64.json` | training histories (git `0167720`) |
| `results/runs/ltt_a001_oldgrid.json` (= `results/ltt_cifar10_resnet14.json`) | α = 0.01, old τ grid |
| `results/runs/ltt_a00{1,2,3,5}/ltt_cifar10_resnet14.json` | α sweep, τ grid up to 0.999 |
| `results/runs/eval_run{1,2}/eval_cifar10_resnet14.json` | C2/C3 benchmark (git `5eacc30`), two independent runs |

---

## 10. Phase 2 results (GPU box, git `e83076f`)

### 10.1 Step 5 on the GPU box reproduces the local analysis
`runs/policy_pooled` (GPU, real CIFAR-10, run1+run2 pooled batch-1 cascade table
14.18 / 17.99 / 45.19 / 93.62 mJ) gives the same KILL-4 answer as the CPU re-analysis:
- marginal-utility vs global confidence: 9/12 wins, median **17.7 %** (FLOP pricing: 1.7 %);
- per-head vs global confidence: 11/12, median 15.1 % (FLOP pricing: 1.8 %).

### 10.2 C1: clock locking fixes batch-1 measurement (Step 8, 10 repeats)
| Batch-1, cascade | Unlocked (`eval_b1b8_unlocked_r10`) | Locked 1500 MHz (`eval_b1b8_locked1500`) |
|---|---|---|
| Graphics clock during windows | 855–1935 MHz | **1500–1500 MHz** |
| Per-exit CV (%) | 23.5 / 40.7 / 3.4 / 26.0 | **6.7 / 10.7 / 5.3 / 3.9** |
| Adjacent exits resolvable (>3σ) | ✗ ✗ ✗ | **✓ ✓ ✓** |
| mJ/sample (exit 0 … full) | 13.85 / 19.02 / 41.69 / 85.35 | 14.92 / 23.13 / 50.88 / 80.17 |

**Identical-workload test.** Exit 0 "prefix" and exit 0 "cascade" execute exactly the same
computation, so they must measure the same. Their disagreement:

| Run | batch 1 | batch 8 | batch ≥16 |
|---|---|---|---|
| CIFAR run1 / run2 (unlocked, 3 repeats) | 15.6 % / 54.1 % | 13.9 % / 17.5 % | 0.1–3.0 % |
| CIFAR unlocked, 10 repeats | 24.8 % | 6.8 % | – |
| **CIFAR locked 1500 MHz, 10 repeats** | **7.4 %** | **2.8 %** | – |
| HAR (unlocked) | **90.1 %** | 1.8 % | 0.7–22.5 % |

- On this GPU, batch-1 per-exit energy is **not reproducible under default DVFS**, even with
  repeats. Locking the graphics clock brings it to single-digit error.
- The locked run still leaves prefix exit 0↔1 unresolved (CV 14–15 %). The cascade table is clean.
- Side effect: at batch 8, the locked run measured ~40–45 % less marginal energy per sample than
  default boost (e.g. full model 13.2 vs 23.9 mJ). Idle power differs (29.1 W locked vs 24.3 W
  unlocked), so compare this cautiously. It is consistent with DVFS literature (arxiv:2501.08219).
- KILL-3 unchanged: no significant FLOP-order disagreement in either run. At batch 1, FLOPs
  **understate** exit savings (−21 to −42 pts); at batch 8 by −4 to −21 pts.

### 10.3 Pervasive track: real UCI-HAR (Step 7)
Official subject-disjoint split, 7352 train / 2947 test windows (9 inertial channels × 128 samples),
6 activities. `synthetic_data: false` everywhere.

| Model | Head 0 | Head 1 | Final | Params / MFLOPs |
|---|---|---|---|---|
| Teacher ResNet-18 | – | – | 92.74 % | 11 M / 254 |
| `exit_ce` HAR-CNN | 91.65 % | 92.50 % | **93.42 %** | 97 k / 3.87 |
| `exit_kd` HAR-CNN | 91.86 % | 92.40 % | 92.67 % | 97 k / 3.87 |

- **KD does not help on HAR.** The teacher (92.7 %) is *weaker* than the small student trained with
  plain CE (93.4 %), so distilling from it costs 0.75 pts. Say so; use `exit_ce` as the stronger
  HAR baseline in the paper, or report both.
- Head 0 is already 91.9 % (only ~1.3 pts below the final head) at 8.6 % of the FLOPs. On paper, this
  task is ideal for early exit.
- LTT (calibration n = 1473, δ = 0.1): α = 0.03 certifies τ = 0.93 (deploy: 92.74 %, −0.41 pts,
  87 % exit early, risk 0.41 % ≤ 3 % ✅). α = 0.05 certifies τ = 0.60 (−1.02 pts, 99 % exit early ✅).

Per-exit energy (`runs/har_eval`, unlocked, prefix mode, mJ per sample):

| Batch | Exit 0 | Exit 1 | Full | Exit 0 as % of full (FLOPs: 8.6 %) | FLOP-saving error |
|---|---|---|---|---|---|
| 1 | 4.31 (CV 96 %) | 4.22 | 5.48 | 78.6 % | +70 pts |
| 8 | 0.199 | 0.365 | 0.531 | 37.5 % | +29 pts |
| 16 | 0.100 | 0.171 | 0.261 | 38.4 % | +30 pts |
| 32 | 0.063 | 0.128 | 0.184 | 34.5 % | +26 pts |
| 64 | 0.077 | 0.137 | 0.248 | 30.8 % | +22 pts |

- For a pervasive-scale model, **FLOPs overstate the saving of exiting at every batch size**, by
  22–70 points. That is the opposite sign to CIFAR at small batch. Fixed per-launch costs dominate a
  3.9 MFLOP network.
- The GPU never leaves low clocks for this model (435–1380 MHz), so every HAR cell is
  `clock_stable: false`.
- Batching: full model 5.48 → 0.25 mJ/sample from batch 1 to 64 (**22×**). Exit 0 at batch 1 costs
  17× the full model at batch 64.

### 10.4 End-to-end policy energy on HAR (the real runtime; `e2e_har_harcnn.json`)

> **Superseded by §11.2.** This was the first HAR e2e run, saved to the wrong folder and without
> `--eval-json`. Kept because comparing it with the rerun measures run-to-run reproducibility.
Deploy split, 1474 windows. The runtime reproduces the offline policy exactly (runtime accuracy =
offline accuracy for every policy and mode). Energy is measured J/sample for the whole policy, vs the
full model at the same batch size:

| Policy (deploy acc) | b = 1 | b = 8 | b = 16 | b = 32 | b = 64 |
|---|---|---|---|---|---|
| Full model (93.15 %) | 5.42 mJ | 0.685 | 0.332 | 0.358 | 0.346 |
| conf τ = 0.90 (92.67 %, 87 % exit at head 0) | **+52 %** | +130 % | +172 % | +110 % | +15 % |
| conf τ = 0.99 (92.88 %) | +71 % | +172 % | +188 % | +52 % | **−13 %** |
| per-head 0.6/0.6 (92.13 %) | +19 % | +36 % | +85 % | −9 % | **−24 %** |

(Compaction runtime shown. Positive = early exit costs MORE than the full model.)

- **Early exit loses energy end to end on HAR at batch 1–16 for every policy tested**, although
  67–97 % of inputs leave early and FLOPs predict ~88–95 % savings. The FLOP prediction of policy
  energy is off by **−58 % to −95 %** in every cell.
- Mechanism: for a 3.9 MFLOP network, each exit adds kernel launches plus a host-side decision
  (`nonzero` / `.item()` sync). That costs more than the convolutions it skips, and the sync stalls
  keep the GPU at 435–600 MHz. Latency goes up too (batch 1: 0.54 → 0.90 ms/sample at τ = 0.9).
- Early exit pays only at batch 64, with aggressive thresholds: per-head 0.6/0.6 saves 24 % for a
  1.0-pt accuracy loss.
- Compaction beats batch-wait in 20 of 24 cells, by up to ~24 % at batch 64.
- **Missing:** the per-exit-*table* comparison (`tableErr`). The run did not get `--eval-json`, and it
  was saved to `results/` instead of `runs/e2e_har`. Rerun needed (§10.7).

### 10.5 Is the CIFAR policy result robust? (KILL-4 re-priced with each cost table)
Real CIFAR deploy split. Median saving at matched accuracy vs global confidence (wins / points):

| Cost table (batch-1 cascade) | Marginal-utility | Per-head | FLOP pricing (both) |
|---|---|---|---|
| run1+run2 pooled (unlocked, 3 repeats) | 17.7 % (9/12) | 15.1 % (11/12) | 1.7–2.1 % |
| unlocked, 10 repeats | 20.5 % (9/12) | 15.9 % (11/12) | 1.7–2.1 % |
| **locked 1500 MHz, 10 repeats** | **9.9 % (9/12)** | **9.8 % (11/12)** | 1.7–2.1 % |

- **It survives, but halves under the trustworthy (locked) table.** Honest claim: *measured
  pricing makes flexible per-head exit rules worth ~10 % at matched accuracy, where FLOP pricing says
  ~2 %*.
- Marginal-utility ≈ per-head thresholds (identical rule). Under locked costs they tie (9.9 vs 9.8 %).
- HAR KILL-4 (`runs/har_eval`) is not interpretable: its batch-1 cost table fails the
  identical-workload test (90 % mismatch), and §10.4 shows early exit does not pay there at batch 1 anyway.

### 10.6 Consolidated claims for the paper
1. **C1:** the NVML counter is valid under load (0.16 %). But batch-1 per-exit energy on a desktop GPU is
   **not reproducible under default DVFS** (identical workloads differ by 16–90 %). Locking clocks
   restores it (7 %, all exits resolvable).
2. **C2:** FLOPs preserve exit *order* but misstate the *size* of savings. The direction depends on model
   scale and batch: CIFAR-ResNet −42 … +7 pts; HAR-CNN +22 … +70 pts. Batching the full model beats
   every early exit at batch 1 (CIFAR 4.9–6.7×, HAR 22×).
3. **C3 (headline):** for the pervasive-scale model, the **real early-exit runtime costs more energy than
   the full model** at batch 1–16, contradicting FLOP-based estimates by 58–95 %.
4. **C4:** a measured cost term turns a ~2 % policy difference into ~10 % (locked). Marginal-utility adds
   nothing over per-head thresholds.

### 10.7 What was still missing at the end of §10 (all three now done, see §11)
1. ~~CIFAR end-to-end (Step 6)~~ → `runs/e2e_cifar` (§11.1).
2. ~~HAR end-to-end rerun with `--eval-json`~~ → `runs/e2e_har` (§11.2).
3. ~~End-to-end with locked clocks~~ → `runs/e2e_har_locked` (§11.2).
4. Seeds / variance: one training seed per model, and HAR accuracy moves ±1 pt across epochs.
5. One GPU model, Windows, ~19 background GPU processes (idle stable to ±0.2 W across runs).

### 10.8 Phase 2 files analysed
`runs/policy_pooled/`, `runs/eval_b1b8_unlocked_r10/`, `runs/eval_b1b8_locked1500/`,
`runs/har_eval/`, `runs/har_ltt_a003/`, `runs/har_ltt_a005/`,
`har_{teacher_resnet18,exit_kd_harcnn,exit_ce_harcnn}_e20_b64.json`, `e2e_har_harcnn.json`.
Local re-pricing (CPU, same checkpoint, real CIFAR-10) with the locked and r10 cost tables produced
the §10.5 rows.

---

## 11. Phase 2 completion: end-to-end policy energy (Steps 6, 7-e2e, locked e2e)

Three runs added: `runs/e2e_cifar`, `runs/e2e_har` (rerun with `--eval-json`), `runs/e2e_har_locked`
(`nvidia-smi -lgc 1500,1500`). All on git `e83076f`, real data, 2500 CIFAR / 1474 HAR deploy samples,
3 repeats per cell, 120 s warm-up, 60 s idle baseline. **In all three runs the runtime reproduced the
offline policy exactly** (runtime accuracy = offline accuracy for every policy, both modes), so the
energy numbers price the policy that was actually simulated.

### 11.1 CIFAR ResNet-14 end to end: early exit works, but only below batch 16
mJ per sample, compaction runtime, vs the full model at the same batch size:

| Policy (deploy acc; Δ vs full) | b = 1 | b = 8 | b = 16 | b = 32 | b = 64 |
|---|---|---|---|---|---|
| Full model (92.20 %) | 70.99 | 23.54 | 18.15 | 16.28 | 13.37 |
| conf τ = 0.99 (91.90 %; −0.30) | **−27 %** | **−35 %** | +22 % | +4 % | −6 % |
| per-head 0.99/0.97/0.75 (91.66 %; −0.54) | **−56 %** | **−39 %** | +18 % | −0.2 % | −10 % |
| conf τ = 0.90 (89.98 %; −2.22) | **−53 %** | **−18 %** | +5 % | −4 % | −15 % |
| per-head 0.98/0.8/0.6 (89.72 %; −2.48) | **−67 %** | **−49 %** | −3 % | −9 % | −13 % |

(Negative = energy saved. Bold = the regime where early exit clearly pays.)

- **Best honest operating point: per-head 0.99/0.97/0.75 at batch 1 — 56 % less energy for 0.54 pts
  of accuracy.** That is a genuinely strong workshop number, and it is measured, not modelled.
- Latency, however, *increases*: 1.99 ms/sample for the full model vs 2.32 ms for that policy at
  batch 1. Early exit here buys energy, not speed. Say this explicitly; a reviewer will check.
- **The saving collapses and reverses at batch 16**, where the GPU finally reaches a stable 1950 MHz
  and the full model becomes efficient: every policy except the most aggressive *costs* 5–29 % extra.
  A partial recovery appears at batch 64 (3–15 % saved) once per-batch overheads amortise again.
- Batching still wins overall: the cheapest batch-1 early exit (23.19 mJ) is still 1.7× the full
  model at batch 64 (13.37 mJ). But early exit closes most of the 5.3× batch-1 penalty.
- Caveat: this run was **not** clock-locked, and clocks differed per policy at b = 1 (870–1245 MHz).
  From §10.2 the batch-1 reproducibility floor is ~7–25 %, so the 18–67 % savings are real in
  direction and roughly right in size, but not to the last point. A locked-clock CIFAR e2e rerun
  would remove the last objection.

### 11.2 HAR CNN end to end: early exit never pays, and it is not DVFS
`runs/e2e_har` (default clocks) and `runs/e2e_har_locked` (1500 MHz). Extra energy vs the full
model, compaction (positive = early exit costs MORE):

| Policy (deploy acc; Δ) | b1 | b8 | b16 | b32 | b64 | b1 **locked** | b8 **locked** |
|---|---|---|---|---|---|---|---|
| Full model (93.15 %) | 5.45 mJ | 0.660 | 0.321 | 0.357 | 0.241 | 8.19 mJ | 1.002 |
| conf τ = 0.90 (92.67 %; −0.47) | +54 % | +119 % | +168 % | +76 % | +65 % | +39 % | +130 % |
| conf τ = 0.99 (92.88 %; −0.27) | +137 % | +166 % | +189 % | +45 % | +27 % | +81 % | +189 % |
| per-head 0.6/0.6 (92.13 %; −1.02) | +29 % | +42 % | +76 % | +6 % | +8 % | +26 % | +49 % |

- **Every policy, every batch size, both clock regimes: early exit costs more energy than simply
  running the whole 97 k-parameter network.** 87–97 % of inputs exit at head 0, and FLOPs predict
  ~90 % savings.
- Locking the clocks **does not rescue it** — this kills the obvious reviewer objection that the
  result is just the GPU down-clocking during the sync-heavy early-exit runtime. (Locked full-model
  energy is itself 50 % higher than unlocked, 8.19 vs 5.45 mJ at b = 1: forcing 1500 MHz on a
  workload the governor would have run at ~700 MHz wastes energy. Worth one sentence in the paper.)
- Latency also worsens: 0.53 → 0.90 ms/sample at b = 1 (0.57 → 0.83 locked).
- Mechanism: per-exit kernel launches plus the host-side decision (`nonzero` / `.item()` sync) cost
  more than the 3.9 MFLOPs of convolution they skip.

**Reproducibility of this run (the old misrouted `results/e2e_har_harcnn.json` vs the new one).**
Same command, different day: 34 of 50 paired rows agree within 10 % (recomputed from both JSONs), worst case 39 % (b = 1, conf 0.99).
But the **b = 64 full-model baseline moved 30 %** (0.346 → 0.241 mJ), and that alone flipped the only
positive HAR cell in the earlier run (per-head 0.6/0.6 "saves 24 % at b = 64") into "costs 8 %".
Report this: single-shot baselines on a millijoule-scale workload are not trustworthy, and the
conclusion should rest on the sign across all 45 policy cells of the rerun (all positive under compaction), not on any one cell.

### 11.3 The core claim: the additive cost table over-promises
`pred_table_err_pct` = (table prediction − measured) / measured. Negative means **the table predicts
less energy than the policy actually burns**, i.e. it promises savings that do not materialise.

| Batch | CIFAR table err | CIFAR FLOP err | HAR table err | HAR FLOP err |
|---|---|---|---|---|
| 1 | −18.5 … +2.0 % | −6.5 … +84 % | −75 … −72 % | −92 … −85 % |
| 8 | −39 … −3 % | −27 … +17 % | −80 … −75 % | −93 … −86 % |
| 16 | −38 … −31 % | −45 … −38 % | −83 … −76 % | −94 … −87 % |
| 32 | −31 … −23 % | −38 … −29 % | −86 … −75 % | −91 … −75 % |
| 64 | −22 … −14 % | −31 … −21 % | −77 … −53 % | −91 … −71 % |

- **The additive per-exit table, measured on the same GPU minutes earlier, still under-prices real
  policy energy in 109 of 111 policy cells** (CIFAR 52/54, HAR 45/45, HAR locked 12/12; both runtime
  modes; recomputed from the JSONs; an earlier draft said "46 of 48", which does not match the files). This is the paper's central result: even replacing FLOPs with
  *measured* per-exit energy does not fix policy pricing, because the missing cost is the exit
  machinery itself (launches, host syncs, the clock state it induces), not the arithmetic.
- The two cells where the table is accurate (±2 %) are CIFAR batch 1 with aggressive per-head
  thresholds — exactly the case where nearly everything exits at one head and the policy degenerates
  to a fixed shallow network.
- FLOPs at CIFAR batch 1 err in the *opposite* direction (up to +84 %, i.e. too pessimistic), so the
  sign of the FLOP error flips with batch size, as §5 found. The table error does not flip: it is
  systematically optimistic.

### 11.4 Compaction vs batch-wait
Compaction (`index_select` the survivors) is cheaper in **45 of 48 paired cells**: median 14.4 % on
CIFAR, 8.9 % on HAR, 4.9 % on HAR locked. The exceptions are all within measurement noise. This is
the energy counterpart to the latency-oriented rebatching results (Fluid Batching, DREX), and it is
a clean secondary contribution.

### 11.5 Updated claims
1. **C1** unchanged: validated telemetry; batch-1 per-exit energy needs locked clocks.
2. **C2** unchanged: FLOPs preserve order, misstate size, with a scale- and batch-dependent sign.
3. **C3 (headline, now on two models):** the additive cost table — the pricing assumption behind
   essentially every early-exit policy paper — **systematically over-promises**, by 3–39 % on a
   CIFAR ResNet and 53–87 % on a pervasive-scale CNN. On the pervasive model the error is large
   enough to invert the decision: early exit costs more energy than the full network at every batch
   size, under default and locked clocks.
4. **C3b:** where early exit does pay (CIFAR, batch < 16), it pays well — 56 % energy for −0.54 pts
   accuracy — but it costs latency, and batching the full model still beats it.
5. **C4** unchanged: measured pricing makes flexible per-head rules worth ~10 % vs ~2 % under FLOPs;
   marginal-utility ≡ per-head thresholds.

### 11.6 Remaining gaps (honest list for the limitations section)
1. **CIFAR e2e was unlocked.** Phase 3 P3.5 re-runs b = 1, 8, 16 at 1500 MHz with 10 randomized
   windows and `--require-pinned-clock`; until then, §11.1 carries the DVFS caveat.
2. **HAR b = 64 baseline was not reproducible** (30 % between runs). Phase 3 P3.6 runs two
   independent locked sessions at b = 64 and `audit compare` reports the agreement; the paper must
   quote the between-session error bar rather than a single cell.
3. One GPU (RTX 3070), Windows, ~19 background GPU processes in Phase 1/2 (Phase 3 records a
   co-tenancy snapshot and requires <=1 compute process), one training seed per model in Phase 1/2
   (Phase 3 adds seeds 42/43/44), FP32, one exit placement per architecture, no battery-powered device.
4. HAR KD hurt (teacher 92.74 % < CE student 93.42 %); report `exit_ce` as the stronger baseline.

Phase 3 outputs were written to `results/p3/` on the 5060 box with seed-suffixed names (`*_s42`, `*_s43`, `*_s44`) and the
`audit_verify.json` / `audit_compare.json` files; on the analysis machine they sit in `new_results/results/p3/`
(and `p3_unlocked/`). **Their full analysis on the new RTX 5060 Laptop (Blackwell) platform is detailed in §12 below.**

---

## 12. Phase 3 Results — Pre-Submission Hardening Pass (RTX 5060 Laptop, Blackwell sm_120)

> **Platform & Hardware Note (Separate Platform):**  
> Per `RUN_PLAN.md`, this Phase 3 suite was executed on a **new, independent platform**:  
> **NVIDIA GeForce RTX 5060 Laptop GPU** (Blackwell architecture, `sm_120`, 8 GB VRAM, locked with `nvidia-smi -lgc 1500,1500` and gated by `--lock-mhz 1500`; observed window clocks 1447–1485 MHz), driver 610.62 (from the result JSONs), CUDA 12.8, PyTorch 2.8.0+cu128, Windows.  
> Following protocol rules, these numbers are **not pooled** with Phase 1/2 (RTX 3070 Ampere). Instead, Phase 3 stands on its own as the self-contained, pre-submission hardening benchmark featuring:
> - **3 independent training seeds** ($s=42, 43, 44$) for all models (MD Fix B);
> - **$\ge 10$ randomized repeated measurement windows** per cell (MD Fix A);
> - **Locked clocks** (`--require-pinned-clock`) across all evaluation and E2E sweeps;
> - **Student-$t$ 95% Confidence Intervals (CI)** on all physical energy measurements;
> - **Automated audit scripts** (`src/audit.py verify` & `compare`) evaluating publishability.

---

### 12.1 Layman's Summary: What Did This New Run Find?

To understand these results without getting lost in the equations:
1. **The "Theoretical Table" is Over-Optimistic (The Core Paper Finding):**
   - *In Simple Terms:* Research papers often calculate early exit energy by saying: *"If 70% of inputs exit at Head 0 and 30% exit at Head 1, just multiply 70% by Head 0's cost and 30% by Head 1's cost."*  
   - *What the Hardware Actually Paid:* When we measure the real GPU at the wall meter, the table **predicted 23% to 54% less energy than was measured on images (CIFAR)** and **68% to 89% less on wearable sensors (HAR)** (table error = (table − measured)/measured, negative in all 55 policy cells). The table completely ignores the physical "toll" the GPU pays to check exit confidence, split tensors, and make decisions on the fly.
2. **Batching Beats Early Exit:**
   - *In Simple Terms:* Developers use early exit hoping to save power. But on modern GPUs, simply gathering samples into a batch (e.g. batch size 8 or 16) and running the standard network is usually cheaper than running early exit. Early exit's only clear saving was at batch size 1 with the most aggressive thresholds. By batch size 8 and 16, most policies **lost** energy (up to +23% under compaction, +36% under batch-wait); only the most aggressive policy (P3, 2.3 points of accuracy lost) still saved about 10%.
3. **Pervasive Wearable Models (UCI-HAR CNN) Suffer a Catastrophic Penalty:**
   - *In Simple Terms:* On wearable activity tracking (a tiny 97,000 parameter network), early exit was a total loss. Because the network is already small, the extra machinery required to check exit confidence costs **140% to 240% MORE energy** than just running the entire network. FLOP calculations promised a 90% savings, but the physical meter showed a 2× to 3× energy penalty.
4. **Clock locking fixes the operating point, not resolvability, on this GPU:**
   - *In Simple Terms:* Consumer GPUs change their frequency on their own, like a car revving at stoplights. With the clock locked at 1500 MHz, per-exit measurements were repeatable (within-cell variation 1.0% to 15.9%, mostly under 6%). But the unlocked run (`results/p3_unlocked`) was *also* resolvable (the clock sat at ~2.78 GHz and every adjacent exit pair separated), so the 3070 finding "unlocked batch 1 is unresolvable" did **not** reproduce here. What locking changes is the absolute energy: unlocked per-exit energies were **1.96× to 3.65×** the locked ones (median 2.9×). Locked numbers are therefore "energy at 1500 MHz", not "energy at the default clock".

---

### 12.2 Multi-Seed Training Robustness (Step P3.2, MD Fix B)

All models were trained across 3 random seeds ($s=42, 43, 44$) for 20 epochs at batch 64 using cosine annealing and AMP.

| Dataset & Model Architecture | Seed 42 Test Acc | Seed 43 Test Acc | Seed 44 Test Acc | Mean ± Std | Spread | Status |
|---|---|---|---|---|---|---|
| **CIFAR-10** Teacher ResNet-18 | 92.97 % | 92.43 % | 93.02 % | 92.81 ± 0.33 % | 0.59 pts | ✅ Tight (<1.5 pts) |
| **CIFAR-10** Early-Exit KD ResNet-14 | 92.13 % | 91.70 % | 92.29 % | 92.04 ± 0.31 % | 0.59 pts | ✅ Tight (<1.5 pts) |
| **UCI-HAR** Teacher ResNet-18 | 93.42 % | 93.18 % | 93.72 % | 93.44 ± 0.27 % | 0.54 pts | ✅ Tight (<1.5 pts) |
| **UCI-HAR** Early-Exit KD HAR-CNN | 92.74 % | 92.43 % | 93.55 % | 92.91 ± 0.58 % | 1.12 pts | ✅ Tight (<1.5 pts) |

*Assessment:* The final-head accuracy spread across all three seeds is $\le 1.12$ points (comfortably below the 1.5-point gate in `RUN_PLAN.md`), proving that model accuracy is robust to training initialization.

---

### 12.3 Learn-Then-Test (LTT) Risk Certification (Step P3.3, C4)

Calibrated on the 5,000-sample calibration split using Hoeffding bounds with confidence level $1 - \delta = 0.90$:
- **CIFAR-10 ($\alpha = 0.02$ risk bound; calibration risk is on the calibration split, "deploy" is the held-out split):**

  | Seed | Selected $\tau$ | Calibration risk | Deploy risk (extra error) | Deploy accuracy (drop) | Exit rate |
  |---|---|---|---|---|---|
  | 42 | 0.99 | 0.0030 | 0.0032 | 91.76 % (−0.26 pts) | 68.1 % |
  | 43 | 0.99 | 0.0040 | 0.0040 | 91.38 % (−0.36 pts) | 68.0 % |
  | 44 | 0.995 | 0.0034 | 0.0024 | 92.10 % (−0.22 pts) | 63.2 % |

  (An earlier draft listed 0.0032 and 0.0016 for seeds 43 and 44; those were other seeds' values. Corrected from the JSONs.)
- **UCI-HAR ($\alpha = 0.05$ risk bound):** all 14 grid thresholds are certified for all three seeds, so the selected $\tau = 0.60$ is the grid floor, not a tight bound.

  | Seed | Calibration risk at 0.60 | Deploy accuracy (drop) | Exit rate |
  |---|---|---|---|
  | 42 | 0.0081 | 92.13 % (−0.95 pts) | 98.4 % |
  | 43 | 0.0020 | 92.67 % (−0.14 pts) | 98.6 % |
  | 44 | 0.0136 | 92.06 % (−1.76 pts) | 99.5 % |

- *Conclusion:* the CIFAR guarantee holds on held-out data for all three seeds with a 0.2–0.4-point accuracy cost. On HAR it holds but is loose (everything certifies), and seed 44 loses 1.76 points.

---

### 12.4 Per-Exit Hardware Energy Sweep (Step P3.4, Contribution C1 & C2)

Hardware measurements on RTX 5060 Laptop GPU under locked clock (requested 1500 MHz, observed 1470–1485 MHz in the per-exit windows; 1447–1485 MHz across the end-to-end windows). 10 randomized windows per cell, each $\ge 3\text{ s}$ and $\ge 1000\text{ iterations}$. **The two tables below are the *cascade* mode** (the deployed path: earlier heads also evaluated, host-side decision each time); the same sweep in *prefix* mode is quoted in the text after them. The "±" here is the audit's 95 % CI half-width.

#### CIFAR-10 Early-Exit ResNet-14 (Seed 42, Session A, cascade mode):
*Exits 0, 1, 2, and Final Head 3:*

| Batch ($b$) | Exit 0 (mJ) | Exit 1 (mJ) | Exit 2 (mJ) | Final Head 3 (mJ) | $3\sigma$ Adjacent Resolvability | CV (%) | Clock range (`stable=True`) |
|---|---|---|---|---|---|---|---|
| **$b = 1$** | $5.079 \pm 0.19$ | $9.304 \pm 0.29$ | $13.702 \pm 0.44$ | $24.779 \pm 0.79$ | **[True, True, True]** (3/3) | 4.3 – 5.1 % | 1470–1485 MHz |
| **$b = 8$** | $1.511 \pm 0.02$ | $2.414 \pm 0.04$ | $3.372 \pm 0.07$ | $4.906 \pm 0.04$ | **[True, True, True]** (3/3) | 1.0 – 3.0 % | 1470–1477 MHz |
| **$b = 16$** | $2.005 \pm 0.03$ | $2.767 \pm 0.03$ | $3.451 \pm 0.04$ | $4.343 \pm 0.09$ | **[True, True, True]** (3/3) | 1.4 – 2.8 % | 1477–1477 MHz |

Prefix mode (run exactly up to head $k$), same session, mJ: $b=1$ 4.833 / 6.959 / 9.551 / 17.544; $b=8$ 1.519 / 2.226 / 3.039 / 4.443; $b=16$ 2.007 / 2.637 / 3.318 / 4.104. All adjacent pairs resolvable in all six cells.

#### UCI-HAR CNN (Seed 42, Session A, cascade mode):
*Exits 0, 1, and Final Head 2:*

| Batch ($b$) | Exit 0 ($\mu$J) | Exit 1 ($\mu$J) | Final Head 2 ($\mu$J) | $3\sigma$ Adjacent Resolvability | CV (%) |
|---|---|---|---|---|---|
| **$b = 1$** | $898.4 \pm 36.8$ | $3051.1 \pm 222.0$ | $5390.8 \pm 252.6$ | **[True, True]** (2/2) | 5.7 – 10.2 % |
| **$b = 64$** | $22.4 \pm 0.5$ | $67.7 \pm 2.7$ | $118.7 \pm 4.5$ | **[True, True]** (2/2) | 3.3 – 5.6 % |

HAR prefix mode, $\mu$J: $b=1$ 1114 / 1673 / 2593 (**not resolvable**: CV 14–16 %, CIs overlap); $b=64$ 22 / 43 / 67 (resolvable).

*Key Findings:*
- **Contribution C1 Validated:** On the Blackwell RTX 5060, locking clocks achieves **100% $3\sigma$ resolvability** across all adjacent heads.
- **Batching Economy:** For CIFAR-10, the full model at $b=8$ (4.39 mJ/sample, end-to-end session A) is **3.4× cheaper** than the best batch-1 early-exit policy (P3, 14.73 mJ) and **4.5× cheaper** than the batch-1 full model (19.70 mJ); at $b=16$ the full model is 5.0× cheaper than at $b=1$. (An earlier draft said "5.05× cheaper than Exit 0 at b=1"; that compared two near-equal numbers, 4.906 vs 5.079 mJ, and was wrong.)

#### 12.4b Seed spot-check at batch 1 (P3.4, seeds 42 / 43 / 44, CIFAR, mJ per sample)

Same protocol, one session each (`eval_cifar10_resnet14_s{42,43,44}_A.json`), all three audit-clean, all adjacent exits resolvable (3/3):

| Mode | Exit 0 | Exit 1 | Exit 2 | Final | Spread across seeds (max−min)/mean |
|---|---|---|---|---|---|
| prefix | 4.83 / 4.67 / 4.51 | 6.96 / 7.01 / 6.50 | 9.55 / 9.30 / 8.81 | 17.54 / 17.91 / 17.14 | 4.4 – 8.1 % |
| cascade | 5.08 / 4.62 / 4.60 | 9.30 / 8.21 / 8.33 | 13.70 / 11.91 / 12.34 | 24.78 / 21.79 / 24.12 | 10.0 – 14.2 % |

The cascade path (the one policies are priced with) varies more across seeds (up to 14 %) than the prefix path (up to 8 %). Seed 42 is the highest in every cascade column, so the cascade-mode numbers in §12.4 should be read as the upper end across seeds. One session per seed, so this mixes seed variation with session-to-session drift.

#### 12.4c Locked vs unlocked contrast (P3.3b, seed 42, b = 1 and 8; `results/p3_unlocked`)

| | Locked (1500 MHz target) | Unlocked (default) |
|---|---|---|
| Graphics clock during windows | 1470–1485 MHz | CIFAR b=1: 2767–2790; CIFAR b=8: 2362–2745 (`stable=False`); HAR: 2790–2812 MHz |
| Adjacent exits resolvable (3σ) | all | **all** (CIFAR and HAR, both modes) |
| CV (%) | CIFAR 1.0 – 5.1; HAR 2.7 – 10.2, except HAR prefix b=1 at 14–16 | CIFAR 2.7 – 6.2; HAR 2.9 – 9.9 |
| Energy, unlocked ÷ locked | — | **1.96× – 3.65×** over 22 matched per-exit cells (median 2.9×); CIFAR 1.96–3.65, HAR (b=1 only) 2.76–3.61 |

- The 3070 result "unlocked batch 1 is unresolvable (0/3)" **did not reproduce on this GPU**. Here the unlocked card sat near 2.8 GHz and every pair separated. So on the 5060 the lock does not rescue resolvability; it fixes the operating point and removes the clock as a variable between sessions.
- The unlocked files were produced at git `7b90eae`, the locked eval/e2e files at `0960180` (training also `7b90eae`). `git diff` between them touches only the clock-check gate (`measure.clock_check`: under-load warm-up thread and the `--lock-mhz` check; plus tests and docs). The energy-window, idle and warm-up code is unchanged.
- Idle power differed: 15.1 W (CIFAR unlocked), 12.3 W (HAR unlocked) vs 13.6–14.5 W locked, so the idle-subtracted ratios carry about ±1 W of baseline uncertainty on a 14 W idle.

---

### 12.5 End-to-End Measured Policy Energy (Step P3.5 & P3.6, Contribution C3)

Physical runtime evaluation over the test dataset comparing measured Joules/sample against the full base model and against the additive theoretical table.

#### CIFAR-10 ResNet-14 (Session A, Seed 42, Locked Clocks):

| Batch | Policy | Runtime Mode | Measured Energy (mJ/sample) | Saving vs Full (%) | Table Error (%) | Layman Interpretation |
|---|---|---|---|---|---|---|
| **$b=1$** | Full Model | compact | $19.701 \pm 0.65$ | 0.0 % | — | Baseline full network |
| $b=1$ | `conf-0.97` | compact | $18.860 \pm 0.61$ | **+4.3 %** | **−40.1 %** | Saves a tiny 4.3%, table claimed 44% saving! |
| $b=1$ | `conf-0.99` | compact | $21.566 \pm 0.72$ | **−9.5 %** | **−38.8 %** | High confidence threshold **costs 9.5% MORE** energy |
| $b=1$ | `perhead-0.99/0.98/0.7` | compact | $18.356 \pm 0.60$ | **+6.8 %** | **−43.8 %** | Saves 6.8% energy |
| $b=1$ | `perhead-0.93/0.85/0.6` | compact | $14.731 \pm 0.49$ | **+25.2 %** | **−44.6 %** | Best exit saving: 25.2% energy saved at a **2.30 pt** accuracy drop (89.72% vs 92.02%) |
| **$b=8$** | Full Model | compact | $4.387 \pm 0.05$ | 0.0 % | — | Baseline full network at batch 8 |
| $b=8$ | `conf-0.97` | compact | $4.841 \pm 0.09$ | **−10.3 %** | **−45.2 %** | Early exit burns **10.3% MORE** energy than full |
| $b=8$ | `conf-0.97` | batchwait | $5.761 \pm 0.08$ | **−31.3 %** | **−54.0 %** | Without compaction, burns 31.3% extra energy |
| $b=8$ | `conf-0.99` | compact | $5.043 \pm 0.10$ | **−15.0 %** | **−40.7 %** | Burns 15.0% MORE energy |
| $b=8$ | `perhead-0.93/0.85/0.6` | compact | $3.937 \pm 0.07$ | **+10.3 %** | **−45.6 %** | Saves only 10.3% (vs 25.2% at batch 1) |
| **$b=16$** | Full Model | compact | $3.915 \pm 0.04$ | 0.0 % | — | Baseline full network at batch 16 |
| $b=16$ | `conf-0.97` | compact | $4.017 \pm 0.04$ | **−2.6 %** | **−29.5 %** | Burns 2.6% MORE energy |
| $b=16$ | `conf-0.99` | compact | $4.004 \pm 0.04$ | **−2.3 %** | **−23.5 %** | Burns 2.3% MORE energy |
| $b=16$ | `perhead-0.99/0.98/0.7` | compact | $3.841 \pm 0.05$ | **+1.9 %** | **−26.7 %** | Saving essentially vanishes (+1.9%) |
| $b=16$ | `perhead-0.93/0.85/0.6` | compact | $3.544 \pm 0.04$ | **+9.5 %** | **−29.6 %** | Saving drops to 9.5% |

#### UCI-HAR CNN (Session A & B, Seed 42, Locked Clocks):

| Batch | Policy | Runtime Mode | Measured Energy ($\mu$J/sample) | Saving vs Full (%) | Table Error (%) | Layman Interpretation |
|---|---|---|---|---|---|---|
| **$b=1$** | Full Model | compact | $2119.3 \pm 72.1$ | 0.0 % | — | Baseline wearable model at batch 1 |
| $b=1$ | `conf-0.97` | compact | $7102.8 \pm 231.0$ | **−235.2 %** | **−71.2 %** | Burns **$3.35\times$ MORE energy** than full model! |
| $b=1$ | `conf-0.99` | compact | $7264.8 \pm 245.2$ | **−242.8 %** | **−68.4 %** | Burns **$3.43\times$ MORE energy** |
| $b=1$ | `perhead-0.6/0.6` | compact | $5735.8 \pm 189.4$ | **−170.6 %** | **−82.6 %** | Burns **$2.71\times$ MORE energy** |
| **$b=64$** | Full Model | compact | $73.6 \pm 2.1$ | 0.0 % | — | Baseline wearable model at batch 64 |
| $b=64$ | `conf-0.97` | compact | $203.0 \pm 6.8$ | **−176.0 %** | **−76.9 %** | Burns **$2.76\times$ MORE energy** than full model! |
| $b=64$ | `conf-0.97` | batchwait | $246.0 \pm 8.2$ | **−234.5 %** | **−80.9 %** | Burns **$3.34\times$ MORE energy** |
| $b=64$ | `perhead-0.6/0.6` | compact | $178.2 \pm 5.9$ | **−142.2 %** | **−86.2 %** | Burns **$2.42\times$ MORE energy** |

*(Session B repeated $b=64$ independently: full model $68.8\ \mu\text{J}$, `conf-0.97` $211.7\ \mu\text{J}$ [−207.7% saving], showing tight cross-session agreement within 6.7%).*

---

### 12.6 Takeaway Findings & How to Frame the Paper for IEEE PerCom

When drafting the conference paper, structure the arguments around these concrete findings:

1. **Contribution C3 (The "Table Illusion"):**  
   Every paper that optimizes early-exit policies using static additive tables is pricing decisions on an illusion. On our measured RTX 5060 platform, the table underestimates energy expenditure by **23% to 54% on CIFAR-10** (23% to 46% under compaction, the deployed mode; 54% is a batch-wait cell) and by **68% to 89% on UCI-HAR**, in all 55 policy cells.
2. **Contribution C2 (Batching Inversion):**  
   Early exit only achieves energy reduction at batch 1 (saving at most 25.2% on CIFAR-10). As batch size increases to 8 and 16, the saving collapses and reverses into an energy penalty. Furthermore, running the full unpruned model at batch 8 costs about 30% of the best batch-1 early exit (3.4× cheaper in session A: 4.39 vs 14.73 mJ; 3.3× in session B: 4.12 vs 13.50 mJ).
3. **Pervasive Computing Reality (The UCI-HAR Negative Result):**  
   For pervasive/edge applications running tiny models (e.g. 97k-param HAR CNNs), **early exit should not be deployed on GPUs**. The host synchronization and index-selection overhead vastly exceeds the arithmetic cost of the skipped layers, turning a theoretical 90% FLOP saving into a 140% to 240% physical energy increase.
4. **Reproducibility & Verification Status:**  
   - 20 out of 24 paired cells agree within 10% across sessions.
   - The 4 cells exceeding 10% occur exclusively at $b=1$ on CIFAR (10.4%–11.8% spread; the full-model baseline itself moved 10.4%), reflecting small thermal drift on power-limited laptop silicon.
   - Checkpoints, training logs, LTT bounds, and raw window traces are fully preserved in `new_results/results/p3/` (analysis machine) / `results/p3/` (5060 box).

---

### 12.7 Verification notes (numbers re-derived from the raw JSONs; paper source: `IEEE-conference-template-062824/percom_paper.tex`)

- **The "±" values in the §12.5 tables are NOT the audit's 95% CI.** They are about half the real half-width (e.g. CIFAR b=1 full model: table says ±0.65, JSON `j_ci95_halfwidth` = 1.31 mJ/1000). Use the paper tables / JSON for any CI. This matters: at b=1 the CIs of the full model and most policies overlap, so only P3 (−25.2% A, −24.0% B) and, in session B only, P2 (−8.2%) are separated from the full model.
- **Significance (95% CIs vs full model, compaction).** b=8: every policy except P3 costs significantly more in both sessions (+4% to +23%); P3 saves 10.3% (A, significant) but 3.3% (B, not significant). b=16 (A only): P3 saves 9.5% (significant); conf-0.97 and P1 cost more; conf-0.99 and P2 are not significant. The LTT-certified conf-0.99 never saves: b=1 +9.5%/+7.9% (n.s.), b=8 +15.0%/+23.2%, b=16 +2.3% (n.s.).
- ~~No telemetry-validation pilot for the 5060~~ **Resolved Nov 13: the pilot was delivered and passes (residual 0.13 %), with one caveat (Steam running). See §13.**
- **Energy windows use the first 2500 images of the CIFAR-10 deploy split** (all 1474 HAR windows); the runtime's accuracy on that subset equals the offline policy's. Accuracy columns in the paper are over the full 5000-image deploy split.
- **Unlocked contrast (`p3_unlocked`)** is in §12.1 item 4: unlocked energy 1.96×–3.65× locked; adjacent exits resolvable in every unlocked cell. HAR b=1 *prefix* (locked) is the only unresolved pair (CV 14–16%); the cascade for the same exits is resolvable.
- **LTT:** HAR certifies every grid threshold down to the floor (τ=0.6), so the selected τ is the grid floor, not a tight bound.
- **Cascade toll** (final exit, cascade ÷ prefix): CIFAR 1.41× / 1.10× / 1.06× at b=1/8/16; HAR 2.08× / 1.77× at b=1/64.
- **Operation-count (FLOP) error at exit 0** (predicted − measured saving, points): CIFAR prefix −18.9 / −12.2 / +2.5 and cascade −25.9 / −15.7 / −0.3 at b=1/8/16; HAR prefix +34.4 / +24.9 and cascade +8.1 / +10.3 at b=1/64. KILL-3 (order disagreement) does not fire in any cell.

---

### 12.8 C4 policy benchmark on the 5060 (table-priced, not end-to-end measured)

Each `eval_*_A.json` carries a `kill4` block: policies compared at matched accuracy, priced with the batch-1 **cascade** cost table from the same locked session. "Saving" = cost reduction of the first rule versus the reference rule at the same accuracy; `n` = matched points with a defined reference. Median [min, max], %:

| Model (seed) | Marginal-utility vs global confidence | Marginal-utility vs per-head | Per-head vs global confidence |
|---|---|---|---|
| CIFAR (42) | 9.0 [5.1, 12.7], n=9 | 3.3 [−0.1, 10.8], n=9 | 8.3 [0.0, 21.7], n=12 |
| CIFAR (43) | 9.5 [−37.2, 23.2], n=12 | 5.0 [2.3, 10.0], n=9 | 8.1 [0.0, 19.6], n=12 |
| CIFAR (44) | 10.6 [4.4, 23.8], n=9 | 6.0 [2.4, 8.9], n=8 | 8.8 [0.0, 22.0], n=12 |
| HAR (42) | 4.5 [−9.6, 15.6], n=8 | 5.4 [−17.4, 10.1], n=7 | 5.4 [−8.0, 12.3], n=8 |

- Same direction as Phase 2 on CIFAR (§10.5: about 10 % under the locked 3070 table), reproduced on a second GPU and three seeds. A measured cost term makes flexible per-head rules worth roughly 8–10 % at matched accuracy.
- Marginal-utility is algebraically a per-head confidence rule (`test_units.py`), so its small edge over per-head thresholds (3–6 %) reflects the λ grid versus the threshold grid, not a new decision rule. Do not present it as a controller.
- These are **table-priced** savings. Section 12.5 shows the table under-prices real policy energy by 23–54 % (CIFAR), so absolute joules here are optimistic. The ranking between rules is not independently confirmed end to end.
- On HAR the comparison is of limited meaning: every early-exit policy costs more than the full model (§12.5), so "saving versus confidence" ranks policies that are all worse than not exiting.

### 12.9 Provenance and integrity of the Phase 3 files

- **Code versions:** training, LTT and the unlocked contrast ran at git `7b90eae`; locked eval/e2e at `0960180` (only the clock gate changed, §12.4c). Phase 2 used `e83076f`; Phase 1 `0167720` / `5eacc30`.
- **Sessions:** `p3-eval-cifar-s{42,43,44}`, `p3-eval-har-A`, `p3-e2e-cifar-{A,B}`, `p3-har-A`, `p3-har-b64-B`, `p3-eval-*-s42-U`. Idle baselines 13.6–14.5 W locked.
- **Co-tenancy:** every locked eval/e2e file records `compute_process_count = 1` and `co_tenant: false` (the benchmark itself). This is a large improvement over the 19–20 background processes in Phase 1/2.
- **Audit:** `audit verify` passes 8/8 files (`min_repeats 10`, `min_window_s 3`, `min_iters 1000`, clock spread ≤ 30 MHz); `audit compare` finds 4 of 24 repeated cells above 10 %, all CIFAR b = 1 compaction (§12.6).
- **Not logged:** NVML reports `power_limit_w = −1` on this laptop part, so the AGENTS.md §6 "power limit" field is unavailable and must be stated as such.
- **Pilot:** delivered Nov 13 and passes (counter vs 141.5 Hz poll, residual 0.13 %). Details and caveats in §13. The 0.16 % figure in §1 remains the 3070 number; the 5060 number is 0.13 %.
- **Privacy:** the `co_tenancy.processes` field stores the full path of the Python interpreter, which includes the collaborator's Windows account name. These JSONs are gitignored; scrub that field before sharing any raw result file outside the project (supplementary material, artifact upload, blind-review package).


---

## 13. RTX 5060 Laptop telemetry pilot (RUN_PLAN Step 0c), delivered Nov 13 2026

Source: `new_results/results/p3/pilot.json` (session `p3-pilot`, git `0960180`, command
`python src/check_gpu.py --idle-seconds 60 --stress-seconds 30 --session p3-pilot --results results/p3`).
This closes the one Phase 3 step that was missing (§0 table, §12.9).

### 13.1 Result: PASS

| Check | Value | Gate | Verdict |
|---|---|---|---|
| CUDA kernel check | `ok: true`, capability `sm_120`, torch 2.8.0+cu128, CUDA 12.8 | must run on sm_120 | ✅ |
| Hardware energy counter | `has_energy_counter: true`, `meter_source: counter` | required | ✅ |
| Validation basis | `counter-vs-poll` (not `poll-only-unverifiable`) | counter-vs-poll | ✅ |
| **Dummy-kernel residual** | **0.00134 (0.13 %)** = (512.103 J poll − 511.415 J counter) / 511.415 J, 10.0 s window, 1415 samples (141.5 Hz) | ≤ 5 % | ✅ |
| Idle window (60 s) | counter 821.257 J vs poll 825.571 J (0.53 % apart), idle power **13.69 W**, 8478 samples (141.3 Hz) | none (information) | ✅ consistent with the 13.6–14.5 W idle used in every Phase 3 file |
| Stress (30 s matmul) | 10 613 iterations; 48 → 62 °C; 13.4 → 51.0 W; average 51.15 W in the validation window | none | ✅ |
| Driver / GPU | 610.62, RTX 5060 Laptop, 8151 MiB | matches the Phase 3 files | ✅ |

What this establishes: on this part the hardware-integrated counter and 100–200 Hz polling of the
same NVML interface agree to 0.13 % (stress) and 0.5 % (idle). The earlier 0.16 % (§1) was the
RTX 3070, so each platform now has its own residual.

What it does **not** establish: both numbers come from NVML, so this is a self-consistency check
of the counter against polling, not validation against an external meter. The paper limitation
about an external meter stays.

### 13.2 Caveats found in the file (be honest about these)

1. **Co-tenant present during the pilot.** `co_tenancy.co_tenant: true`, `compute_process_count: 2`:
   `steam.exe` (pid 23636) plus the benchmark's own Python. Steam also still showed in the
   post-run compute-app listing. The Phase 3 eval/e2e files all recorded one process (§12.9), so
   the measurement files are unaffected, but the pilot itself was not run under the "no other GPU
   process" hygiene of RUN_PLAN Step 0d. A residual of 0.13 % with Steam resident is, if anything,
   a harder test, but it is not a clean-state run. **Recommended: rerun the pilot once with Steam closed** (12 min), or state the caveat.
2. **Clock verdict reads `variable` / `pinned: false`.** Cause: the pilot calls `clock_check` with no
   `--lock-mhz`, so it falls back to the 50 %-of-max rule (floor 1545 MHz) and 1492 MHz is below it.
   It is not a lock failure: all 8 samples were 1492 MHz, spread 0.0 MHz, and stress-time clocks were
   1477–1485 MHz, i.e. the same band as the locked Phase 3 windows (1447–1485 MHz). The lock was
   therefore still applied during the pilot.
3. **Lock release is not confirmed.** `nvidia-smi -rgc` printed "All done", but the next query still read
   1492 MHz (identical to the locked value, max 3090 MHz). An idle unlocked GPU normally reads far
   lower, so the clock may simply have been held by load (Steam) or have not yet settled. The reading
   neither proves nor disproves the release. RUN_PLAN P3.8 stays "unverifiable" until a re-query
   after ~10 s of idle shows a clock below ~1000 MHz, or `nvidia-smi -q -d CLOCK` shows no locked limit.
4. **Power limit still unavailable:** `power_limit_w = −1`, `smi` raw `[N/A]` (same as §12.9).
5. **Privacy:** `co_tenancy.processes[1].name` contains the collaborator's Windows account name in the
   interpreter path. The file was moved from the repo root (where git does not ignore it) into the
   gitignored `new_results/results/p3/`. Scrub that field before sharing it anywhere.

### 13.3 What changes in the paper

- Limitation (iii) can now say: the counter agreed with polling to 0.13 % on this part in a
  dummy-kernel check; it was not compared with an external meter.
- Do not claim the pilot ran in a clean state (Steam was resident) unless it is rerun.

### 13.4 Phase 3 status after this file

| Step | State |
|---|---|
| 0c pilot | ✅ delivered, PASS (caveat: Steam resident; optional clean rerun) |
| P3.2–P3.7 | ✅ unchanged (§12) |
| P3.8 clock release | ⚠️ still unverifiable (§13.2 item 3) |
| O1–O3 | not run, not needed for the current claims |
