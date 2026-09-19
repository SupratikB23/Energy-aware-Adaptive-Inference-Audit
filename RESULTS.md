# RESULTS — measured-energy audit of early-exit policies (CIFAR-10, RTX 3070)

Scope: first complete pass of RUN_PLAN Steps 1–4 on one GPU box. All numbers below are
read from `results/` (files listed in §9). Research framing and kill criteria:
[percom2027_literature_matrix_and_verdict.md](percom2027_literature_matrix_and_verdict.md).

**Setup.** NVIDIA GeForce RTX 3070 (8 GB, sm_86, 270 W limit), driver 591.86, torch 2.7.0+cu126,
Python 3.11.9, Windows. FP32 inference (cuDNN TF32 on, matmul TF32 off, cudnn.benchmark on).
Real CIFAR-10 everywhere (`synthetic_data: false` in every file). Seed 42.
Energy = NVML hardware energy counter, idle-subtracted, J **per sample**.
Protocol: 60 s idle baseline → 120 s warmup → per exit 3 repeated windows, each ≥1000 iters
and ≥3 s, round-robin order. Two independent full benchmark runs (`eval_run1`, `eval_run2`).

---

## 0. Verdict in one table

| Gate | Result | Status |
|---|---|---|
| **KILL-1** telemetry valid | counter vs poll residual **0.16 %** (limit 5 %) | ✅ PASS |
| **KILL-2** adjacent exits resolvable (>3σ) at batch 1 | exit 0↔1 **not** resolvable (both runs, both modes); 1↔2 resolvable; 2↔3 mixed | ⚠️ PARTIAL FAIL |
| KILL-2 at batch ≥16 | all adjacent exits resolvable, both runs, both modes | ✅ |
| **KILL-3** measured Joule order ≠ FLOP order | **no** significant disagreement at any batch, either run | ❌ FIRES → drop the "FLOPs get the *order* wrong" claim |
| FLOPs predict the *size* of savings | **no**: off by −7 to +17 pts, sign flips with batch size (§4) | ✅ usable claim |
| **KILL-4** marginal-utility beats confidence at matched accuracy | 10–31 % cheaper in both runs (mid/low-accuracy range), with serious caveats (§6) | ⚠️ PASS, weak |
| LTT accuracy-risk guarantee holds on held-out split | yes for α = 0.02 / 0.03 / 0.05; α = 0.01 uncertifiable | ✅ |

**Headline:** the measurement protocol works (C1), and the regime map (C2) has a clear, reproducible
story. But **batching, not early exit, dominates energy per sample on this GPU**, and batch-1
per-exit energy is too noisy to support fine-grained per-exit policy claims.

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

## 8. Recommended next runs (in priority order)
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
