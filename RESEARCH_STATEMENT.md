# Research statement (v2, after the first RTX 3070 results)

Controlling verdict and full literature matrix:
[percom2027_literature_matrix_and_verdict.md](percom2027_literature_matrix_and_verdict.md).
First results: [RESULTS.md](RESULTS.md). Run order: [RUN_PLAN.md](RUN_PLAN.md).

## Working title
**Priced by the Table, Paid at the Meter: a measured-energy audit of early-exit inference on a commodity GPU**

## One-paragraph pitch
Early-exit networks are sold on savings that are almost never measured. Papers price an
early-exit policy with an additive table, E_policy ≈ Σ_h P(exit at h)·cost(h), where cost(h) is
FLOPs or a per-exit energy profile, and then optimise decision rules against that price.

We audit the price itself on a commodity GPU, with telemetry we validate first
(hardware counter vs 142 Hz polling, 0.16 % residual). We then measure early-exit energy per
exit, per policy and end to end, across batch sizes. We cover a CIFAR-10 ResNet and a
pervasive-scale HAR CNN (97 k params, 3.9 MFLOPs, real UCI-HAR smartphone data).

We report:
- when per-exit energy is resolvable at all;
- where FLOP accounting over- or under-states the saving (the sign flips with batch size);
- whether the additive table predicts what a real early-exit runtime pays;
- whether batching erases early-exit savings altogether.

The marginal-utility controller is audited as an instance of rational metareasoning; the
decision rule is not the contribution.

## Research questions
| RQ | Question | Evidence so far | Pending run |
|---|---|---|---|
| RQ1 (C1) | Can per-exit GPU energy be measured, and in which regime? | Counter vs poll 0.16 %. All adjacent exits resolvable (>3σ) at batch ≥16; exits 0↔1 not resolvable at batch 1, where clocks wander 675–1470 MHz. | Clock-locked batch-1/8 rerun, 10 repeats (Step 6) |
| RQ2 (C2) | When do FLOPs mispredict early-exit savings? | Order always preserved (KILL-3 fired), but the size of the saving is off by −16.6 to +7.2 pts, with a sign flip between batch 8 and 16. | HAR (tiny model; launch-bound regime) (Step 8) |
| RQ3 (C2/C3, new) | Does the additive table predict what a real early-exit runtime costs, and does batching erase the savings? | Full model at batch 64 is cheaper per sample than the earliest exit at batch 1. | End-to-end policy energy: compaction vs batch-wait vs full-model batching (Steps 7, 8) |
| RQ4 (C3) | Does a measured cost term change policy decisions? | Marginal-utility is algebraically per-head confidence thresholds (proved in `test_units.py`). It beat global confidence by 10–31 %, but using a noisy cost table. | Per-head baseline + FLOP-cost sensitivity + pooled costs (Step 5) |
| RQ5 (C4-lite) | Are certified operating points meaningful energetically? | LTT certifies τ=0.99 at α=0.02 (68 % exit early, −0.3 pts). α=0.01 is uncertifiable at n=5000 by Hoeffding. | Price certified τ end to end (Step 7 uses τ 0.9 / 0.97 / 0.99) |

## Contributions (what we claim)
1. **C1: a validated per-exit energy protocol**, and the regime where it fails. The NVML counter is
   cross-checked against polling. Resolvability is reported per batch size with 3σ tests and clock
   logs, and batch-1 DVFS non-stationarity is characterised (clock-locked vs unlocked).
2. **C2: a measured regime map of FLOP accounting.** At batch ≥16, energy is affine in FLOPs
   (R² ≥ 0.998) with a fixed overhead, so FLOPs overstate savings. At small batch, energy is convex
   in depth, so FLOPs understate them. This is measured on a mid-size and a pervasive-scale model.
3. **C3: an end-to-end audit of the additive policy-pricing assumption.** We measure the real
   early-exit runtime (per-sample exits with compaction, and batch-wait) against the per-exit-table
   and FLOP predictions, and against simply batching the full model.
4. **C4: an honest policy benchmark.** Marginal-utility (rational metareasoning with a measured
   cost term) vs global and per-head confidence, entropy, and EEFP-style rules. It is priced under
   measured and FLOP costs, so we can say whether measurement changes decisions, and every
   operating point comes with a certified accuracy-risk guarantee.

## Explicit non-claims (reviewer-proofing)
- We do **not** propose a new exit rule. Continue iff E[gain]/cost > λ is rational metareasoning
  (Russell & Wefald). EEFP (arxiv:2508.21495) already makes exits cost-aware, and Green MLOps
  (arxiv:2601.04250) uses marginal energy.
- We do **not** claim FLOPs get the energy *ordering* wrong (KILL-3 fired). We claim they get the
  *size* of savings wrong, in a batch-dependent direction.
- "Accuracy-per-Joule" is not our metric name (PMC12899382). LTT on exits is not ours (SAFE-KD).

## Delta against the closest work
| Work | What it does | What it leaves open, which we measure |
|---|---|---|
| Predictive Exit (arxiv:2206.04685); E4 (AAAI 2025) | Early exit + DVFS for energy on embedded/edge | Whether the per-exit cost model used to price policies matches end-to-end measured energy; batch regime |
| Green MLOps (arxiv:2601.04250) | Marginal-energy utility at request admission | Per-input exits; validated telemetry |
| EEFP (arxiv:2508.21495) | Exit decision uses correctness + abstract compute cost | Physical, measured cost term |
| HW co-opt of EENNs (arxiv:2512.04705) | Shows MACs are insufficient for EENNs, **analytically** | The measured version, with a batch-dependent sign flip |
| Fluid Batching (arxiv:2209.13443), DREX (arxiv:2512.15705), SEEB-GPU (SEC 2025) | Batching vs early-exit conflict; rebatching / scheduling for **latency / throughput** (NPU, LLM, edge GPU) | **Energy** of compaction vs batch-wait vs full-model batching for CNN early exits, measured |
| Part-time power measurements (arxiv:2312.02741) | NVML sampling pathologies on A100/H100 | Ampere consumer GPU validation at per-exit granularity; DVFS confound at batch 1 |
| Multiuser edge inference with batching + early exiting (arxiv:2204.05223) | Joint resource allocation, optimisation-based | Measurement of whether its cost assumptions hold |

## Venue fit (PerCom workshop, e.g. PerConAI)
- The pervasive workload is real UCI-HAR smartphone inertial data with a 97 k-param CNN.
  Pervasive-scale models are exactly where GPU fixed overheads dominate, so FLOP-based savings are
  least trustworthy there.
- A commodity 8 GB RTX GPU stands in for an edge-server / home-hub accelerator. The limitation to
  state honestly: no battery-powered device yet. A Jetson / RPi+INA219 cross-check is future work.

## Threats to validity (state them)
- Single GPU model (RTX 3070), Windows with ~19 background GPU processes. Idle is stable to 0.2 %
  across runs, but this remains a limitation.
- Batch-1 per-exit numbers are DVFS-confounded unless clocks are locked (Step 6 tests this).
- CIFAR-10 / UCI-HAR only; FP32; one exit placement per architecture.
