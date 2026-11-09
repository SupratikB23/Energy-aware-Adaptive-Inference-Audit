# Research statement (v3, after the Phase 2 end-to-end audit)

Controlling verdict and full literature matrix:
[percom2027_literature_matrix_and_verdict.md](percom2027_literature_matrix_and_verdict.md).
Measured results: [RESULTS.md](RESULTS.md). Run order: [RUN_PLAN.md](RUN_PLAN.md).

> **Phase 3 status (Nov 9, 2026).** P3.1-P3.3b are reported complete on the RTX 5060 (3 training
> seeds, calibration, unlocked b=1/8 contrast). The locked core P3.4-P3.7 is still pending, so every
> Phase 2 number below is preliminary and the locked/10-randomized-window protocol is a target, not
> yet a published result.

## Working title
**Priced by the Table, Paid at the Meter: a measured-energy audit of early-exit inference on a commodity GPU**

## One-paragraph pitch
Early-exit networks are sold on savings that are almost never measured. Papers price an
early-exit policy with an additive table, E_policy ~= sum_h P(exit at h) * cost(h), where cost(h)
is FLOPs or a per-exit energy profile, and then optimise decision rules against that price.

We audit the price itself on a commodity GPU, with telemetry we validate first
(hardware counter vs 142 Hz polling, 0.16 % residual). Phase 3 measures early-exit energy per
exit, per policy and end to end, across batch sizes, under locked clocks and with 10 randomized
windows per cell. We cover a CIFAR-10 ResNet and a pervasive-scale HAR CNN (97 k params,
3.9 MFLOPs, real UCI-HAR smartphone data).

We report:
- when per-exit energy is resolvable at all, and that this requires a locked clock;
- where FLOP accounting over- or under-states the saving;
- that the additive per-exit table under-predicts what a real early-exit runtime pays
  (3-39 % on CIFAR, 53-87 % on HAR in the Phase 2 runs);
- that batching erases early-exit savings by batch 16 on CIFAR, and that the HAR CNN never
  saves energy at any batch size under the default and locked clocks tested so far.

The marginal-utility controller is audited as an instance of rational metareasoning; the
decision rule is not the contribution.

## Research questions
| RQ | Question | Evidence so far | Pending run |
|---|---|---|---|
| RQ1 (C1) | Can per-exit GPU energy be measured, and in which regime? | Counter vs poll 0.16 %. Unlocked batch-1: clocks 855-1935 MHz, CV 3-41 %, adjacent exits 0/3 resolvable. **Locked 1500 MHz: clocks 1500-1500, CV 3.9-10.7 %, adjacent exits 3/3 resolvable**, identical-workload mismatch 16-54 % -> 7 %. | Phase 3: 10 repeats + randomized order + 2 independent sessions per headline cell, CIFAR and HAR |
| RQ2 (C2) | When do FLOPs mispredict early-exit savings? | Order preserved (KILL-3 fired), but the *size* is off: CIFAR -45 to +84 %, HAR -71 to -94 %. Full model at b=64 is cheaper per sample than the earliest exit at b=1. | Locked-clock CIFAR e2e b=1/8/16/64 (Phase 3, laptop-lite) |
| RQ3 (C2/C3) | Does the additive table predict what a real early-exit runtime costs, and does batching erase the savings? | **No.** Table under-predicts by 3-39 % (CIFAR) and 53-87 % (HAR). CIFAR early exit saves 18-67 % at b=1 and 18-49 % at b=8; at b=16 it mostly reverses (costs up to 29 %, best case saves 3 %), and b=64 recovers only to 3-15 % saved. HAR costs 6-241 % *more* than full at every batch. Compaction beats batch-wait in 45/48 cells. | Phase 3 (laptop-lite): locked CIFAR e2e b=1/8/16/64; HAR e2e b=1 and 64 (b=64 in two sessions) |
| RQ4 (C3) | Does a measured cost term change policy decisions? | Marginal-utility is algebraically per-head confidence thresholds (proved in `test_units.py`). Against per-head it ties (11/12, median 15.1 % under measured cost; 1.8 % under FLOP cost). The measured cost term does not create a new controller; it changes the *price* of the operating points. | 3 seeds; b=1/64 cost table for seeds 43/44 (Phase 3) |
| RQ5 (C4-lite) | Are certified operating points meaningful energetically? | LTT certifies tau=0.99 at alpha=0.02 (68 % exit early, -0.3 pts). alpha=0.01 is uncertifiable at n=5000 by Hoeffding. | Price certified tau end to end (Phase 3 `--taus 0.97 0.99`) |

## Contributions (what we claim)
1. **C1: a validated per-exit energy protocol**, and the regime where it fails. The NVML counter is
   cross-checked against polling. Resolvability is reported per batch size with 3-sigma tests,
   randomized window order, >=10 repeats, Student-t 95 % CIs, pinned-clock evidence, co-tenancy
   snapshots and cross-session agreement. The locked-vs-unlocked batch-1 experiment is the
   protocol's own stress test: locking turns 0/3 resolvable adjacent exits into 3/3.
2. **C2: a measured regime map of FLOP accounting.** At batch >=16, energy is affine in FLOPs
   (R^2 >= 0.998) with a fixed overhead, so FLOPs overstate savings. At small batch, energy is
   convex in depth, so FLOPs understate them. Measured on a mid-size and a pervasive-scale model.
3. **C3: an end-to-end audit of the additive policy-pricing assumption.** We measure the real
   early-exit runtime (per-sample exits with compaction, and batch-wait) against the per-exit-table
   and FLOP predictions, and against simply batching the full model. The table under-predicts the
   real cost in every measured cell so far.
4. **C4: an honest policy benchmark.** Marginal-utility (rational metareasoning with a measured
   cost term) vs global and per-head confidence, entropy, and EEFP-style rules. It is priced under
   measured and FLOP costs, so we can say whether measurement changes decisions, and every
   operating point comes with a certified accuracy-risk guarantee. The honest answer so far is
   that marginal-utility is a re-parameterisation of per-head thresholds, not a new controller.

## Explicit non-claims (reviewer-proofing)
- We do **not** propose a new exit rule. Continue iff E[gain]/cost > lambda is rational metareasoning
  (Russell & Wefald). EEFP (arxiv:2508.21495) already makes exits cost-aware, and Green MLOps
  (arxiv:2601.04250) uses marginal energy.
- We do **not** claim FLOPs get the energy *ordering* wrong (KILL-3 fired). We claim they get the
  *size* of savings wrong, in a batch-dependent direction.
- "Accuracy-per-Joule" is not our metric name (PMC12899382). LTT on exits is not ours (SAFE-KD).
- The HAR "no saving" result is a measured property of this 97 k-param CNN on this GPU, not a
  universal claim about all pervasive models.

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
  least trustworthy there. The measured answer (no saving at any batch) is a useful negative result
  for the community.
- A commodity 8 GB RTX GPU stands in for an edge-server / home-hub accelerator. The limitation to
  state honestly: no battery-powered device yet. A Jetson / RPi+INA219 cross-check is future work.

## Threats to validity (state them)
- Single headline GPU, Windows. Phase 1/2 ran on an RTX 3070 with ~19 background GPU processes;
  Phase 3 may run on a different card (e.g. RTX 5060). Phase 3 is then a self-contained platform
  (own checkpoints, own locked clock `$LOCK`, never pooled with 3070 numbers); the 3070 data is
  reported only as a labelled, preliminary secondary platform. Phase 3 closes background
  processes and records a co-tenancy snapshot per run. No battery-powered device.
- Overhead is quantified as measured end-to-end energy minus the additive per-exit prediction
  (`pred_table_err_pct`, a per-cell point value; the measured side carries the 95 % CI), and compaction vs batch-wait. We do **not** decompose it into
  kernel-launch / synchronisation / host-decision time, so any such mechanism is a hypothesis,
  not a measured result.
- Batch-1 per-exit numbers are DVFS-confounded unless clocks are locked. Phase 2 showed that
  locking at 1500 MHz fixes resolvability; Phase 3 re-runs every headline cell locked, with
  `--require-pinned-clock` as a hard gate.
- Phase 2 used 3 repeats and a rotated (not randomized) order. Phase 3 uses >=10 repeats,
  seeded randomized order, Student-t CIs and cross-session comparison. Phase 2 numbers are
  therefore preliminary and must be reported as such if the Phase 3 runs are not finished.
- One training seed in Phase 1/2. Phase 3 trains 3 seeds for CIFAR and HAR and reports the seed
  spread; hardware measurements are repeated for seed 42 and spot-checked at b=1 for seeds 43/44. Because the measurement GPU is a power-limited laptop part, the
  locked-clock set is reduced to CIFAR b=1/8/16/64 and HAR b=1/64 (optional extensions cover b=32 and the HAR middle batches);
  claims are worded only for the batches measured ("HAR never saves at b=1 and b=64").
- CIFAR-10 / UCI-HAR only; FP32; one exit placement per architecture.
