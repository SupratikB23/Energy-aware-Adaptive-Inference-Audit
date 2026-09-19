# Literature Matrix and Novelty Verdict
## Energy-aware adaptive inference: is "marginal accuracy gain per marginal Joule" open?

**Question asked:** build a literature matrix across SAFE-KD, energy-aware KD, dynamic early exit, hardware-aware adaptive inference, energy-aware inference, and real-device measurement. Then decide whether the proposed controller is genuinely underexplored or already exists under another name.

**Date:** 2026-09-14
**Method:** Firecrawl paper-index searches across six framings, plus targeted abstract inspection of the nine closest papers.

---

# 0. VERDICT FIRST

## 0.1 The controller is not new. It has a 35-year-old name.

The rule

```
continue computing  iff  expected accuracy gain / marginal cost  >  lambda
```

is **rational metareasoning**, also called the **value of computation** (Russell and Wefald, early 1990s). It is not a variant of it. It is the definition of it. The literature on it is continuous from 1991 to 2026, and there are live 2026 instantiations in exactly your problem space.

Do not write a paper whose headline claim is "we propose a marginal-utility controller for adaptive inference." A reviewer who knows the metareasoning literature will reject it in one paragraph.

## 0.2 Five papers that already occupy parts of the claim

| Paper | ID | What it already does | Cost unit |
|---|---|---|---|
| **Green MLOps** | arxiv:2601.04250, Jan 2026 | Admits a request only when the **expected utility-to-energy trade-off** is favourable, defined on high confidence at **low marginal energy** and congestion. Closed-loop decaying threshold. Served on an RTX 4000 Ada. | Marginal energy, closed-loop, on a real GPU |
| **Rethinking Calibration for EENNs (EEFP)** | arxiv:2508.21495, updated May 2026 | Shows calibration alone is insufficient for early-exit networks. Introduces Early-Exit Failure Prediction, which "accounts for both prediction correctness **and the cost of further computation**." | Abstract compute cost |
| **Calibrate-Then-Delegate** | arxiv:2604.14251, 2026 | LTT-style cascade with finite-sample guarantees on **cost or performance**. States explicitly that uncertainty is a poor proxy for the **utility of an expert call**, because it ignores whether the expert would improve the prediction. | Delegation rate (call count) |
| **Rational Metareasoning for LLMs** | arxiv:2410.05563 | Trains an LLM to use reasoning steps only when the expected benefit exceeds their cost, using explicit metareasoning reward functions. | Tokens |
| **ROI-Reasoning** | arxiv:2601.03822, 2026 | Formalizes budgeted inference as a knapsack problem, estimating per-task **return on investment** before allocating compute. | Tokens under a global budget |

Also relevant: the metric name is taken too. **Accuracy-per-Joule (APJ)** already appears as a defined, named evaluation metric in an energy-aware IoT intrusion-detection benchmark (pmcid:PMC12899382).

## 0.3 What is actually still open

Three narrow things survive. All three are about **measurement**, not about the decision rule.

| # | Open question | Why it is open | Confidence |
|---|---|---|---|
| **O1** | Does the marginal-utility policy, with its cost term filled in by **measured per-exit Joules on real hardware**, beat a confidence policy? | Every paper above fills the cost term with tokens, call counts, MACs, or an analytical accelerator model. Green MLOps comes closest but works at request-admission level, not per-input exit level, and does not validate its energy telemetry. | High |
| **O2** | Are GPU energy claims about adaptive inference even **measurable** with standard tooling? | nvidia-smi/NVML has documented sampling pathologies (Section 4). Nobody has audited whether per-exit energy differences are resolvable above that noise floor. | High |
| **O3** | In which **regime** does FLOP-optimal diverge from energy-optimal for early-exit networks? | Established in direction, uncharacterized in magnitude. The one paper that studies it for EENNs (arxiv:2512.04705) uses analytical design-space exploration, not measurement. | Medium-high |

**O2 is the strongest and the one nobody can scoop quickly**, because it requires doing the measurement work honestly rather than proposing another policy.

## 0.4 Recommended reframe

Stop proposing the controller. Propose the audit, and include the controller as one of the policies audited.

> **"When is adaptive inference actually worth its energy? A measured-energy audit of early-exit policies."**

Contributions in order:
1. A validated measurement protocol for per-exit marginal energy on modern NVIDIA hardware, with the NVML sampling artifacts characterized and corrected.
2. A regime map: batch size, model size, and exit overhead against the gap between FLOP-predicted and measured energy savings.
3. The marginal-utility-per-measured-Joule policy, presented as **an instantiation of known metareasoning with a measured cost term**, benchmarked against confidence-threshold exit, and reported honestly including where it fails to help.

This survives a negative result. The current framing does not.

---

# 1. THE LITERATURE MATRIX

Columns that matter: what decides the exit, what unit the cost is counted in, and whether energy was **measured** or **modelled**. The third column is where the field is thin.

## 1.1 Cluster A - Distillation into early-exit students (SAFE-KD line)

| Paper | ID | Exit decided by | Cost unit | Energy measured? | Leaves open |
|---|---|---|---|---|---|
| SAFE-KD | arxiv:2602.03043 | Conformal risk-calibrated threshold (LTT) | Risk level, not cost | No | Target is statistical misclassification risk. No energy term at all. |
| ERDE | arxiv:2510.04856 | Entropy-regularized confidence | None | No | No cost objective |
| LEAP | arxiv:2605.01058 | Convergence-based | None | No | **Important warning:** shows layer-aligned distillation *suppresses* the representational convergence early exit relies on. Read before designing any KD + exit objective. |
| CAPEEN | arxiv:2410.04433 | Confidence, KD-stabilized | Latency | No | Task-specific |
| Exit-ensemble distillation | arxiv:2104.00299 | Fixed | None | No | Accuracy only |
| NEO-KD | arxiv:2311.00428 | Confidence | None | No | Robustness objective |

**Cluster verdict:** KD-into-exits is closed as a contribution. It is a component, not a claim. Note that SAFE-KD already uses LTT, so "apply LTT to early exits" is also gone.

## 1.2 Cluster B - Dynamic early exit, decision rules

| Paper | ID | Exit decided by | Cost unit | Energy measured? | Leaves open |
|---|---|---|---|---|---|
| BranchyNet | arxiv:1709.01686 | Entropy threshold | FLOPs | No | The baseline everyone compares to |
| DART | arxiv:2603.12269 | Learned input-difficulty estimator + DP-optimized thresholds | Accelerator energy model | **Modelled** | Difficulty read from input only. Energy never measured. |
| **EEFP** | arxiv:2508.21495 | Failure prediction accounting for correctness **and cost of continuing** | Abstract compute | No | **Closest decision-rule prior art.** Cost term is not physical. |
| Optimal Depth of Neural Networks | arxiv:2506.16862 | Optimal stopping formulation of depth | Theoretical | No | Pure theory, no hardware |
| QuEE | arxiv:2406.14404 | Predicted error probability, combines quantization + exit | Compute | No | Same missing physical term |
| The Right Tool for the Job | arxiv:2004.07453 | Calibrated confidence | Budget (abstract) | No | NLP, classic |
| Performance Control in Early Exiting | arxiv:2412.19325 | Confidence with performance control | Compute cost | No | Control target is accuracy |
| Temporal Decisions | arxiv:2403.07958 | Temporal correlation between frames | Latency | No | Different signal source, embedded setting |
| EENets | arxiv:2409.09195 | Learned confidence branch | FLOPs | No | Standard |

**Cluster verdict:** the decision-rule space is saturated. EEFP already frames exit as correctness-versus-cost-of-continuing. The only unclaimed axis is what fills in "cost."

## 1.3 Cluster C - Metareasoning and value of computation (the name you were missing)

| Paper | ID | Relevance |
|---|---|---|
| Learning to select computations (BMPS) | arxiv:1711.06892 | Domain-general learning algorithm approximating optimal computation selection under rational metareasoning |
| Rational Metareasoning for LLMs | arxiv:2410.05563 | Trains selective use of reasoning steps from a metareasoning reward. Direct modern instantiation. |
| ROI-Reasoning | arxiv:2601.03822 | Budgeted inference as a stochastic knapsack with per-task ROI estimation |
| Metareasoning for Planning Under Uncertainty | arxiv:1505.00399 | Classic formulation of trading planning cost against policy improvement |
| Ideal Partition of Resources for Metareasoning | arxiv:2110.09624 | The overhead-of-metareasoning problem. **Relevant to your allocator overhead.** |
| BEACON: Bayesian Optimal Stopping for LLM Sampling | arxiv:2510.15945 | Stops sampling when marginal accuracy gain no longer justifies compute |
| Optimal Stopping of Self-Refining Foundation Models | arxiv:2608.10729 | Stopping by expected improvement relative to cost |
| Adaptive Test-Time Compute via Constrained Policy Optimization | arxiv:2604.14853 | Maximize accuracy subject to an average compute budget, solve-then-learn |
| Conformal Thinking | arxiv:2602.03814 | Budget setting for reasoning reframed as risk control |
| Certaindex | arxiv:2412.20993 | Answer-stabilization metric signalling when more compute will not change the result |

**Cluster verdict:** this is the cluster you had not searched, and it is the one that decides the question. Your controller is a known object here. Cite this cluster explicitly and position against it, or a reviewer will do it for you.

## 1.4 Cluster D - Hardware-aware and energy-aware adaptive inference

| Paper | ID | Exit decided by | Cost unit | Energy measured? | Leaves open |
|---|---|---|---|---|---|
| **Predictive Exit** | arxiv:2206.04685 | Low-cost prediction engine forecasts the exit in advance, **then sets voltage and frequency** accordingly | Energy | Partly, embedded platform | **Closest systems prior art.** Reports 72.9% energy saving vs static, 37.6% vs SOTA exit strategies. Decision is still prediction/confidence-based, not marginal-utility. |
| **Green MLOps** | arxiv:2601.04250 | Expected utility-to-energy trade-off, decaying closed-loop threshold | **Marginal energy** | On a real GPU, lightly validated | **Closest controller prior art.** Request-admission granularity, not per-input exits. Bio-inspired framing, weak measurement rigour. |
| Sustainable Edge Intelligence via Energy-Aware Early Exiting | arxiv:2305.14094 | Harvested-energy device state | Analytic energy model | **Modelled** | No per-input difficulty, no teacher |
| Energy-Aware Dynamic Neural Inference | arxiv:2411.02471 | Stochastic harvester state | Modelled | **Modelled** | Same |
| HAPI | arxiv:2008.03997 | Design-time exit configuration per platform | Latency | Measured latency | No runtime policy |
| HW-Algorithm Co-Optimization of EENNs | arxiv:2512.04705 | NAS over exit placement, quantization, mapping | Energy-latency product | **Analytically modelled** | States plainly that MACs are insufficient for EENN deployment. Occupies the FLOPs-are-wrong claim **analytically**, leaving the measured version open. |
| ALERT | arxiv:1911.00119 | Runtime controller switching model variants | Latency, accuracy, energy | Measured | System-level, global, not per-input |
| Hierarchical adaptive control at the edge | arxiv:2604.26470 | Runtime hyperparameter control | Latency, energy, memory | Partly | No learned per-input allocation |
| Runtime-Throttleable NNs | arxiv:1905.13179, arxiv:2011.02836 | Single global utilization knob | Utilization | No | Input-independent |
| NeuralPower | arxiv:1710.05420 | N/A, predictor | Predicted energy | Measured to fit the model | Layer-wise energy prediction. Useful as a baseline predictor for your marginal-cost term. |

**Cluster verdict:** Predictive Exit and Green MLOps between them cover most of what you were going to propose. Predictive Exit has the energy and the DVFS; Green MLOps has the marginal-energy utility rule. Neither has both, at per-input granularity, with validated measurement. That intersection is the remaining gap, and it is narrow.

## 1.5 Cluster E - Real-device energy measurement, and why this cluster matters most

| Paper | ID | Finding you must not ignore |
|---|---|---|
| **Part-time Power Measurements: nvidia-smi's Lack of Attention** | arxiv:2312.02741 | Benchmarked over 70 GPUs. On A100 and H100, **only about 25% of runtime is actually sampled** for power; during the other 75% the GPU may draw drastically different power invisibly. Following their corrected practices reduced energy measurement error by 35% on average, up to 65%. **This is the single most important paper for your project.** |
| FinGraV | arxiv:2412.12426 | Fine-grain GPU power visibility for sub-millisecond to millisecond executions. Your per-exit differences live exactly in this range. |
| Verified Instruction-Level Energy Measurement for NVIDIA GPUs | arxiv:2002.07795 | Per-instruction energy across four GPU generations |
| Wattchmen | arxiv:2603.26435 | Per-instruction energy model for attribution and prediction |
| Accurate Calibration of Jetson Internal Power Sensors | arxiv:2306.13107 | Jetson built-in sensors need regression calibration against external hardware to be trusted |
| Where Do the Joules Go? | arxiv:2601.22076 | Large-scale inference energy measurement on H100 and B200, 1,858 configurations |
| The Illusion of Power Capping in LLM Decode | arxiv:2605.11999 | Power capping appears to work but does not, in memory-bound phases. Cautionary for any DVFS-based claim. |
| Characterization of Request and Token Energy Costs | arxiv:2608.28044 | Decomposes fixed setup energy from **marginal step energy** on H100/H200. Direct methodological template for measuring your marginal per-exit energy. |
| CodeGreen | arxiv:2603.17924 | Modular measurement platform decoupling instrumentation from sampling, supports NVML |
| Double-Exponential Increases in Inference Energy | arxiv:2412.09731 | 1,200 ImageNet models measured. Steep diminishing accuracy returns per unit energy. **Good motivating citation for your introduction.** |
| Watt For What | arxiv:2310.06522 | Proposes an accuracy metric penalized by electricity consumption |

**Cluster verdict:** this cluster is where your contribution can be real, and it is also where your current plan is weakest. You intended to measure energy on a Blackwell via NVML. The tooling you planned to trust has a documented 25%-duty-cycle sampling problem on the immediately preceding architecture generation.

## 1.6 Cluster F - The batching problem (a threat to the whole premise)

| Paper | ID | Finding |
|---|---|---|
| Fluid Batching | arxiv:2209.13443 | Exit-aware preemptive serving for early-exit networks on edge NPUs; batching and dynamic exits fight each other |
| Apparate | arxiv:2312.05385 | Early exits in serving, latency-throughput tension |
| DREX | arxiv:2512.15705 | Dynamic rebatching at each exit point, because batches cannot exit uniformly |
| HELIOS | arxiv:2504.10724 | EE-LLM throughput limited despite theoretical savings |
| Accelerating LLM Inference via Early-Exiting | arxiv:2509.05915 | States the conflict directly: per-token dynamism intended to save compute creates system bottlenecks that can **paradoxically reduce throughput** in batched inference |
| Continuous Depth Batching | arxiv:2608.09444 | Depth adaptivity breaks standard batching |

**Cluster verdict:** at batch size greater than 1, adaptive exits can lose. This is well established on the LLM side and comparatively unexamined for CNNs. **This is a threat to your premise and simultaneously your best experiment**, because measuring where the crossover happens on a modern GPU is exactly O3.

---

# 2. THE DIRECT ANSWER TO YOUR QUESTION

> Is "marginal accuracy gain per marginal Joule" genuinely underexplored, or has someone proposed essentially the same controller under another name?

**Someone has proposed essentially the same controller under another name. Several someones. Repeatedly, since 1991.**

Broken into parts:

| Component of your idea | Status | Occupied by |
|---|---|---|
| Continue iff expected gain exceeds cost | **Closed.** 35 years old. | Rational metareasoning / value of computation |
| Applied to early exits specifically | **Closed.** | EEFP (arxiv:2508.21495) |
| Applied to cascades with a budget guarantee | **Closed.** | Calibrate-Then-Delegate (arxiv:2604.14251) |
| With the cost term being energy | **Nearly closed.** | Green MLOps (arxiv:2601.04250) uses marginal energy in the utility rule |
| With DVFS coupled to the exit decision | **Closed.** | Predictive Exit (arxiv:2206.04685) |
| The Accuracy-per-Joule metric | **Closed.** Name already used. | pmcid:PMC12899382 |
| With the cost term being **validated measured** per-exit marginal Joules, at per-input granularity, on a modern GPU, with the measurement itself audited | **OPEN** | Nobody |
| Characterizing **when** measured energy ordering diverges from FLOP ordering across batch size and model scale | **OPEN empirically**, closed analytically | arxiv:2512.04705 did it analytically only |
| Whether an energy-risk guarantee survives device-state shift (exchangeability violation) | **OPEN** | Nobody |

So: the idea is not dead, but the novelty has moved. It is no longer in the controller. It is in the **measurement, the validation, and the regime characterization**. That is a less glamorous paper and a more defensible one.

---

# 3. WHAT TO BUILD INSTEAD

## 3.1 Revised contribution list

**C1 (measurement, primary).** A validated protocol for resolving per-exit marginal energy on a modern NVIDIA GPU. Characterize the NVML sampling behaviour on your Blackwell part the way arxiv:2312.02741 did for earlier generations, establish the noise floor, and state explicitly whether per-exit energy differences are resolvable above it. If they are not, that is a publishable finding and it saves the field wasted effort.

**C2 (characterization).** The regime map. Sweep batch size (1, 8, 16, 32, 64), model scale, and exit overhead. Plot where FLOP-predicted savings and measured savings diverge, and where adaptive exiting stops paying at all. This directly addresses the batching threat in Cluster F rather than ignoring it.

**C3 (method, secondary).** The marginal-utility policy with a measured cost term, positioned honestly as metareasoning instantiated with physical cost. Compare against confidence-threshold exit, EEFP-style cost-aware exit, and DART-style difficulty thresholds. Report where it wins and where it does not.

**C4 (deferred, only if C1-C3 land early).** LTT calibration against an energy risk, plus the exchangeability-violation experiment: calibrate on a cool, idle GPU, deploy on a hot, loaded one, and show how the guarantee degrades. Do not start here.

## 3.2 Why this ordering

C1 must come first because C3 is unmeasurable without it, and because C1 is the one contribution nobody can take from you by publishing next month. If C1 shows the differences are not resolvable on a shared cloud GPU, you learn that in week 2 rather than week 7.

---

# 4. THE MEASUREMENT PROBLEM, IN DETAIL

This section exists because your plan depends on NVML on a MoLab Blackwell and that dependency is riskier than it looks.

## 4.1 Known NVML pathologies

- On A100 and H100, the built-in sensor samples roughly **25% of the runtime**. The unsampled 75% can draw very different power (arxiv:2312.02741).
- The reading returned is not an instantaneous value but a hardware-side average over an undocumented window, with undocumented update intervals that vary by architecture.
- Blackwell behaviour is **not characterized in the published literature**. You would be establishing it. That is the opportunity and the risk in the same fact.

## 4.2 Protocol

1. **Prefer the energy counter over power polling.** Use `nvmlDeviceGetTotalEnergyConsumption`, which returns cumulative millijoules since driver load, if it is exposed on your part. It integrates in hardware and sidesteps the polling problem. Verify it is present before building anything on top of it.
2. **If only power polling is available**, poll at 100 to 200 Hz in a background thread and integrate with the trapezoidal rule. Then validate that integration against the energy counter or an external meter on a known workload.
3. **Synchronize CUDA** before and after the measured region. Without `torch.cuda.synchronize()`, you are timing kernel launches, not execution.
4. **Loop, do not single-shot.** A single inference at 5 to 20 ms is below the reliable resolution. Run at least 1000 inferences, integrate across the window, divide.
5. **Subtract idle.** Measure the idle baseline for 60 s at the same clock and memory-allocation state. Report total and marginal energy separately. Note that idle draw depends on VRAM allocation (arxiv:2605.23918).
6. **Validate with a dummy kernel.** Run a fixed synthetic load whose energy you can predict, and check that your pipeline recovers it. Report the residual error. This single check is what separates a credible measurement section from a rejected one.
7. **Record device state** with every measurement: temperature, clocks, utilization, power limit, and any co-tenancy. On a shared MoLab instance, co-tenancy is not controllable and must be reported as a limitation.

## 4.3 The MoLab-specific risk

A shared cloud notebook cannot reproduce an exact hardware state between runs. Your calibration and deployment conditions differ in ways you do not observe. State this as a limitation, and turn it into the C4 experiment rather than pretending it away.

---

# 5. VENUE FIT WARNING

PerConAI is a **pervasive and resource-constrained AI** workshop at PerCom. A results section measured entirely on a workstation-class Blackwell GPU in a cloud notebook is a poor fit for that audience, whatever its technical quality. Reviewers will ask why this is a PerCom paper and not an MLSys or systems-conference paper.

Two ways to fix it, in order of preference:

1. **Add one genuinely constrained device.** A Jetson (with the sensor calibration caveat from arxiv:2306.13107), a Raspberry Pi with an INA219, or an Android phone. Then Blackwell becomes the development platform and the constrained device carries the pervasive claim. Cross-device policy transfer becomes a contribution rather than a gap.
2. **Reframe the workload as pervasive.** Use wearable HAR or keyword spotting rather than CIFAR, so the task is pervasive even if the measurement platform is not. Weaker, but it helps.

Doing neither leaves you with a good paper at the wrong venue.

---

# 6. KILL CRITERIA

Check these in order. Stop at the first failure.

| Week | Check | If it fails |
|---|---|---|
| 1 | Is `nvmlDeviceGetTotalEnergyConsumption` available on the MoLab Blackwell, and does the dummy-kernel validation close to within a stated error? | Switch measurement platform or switch topic. Do not proceed on unvalidated telemetry. |
| 2 | At batch size 1, is the measured energy difference between adjacent exits larger than the measurement noise floor? | The per-exit policy question is unmeasurable on this hardware. Pivot to C1 plus C2 as a measurement-methodology paper, which is still viable. |
| 3 | Does measured energy ordering across exits ever disagree with FLOP ordering, in any tested configuration? | The FLOPs-are-wrong claim does not hold here. Drop it from the abstract and lean on C2's batching results. |
| 5 | Does the marginal-utility policy beat confidence thresholding at matched accuracy, on measured energy? | Report the negative result as C3. The paper becomes an audit paper. This is acceptable and still publishable. |

Every one of these failures has a defined landing place. That is the point of ordering the contributions this way.

---

# 7. REFERENCES BY CLUSTER

**A. KD into early exits**
arxiv:2602.03043 (SAFE-KD), arxiv:2510.04856, arxiv:2605.01058 (LEAP), arxiv:2410.04433, arxiv:2104.00299, arxiv:2311.00428

**B. Early-exit decision rules**
arxiv:1709.01686 (BranchyNet), arxiv:2603.12269 (DART), arxiv:2508.21495 (EEFP), arxiv:2506.16862, arxiv:2406.14404, arxiv:2004.07453, arxiv:2412.19325, arxiv:2403.07958, arxiv:2409.09195

**C. Metareasoning and value of computation**
arxiv:1711.06892, arxiv:2410.05563, arxiv:2601.03822, arxiv:1505.00399, arxiv:2110.09624, arxiv:2510.15945, arxiv:2608.10729, arxiv:2604.14853, arxiv:2602.03814, arxiv:2412.20993

**D. Hardware-aware and energy-aware adaptive inference**
arxiv:2206.04685 (Predictive Exit), arxiv:2601.04250 (Green MLOps), arxiv:2305.14094, arxiv:2411.02471, arxiv:2008.03997, arxiv:2512.04705, arxiv:1911.00119, arxiv:2604.26470, arxiv:1905.13179, arxiv:2011.02836, arxiv:1710.05420

**E. Real-device energy measurement**
arxiv:2312.02741 (**read first**), arxiv:2412.12426, arxiv:2002.07795, arxiv:2603.26435, arxiv:2306.13107, arxiv:2601.22076, arxiv:2605.11999, arxiv:2608.28044, arxiv:2603.17924, arxiv:2412.09731, arxiv:2310.06522, arxiv:2605.23918

**F. Batching and serving**
arxiv:2209.13443 (Fluid Batching), arxiv:2312.05385 (Apparate), arxiv:2512.15705 (DREX), arxiv:2504.10724, arxiv:2509.05915, arxiv:2608.09444

**G. Risk control**
arxiv:2604.14251 (Calibrate-Then-Delegate), arxiv:2405.20915, arxiv:2104.08803, arxiv:2602.03814

---

# 8. THE FOUR PAPERS TO READ THIS WEEK

In this order, before writing any code:

1. **arxiv:2312.02741** - nvidia-smi power measurement pathologies. Decides whether your measurement plan is viable at all.
2. **arxiv:2601.04250** - Green MLOps. The closest thing to your proposed controller. Establish exactly what it does and does not do.
3. **arxiv:2508.21495** - EEFP. The closest decision-rule prior art for early exits specifically.
4. **arxiv:2206.04685** - Predictive Exit. The closest energy-aware exit system, including DVFS.

After reading those four, you will be able to write the delta paragraph in one sitting, or you will know the idea needs to change. Either outcome is worth four papers.
