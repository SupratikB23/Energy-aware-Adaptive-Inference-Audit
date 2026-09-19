# RUN_PLAN — GPU BOX runbook (RTX 8GB, copy this repo + run top to bottom)

> Do the steps in order. Each step gates the next (kill criteria at the end).
> On THIS CPU-only machine only Step 0b applies. Everything else runs on the GPU box.
>
> **STATUS (RTX 3070):** Steps 0–4 DONE, see [RESULTS.md](RESULTS.md).
> **Next: PHASE 2 (Steps 5–9, ~60 min total), further down this file.**
> Research framing: [RESEARCH_STATEMENT.md](RESEARCH_STATEMENT.md).

## Step 0a. Setup (GPU box, once)
```bash
cd "D:\Users\SUPRATIK\IEEE PERCOM"   # or wherever you cloned it on the box
# 1) CUDA torch FIRST (CPU wheels have no GPU support):
#    RTX 20/30/40:  pip install "torch>=2.7,<2.9" "torchvision>=0.22,<0.24" --index-url https://download.pytorch.org/whl/cu126
#    RTX 50 (sm_120, Blackwell): MUST use cu128:  ... --index-url https://download.pytorch.org/whl/cu128
# 2) the rest:
pip install -r requirements.txt
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi   # note GPU name, driver, power limit
```

## Step 0b. Smoke (both machines)
```bash
python src/test_smoke.py
# expect: ALL PASS (GPU checks SKIP on CPU box). If any FAIL here, fix before GPU work.
```

## Step 1. Pilot — can this box do the science? (10 min)
```bash
python src/check_gpu.py --idle-seconds 60 --stress-seconds 30
cat results/pilot.json
```
- Expect: `cuda_kernel_check.ok: true`, `has_energy_counter: true` (all RTX 20/30/40/50 have it),
  `validation.passed: true`, `validation.basis: counter-vs-poll`, `residual <= 0.05`.
- `basis: poll-only-unverifiable` = no hardware counter → no independent reference → FAIL by design.
- Expect stress to move power/temp (e.g. idle ~20W → load 80W+).
- **KILL-1: residual > 5%, no power samples, or poll-only → STOP. Do not benchmark. Report telemetry.**

## Step 2. Train baselines (8GB-safe; AMP on, batch 64)
```bash
python src/train.py --dataset cifar10 --mode teacher --epochs 20 --batch-size 64
python src/train.py --dataset cifar10 --mode exit_kd --teacher-ckpt results/cifar10_teacher_resnet18_e20_b64.pt --epochs 20 --batch-size 64
# optional ablations:
python src/train.py --dataset cifar10 --mode exit_ce --epochs 20 --batch-size 64
python src/train.py --dataset cifar10 --mode student --epochs 20 --batch-size 64
```
- Expect: `results/*.pt + *.json` with per-head accuracy. Early heads < final head.
- OOM (exit 2)? Lower `--batch-size 32`, keep `--grad-accum` default; never silently shrink — record it.

## Step 3. Calibrate (LTT accuracy-risk, C4 energy-risk deferred)
```bash
python src/calibrate.py --ckpt results/cifar10_exit_kd_resnet14_e20_b64.pt --dataset cifar10
# (arch is read from the checkpoint; --arch only to override)
cat results/ltt_cifar10_resnet14.json
```
- Expect: `selected` tau (most aggressive certified) + `deploy_check` on held-out split.
- Risk = P(policy wrong AND full right) (bounded {0,1} loss → Hoeffding valid).
- `selected: null` = nothing certifiable at this alpha — loosen alpha or accept negative result.
- **alpha=0.01 is uncertifiable at n=5000 by construction**: Hoeffding needs
  emp_risk <= alpha - sqrt(ln(1/delta)/(2n)) = 0.01 - 0.0152 < 0. Minimum certifiable alpha ~0.016.
  Report this in the paper; then run the alpha sensitivity sweep (copy after each run, same output file):
```powershell
python src/calibrate.py --ckpt results/cifar10_exit_kd_resnet14_e20_b64.pt --dataset cifar10 --alpha 0.05
Copy-Item results/ltt_cifar10_resnet14.json results/ltt_cifar10_resnet14_a005.json
python src/calibrate.py --ckpt results/cifar10_exit_kd_resnet14_e20_b64.pt --dataset cifar10 --alpha 0.03
Copy-Item results/ltt_cifar10_resnet14.json results/ltt_cifar10_resnet14_a003.json
python src/calibrate.py --ckpt results/cifar10_exit_kd_resnet14_e20_b64.pt --dataset cifar10 --alpha 0.02
Copy-Item results/ltt_cifar10_resnet14.json results/ltt_cifar10_resnet14_a002.json
```
- Expect: alpha 0.05 certifies tau ~0.85-0.9+; alpha 0.02 needs emp_risk <= ~0.5% (very high tau or null).
  Thresholds grid (config `eval.thresholds`) goes up to 0.999 because head 0 is overconfident.

## Step 4. Benchmark C2+C3 (the audit; use >=1000 energy iters)
```bash
python src/evaluate.py --ckpt results/cifar10_exit_kd_resnet14_e20_b64.pt --dataset cifar10 --batches 1 8 16 32 64
cat results/eval_cifar10_resnet14.json
```
- Runtime ≈ 60s idle + 120s warmup + 5 batches × 2 modes × 4 exits × 3 repeats × ≥3s ≈ 10-12 min.
- Per batch row: `energy_prefix` / `energy_cascade` → `mean_j`, `std_j` (J per SAMPLE, idle-subtracted),
  `adjacent_resolvable_3sigma`; `divergence_prefix` / `divergence_cascade`.
  `cascade` = deployed path (earlier heads + host exit decisions) and is what policies are costed with.
- Expect: `synthetic_data: false` (if true, numbers are plumbing only), `flops_per_exit` increasing.
- **KILL-2: `adjacent_resolvable_3sigma` has False at batch 1 → per-exit policy unmeasurable. Pivot to C1+C2.**
- **KILL-3: `significant_disagreements_3sigma` empty for every batch → drop FLOPs-are-wrong from abstract.**
- **KILL-4: marginal-utility never beats confidence at matched acc → report negative C3 (still publishable).**

---

# PHASE 2 — strengthen the workshop paper (run in this order, ~60 min)

All commands are one line each: copy-paste them as-is into PowerShell.

Before you start (3 min):
```powershell
git pull
python src/test_smoke.py
python -m pytest src/test_units.py -q
nvidia-smi
```
- Expect smoke `11 pass, 0 fail` and pytest all passed.
- Close browsers / Discord / game launchers. Keep the PC idle, plugged in, power mode "Best performance".
- Every step writes to its OWN folder under `results/runs/`, so nothing overwrites anything.

## Step 5. Policy re-analysis with a pooled, fair baseline (CIFAR, ~2 min, no energy run)
Re-prices every policy with the run1+run2 **pooled** batch-1 cost table. Adds the **per-head
threshold baseline** (the fair comparison for marginal-utility) and a **FLOP-cost sensitivity**
(does MU only win because of the measured cost?).
```powershell
python src/evaluate.py --ckpt results/cifar10_exit_kd_resnet14_e20_b64.pt --dataset cifar10 --skip-energy --cost-json results/runs/eval_run1/eval_cifar10_resnet14.json results/runs/eval_run2/eval_cifar10_resnet14.json --results results/runs/policy_pooled
```
Look at the `[KILL-4 measured]` / `[KILL-4 flops]` lines at the end:
- `mu_vs_perhead` ≈ 0 wins → MU is just per-head thresholds (expected; report it).
- `mu_vs_confidence` wins under **measured** but not under **flops** → the measured cost term changed
  the decision (the C3 claim). Wins under both → the win comes from per-head flexibility, not measurement.

## Step 6. End-to-end policy energy on CIFAR (THE new experiment, ~15 min)
Runs the real early-exit runtime on 2500 real deploy images at every batch size, and compares
the measured J/sample with the per-exit **table** prediction and the **FLOP** prediction.
Compaction (drop exited samples) vs batch-wait vs just batching the full model.
```powershell
python src/e2e_policy.py --ckpt results/cifar10_exit_kd_resnet14_e20_b64.pt --dataset cifar10 --batches 1 8 16 32 64 --taus 0.9 0.97 0.99 --perhead-from results/runs/policy_pooled/eval_cifar10_resnet14.json --eval-json results/runs/eval_run1/eval_cifar10_resnet14.json results/runs/eval_run2/eval_cifar10_resnet14.json --results results/runs/e2e_cifar
```
- Console line per cell: `E=… mJ save=…% tableErr=…% flopErr=…%`.
- `[e2e] WARNING runtime != offline` must NOT appear (the runtime must reproduce the offline policy).

## Step 7. Pervasive track on REAL UCI-HAR (~25 min)
The first command auto-downloads UCI-HAR (58 MB, SHA-256 pinned) into `data/`.
```powershell
python src/train.py --dataset har --mode teacher --arch resnet18 --epochs 20
python src/train.py --dataset har --mode exit_kd --arch harcnn --teacher-ckpt results/har_teacher_resnet18_e20_b64.pt --epochs 20
python src/train.py --dataset har --mode exit_ce --arch harcnn --epochs 20
python src/calibrate.py --ckpt results/har_exit_kd_harcnn_e20_b64.pt --dataset har --alpha 0.05 --results results/runs/har_ltt_a005
python src/calibrate.py --ckpt results/har_exit_kd_harcnn_e20_b64.pt --dataset har --alpha 0.03 --results results/runs/har_ltt_a003
python src/evaluate.py --ckpt results/har_exit_kd_harcnn_e20_b64.pt --dataset har --batches 1 8 16 32 64 --results results/runs/har_eval
python src/e2e_policy.py --ckpt results/har_exit_kd_harcnn_e20_b64.pt --dataset har --batches 1 8 16 32 64 --taus 0.9 0.97 0.99 --perhead-from results/runs/har_eval/eval_har_harcnn.json --eval-json results/runs/har_eval/eval_har_harcnn.json --results results/runs/e2e_har
```
- Expect: HAR final head ≈ 92–95 % (CPU check here: 93.0 %). `synthetic_data: false` everywhere.
- HAR calibration n = 1473, so the smallest certifiable α ≈ 0.03 (Hoeffding). Don't bother with α ≤ 0.02.
- The HAR model is 97 k params / 3.9 MFLOPs: FLOPs predict ~91 % saving at exit 0. The measured
  number is the pervasive-scale C2 result.

## Step 8. Is batch-1 noise DVFS? Clock-locked vs unlocked (CIFAR, ~20 min)
First an unlocked run with 10 repeats (normal PowerShell):
```powershell
python src/evaluate.py --ckpt results/cifar10_exit_kd_resnet14_e20_b64.pt --dataset cifar10 --batches 1 8 --repeats 10 --results results/runs/eval_b1b8_unlocked_r10
```
Then, in **PowerShell as Administrator**: lock the graphics clock, run the same thing, and UNLOCK:
```powershell
nvidia-smi -lgc 1500,1500
python src/evaluate.py --ckpt results/cifar10_exit_kd_resnet14_e20_b64.pt --dataset cifar10 --batches 1 8 --repeats 10 --results results/runs/eval_b1b8_locked1500
nvidia-smi -rgc
```
- Compare `clock_stable`, `cv%` and `resolvable` between the two runs (printed per batch).
  If exits 0↔1 become resolvable when locked → C1 finding: "batch-1 per-exit energy needs locked clocks".
- If `nvidia-smi -lgc` is refused: skip the locked run, keep the unlocked r10 run, note the limitation.
- **Always run `nvidia-smi -rgc` afterwards**, even if the run fails.

## Step 9. Copy back
Copy the whole `results/runs/` folder plus the new `results/har_*.json` to the analysis machine.
Never commit `data/`, `results/`, `*.pt`.

### Phase 2 decision table
| Check | Where | Meaning |
|---|---|---|
| Runtime reproduces offline policy | Step 6/7: no `WARNING runtime != offline` | If it appears, stop: runtime numbers are invalid |
| Table prediction error | `tableErr` in e2e output | Large error at any batch = additive per-exit tables mis-price policies (C3) |
| Early exit vs full model at the same batch | `save=` at batch ≥16 | ≤ 0 = batching erases early-exit savings (Cluster F, measured) |
| MU vs per-head baseline | `[KILL-4 …] mu_vs_perhead` | Expected ≈ tie: MU = per-head thresholds. Report as equivalence |
| Locked clocks fix batch-1 resolvability | Step 8 | If not: batch-1 per-exit energy is intrinsically unresolvable on this GPU |

## 8GB cheat sheet
- Train batch 64; eval sweep `--batches 1 8 16 32 64`; `--num-workers 0` on Windows.
- Every run logs VRAM (`vram` in JSON) — allocation shifts idle power, keep it constant across comparisons.
- `evaluate.py` does a settled 60s idle baseline, then a 120s GPU warmup, then round-robin repeated windows.
  Don't pass `--warmup-seconds 0` / `--idle-seconds` < 60 for real numbers. Close browsers/miners; record `process_count`.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `pynvml not installed` | `pip install -r requirements.txt` on GPU box (optional on CPU box) |
| `nvmlInit() failed` | driver/CUDA mismatch; reinstall driver-matched torch; retry `nvidia-smi` |
| `has_energy_counter: false` | Poll-only cannot be self-validated → KILL-1. Needs an external meter (should not happen on RTX 20+) |
| `no kernel image is available` / kernel check fails | torch wheel lacks your GPU arch; RTX 50 → reinstall torch from the cu128 index |
| `CIFAR10 load/download failed` | box has no internet: copy `data/cifar-10-batches-py/` over; never use `--synthetic` for real runs |
| `C2cModeInfoV` errors (old MoLab bug) | gone — we never call it; if Zeus reintroduces it, don't shim energy paths |
| OOM exit 2 | `--batch-size 32`, close other procs, keep AMP on |
| `selected: null` in LTT | expected at strict alpha; report honestly, try alpha 0.02/0.05 as sensitivity |
| `UCI-HAR ... sha256 mismatch` | delete `data/uci_har.zip` and rerun; if it persists, the download is corrupted/tampered |
| `UCI-HAR load/download failed` | no internet: get the zip from archive.ics.uci.edu (dataset 240) and save it as `data/uci_har.zip` |
| `nvidia-smi -lgc` refused | needs Administrator PowerShell; if still refused, skip the Step 8 locked run (limitation) |
