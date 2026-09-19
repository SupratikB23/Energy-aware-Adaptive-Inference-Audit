# RUN_PLAN — GPU BOX runbook (RTX 8GB, copy this repo + run top to bottom)

> Do the steps in order. Each step gates the next (kill criteria at the end).
> On THIS CPU-only machine only Step 0b applies. Everything else runs on the GPU box.

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

## Step 5. Pervasive track (PerConAI fit; same pipeline, small nets)
```bash
python src/train.py --dataset kws --mode teacher --epochs 20
python src/train.py --dataset kws --mode exit_kd --arch dscnn-s --teacher-ckpt results/kws_teacher_resnet18_e20_b64.pt --epochs 20
python src/evaluate.py --ckpt results/kws_exit_kd_dscnn-s_e20_b64.pt --dataset kws --batches 1 8 16 32
# har mirrors kws with --dataset har
```
- KWS/HAR use synthetic-shaped RANDOM windows until real loaders exist (outputs say `synthetic_data: true`):
  energy/latency numbers are valid (shape-dependent only); accuracy/policy numbers are NOT.
- One constrained device (Jetson/RPi) later = cross-device transfer experiment; until then, state the limitation.

## Step 6. What to copy back
- `results/pilot.json`, `results/eval_*.json`, `results/ltt_*.json` (+ `*.pt` only if needed).
- Never commit `data/`, `results/`, `*.csv`, `*.pt` (see `.gitignore`).

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
