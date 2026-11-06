# RUN_PLAN — GPU BOX runbook (RTX 8GB, copy this repo + run top to bottom)

> Research framing: [RESEARCH_STATEMENT.md](RESEARCH_STATEMENT.md).
> Completed results: [RESULTS.md](RESULTS.md).
>
> **STATUS.** Phase 1 (telemetry + C2/C3) and Phase 2 (locked-clock batch-1, CIFAR/HAR
> end-to-end) are DONE. **Phase 3 (this file) is the pre-submission hardening pass** and
> implements the five fixes in the ChatGPT feasibility review:
>
> | MD fix | Requirement | Where in Phase 3 |
> |---|---|---|
> | **A** | locked clocks, >=10 repeats, randomized order, close background GPU work, mean/SD/95% CI, repeat across sessions | Steps P3.1, P3.3, P3.4, P3.6 |
> | **B** | >=3 training seeds for the main architectures | Step P3.2 |
> | **C** | remaining experiments: locked-clock CIFAR e2e at b=1/8/16, more UCI-HAR b=64 repeats, reproducibility tests | Steps P3.3b, P3.4, P3.5, P3.5b, P3.6, P3.6b, P3.7 |
> | **D** | remove any controller-novelty claim | already enforced in AGENTS.md / RESEARCH_STATEMENT.md |
> | **E** | update the research statement with the new evidence | RESEARCH_STATEMENT.md v3 |
>
> Every command writes to `results/p3/` and every output name carries the training seed
> (`_s42`, `_s43`, `_s44`) and a `session_id`. Nothing from Phase 1/2 is overwritten.
>
> **PLATFORM NOTE (read first).** Phase 1/2 ran on an RTX 3070. If Phase 3 runs on a
> *different GPU* (e.g. RTX 5060, Blackwell), Phase 3 is a **new, self-contained platform**:
> it trains its own checkpoints and never reads Phase 1/2 files. Do NOT copy 3070 JSONs into
> `results/p3/` and do not pool 3070 and new-GPU numbers. Phase 2 stays as a labelled,
> preliminary secondary platform. The locked clock is no longer a fixed 1500 MHz: it is
> `$LOCK`, derived per GPU in Step 0d, and must be identical in every locked block.
> Phase 3 is *complete on its own*: it re-covers every Phase 2 claim (CIFAR e2e b=1/8/16/32/64,
> HAR e2e b=1/8/16/32/64, unlocked-vs-locked contrast), so Phase 2 does not have to be rerun.

## Step 0a. Setup (GPU box, once)
```bash
cd "D:\Users\SUPRATIK\IEEE PERCOM"   # or wherever you cloned it on the box
# 1) CUDA torch FIRST (CPU wheels have no GPU support):
#    RTX 20/30/40:  pip install "torch>=2.7,<2.9" "torchvision>=0.22,<0.24" --index-url https://download.pytorch.org/whl/cu126
#    RTX 50 (sm_120, Blackwell): MUST use cu128:  ... --index-url https://download.pytorch.org/whl/cu128
# 2) the rest:
pip install -r requirements.txt
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
nvidia-smi   # note GPU name, driver, power limit
```
**RTX 5060 (this run's target platform, Blackwell sm_120, 8 GB):**
- Install the **cu128** wheels (second option above). A cu126 wheel imports fine but fails at the
  first kernel with `no kernel image is available`.
- The printed arch list MUST contain `sm_120`. If it does not, the wheel is wrong: reinstall.
- Use a current Game Ready / Studio driver (Blackwell needs R570 or newer). Record its version.
- Confirm it is the desktop 8 GB card, not a laptop part: a laptop GPU shares power and thermals
  with the CPU, throttles under a locked clock, and makes the controlled-run claims much weaker.
  If it is a laptop, say so in the paper and expect `$LOCK` to need lowering.
- Confirm `nvidia-smi` reports an 8 GB card: the 8 GB VRAM ceiling in AGENTS.md matches; do not raise batch sizes.

## Step 0b. Smoke + unit tests (both machines)
```bash
python src/test_smoke.py
python -m pytest src/test_units.py -q
# expect: smoke "11 pass, 0 fail, 1 skip" and pytest all passed.
```
If either fails here, fix it before touching the GPU.

## Step 0c. Pilot — can this box still do the science? (~12 min)
```bash
python src/check_gpu.py --idle-seconds 60 --stress-seconds 30 --session p3-pilot
```
- Expect: `cuda_kernel_check.ok: true`, `has_energy_counter: true`,
  `validation.passed: true`, `validation.basis: counter-vs-poll`, `residual <= 0.05`.
- `basis: poll-only-unverifiable` = no hardware counter -> no independent reference -> FAIL by design.
- **KILL-1: residual > 5 %, no power samples, or poll-only -> STOP. Do not benchmark.**
- **5060 risk:** the 3070 passed this pilot with a hardware energy counter. A new Blackwell card
  on Windows is untested here. This pilot is the single most important result of the whole
  plan: if it fails, Phase 3 energy claims are blocked on this card and the options are an
  updated driver/`nvidia-ml-py`, or falling back to the 3070 for the headline platform.
  Never relax the check or fall back to 1 Hz / synthetic Joules.
- Log the 5060 idle power, power limit (nominal TGP about 145 W) and idle clock from `pilot.json`.

## Step 0d. Controlled-run hygiene (do this before EVERY locked block)
```powershell
# 1) Close browsers, Discord, game launchers, OBS, other Python/CUDA jobs.
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
# expect: no rows (or only your own shell). Record the output in your lab notes.
# 2) Plug the laptop in / desktop on "Best performance".
# 3) Choose the lock ONCE per GPU and reuse it everywhere (Administrator PowerShell).
#    Rule: >= 55 % of clocks.max.gr (the pinned-clock gate needs >= 50 %) and a clock the card
#    holds under load without power/thermal throttling. 70 % of max, rounded to 15 MHz, is a good start.
$max  = [int](nvidia-smi --query-gpu=clocks.max.gr --format=csv,noheader,nounits)
$LOCK = [int]([math]::Floor(0.70 * $max / 15) * 15)
"max=$max  LOCK=$LOCK"        # RTX 3070 Phase 2 used 1500; write the new value in your lab notes
nvidia-smi -lgc "$LOCK,$LOCK"
nvidia-smi --query-gpu=clocks.gr,clocks.max.gr --format=csv
# expect clocks.gr ~= $LOCK and NOT ~210 (a stable idle clock is not a lock).
```
Do **training (P3.2) and calibration (P3.3) and the unlocked contrast (P3.3b) with the clock
UNLOCKED**; lock only at the start of P3.4. Run every locked block in that **same
Administrator shell** (so `$LOCK` stays defined). If the pilot or a quick load test shows
the card cannot hold `$LOCK` (clocks sag, power-limit throttling), lower it and use the new
value for *all* locked blocks. At the very end:
```powershell
nvidia-smi -rgc
```
If `-lgc` is refused even as Administrator, stop and report the limitation; do not
pretend an unlocked run is a controlled run. `--require-pinned-clock` exists to make
that mistake impossible.

---

# PHASE 3 — pre-submission hardening (run in this order, ~4–5 h)

Set one session id per block so the audit can group the runs:
```powershell
$env:PERCOM_SESSION = "p3-default"
```

## Step P3.1. Sanity: one fast CPU-only command on the GPU box (1 min)
```powershell
python src/test_smoke.py
python -m pytest src/test_units.py -q
```
Both must pass on the GPU box too before training.

## Step P3.2. Train 3 seeds (MD fix B, ~85 min)

CIFAR-10 (teacher ResNet-18 + exit-aware KD ResNet-14), 3 seeds:
```powershell
foreach ($s in 42,43,44) {
  python src/train.py --dataset cifar10 --mode teacher --epochs 20 --batch-size 64 --seed $s --results results/p3
  python src/train.py --dataset cifar10 --mode exit_kd --arch resnet14 --teacher-ckpt "results/p3/cifar10_teacher_resnet18_e20_b64_s$s.pt" --epochs 20 --batch-size 64 --seed $s --results results/p3
}
```
UCI-HAR (teacher ResNet-18 + exit-aware KD HAR-CNN), 3 seeds (~5 min total):
```powershell
foreach ($s in 42,43,44) {
  python src/train.py --dataset har --mode teacher --arch resnet18 --epochs 20 --batch-size 64 --seed $s --results results/p3
  python src/train.py --dataset har --mode exit_kd --arch harcnn --teacher-ckpt "results/p3/har_teacher_resnet18_e20_b64_s$s.pt" --epochs 20 --batch-size 64 --seed $s --results results/p3
}
```
- Expect `results/p3/*_s42.pt`, `*_s43.pt`, `*_s44.pt`, each with a matching `.json`.
- Expect CIFAR final head ~92 %, HAR final head ~92–95 %.
- OOM (exit 2)? Lower `--batch-size 32` for **all** seeds and record the change. Never shrink silently.
- Check the seed spread before spending GPU time on measurement:
```powershell
python -c "import json,glob; [print(sorted((j['seed'], j['history'][-1]['acc']) for j in (json.load(open(p)) for p in glob.glob('results/p3/cifar10_exit_kd_*_s*.json'))))]"
```
If the final-head seed spread is > 1.5 accuracy points, report it; the energy story may still hold but the accuracy claim must be softened.

## Step P3.3. Calibrate each seed (C4, ~1 min)
```powershell
foreach ($s in 42,43,44) {
  python src/calibrate.py --ckpt "results/p3/cifar10_exit_kd_resnet14_e20_b64_s$s.pt" --dataset cifar10 --alpha 0.02 --session "p3-cal-cifar-s$s" --results results/p3
  python src/calibrate.py --ckpt "results/p3/har_exit_kd_harcnn_e20_b64_s$s.pt" --dataset har --alpha 0.05 --session "p3-cal-har-$s" --results results/p3
}
```
- Expect `ltt_*_s42.json`, `_s43.json`, `_s44.json`.
- `selected: null` at strict alpha is a valid result; record it as-is.

## Step P3.3b. Unlocked contrast, batch 1 and 8 (C1 stress test on THIS GPU, ~20 min)
Run **before** locking (clock must still be unlocked: `nvidia-smi -rgc` if in doubt). It goes to
a *separate* folder because the unlocked files intentionally fail the `audit verify` pinned-clock
gate and must never sit in `results/p3/`:
```powershell
python src/evaluate.py --ckpt results/p3/cifar10_exit_kd_resnet14_e20_b64_s42.pt --dataset cifar10 --batches 1 8 --repeats 10 --session p3-eval-cifar-s42-U --run-tag U --results results/p3_unlocked
python src/evaluate.py --ckpt results/p3/har_exit_kd_harcnn_e20_b64_s42.pt --dataset har --batches 1 8 --repeats 10 --session p3-eval-har-s42-U --run-tag U --results results/p3_unlocked
```
- Compare with the locked `eval_*_s42_A.json` rows at b=1/8 (same checkpoint, same repeats): clock
  range, CV %, and `adjacent_resolvable_3sigma` unlocked vs locked. This is the paper's
  "locking turns unresolvable exits into resolvable ones" figure, now measured on the new GPU.
- `results/p3_unlocked` is a subfolder of the gitignored `results/`, not a new top-level folder.
- If the unlocked clock happens to be flat (rare), report that honestly; do not rerun to find a nicer one.

## Step P3.4. CIFAR per-exit energy at 10 repeats, locked clocks (MD fix A, ~65 min)
**Lock now** (Step 0d, `nvidia-smi -lgc "$LOCK,$LOCK"`), then:
```powershell
# full regime map, seed 42 only (hardware measurement need not repeat for every seed)
python src/evaluate.py --ckpt results/p3/cifar10_exit_kd_resnet14_e20_b64_s42.pt --dataset cifar10 --batches 1 8 16 32 64 --repeats 10 --require-pinned-clock --session p3-eval-cifar-s42-A --run-tag A --results results/p3
# seed robustness: the policy cost table (b=1) and the large-batch regime (b=64)
foreach ($s in 43,44) {
  python src/evaluate.py --ckpt "results/p3/cifar10_exit_kd_resnet14_e20_b64_s$s.pt" --dataset cifar10 --batches 1 64 --repeats 10 --require-pinned-clock --session "p3-eval-cifar-s$s" --run-tag A --results results/p3
}
# cross-session repeat of the headline cells with the SAME seed-42 checkpoint (~14 min)
python src/evaluate.py --ckpt results/p3/cifar10_exit_kd_resnet14_e20_b64_s42.pt --dataset cifar10 --batches 1 8 16 --repeats 10 --require-pinned-clock --session p3-eval-cifar-s42-B --run-tag B --results results/p3
```
- Each row must show `clock ... <LOCK>-<LOCK> MHz stable=True`, `cv%` in the single digits,
  and `repeats: 10`, `randomized_order: true` in the JSON.
- `--require-pinned-clock` exits 2 if the lock is not active. That is intentional: fix the lock, rerun.
- `--run-tag` keeps session A and session B from overwriting each other; `audit compare` then tests
  the same checkpoint across the two sessions (it groups by `train_seed`, so seeds 43/44 are not
  mistaken for session repeats).
- **KILL-2** still applies: if batch-1 adjacent exits are not resolvable at 10 repeats, report it as a C1 limit rather than forcing a per-exit policy claim.

## Step P3.5. CIFAR end-to-end at b = 1, 8, 16, locked clocks (MD fix C, ~50 min)
Two independent sessions, same checkpoint:
```powershell
python src/e2e_policy.py --ckpt results/p3/cifar10_exit_kd_resnet14_e20_b64_s42.pt --dataset cifar10 --batches 1 8 16 --taus 0.9 0.97 0.99 --perhead-from results/p3/eval_cifar10_resnet14_s42_A.json --eval-json results/p3/eval_cifar10_resnet14_s42_A.json --repeats 10 --require-pinned-clock --session p3-e2e-cifar-A --run-tag A --results results/p3
python src/e2e_policy.py --ckpt results/p3/cifar10_exit_kd_resnet14_e20_b64_s42.pt --dataset cifar10 --batches 1 8 16 --taus 0.9 0.97 0.99 --perhead-from results/p3/eval_cifar10_resnet14_s42_A.json --eval-json results/p3/eval_cifar10_resnet14_s42_A.json --repeats 10 --require-pinned-clock --session p3-e2e-cifar-B --run-tag B --results results/p3
```
- Expect `[e2e] WARNING runtime != offline` to be **absent**; if present, stop — the runtime is not the measured policy.
- Every row must have `j_ci95_halfwidth`, `n_windows: 10`, `cv_pct`, `min_window_s >= 3`, `clock_mhz` ~`$LOCK`.
- This is the run that answers "does early exit still save energy at b = 16 under locked clocks?"
  Phase 2 (unlocked) said the saving disappears by b = 16; this run is the controlled test.

### Step P3.5b. CIFAR end-to-end at the large batches b = 32, 64 (~17 min)
The statement claims the saving reverses to a cost at b = 64. Phase 3 must measure that locked,
not inherit it from Phase 2. One session, a different tag so it cannot overwrite session A:
```powershell
python src/e2e_policy.py --ckpt results/p3/cifar10_exit_kd_resnet14_e20_b64_s42.pt --dataset cifar10 --batches 32 64 --taus 0.9 0.97 0.99 --perhead-from results/p3/eval_cifar10_resnet14_s42_A.json --eval-json results/p3/eval_cifar10_resnet14_s42_A.json --repeats 10 --require-pinned-clock --session p3-e2e-cifar-hi --run-tag Ahi --results results/p3
```
- Writes `e2e_cifar10_resnet14_s42_Ahi.json`; single-session, so `audit compare` reports but does
  not count it as agreement. Same gates apply (`audit verify`).

## Step P3.6. UCI-HAR b = 64 repeat sessions (MD fix C, ~25 min)
Two independent locked sessions of the same cell, to test the Phase 2 "HAR b=64 is
unreproducible" gap:
```powershell
python src/evaluate.py --ckpt results/p3/har_exit_kd_harcnn_e20_b64_s42.pt --dataset har --batches 1 8 16 32 64 --repeats 10 --require-pinned-clock --session p3-eval-har-A --run-tag A --results results/p3
python src/e2e_policy.py --ckpt results/p3/har_exit_kd_harcnn_e20_b64_s42.pt --dataset har --batches 64 --taus 0.9 0.97 0.99 --perhead-from results/p3/eval_har_harcnn_s42_A.json --eval-json results/p3/eval_har_harcnn_s42_A.json --repeats 10 --require-pinned-clock --session p3-har-b64-A --run-tag A --results results/p3
python src/e2e_policy.py --ckpt results/p3/har_exit_kd_harcnn_e20_b64_s42.pt --dataset har --batches 64 --taus 0.9 0.97 0.99 --perhead-from results/p3/eval_har_harcnn_s42_A.json --eval-json results/p3/eval_har_harcnn_s42_A.json --repeats 10 --require-pinned-clock --session p3-har-b64-B --run-tag B --results results/p3
```
- The two e2e runs write `..._A.json` and `..._B.json`, so neither overwrites the other.
- The whole point is that the two sessions must land within 10 % of each other (`audit compare`).
  If they do not, that is a C1 finding, and the Phase 2 HAR numbers must be reported with that error bar.
- The HAR eval above covers b = 1/8/16/32/64 on purpose: `e2e_policy.py` gets the additive
  per-exit prediction (C3) from the eval table of the *same batch*, so a missing batch would
  silently drop the table-error column for that batch.

### Step P3.6b. UCI-HAR end-to-end at b = 1, 8, 16, 32 (~30 min)
Backs the claim "HAR never saves energy at any batch size" with locked, 10-repeat evidence
(b = 64 is already covered by sessions A/B above):
```powershell
python src/e2e_policy.py --ckpt results/p3/har_exit_kd_harcnn_e20_b64_s42.pt --dataset har --batches 1 8 16 32 --taus 0.9 0.97 0.99 --perhead-from results/p3/eval_har_harcnn_s42_A.json --eval-json results/p3/eval_har_harcnn_s42_A.json --repeats 10 --require-pinned-clock --session p3-har-lo --run-tag Alo --results results/p3
```
- Writes `e2e_har_harcnn_s42_Alo.json` (single session; reported, not counted as agreement).

## Step P3.7. Audit the runs (MD fix A/C, 1 min)
```powershell
# hard gates: >=10 windows, >=1000 iters, >=3 s, pinned clock, no co-tenant, real data, session id
python src/audit.py verify --results results/p3
# cross-session agreement: the same cell measured in >=2 sessions must agree within 10 %
python src/audit.py compare --results results/p3 --within 10
```
- `audit_verify.json` lists every failure per file. **Do not put a file in the paper if it says FAIL.**
- `audit_compare.json` lists every repeated cell; `agrees: true` is the reproducibility claim.
- A single-session cell is reported but not counted as agreement. That is expected.

## Step P3.8. Copy back and unlock
```powershell
nvidia-smi -rgc
# copy results/p3/ + the two audit JSONs to the analysis machine
```
Never commit `data/`, `results/`, `*.pt`, `*.csv`.

---

## Time budget (RTX 3070, measured Phase 1/2 speeds)

| Block | What | Estimate |
|---|---|---|
| P3.2 | 3 seeds x (CIFAR teacher + exit_kd) | 80–90 min |
| P3.2 | 3 seeds x (HAR teacher + exit_kd) | 5–10 min |
| P3.3 | 6 calibrations | 2 min |
| P3.3b | unlocked contrast, CIFAR + HAR, b=1/8 | 20 min |
| P3.4 | CIFAR eval: full sweep A + 2 x 2-batch + headline repeat B | 60–70 min |
| P3.5 | CIFAR e2e b=1/8/16, 10 repeats, sessions A + B | 45–50 min |
| P3.5b | CIFAR e2e b=32/64, one session | 17 min |
| P3.6 | HAR eval b=1/8/16/32/64 + 2 x e2e b=64 | 30–35 min |
| P3.6b | HAR e2e b=1/8/16/32, one session | 30 min |
| P3.7 | audit | 1 min |
| **Total** | first clean pass (incl. 12 min pilot) | **~5.5–6.5 h** |
| worst case | one retry per block + OOM/slow disk | **~8–9 h** |

### RTX 5060 estimate (what to expect on this card)
The table above uses RTX 3070 timings. No 5060 timing exists yet, so this is a scaled
estimate, not a measurement. Update it after the first seed trains.

| Part | Why it scales (or not) | 5060 estimate |
|---|---|---|
| Training (P3.2, CIFAR 3 seeds) | Compute-bound: 5060 has 3840 CUDA cores vs 3070's 5888, higher clocks, same ~448 GB/s memory and 8 GB, newer tensor cores + AMP. Expect 0.9–1.3x of the 3070's 1608 s/seed | **80–110 min** |
| Training (HAR 3 seeds) | Tiny model, launch-bound | 5–10 min |
| All measurement blocks (P3.3b–P3.6b) | **Wall-clock fixed**: every window is >= 3 s, idle 60 s, warmup 120 s, 10 randomized windows per cell. GPU speed barely changes duration; a lower locked clock only changes iterations per window | same as table, **~3.5 h** of locked/unlocked measurement |
| Pilot + lock + hygiene + copy-back | manual steps | 30–40 min |
| **Total, clean pass** | | **~5.5–6.5 h** (low end if training is fast, high end if slow) |
| **Worst case** | one retry of a locked block (pinned-clock refused, window short) + slower training | **~8–9 h** |
| Desktop vs laptop | laptop 5060 may throttle at `$LOCK`; blocks then need `$LOCK` lowered and rerun | +1–2 h |

Plan for it as **two sessions**: (1) pilot, unlocked training for seed 42, calibration, P3.3b;
(2) one continuous Administrator shell for all locked blocks P3.4–P3.7 (~3.5 h). The P3.4 seed
loop evaluates seeds 43/44, so those checkpoints must exist first: train all 3 seeds (unlocked,
~55–75 min more) before session 2. If time is short, skip that loop in session 2 and run it
later in a third, shorter locked session (re-lock with the same `$LOCK`).

Suggested split if time is short: day 1 = pilot + P3.2 seed 42 + P3.3 + P3.3b; day 2 = lock,
P3.4, P3.5, P3.5b, P3.6, P3.6b, P3.7; then seeds 43/44 (P3.2, P3.4 seed loop) last.

Per-run detail (why the numbers are what they are):
- `evaluate` full sweep: 60 s idle + 120 s warmup + 5 batches x 2 modes x 4 exits x 10 windows x >=3 s = ~20–22 min.
- `evaluate` 2-batch run: ~8–10 min; the 3-batch headline repeat B: ~12–14 min.
- `e2e_policy` 3-batch run: ~13 cells/batch x 3 batches x 10 windows x >=3 s + warmup = ~22–25 min.
- `e2e_policy` HAR b=64: ~13 cells x 10 x >=3 s + warmup = ~8–10 min per session.
- CIFAR training: 778 s teacher + 830 s exit_kd per seed (measured), so 3 seeds ~80 min.

If you also want the **full** Phase 2 matrix re-run at 10 repeats for both workloads
(all e2e batches 1/8/16/32/64, both modes, all seeds), budget **7–9 h**.

## Phase 3 decision table
| Check | Where | Meaning |
|---|---|---|
| Telemetry still valid | Step 0c `check_gpu.py` | If KILL-1 fires, stop: no policy runs on bad telemetry |
| Clock actually locked | Step 0d + every JSON `clock_check.pinned` | `--require-pinned-clock` refuses to run otherwise |
| No co-tenant | `co_tenancy.compute_process_count <= 1` | >1 means another GPU process is shifting idle power |
| Seed robustness | P3.2 JSONs, 3 seeds | Final-head spread > 1.5 pts => soften accuracy claims |
| Per-exit resolvable at b=1 | `adjacent_resolvable_3sigma` | If false at 10 locked repeats, it is a C1 limit, not a bug |
| Additive table error | `e2e` `tableErr` | Large error at any batch = per-exit tables mis-price policies (C3) |
| Locking helps (C1 on this GPU) | P3.3b `results/p3_unlocked` vs locked `eval_*_A` | Clock range + `adjacent_resolvable_3sigma` unlocked vs locked |
| Early exit vs full at same batch | `e2e` `save=` | <=0 at b>=16 = batching erases savings (the central C2 result) |
| Runtime == offline policy | no `WARNING runtime != offline` | If it appears, the e2e numbers are invalid |
| HAR b=64 reproducible | P3.6 two sessions + `audit compare` | >10 % disagreement is a C1 finding to report |
| Every file publishable | `audit verify` all PASS | Any FAIL file is excluded from the paper |

## 8GB cheat sheet
- Train batch 64; eval sweep `--batches 1 8 16 32 64`; `--num-workers 0` on Windows.
- Every run logs VRAM (`vram` in JSON); allocation shifts idle power, keep it constant across comparisons.
- `evaluate.py` does a settled 60 s idle baseline, then a 120 s GPU warmup, then randomized repeated windows.
- Do not pass `--warmup-seconds 0` or `--idle-seconds < 60` for real numbers.
- `--repeats < 10` prints a warning and is exploratory only; `audit verify` will fail it.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `pynvml not installed` | `pip install -r requirements.txt` on the GPU box |
| `nvmlInit() failed` | driver/CUDA mismatch; reinstall driver-matched torch; retry `nvidia-smi` |
| `has_energy_counter: false` | Poll-only cannot be self-validated -> KILL-1. Needs an external meter |
| `--require-pinned-clock` exits 2 | Run `nvidia-smi -lgc "$LOCK,$LOCK"` in an Administrator shell first; confirm `clocks.gr ~= $LOCK`, not ~210. If `$LOCK` < 50 % of `clocks.max.gr` the gate rejects it by design: raise it |
| Clock sags below `$LOCK` under load | Power/thermal throttle: lower `$LOCK` for ALL locked blocks (and re-run any already done) |
| Blackwell (RTX 50) cannot give a hardware energy counter | Pilot fails (KILL-1). Energy claims on that card are blocked; do not fall back to 1 Hz or synthetic Joules |
| Clock reads ~210 after `-lgc` | Lock did not take (driver/permission). Do not disable the gate; fix the lock |
| `--require-pinned-clock` passes but clocks drift later | Re-run; `clock_check` is a start-of-run snapshot, per-window clocks are in every JSON |
| `no kernel image is available` | torch wheel lacks your GPU arch; RTX 50 -> reinstall from the cu128 index |
| CIFAR download fails | copy `data/cifar-10-batches-py/` over; never use `--synthetic` for real runs |
| UCI-HAR `sha256 mismatch` | delete `data/uci_har.zip` and rerun; if it persists, the download is corrupted/tampered |
| OOM exit 2 | `--batch-size 32` for all seeds, close other processes, keep AMP on, record the change |
| `selected: null` in LTT | expected at strict alpha; report honestly, try alpha 0.02/0.05 as sensitivity |
| `audit verify` says `n_windows < 10` | you ran with `--repeats < 10`; rerun at 10 for publishable numbers |
| `audit compare` disagreement > 10 % | treat as a C1 result: report the between-session error bar, do not average it away |
