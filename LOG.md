# LOG

Dated lab notebook. One entry per working session. Keep it terse: what was run,
what came out, what it changed. Durable operational facts belong in
`RUNBOOK.md`; this file is the narrative of how we got there.

Entry template:

```
## YYYY-MM-DD (Day)

**Goal.**
**Ran.**
**Result.**
**Decided.**
**Next.**
```

---

## 2026-08-27 (Thu)

**Goal.** Stand up the repo scaffold so the science can go straight into it.

**Ran.** Scaffolded `src/xconformal/{config,aci,thresholds,binning,coverage}.py`,
`scripts/00`–`04`, `tests/`. `pytest -q`.

**Result.** Scaffold in place; stubs raise `NotImplementedError` with an
inputs/outputs spec. `tests/test_aci_synthetic.py` is written in full and marked
`xfail` — it is the contract `aci.py` gets implemented against.

Checked against the pod (see `00_inspect_data.py` for the live version):
- `$MODEL_ROOT` is **empty** — no checkpoints yet.
- `$DATA_ROOT/era5_240/era5_240_clim.nc` **does not exist** yet.
- ERA5 2020 and 2021 present (4 files each, 0/6/12/18Z); 2020 was still being
  written at the time of checking. No 2022 files.

A reference implementation of the ACI loop on a synthetic under-dispersed
stream (σ=0.5, 20 members, 20 000 steps, τ=5) reached 0.8974 coverage in
**variable space** but stalled at **0.634** in **quantile space**, with the
effective level clipped at zero on 68% of steps — the interval was already
[min, max] and could not widen. Worth confirming on real data; if it survives,
it is a result, not a nuisance.

**Decided.** Nothing scientific. Adaptation space, target variable, threshold
source and day-of-year smoothing all remain open — see RUNBOOK "Decisions".

**Next.** Fetch checkpoints and climatology, run `00_inspect_data.py`, implement
`aci.py` against the test.


## 2026-08-27 (Thu) — GPU bring-up, sizing, batch launch

**Goal.** Get ensemble generation launched and verified.

**Environment.** The pod is an **RTX 4090 (24 GB)**, not an A100 — every sizing
number below is measured on it, not scaled from A100 figures. `tmux` is present
but lives on the ephemeral disk, so it is a per-pod reinstall. `$HF_TOKEN` was
not exported; a placeholder was added to `env.sh` and flagged. Nothing is
blocked by it — the ArchesWeather repo and the stats files are public, and an
empty value keeps `hf_hub` anonymous rather than sending a bad Authorization
header.

**Quantiles file — verdict: unusable.**
`era5-quantiles-2016_2022.nc` (133.8 MB, resolved via
`geoarches.stats.resolve_quantiles_file`) has dims
`(quantile: 6, longitude: 240, latitude: 121, level: 13)`. It fails **both**
arms of the decision rule: there is **no day-of-year dimension**, and the
quantile levels are `[1e-4, 1e-3, 0.01, 0.99, 0.999, 0.9999]` with **no 0.95**.
It pools all seasons together, so its upper quantiles are effectively a summer
threshold no winter day can reach.

**Threshold plan — built today, not deferred to Friday.** `dl_era` was the
wrong tool: it pulls every variable and pressure level, so 2011–2019 at four
synoptic hours would have been ~165 GB on disk (total projection 222 GB, over
the 200 GB escalation ceiling) to produce a 42 MB field. Instead, streamed only
`2m_temperature` at 0Z from the WB2 zarr:

| field | value |
|---|---|
| source | `gs://weatherbench2/.../1959-2022-6h-240x121_...zarr` |
| years / hour | 1990–2019, 0Z only |
| through memory | 10 957 timesteps, 1.27 GB (raw slice never persisted) |
| window | circular ±15 days → **907–930 samples** per gridpoint per day-of-year |
| output | `t2m_p95_clim_1990-2019_0z.nc`, 42.5 MB, 158 s |
| sanity | annual range 23.3 K at 30N/0E vs 4.0 K at the equator — seasonality present |

Conditioning on valid hour is not optional: t2m has a large diurnal cycle, so
pooling synoptic hours samples a *mixture* distribution rather than giving a
bigger sample of the same one.

Also built `t2m_std_2020_0z.nc` (0.13 MB), the per-gridpoint std of the
calibration year, as the denominator for standardized variable-space ACI. It
spans **0.46–23.44 K** — a 50× range, which is exactly why a single unstandardized
`eta` would be hopeless in the tropics.

**GPU gate — passed.**

| item | value |
|---|---|
| t2m | surface index **2** (`T2m`), confirmed in `era5.surface_variables` |
| det backbone | `AvgModule` over the 4 `archesweather-m-*` checkpoints |
| `num_steps` | **25** |
| `cf_guidance` / `s_churn` | **1** / **0.0** |
| `scale_input_noise` | **absent from the config → `None`**, so `sample()` applies no rescaling |
| scheduler / prediction_type | `flow` / `sample` |
| single 24 h step | **2.35 s** (median of 2, warm-up discarded, CUDA-synced) |
| 5-day rollout, 1 member | **14.72 s** |
| output range | 220.4–313.7 K, no NaNs |

**Benchmark — batching abandoned.**

| batch | s/pass | s/member | VRAM GB | members distinct | member 0 == serial |
|------:|-------:|---------:|--------:|---|---|
| 1 | 15.49 | **15.49** | 3.5 | n/a | n/a |
| 2 | 32.27 | 16.14 | 5.5 | yes | no (0.01 K) |
| 4 | 66.45 | 16.61 | 9.8 | yes | no (0.01 K) |
| 8 | 132.37 | 16.55 | 18.6 | yes | no (12.45 K) |

Batching is **not faster** — the GPU is already saturated at batch 1 (100 %
utilisation on 3.5 GB of 24 GB), so replicating the state just serialises the
same work with more memory traffic. It also fails the reproducibility check:
geoarches draws a batch's initial noise from one generator in a single stream,
so a member's noise depends on how many members share its pass. Two independent
reasons to drop it, so `SAMPLING_BATCH_SIZE = 1` and the run is serial.

**Sizing.** At 15.49 s/member over 20 members:

| grid | inits | members | wall-clock h | disk GB |
|---|---:|---:|---:|---:|
| 2020 daily | 725 | 14 500 | 62.4 | 8.4 |
| 2020 every 2d | 543 | 10 860 | 46.7 | 6.3 |
| 2020 every 3d | 482 | 9 640 | 41.5 | 5.6 |
| **2020 every 4d** | **452** | **9 040** | **38.9** | **5.3** |

Sizing rule (≤ 40 h, never thin 2021) selects **2020 every 4 days, 2021 daily**.
Thinning costs warm-up samples, not evaluation samples; 92 calibration inits
still give the controller ~18× tau of settling before the reported window opens.

**Launched.** tmux session `gen`, chronological (2020 first). Measured
**314 s/init**, ETA **39.2 h** → finishes ~Sat 00:30. Files are 6.8 MB
compressed (11.6 MB raw), so the archive lands at **~3.1 GB**, well under the
5.3 GB projection.

Restarted once, three minutes in, to fix `lead`'s `units: days` attribute —
xarray decodes a timedelta-like `units` into `timedelta64` on open, so a later
`ds.sel(lead=5)` would have raised. Cheap to fix now, a Saturday landmine
otherwise. Seeds are date-derived, so the two discarded files regenerate
bit-identically.

**Verified.** First 10 files, 10:06 UTC: all pass. Shapes (20, 5, 121, 240)
float32, no NaNs, ranges 217.4-315.3 K, day-5 member spread 0.90-1.03 K
(non-zero, and under-dispersed as the paper expects), full provenance attrs
present. `lead` decodes as int8 so `ds.sel(lead=5)` works. Left running.

**Decided.** Init window, target variable, adaptation space, threshold method
and init grid — all recorded in RUNBOOK "Decisions" with the rule that
triggered each.

**Next (Fri).** Implement `aci.py` against `tests/test_aci_synthetic.py`; wire
`03_verify.py` to the real archive; decide what a missing init means to a
sequential controller.


## 2026-08-28 (Fri) — ACI implemented, contract green

**Task 0 — data hygiene.** All four re-downloaded 2020 ERA5 files pass: 4.59 GB
each (≥ 4.58 GB), `time == 366`, full `.load()` succeeds, t2m physical
(196.1–325.3 K across the four), no NaNs, timestamps exactly 1 day apart.

Generation at 10:21 UTC: session `gen` alive, **12/452 inits**, **314 s/init**,
ETA 38.4 h → finishes ~**Sat 00:45 UTC**. On pace.

**Task 1 — spacing contract, written before the implementation.** Four tests,
covering properties (a)/(b)/(c) of the spacing contract plus a units guard:
`test_irregular_spacing_still_converges`,
`test_no_update_is_applied_before_its_verification_time`,
`test_spacing_change_leaves_no_trace_beyond_updates_in_flight`,
`test_tau_is_physical_days_not_array_rows`. Split into four rather than one so a
failure names which property broke.

**Task 2 — aci.py.** 11 of 12 contract tests passed on the first run. The one
failure was a **wrong assertion of mine from Thursday**, not a bug: it claimed
every corrected interval is at least as wide as raw, which the update rule
contradicts — the first update on a *covered* outcome sets
`c = eta*(0 − alpha) = −0.002`, so at least one early interval is necessarily
narrower. Measured: exactly 1 step of 20 000, at step 5. Escalated rather than
weakened; replaced with the aggregate claim (mean width 1.4322 → 3.3060).

Reference numbers, σ = 0.5 under-dispersed stream, 20 000 steps, τ = 5 d:
raw coverage **0.5173** → ACI **0.8975**, final c **0.97**, c range
**−0.002 to 1.178**.

**pytest: 28 passed, 0 xfailed, 0 failed.** All xfail markers dropped.


### Fri, later — verification on the partial archive

**A silent grid bug, found by check (1).** First run gave raw coverage 0.12 and
32% of samples in the top `p_t` bin. Diagnosed by round-tripping the model's own
*input state* — which is ERA5 at the init time — through the same denormalize
and extract path: 12.7 K RMSE against ERA5, with min and max matching to the
last bit. Identical values, different arrangement ⇒ a permutation. Brute-forcing
latitude-flip × longitude-roll found an exact match at **RMSE 0.000000**:
`model = roll(flip(era5, lat), 120, lon)`.

| quantity | before | after |
|---|---:|---:|
| day-1 ensemble-mean RMSE | 12.68 K | **0.73 K** |
| day-5 ensemble-mean RMSE | 12.91 K | **1.54 K** |
| mean `p_t` | 0.350 | **0.072** |
| raw 5–95 coverage | 0.114 | **0.705** |

**The archive is correct and needs no regeneration** — this was read-side only.
`01_generate.py` untouched (run is live); fix lives in the new
`xconformal/grid.py`. Every wrong number above was *plausible*, so
`grid.check_alignment` now runs on every archive load and raises above 3 K.

**Check (2): the calibration year was too short, and is now fixed.**
Equilibrium padding `c* = 0.1137` (median 0.44 K); `dcoverage/dc ≈ 1.25`; ACI
time constant `1/(eta·slope) ≈ 80 inits` at `eta = 0.01`. A single 92-init pass
reaches only **68%** of `c*` → ~0.868 coverage, transient alive well into 2021.

Fixed by cycling the calibration year to equilibrium (`aci.warm_start`). Passes
carry `c` but **flush-and-clear the queue at each boundary** — otherwise
December verification times are still pending when the next pass restarts in
January and all come due at once, in the wrong order relative to that pass's own
inits. Passes run over **whole years** deliberately: the final pass ends on
late-December conditions, which is the correct seasonal phase for entering
January of the evaluation year. Stopping mid-pass would hand 2021 a `c` tuned to
July.

Caveat carried forward: `c* = 0.114` was estimated from **winter inits only**
and is seasonal. Expect the full-year `c_t` trajectories to oscillate — that is
the controller tracking seasonal miscalibration, and it is a figure-worthy
observation, not an anomaly.

**Six probe gridpoints** for `c_t` (nearest grid cell): tropics ocean
(0.0, 199.5) central Pacific · tropics land (0.0, 25.5) Congo · midlat ocean
(45.0, 330.0) N Atlantic · midlat land (45.0, 265.5) US Great Plains · high-lat
(75.0, 90.0) Siberian Arctic · desert (25.5, 15.0) central Sahara.

**pytest: 30 passed, 0 failed.**


## 2026-08-29 (Sat) — final measurement: archive audit clean, CHECK 2 fails

**Goal.** Audit the finished archive, warm-start on the full calibration year,
and gate the paper's results on two coverage checks.

**Phase 0 — archive audit. PASSED, every item.**

| item | result |
|---|---|
| files on disk | 452, exactly the locked schedule (92 in 2020 at stride 4 from 01-02, 360 daily in 2021 to 12-26) |
| missing / extra / duplicate dates | 0 / 0 / 0 |
| init spacing | 2020 gaps == {4 d}, 2021 gaps == {1 d} — no cadence surprise |
| open + `.load()` + M == 20 + lead 5 present | 452 / 452 |
| finite, physical | 0 NaN/Inf anywhere; t2m 193.3–321.7 K over the whole archive |
| day-5 member spread | 0.85–1.17 K, non-zero everywhere |
| `grid.check_alignment`, 20 random files across both years | all pass, day-1 ens-mean RMSE 0.604–0.760 K (guard raises above 3 K) |
| reproducibility | 2021-06-15 regenerated on the GPU: **bit-identical**, max abs diff 0.0, sha256 2691ffeeff18e398 both |
| disk | archive 2.9 GB, `/workspace` 52 GB total, container disk 22% used — nowhere near a limit |

New: `scripts/08_audit_archive.py` runs the whole thing and exits non-zero if
any phase fails.

**Phase 1 — warm start on all 92 calibration inits.** Converged in **8 passes**
at the shipped `tol = 0.01`, pass-mean c **+0.12330**; queue empty at the
2020→2021 boundary. Friday's 11-init preview gave +0.0838, so the full
calibration year lands **+0.0395 higher** — Friday's value was winter-biased, as
flagged. The 1% stopping rule stops slightly short of the true fixed point: 60
passes with no early stop gives **+0.12544** (c_0 field mean, area-weighted,
+0.13550). The difference is immaterial to the checks — see below.

**Phase 2 — CHECK 1 PASSED, CHECK 2 FAILED.**

| check | number | verdict |
|---|---|---|
| marginal ACI coverage, area-weighted | **0.8971** | PASS, inside [0.885, 0.915] |
| marginal raw coverage, area-weighted | 0.7409 (unweighted 0.7397) | reproduction number, ~0.045 below Asch's ~0.785/0.794 |
| Jan–Feb vs the other months | Jan 0.8901, Feb 0.8906 vs Mar–Dec 0.8935–0.9057 | **FAIL** |

**Diagnosis of CHECK 2 — not a bug, and not removable by the warm start.**
Four controls, in order:

1. *Warm start not converged?* No. Tightening `tol` to 1e-3, 1e-4 and running
   60 passes with no early stop moves c_0 from +0.12330 to +0.12544 and lifts
   every month by ~0.001. The gap is unchanged and the check still fails.
2. *Is Jan–Feb 2021 intrinsically harder?* No. With c FROZEN at the exact field
   the warm start hands over (controller off), Jan 0.8890 and Feb 0.8856 sit
   comfortably inside the Mar–Dec range 0.8835–0.8979.
3. *Does the calibration year want the same padding as the verification year?*
   No. Per-gridpoint equilibrium padding (the 90th percentile of the
   standardized required padding — static, no dynamics) is **0.13068** on 2020
   against **0.13973** on 2021, area-weighted; the two fields correlate only
   **0.747** across the grid.
4. *Oracle starts.* Starting the evaluation run at the 2021 fixed-point field
   gives Jan 0.8976 / Feb 0.8951 → PASS; cycling 2021 to its own equilibrium
   gives Jan 0.8989 / Feb 0.8972 → PASS. The only thing that changes between
   these and the paper's configuration is where c_0 came from.

So the residual is the controller climbing from the **calibration** year's
equilibrium to the **verification** year's, a gap of ~0.009 in c, at a time
constant of 83 inits ≈ 2.8 months. Cycling 2020 removed the c = 0 → 2020
transient (the Friday problem, 68% of the way); what is left is a smaller
transient that no amount of calibration-year data can remove, because 2020 does
not know what 2021 needs. Cost against the oracle: January **−0.0078**,
February −0.0059, March −0.0042, June −0.0016, September −0.0006.

**Decided.** Nothing yet. The response to a failed gate is a scientific
decision, and every option (disclose as-is, shorten the
reported window, revisit eta) changes what §3 and §4 claim.

**STOPPED before Phase 3 and Phase 4.** The per-bin, land/ocean, width, c_t and
worse-than-raw numbers all computed cleanly in the same run and are held
pending the decision. No figure written, no RESULTS.md, nothing pushed.

**Next.** Decide CHECK 2, then Phase 3 + 4 unchanged.


### Sat, later — decision on CHECK 2, and release

**Decided.** Option 1: **report all 12 months.** The Jan-Feb residual is
disclosed rather than windowed away, and is **reframed as a measurement** — the
controller's adaptation lag under a real year-to-year shift in required padding
(~0.009 in c, tracked at the 83-init time constant, cost decaying Jan −0.0078 to
Sep −0.0006 against the oracle start). `eta` stays 0.01. No window shortened.
The `tol` protocol is recorded exactly: 8 passes at tol 0.01, fixed point
verified at +0.12544 over 60 passes.

Paper allocation: one clause in §3 (warm-start protocol + residual), one short
paragraph in limitations, one sentence in the outlook connecting the measured
lag to the top-bin time constant via `1/w_k`.

**Phase 3 — released.** Headline per-bin coverage, area-weighted, 2021 only:

| p_t bin | n | raw | ACI |
|---|---:|---:|---:|
| [0, 0.1) | 8,885,421 | 0.7377 | 0.8950 |
| [0.1, 0.5) | 1,082,162 | 0.7694 | 0.9105 |
| [0.5, 0.9) | 348,395 | 0.7430 | 0.9062 |
| [0.9, 1] | 138,422 | **0.6971** | **0.8961** |

The land/ocean split sharpens it: the land top bin is the weakest cell in the
table at **0.8809** ACI (raw 0.6612), against 0.9030 on ocean. Bins where ACI
covers less often than raw: **none**, anywhere. Width cost 0.83–0.92 K
(1.36x–1.56x), largest ratio in the top bin (1.564x) — it pays most where it
helps least. `c_t` oscillates seasonally at all six probes with net drift ≤0.055
over the year and no non-finite values; final c over the grid min −0.0110,
median +0.1090, max +0.8690.

**Appendix ablation ran (time permitted).** Quantile space on the full archive:
**`saturated_frac` 95.61%**, marginal coverage stalls at **0.8444**, the warm
start never converges (25-pass ceiling, pass-mean c +1.354) and c runs away to
**+16.28** at the worst gridpoint. This is Thursday's synthetic prediction
(0.634, clipped on 68% of steps) confirmed on real data, and it is the argument
for variable space stated as a result rather than an assumption.

**Phase 4.** Figure form signed off — paired dots, target rule, per-bin counts on
the axes, hand-set ylim (0.63, 0.95) because autoscale collided the top bin's
0.697 label with the count row. `DEFAULT_YLIM` now lives in `04_figure.py` so the
figure is reproducible without remembering a flag. PDF + PNG + the exact CSV
written side by side. `$EVAL_ROOT/RESULTS.md` written as the paper's source of
truth, carrying the commit hash, the 452-file archive count, `c_0`, all Phase 2
and Phase 3 tables, the four-control diagnosis, and the `1/w_k` arithmetic for
the outlook.

**Next.** Frozen. No code changes unless a bug is found; improvements go to
RUNBOOK "post-deadline".
