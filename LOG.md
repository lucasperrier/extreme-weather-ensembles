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
