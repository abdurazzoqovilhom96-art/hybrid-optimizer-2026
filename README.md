# hybrid-optimizer-2026

Benchmark study of **TEMOA_V10_HYBRID**, a hybrid metaheuristic combining an
L-SHADE/jSO core with GWO-, WOA- and HHO-inspired operators, an eigenbasis
crossover and a (1+1)-ES refinement phase.

This repository contains the original algorithm, a repaired variant
(**TEMOA_V11**), fair comparators, and a reproducible experimental protocol.

> **Where this stands.** On the original twelve-function suite TEMOA_V10 is good
> on average — better mean rank than jSO and L-SHADE — but collapses on two
> functions: Schwefel, where it is ~78 000× behind L-SHADE, and RotatedElliptic,
> ~43× behind jSO. Both causes were diagnosed and repaired in `TEMOA_V11`.
>
> That suite is not, however, a basis for any claim: no difference among the
> DE-family algorithms is statistically significant on 12 functions, and the
> eigenbasis crossover that gives TEMOA its one real strength is already the core
> of EA4eig, the CEC'2022 winner. The CEC'2017 gate exists to find out what
> survives against real competitors. Read `reports/TAHLIL_UZ.md` and
> `reports/PRIOR_ART.md` before using any number from this repository.

## Quick start — one file, one command

```
python START.py
```

That is the whole thing. `START.py` checks the dependencies, installs anything
missing, runs all 41 tests, and **refuses to start the experiment if any test
fails** — a benchmark that is wrong is worse than no benchmark. Then it runs the
gate and the analysis.

On Windows, double-click `run_windows.bat` instead; it only sets the thread
environment and calls `START.py`.

```
python START.py --check     # dependencies and tests only (~4 min)
python START.py --smoke     # tiny end-to-end run (~2 min)
python START.py --jobs 20   # set the worker count (default: 80% of cores)
```

Interrupt with Ctrl+C at any point and run the same command again: finished runs
are skipped and the result is bit-identical to an uninterrupted run (there is a
test for that).

### What it runs by default, and what it deliberately does not

The default is the **decision gate**: CEC'2017 at D=10, 29 functions, 51 runs,
100 000 evaluations each — 1.18e9 evaluations, about **2.5 hours on 20 workers**.
That single run answers the only question worth answering now: where, if
anywhere, is this algorithm competitive against CMA-ES and the adaptive-DE
lineage it was built from.

Higher dimensions are **not** the default, and that is on purpose. Measured from
this machine's throughput, D=30 costs a further ~11 hours and D=50 about
~27 hours on 20 workers. Run them only after reading the gate:

```
python START.py --dims 10 30
```

### The protocol comes from the suite, not from you

`--suite cec2017` means 51 runs at 10000·D evaluations because that is what the
competition specifies. Overrides are allowed, but they are recorded in
`manifest.json` and printed as a warning, because a result measured under a
non-standard protocol cannot be compared with any published table. The original
study's 30 runs at 3000·D matched no competition, which is why its numbers could
never be placed beside a published one.

## Layout

| Path | What it is |
|---|---|
| `hybrid 2026.py` | the original single-file study, kept unmodified for reference |
| `temoa/problems.py` | benchmark suite, error values, rotated and shift-only tracks |
| `temoa/tracker.py` | FES accounting; offline-error and noisy-recommendation scoring |
| `temoa/algorithms/temoa_v10.py` | the original algorithm (RNG injected, otherwise identical) |
| `temoa/algorithms/temoa_v11.py` | the repaired variant, with the evidence for each change |
| `temoa/suites.py` | CEC'2017 / CEC'2022 suites, official numbering, per-suite protocol |
| `temoa/algorithms/modern.py` | L-SHADE, jSO, and the CMA-ES family — the comparators the original study omits |
| `temoa/registry.py` | two tiers: `MODERN` (what counts) and `LEGACY_SWARM` (secondary) |
| `temoa/algorithms/baselines.py` | DE, GWO, WOA, SCA, PSO, HHO (+ the original weakened PSO/HHO) |
| `temoa/stats.py` | Friedman, Iman-Davenport, Holm, Nemenyi, signed-rank, Vargha-Delaney A12 |
| `run_all.py` / `analyze.py` | experiment driver / reporting |
| `tests/test_port_fidelity.py` | proves the ported V10 matches `hybrid 2026.py` |
| `reports/TAHLIL_UZ.md` | **the analysis — start here** |
| `reports/QAMROV_UZ.md` | experimental scope of the legacy study |

## Protocol

- **Suite**: 12 functions, shifted and rotated (CEC style). `--suite shift`
  reproduces the original's shift-only suite, which rotates only 1 of 12 and so
  is nearly separable.
- **Budget**: 3000·D evaluations; D ∈ {30, 50, 100}; 30 independent runs.
- **Metric**: error `f(x) − f*`. Dynamic problems use windowed offline error;
  noisy problems use a re-evaluation recommendation rule (see `tracker.py`).
- **Statistics**: best/worst/mean/std/median/IQR; Friedman **per dimension**
  with the Iman-Davenport correction; Holm post-hoc against a control; Nemenyi
  critical difference; paired Wilcoxon signed-rank with Holm applied both within
  each cell and over the whole family; Vargha-Delaney Â₁₂ effect sizes.
- **Reproducibility**: no global RNG. Algorithm stream is
  `default_rng([42, dim, run, alg_index])`, noise stream is
  `default_rng([99, dim, run])` and is matched across algorithms within a run.
  Every run records its wall-clock time; `results/manifest.json` records
  versions, platform, seeds and settings.

Note on statistical power: the Wilcoxon signed-rank test on *n* runs cannot
produce a p-value below `2/2^n`, so with 3 runs (the smoke setting) nothing can
reach significance. Use 30 runs for any reported claim.

## On the GPU

The RTX 4060 Ti is not used, and adding it would not help. At 30–100 dimensions
each function evaluation is a vector operation on a few hundred floats; the cost
is kernel-launch latency and Python overhead, not arithmetic. The only way a GPU
would pay here is to evaluate all 30 runs as one batched population, which means
rewriting every algorithm as a batched kernel and changing their semantics. The
speed-up on this workload comes from CPU process parallelism.
