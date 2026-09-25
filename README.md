# hybrid-optimizer-2026

Benchmark study of **TEMOA_V10_HYBRID**, a hybrid metaheuristic combining an
L-SHADE/jSO core with GWO-, WOA- and HHO-inspired operators, an eigenbasis
crossover and a (1+1)-ES refinement phase.

This repository contains the original algorithm, a repaired variant
(**TEMOA_V11**), fair comparators, and a reproducible experimental protocol.

> **Main finding.** As published, TEMOA_V10 is outperformed by plain L-SHADE and
> jSO — the very algorithms its own header comment names as its core — on most of
> the suite, and collapses by roughly six orders of magnitude on Schwefel. The
> cause is diagnosed in `reports/TAHLIL_UZ.md` and repaired in `TEMOA_V11`.
> Read that report before using any number from the original study.

## Quick start

### Windows (one click)

```
run_windows.bat
```

Installs dependencies, runs the tests and the smoke test, then the full study
and the analysis. Safe to interrupt: re-running resumes where it stopped.

### Any platform

```bash
pip install -r requirements.txt
python tests/test_all.py                                    # 19 correctness tests
python tests/test_port_fidelity.py                          # port == original (~4 min)
python run_all.py --smoke --jobs 20                         # ~3 min end-to-end check
python run_all.py --dims 30 50 100 --runs 30 --jobs 20      # main study
python analyze.py --out results --control TEMOA_V11         # tables, stats, figures
```

Optional deeper studies:

```bash
python experiments/ablation.py    --dim 30 --runs 15 --jobs 20
python experiments/sensitivity.py --dim 30 --runs 15 --jobs 20
```

**`--jobs`**: use about 80% of your logical cores (20 on a 24-thread CPU). Each
worker is pinned to one BLAS thread; without that pinning NumPy's internal
threads oversubscribe the CPU and more jobs makes the run *slower*.

Reference timing: the full 30D/50D/100D protocol takes roughly **2 hours** on a
16-core / 24-thread desktop, or about 6 hours on 4 cores.

## Layout

| Path | What it is |
|---|---|
| `hybrid 2026.py` | the original single-file study, kept unmodified for reference |
| `temoa/problems.py` | benchmark suite, error values, rotated and shift-only tracks |
| `temoa/tracker.py` | FES accounting; offline-error and noisy-recommendation scoring |
| `temoa/algorithms/temoa_v10.py` | the original algorithm (RNG injected, otherwise identical) |
| `temoa/algorithms/temoa_v11.py` | the repaired variant, with the evidence for each change |
| `temoa/algorithms/modern.py` | L-SHADE and jSO — the comparators the original study omits |
| `temoa/algorithms/baselines.py` | DE, GWO, WOA, SCA, PSO, HHO (+ the original weakened PSO/HHO) |
| `temoa/stats.py` | Friedman, Iman-Davenport, Holm, Nemenyi, signed-rank, Vargha-Delaney A12 |
| `run_all.py` / `analyze.py` | experiment driver / reporting |
| `tests/test_port_fidelity.py` | proves the ported V10 matches `hybrid 2026.py` |
| `reports/TAHLIL_UZ.md` | **the analysis — start here** |

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
