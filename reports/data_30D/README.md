# Preliminary 30D results

Produced by:

```
python run_all.py --dims 30 --runs 12 --jobs 4 --out results_prelim --with-weakened
python analyze.py --out results_prelim --control TEMOA_V11
```

12 algorithms × 12 functions × 12 runs = 1728 runs, rotated suite, 90 000 FES
each, 23.9 min on 4 cores. `manifest.json` records versions, seeds and platform.

**These numbers are not publication-ready.** With 12 runs the Wilcoxon
signed-rank test cannot produce a p-value below 2/2^12 = 4.9e-04, while
family-wide Holm over 132 tests requires p < 3.8e-04 — so nothing can reach
significance regardless of effect size. Re-run with `--runs 30` (minimum
p = 1.9e-09) before quoting any p-value. See §4.3 of `../TAHLIL_UZ.md`.
