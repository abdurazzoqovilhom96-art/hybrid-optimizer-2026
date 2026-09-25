"""Per-class Friedman ranks for CEC'2017.

The overall rank across 29 functions hides the only thing worth knowing. The
suite is four classes with different structure, and an algorithm can lead on one
while trailing on another -- which is exactly what the preliminary gate measured
for this family: first on the hybrid class, fourth on composition. Reporting one
average rank would have concealed both facts.

The classes are the competition's own, with F2 withdrawn:

    unimodal          F1, F3                  2 functions
    simple multimodal F4-F10                  7
    hybrid            F11-F20                10
    composition       F21-F30                10

A caution that belongs with every number this prints: with 2 functions the
unimodal block cannot support a Friedman test at all, and with 7 or 10 it
supports one only weakly. Friedman's statistic assumes the blocks are the
sample, so a class with ten functions is a sample of ten. The per-class ranks
are a description of where an algorithm stands; the significance claim belongs
to the full 29-function test.

    python tools/class_ranks.py --out results_cec2017 [--dim 10]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temoa.registry import TARGET                                    # noqa: E402
from temoa.stats import friedman, friedman_posthoc_holm, nemenyi_cd  # noqa: E402

CLASSES = {
    "unimodal":          [1, 3],
    "simple multimodal": list(range(4, 11)),
    "hybrid":            list(range(11, 21)),
    "composition":       list(range(21, 31)),
}
MIN_BLOCKS_FOR_TEST = 5


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="results directory")
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--control", default=TARGET)
    args = ap.parse_args(argv)

    raw = Path(args.out) / "raw" / "results.csv"
    df = pd.read_csv(raw, float_precision="round_trip")
    dims = [args.dim] if args.dim else sorted(df["Dimension"].unique())

    for dim in dims:
        sel = df[df["Dimension"] == dim]
        piv = sel.pivot_table(index="Function", columns="Algorithm",
                             values="Error", aggfunc="median")
        fid = {f"F{i}": i for i in range(1, 31)}
        piv = piv[piv.index.isin(fid)]
        piv["_id"] = [fid[f] for f in piv.index]

        print(f"\n{'=' * 78}\n  D = {dim}   ({sel['Run'].nunique()} runs per cell)\n{'=' * 78}")

        rows = {}
        for name, ids in list(CLASSES.items()) + [("ALL 29", list(range(1, 31)))]:
            block = piv[piv["_id"].isin(ids)].drop(columns="_id").dropna()
            if block.empty:
                continue
            fr = friedman(block)
            rows[name] = fr["avg_ranks"]
            n = fr["N_blocks"]
            head = f"-- {name}  ({n} functions)"
            if n < MIN_BLOCKS_FOR_TEST:
                print(f"{head}  -- too few functions for a meaningful test; ranks only")
            else:
                cd = ""
                try:
                    cd = f"   Nemenyi CD = {nemenyi_cd(fr['k_algorithms'], n):.3f}"
                except ValueError:
                    pass
                print(f"{head}   Friedman chi2 = {fr['chi2']:.3f}, p = {fr['p_chi2']:.3e}"
                      f"   Iman-Davenport p = {fr['p_F']:.3e}{cd}")
            print(fr["avg_ranks"].sort_values().to_string())
            if n >= MIN_BLOCKS_FOR_TEST and args.control in block.columns:
                ph = friedman_posthoc_holm(fr["avg_ranks"], n, args.control)
                sig = ph[ph["p_holm"] < 0.05] if "p_holm" in ph else ph
                print(f"   Holm vs {args.control}: "
                      f"{len(sig)} of {len(ph)} comparisons significant at 0.05")
            print()

        table = pd.DataFrame(rows)
        table = table.loc[table["ALL 29"].sort_values().index] if "ALL 29" in table else table
        print("-- average ranks, all classes side by side (1 = best)")
        print(table.round(2).to_string())
        tdir = Path(args.out) / "tables"
        tdir.mkdir(parents=True, exist_ok=True)
        table.round(4).to_csv(tdir / f"class_ranks_{dim}D.csv")
        print(f"\n[ok] written to {tdir / f'class_ranks_{dim}D.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
