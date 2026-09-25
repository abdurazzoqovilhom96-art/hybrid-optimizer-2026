"""Compare two runs of the same protocol, row for row.

The paper claims the study is reproducible. That claim is worth only as much as
the check behind it, so this compares two independently produced results files
and requires the error of every (algorithm, function, dimension, run) cell to be
bit-identical -- not close, identical.

It is meant for a run on one machine against a run on another. A mismatch is
informative either way: it means either a seed that depends on something it
should not (the line-up, the function, the shard), or a genuine platform
difference in floating-point arithmetic, and the two are told apart by whether
the mismatch is systematic or confined to a few cells.

    python tools/compare_runs.py --a results_cec2017/raw/results.csv \
                                 --b artifacts/results.csv
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

KEY = ["Algorithm", "Function", "Dimension", "Run"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args(argv)

    a = pd.read_csv(args.a, float_precision="round_trip")
    b = pd.read_csv(args.b, float_precision="round_trip")

    ka, kb = set(map(tuple, a[KEY].to_numpy())), set(map(tuple, b[KEY].to_numpy()))
    common = ka & kb
    print(f"[*] {args.label_a}: {len(a)} rows   {args.label_b}: {len(b)} rows   "
          f"common cells: {len(common)}")
    if not common:
        print("[X] the two files share no cell; they are not the same protocol.")
        return 1
    for label, only in ((args.label_a, ka - kb), (args.label_b, kb - ka)):
        if only:
            print(f"[!] {len(only)} cells only in {label}, e.g. {sorted(only)[:3]}")

    m = a.merge(b, on=KEY, suffixes=("_a", "_b"))
    same = m["Error_a"].to_numpy() == m["Error_b"].to_numpy()
    n_bad = int((~same).sum())

    print(f"[*] bit-identical: {int(same.sum())}/{len(m)} "
          f"({100 * same.mean():.4f}%)")
    if n_bad == 0:
        print(f"[ok] {args.label_a} and {args.label_b} agree exactly on every shared cell.")
        return 0

    bad = m[~same].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.abs(bad["Error_b"].to_numpy())
        bad["rel"] = np.where(denom > 0,
                              np.abs(bad["Error_a"] - bad["Error_b"]) / denom, np.inf)
    print(f"[X] {n_bad} cells differ. Largest relative differences:")
    cols = KEY + ["Error_a", "Error_b", "rel"]
    print(bad.sort_values("rel", ascending=False)[cols].head(15).to_string(index=False))
    print("\n    differing cells per algorithm:")
    print(bad.groupby("Algorithm").size().to_string())
    print(f"\n    max relative difference: {bad['rel'].max():.3e}")
    print("    A difference at the 1e-16 level on a few cells is platform "
          "floating-point.\n    A systematic one is a seed that depends on "
          "something it should not.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
