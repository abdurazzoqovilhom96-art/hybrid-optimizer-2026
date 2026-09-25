"""Merge sharded results back into one results directory.

The gate is split across CI jobs by function, one shard per function. The split
changes nothing about what is computed: every seed is
``default_rng([SEED_BASE, dim, run, algorithm_seed(name)])``, which contains no
term for the function or for the shard, so a sharded run and a single-process
run produce the same numbers. This script only reassembles the pieces.

It refuses to merge a partial set. A missing shard would produce a results file
that looks complete and is quietly missing a function -- the kind of fault that
survives into a published table.

    python tools/merge_shards.py --shards artifacts --out results_cec2017 \
                                 --expect-functions 29 --expect-rows 14790
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS = ["Algorithm", "Function", "Dimension", "Run", "Error", "Seconds", "FES", "Overrun"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True, help="directory holding the downloaded shards")
    ap.add_argument("--out", required=True, help="results directory to assemble")
    ap.add_argument("--expect-functions", type=int, default=None)
    ap.add_argument("--expect-rows", type=int, default=None)
    args = ap.parse_args(argv)

    shards = sorted(Path(args.shards).glob("**/raw/results.csv"))
    if not shards:
        ap.error(f"no shard results found under {args.shards}")

    frames = [pd.read_csv(p, float_precision="round_trip") for p in shards]
    df = pd.concat(frames, ignore_index=True)

    dup = df.duplicated(subset=["Algorithm", "Function", "Dimension", "Run"])
    if dup.any():
        print(f"[X] {int(dup.sum())} duplicated (algorithm, function, dim, run) rows across shards.")
        print(df[dup].head(10).to_string())
        return 1

    df = df.sort_values(["Dimension", "Function", "Algorithm", "Run"], kind="stable")

    n_fun = df["Function"].nunique()
    print(f"[*] {len(shards)} shards -> {len(df)} rows, {n_fun} functions, "
          f"{df['Algorithm'].nunique()} algorithms, dims {sorted(df['Dimension'].unique())}")

    ok = True
    if args.expect_functions is not None and n_fun != args.expect_functions:
        got = sorted(df["Function"].unique())
        print(f"[X] expected {args.expect_functions} functions, got {n_fun}: {got}")
        ok = False
    if args.expect_rows is not None and len(df) != args.expect_rows:
        print(f"[X] expected {args.expect_rows} rows, got {len(df)}")
        # name the incomplete cells rather than only the total
        counts = df.groupby(["Function", "Algorithm"]).size()
        short = counts[counts != counts.max()]
        if len(short):
            print("    incomplete cells (function, algorithm -> runs):")
            print(short.head(20).to_string())
        ok = False
    if int(df["Overrun"].sum()) != 0:
        bad = df[df["Overrun"] != 0]
        print(f"[X] {len(bad)} runs exceeded their evaluation budget:")
        print(bad[["Algorithm", "Function", "Run", "FES", "Overrun"]].head(10).to_string())
        ok = False
    if not np.isfinite(df["Error"]).all():
        print(f"[X] {int((~np.isfinite(df['Error'])).sum())} non-finite error values")
        ok = False
    if not ok:
        print("[X] refusing to write an incomplete results file.")
        return 1

    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "curves").mkdir(parents=True, exist_ok=True)
    df[RAW_COLUMNS].to_csv(out / "raw" / "results.csv", index=False)

    curves = {}
    for p in sorted(Path(args.shards).glob("**/curves/curves_*D.npz")):
        with np.load(p) as z:
            curves.setdefault(p.name, {}).update({k: z[k] for k in z.files})
    for name, data in curves.items():
        np.savez_compressed(out / "curves" / name, **data)
        print(f"[*] {name}: {len(data)} curves")

    manifests = sorted(Path(args.shards).glob("**/manifest.json"))
    base = json.loads(manifests[0].read_text(encoding="utf-8"))
    base["sharded"] = {
        "n_shards": len(shards),
        "note": "run split across CI jobs by function; seeds contain no function or "
                "shard term, so this is identical to a single-process run",
        "functions": sorted(df["Function"].unique().tolist()),
    }
    (out / "manifest.json").write_text(json.dumps(base, indent=2), encoding="utf-8")

    print(f"[ok] {out}/raw/results.csv written ({len(df)} rows)")
    print(f"[*] next: python analyze.py --out {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
