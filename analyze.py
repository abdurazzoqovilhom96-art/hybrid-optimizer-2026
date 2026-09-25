"""Turn raw results into tables, statistics and figures.

Separate from ``run_all.py`` on purpose: analysis is cheap and gets re-run many
times, the experiment is expensive and gets run once.

    python analyze.py --out results --control TEMOA_V11
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from temoa.stats import (a12_magnitude, descriptive, friedman,
                         friedman_posthoc_holm, holm, mannwhitney, nemenyi_cd,
                         paired_wilcoxon, vargha_delaney_a12)

ALPHA = 0.05


def load(out: Path) -> pd.DataFrame:
    # float_precision="round_trip" is required: the default CSV parser is off by
    # up to one ULP, which matters when algorithms are separated at 1e-20.
    df = pd.read_csv(out / "raw" / "results.csv", float_precision="round_trip")
    # a resumed run can append a duplicate block; keep the last of each key
    return df.drop_duplicates(subset=["Algorithm", "Function", "Dimension", "Run"], keep="last")


def table_descriptive(df, tdir):
    rows = []
    for (f, d, a), g in df.groupby(["Function", "Dimension", "Algorithm"]):
        rows.append({"Function": f, "Dimension": d, "Algorithm": a,
                     **descriptive(g["Error"].to_numpy())})
    out = pd.DataFrame(rows).sort_values(["Dimension", "Function", "median"])
    out.to_csv(tdir / "descriptive_stats.csv", index=False)
    return out


def median_pivot(df, dim):
    sel = df[df["Dimension"] == dim]
    return sel.pivot_table(index="Function", columns="Algorithm", values="Error", aggfunc="median")


def stats_per_dimension(df, tdir, control, algos):
    lines, cd_data = [], {}
    for dim in sorted(df["Dimension"].unique()):
        present = [a for a in algos if a in median_pivot(df, dim).columns]
        piv = median_pivot(df, dim)[present].dropna()
        if piv.empty:
            continue
        fr = friedman(piv)
        ranks = fr["avg_ranks"]
        cd_data[dim] = (ranks, fr["N_blocks"])

        lines.append(f"=== {dim}D  ({fr['N_blocks']} functions, {fr['k_algorithms']} algorithms) ===")
        lines.append(f"Friedman chi2 = {fr['chi2']:.4f}, p = {fr['p_chi2']:.4e}")
        lines.append(f"Iman-Davenport F = {fr['iman_davenport_F']:.4f}, p = {fr['p_F']:.4e}")
        try:
            cd = nemenyi_cd(fr["k_algorithms"], fr["N_blocks"], ALPHA)
            lines.append(f"Nemenyi critical difference (alpha={ALPHA}) = {cd:.4f}")
        except ValueError as e:
            lines.append(f"Nemenyi CD unavailable: {e}")
        lines.append("Average ranks (1 = best):")
        lines.append(ranks.to_string())

        ph = friedman_posthoc_holm(ranks, fr["N_blocks"], control)
        ph.to_csv(tdir / f"posthoc_holm_{dim}D.csv", index=False)
        lines.append(f"\nHolm post-hoc, control = {control}:")
        lines.append(ph.to_string(index=False))
        lines.append("")

    (tdir / "friedman_per_dimension.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return cd_data


def pairwise(df, tdir, control, algos):
    """Signed-rank + Mann-Whitney + A12, Holm within cell and over the whole family."""
    competitors = [a for a in algos if a != control]
    rows = []
    for dim in sorted(df["Dimension"].unique()):
        for f in sorted(df["Function"].unique()):
            sel = df[(df["Dimension"] == dim) & (df["Function"] == f)]
            ctl = sel[sel["Algorithm"] == control].sort_values("Run")["Error"].to_numpy()
            if ctl.size == 0:
                continue
            cell = []
            for c in competitors:
                cmp_ = sel[sel["Algorithm"] == c].sort_values("Run")["Error"].to_numpy()
                if cmp_.size != ctl.size:
                    continue
                a12 = vargha_delaney_a12(ctl, cmp_)
                cell.append({"Dimension": dim, "Function": f, "Control": control,
                             "Competitor": c,
                             "median_control": float(np.median(ctl)),
                             "median_competitor": float(np.median(cmp_)),
                             "p_signedrank": paired_wilcoxon(ctl, cmp_),
                             "p_mannwhitney": mannwhitney(ctl, cmp_),
                             "A12": a12, "A12_magnitude": a12_magnitude(a12)})
            if not cell:
                continue
            adj = holm([r["p_signedrank"] for r in cell])
            for r, p in zip(cell, adj):
                r["p_holm_cell"] = p
            rows.extend(cell)

    res = pd.DataFrame(rows)
    if res.empty:
        return res
    # multiplicity control over the entire family of tests actually performed
    res["p_holm_family"] = holm(res["p_signedrank"].to_numpy())

    def verdict(r):
        if r["p_holm_cell"] >= ALPHA:
            return "="
        return "+" if r["median_control"] < r["median_competitor"] else "-"

    res["outcome"] = res.apply(verdict, axis=1)
    res.to_csv(tdir / "pairwise_tests.csv", index=False)

    wtl = (res.groupby(["Competitor", "outcome"]).size().unstack(fill_value=0)
           .reindex(columns=["+", "=", "-"], fill_value=0))
    wtl.to_csv(tdir / "win_tie_loss.csv")
    print(f"\n{control} vs competitors (+ win / = tie / - loss), Holm within cell, alpha={ALPHA}:")
    print(wtl.to_string())
    return res


def runtime_table(df, tdir):
    rt = df.groupby(["Algorithm", "Dimension"])["Seconds"].agg(["mean", "std"]).reset_index()
    rt.to_csv(tdir / "runtime_seconds.csv", index=False)
    piv = rt.pivot(index="Algorithm", columns="Dimension", values="mean")
    print("\nMean wall-clock seconds per run:")
    print(piv.round(2).to_string())
    return rt


def fes_per_dim(out: Path, default: int = 3000) -> int:
    mf = out / "manifest.json"
    if mf.exists():
        import json
        return int(json.loads(mf.read_text(encoding="utf-8")).get("fes_per_dim", default))
    return default


def figures(df, out, algos, control):
    fdir = out / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    cdir = out / "curves"
    budget_per_dim = fes_per_dim(out)

    for dim in sorted(df["Dimension"].unique()):
        path = cdir / f"curves_{dim}D.npz"
        if not path.exists():
            continue
        data = np.load(path)
        fes_axis = None
        for f in sorted(df["Function"].unique()):
            plt.figure(figsize=(9, 5.5))
            plotted = False
            for a in algos:
                keys = [k for k in data.files if k.startswith(f"{a}|{f}|")]
                if not keys:
                    continue
                arr = np.vstack([data[k] for k in keys])
                med = np.median(arr, axis=0)         # median, not mean: robust to outlier runs
                if fes_axis is None or len(fes_axis) != len(med):
                    total = dim * budget_per_dim
                    fes_axis = np.linspace(total / len(med), total, len(med))
                plt.plot(fes_axis, np.maximum(med, 1e-300), label=a,
                         linewidth=2.5 if a == control else 1.2)
                plotted = True
            if not plotted:
                plt.close()
                continue
            plt.yscale("log")
            plt.xlabel("Function evaluations (FES)")
            plt.ylabel("Median error  f(x) - f*")
            plt.title(f"{f} ({dim}D)")
            plt.legend(fontsize=8, ncol=2)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(fdir / f"Conv_{f}_{dim}D.png", dpi=200)
            plt.close()


def cd_diagram(cd_data, out, alpha=ALPHA):
    fdir = out / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    for dim, (ranks, n_blocks) in cd_data.items():
        k = len(ranks)
        try:
            cd = nemenyi_cd(k, n_blocks, alpha)
        except ValueError:
            continue
        fig, ax = plt.subplots(figsize=(9, 0.42 * k + 1.8))
        y = np.arange(k)[::-1]
        ax.barh(y, ranks.to_numpy(), color="#4C78A8", height=0.55)
        ax.set_yticks(y)
        ax.set_yticklabels(ranks.index)
        best = ranks.min()
        ax.axvline(best, color="#333", lw=1)
        ax.axvline(best + cd, color="#E45756", ls="--", lw=1.4,
                   label=f"critical difference = {cd:.2f}")
        ax.set_xlabel("Average Friedman rank (1 = best)")
        ax.set_title(f"{dim}D  -  Nemenyi CD, alpha={alpha}, N={n_blocks}")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(fdir / f"CD_{dim}D.png", dpi=200)
        plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--control", default="TEMOA_V11")
    ap.add_argument("--algos", nargs="+", default=None)
    args = ap.parse_args(argv)

    out = Path(args.out)
    tdir = out / "tables"
    tdir.mkdir(parents=True, exist_ok=True)

    df = load(out)
    algos = args.algos or sorted(df["Algorithm"].unique())
    if args.control not in algos:
        ap.error(f"control {args.control!r} not present. Available: {algos}")

    over = df[df["Overrun"] > 0]
    if not over.empty:
        print(f"[!] {len(over)} runs exceeded the FES budget -- investigate before reporting:")
        print(over.groupby("Algorithm")["Overrun"].max().to_string())

    table_descriptive(df, tdir)
    for dim in sorted(df["Dimension"].unique()):
        median_pivot(df, dim).to_csv(tdir / f"median_error_{dim}D.csv")
    cd_data = stats_per_dimension(df, tdir, args.control, algos)
    pairwise(df, tdir, args.control, algos)
    runtime_table(df, tdir)
    figures(df, out, algos, args.control)
    cd_diagram(cd_data, out)
    print(f"\n[*] tables -> {tdir}\n[*] figures -> {out / 'figures'}")


if __name__ == "__main__":
    main()
