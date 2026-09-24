"""Natijalarni jadvalga aylantirish. V1 chiqish shakli saqlanadi + kategoriya tahlili.

Hosil bo'ladigan fayllar (`docs/ACE_SHADE_PROMPT.md` 5-bo'lim):

  tables/summary_mean_std.csv       tables/friedman_test.txt
  tables/summary_results.csv        tables/friedman_posthoc.csv
  tables/summary_table.tex          tables/wilcoxon_pvalues.csv
  tables/overall_average_ranks.csv  tables/win_tie_loss.csv
  tables/category_ranks.csv         tables/category_wtl_<raqib>.csv
  tables/category_summary.md        tables/dimension_category.csv
  figures/Conv_<F>_<D>.png          raw_data/full_raw_results.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, mannwhitneyu, norm

from .benchmarks import CEC2022_CATEGORIES, category_of

PRECISION_FLOOR = float(os.environ.get("PRECISION_FLOOR", "1e-8"))


def holm(pvals):
    """Holm-Bonferroni tuzatishi."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, p[i] * (n - rank)))
        adj[i] = run
    return adj


def apply_floor(df_raw):
    """CEC konventsiyasi: xato < 1e-8 -> 0. Xom qiymatlar alohida saqlanadi."""
    out = df_raw.copy()
    n = int((out["Best"] < PRECISION_FLOOR).sum())
    out["Best"] = out["Best"].where(out["Best"] >= PRECISION_FLOOR, 0.0)
    return out, n


def _ranks(pivot):
    return pivot.rank(axis=1)


def posthoc(pivot, target):
    """Friedman post-hoc, nazorat = target. z, tuzatilmagan p, Holm p."""
    algs = list(pivot.columns)
    k, N = len(algs), len(pivot)
    avg = _ranks(pivot).mean()
    se = np.sqrt(k * (k + 1) / (6.0 * N))
    rows = []
    for a in algs:
        if a == target:
            continue
        diff = avg[a] - avg[target]
        z = diff / se
        rows.append({"Algorithm": a, "Average_Rank": avg[a],
                     "Rank_diff_vs_control": diff, "z": z,
                     "p_unadjusted": 2 * (1 - norm.cdf(abs(z)))})
    df = pd.DataFrame(rows)
    df["p_Holm"] = holm(df["p_unadjusted"].values)
    df["significant_0.05"] = df["p_Holm"] < 0.05
    return df.sort_values("p_Holm").reset_index(drop=True)


def win_tie_loss(df_raw, target, algs, funcs, dims):
    """Instansiya bo'yicha Mann-Whitney + Holm. `+/=/-` va p qiymatlari."""
    rivals = [a for a in algs if a != target]
    pvals, wtl = [], {c: {"+": 0, "=": 0, "-": 0} for c in rivals}
    for f in funcs:
        for d in dims:
            sel = (df_raw["Function"] == f) & (df_raw["Dimension"] == d)
            vt = df_raw[sel & (df_raw["Algorithm"] == target)]["Best"].values
            raw_p, better = [], []
            for c in rivals:
                vc = df_raw[sel & (df_raw["Algorithm"] == c)]["Best"].values
                try:
                    _, p = mannwhitneyu(vt, vc, alternative="two-sided")
                except ValueError:
                    p = 1.0
                raw_p.append(1.0 if np.isnan(p) else p)
                better.append(np.median(vt) < np.median(vc))
            adj = holm(raw_p)
            row = {"Function": f, "Dimension": d}
            for c, p, b in zip(rivals, adj, better):
                row[c] = p
                wtl[c]["=" if p >= 0.05 else ("+" if b else "-")] += 1
            pvals.append(row)
    return pd.DataFrame(pvals), pd.DataFrame(wtl).T[["+", "=", "-"]]


def category_tables(df_raw, pivot, target, out_dir):
    """Kategoriya x raqib bo'yicha to'liq tahlil (4.3-bo'lim)."""
    algs = list(pivot.columns)
    rivals = [a for a in algs if a != target]
    dims = sorted(df_raw["Dimension"].unique())

    # 1) Kategoriya x algoritm: o'rtacha rank
    cat_rows = []
    for cat, members in CEC2022_CATEGORIES.items():
        sub = pivot[pivot.index.get_level_values("Function").isin(members)]
        if len(sub) == 0:
            continue
        r = _ranks(sub).mean()
        cat_rows.append({"Category": cat, "n_instances": len(sub), **r.to_dict()})
    df_cat = pd.DataFrame(cat_rows)
    df_cat.to_csv(f"{out_dir}/tables/category_ranks.csv", index=False)

    # 2) Kategoriya x o'lcham kesimi
    dim_rows = []
    for cat, members in CEC2022_CATEGORIES.items():
        for d in dims:
            sub = pivot[pivot.index.get_level_values("Function").isin(members)
                        & (pivot.index.get_level_values("Dimension") == d)]
            if len(sub) == 0:
                continue
            dim_rows.append({"Category": cat, "Dimension": d,
                             "n_instances": len(sub), **_ranks(sub).mean().to_dict()})
    pd.DataFrame(dim_rows).to_csv(f"{out_dir}/tables/dimension_category.csv", index=False)

    # 3) Har raqib uchun kategoriya bo'yicha +/=/-
    lines = ["# Kategoriya bo'yicha tahlil", "",
             f"Nazorat: **{target}**. `YUTDIK` faqat Holm tuzatilgan "
             "`p < 0.05` bo'lganda - o'rtacha kichikroq bo'lishi yetarli emas.",
             ""]
    for cat, members in CEC2022_CATEGORIES.items():
        funcs = [f for f in members if f in set(pivot.index.get_level_values("Function"))]
        if not funcs:
            continue
        sub = pivot[pivot.index.get_level_values("Function").isin(funcs)]
        ranks = _ranks(sub).mean()
        pv, wtl = win_tie_loss(df_raw[df_raw["Function"].isin(funcs)], target, algs, funcs, dims)
        wtl.to_csv(f"{out_dir}/tables/category_wtl_{cat.replace(' ', '_')}.csv")

        lines += [f"## {cat} ({', '.join(funcs)})", "",
                  f"Instansiyalar: {len(sub)} ta. "
                  f"{target} o'rtacha ranki: **{ranks[target]:.2f}**", "",
                  "| Raqib | + | = | − | Bizning rank | Uning ranki | Holat |",
                  "|---|---|---|---|---|---|---|"]
        for c in rivals:
            w, t_, l = wtl.loc[c, "+"], wtl.loc[c, "="], wtl.loc[c, "-"]
            if w > l and w > 0:
                status = "**YUTDIK**"
            elif l > w:
                status = "**YUTQAZDIK**"
            else:
                status = "teng"
            lines.append(f"| {c} | {w} | {t_} | {l} | {ranks[target]:.2f} "
                         f"| {ranks[c]:.2f} | {status} |")
        lines.append("")

        # Funksiya darajasidagi tafsilot
        lines += ["| Instansiya | " + " | ".join(algs) + " |",
                  "|---|" + "---|" * len(algs)]
        for ix in sub.index:
            row = sub.loc[ix]
            best = row.min()
            cells = [("**%.2e**" if row[a] <= best + 1e-300 else "%.2e") % row[a]
                     if row[a] > 0 else ("**0**" if row[a] <= best else "0") for a in algs]
            lines.append(f"| {ix[0]} {ix[1]}D | " + " | ".join(cells) + " |")
        lines.append("")

    with open(f"{out_dir}/tables/category_summary.md", "w") as fh:
        fh.write("\n".join(lines))
    return df_cat


def analyse(df_raw, target, out_dir, curves=None):
    """To'liq tahlil quvuri. `df_raw`: Algorithm, Function, Dimension, Run, Best."""
    for sub in ("tables", "figures", "raw_data"):
        os.makedirs(f"{out_dir}/{sub}", exist_ok=True)
    df_raw.to_csv(f"{out_dir}/raw_data/full_raw_results.csv", index=False)

    df, n_floor = apply_floor(df_raw)
    print(f"[*] CEC aniqlik chegarasi {PRECISION_FLOOR:g}: {n_floor} ta natija 0 ga tenglashtirildi")

    algs = [a for a in df["Algorithm"].unique()]
    algs = [target] + [a for a in algs if a != target]
    funcs = sorted(df["Function"].unique(), key=lambda s: int(s[1:]))
    dims = sorted(df["Dimension"].unique())

    summ = df.groupby(["Function", "Dimension", "Algorithm"])["Best"].agg(["mean", "std"]).reset_index()
    summ.to_csv(f"{out_dir}/tables/summary_mean_std.csv", index=False)
    pivot = summ.pivot(index=["Function", "Dimension"], columns="Algorithm", values="mean")[algs].dropna()
    pivot.to_csv(f"{out_dir}/tables/summary_results.csv")
    try:
        pivot.to_latex(f"{out_dir}/tables/summary_table.tex", float_format="%.2e")
    except Exception as e:
        print(f"[!] LaTeX jadval yozilmadi: {e}")

    chi2, p = friedmanchisquare(*[pivot[a].values for a in algs])
    avg = _ranks(pivot).mean().sort_values()
    with open(f"{out_dir}/tables/friedman_test.txt", "w") as fh:
        fh.write(f"Friedman chi2 = {chi2:.4f}, p = {p:.4e}\n")
        fh.write(f"instansiyalar = {len(pivot)}, algoritmlar = {len(algs)}\n\n")
        fh.write(avg.to_string())
    avg.reset_index().rename(columns={0: "Average_Rank", "index": "Algorithm"}) \
       .to_csv(f"{out_dir}/tables/overall_average_ranks.csv", index=False)

    posthoc(pivot, target).to_csv(f"{out_dir}/tables/friedman_posthoc.csv", index=False)
    pv, wtl = win_tie_loss(df, target, algs, funcs, dims)
    pv.to_csv(f"{out_dir}/tables/wilcoxon_pvalues.csv", index=False)
    wtl.to_csv(f"{out_dir}/tables/win_tie_loss.csv")

    category_tables(df, pivot, target, out_dir)

    if curves is not None:
        _plot(curves, out_dir, algs)
    print(f"[*] Tahlil tayyor: {out_dir}/tables/")
    return pivot


def _plot(curves, out_dir, algs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = sorted({(f, d) for (_, f, d) in curves})
    for f, d in keys:
        plt.figure(figsize=(6, 4.2))
        for a in algs:
            cs = [v for (aa, ff, dd), v in curves.items() if aa == a and ff == f and dd == d]
            if not cs:
                continue
            med = np.nanmedian(np.vstack(cs), axis=0)
            x = np.linspace(0, 1, len(med))
            plt.semilogy(x, np.maximum(med, 1e-12), label=a, lw=1.3)
        plt.xlabel("byudjet ulushi"); plt.ylabel("xato (median)")
        plt.title(f"{f}, D={d}"); plt.grid(alpha=.3); plt.legend(fontsize=7)
        plt.tight_layout(); plt.savefig(f"{out_dir}/figures/Conv_{f}_{d}D.png", dpi=110)
        plt.close()
