"""CBA-SHADE zaif tomonlarini natijalar jadvalidan ajratib oladi.

Yakuniy hisobotning eng muhim bo'limi uchun: qaysi instansiyalarda taklif
etilayotgan usul yutqazdi yoki teng chiqdi, qanchaga va kimdan. Hisobotni
bezash o'rniga bu raqamlarni ochiq keltirish maqsad qilingan.

Ishlatish:
    python3 scripts/weaknesses.py results/full/tables
"""
import sys
import pandas as pd

TARGET = "CBA-SHADE"


def load(tables_dir):
    means = pd.read_csv(f"{tables_dir}/summary_results.csv", index_col=[0, 1])
    pvals = pd.read_csv(f"{tables_dir}/wilcoxon_pvalues.csv").set_index(["Function", "Dimension"])
    return means, pvals


def analyse(means, pvals):
    """Har bir instansiya uchun eng yaxshi raqib va statistik ahamiyatni qaytaradi."""
    competitors = [c for c in means.columns if c != TARGET]
    rows = []
    for idx, row in means.iterrows():
        ours = row[TARGET]
        best_rival = row[competitors].idxmin()
        theirs = row[best_rival]
        # Holm bilan tuzatilgan p — wilcoxon_pvalues.csv da raqib ustunida.
        p = pvals.loc[idx, best_rival] if idx in pvals.index else float("nan")
        # Ahamiyatsiz farq g'alaba emas: p >= 0.05 bo'lsa natija teng deb
        # sanaladi, aks holda hisobot ustunlikni ortiqcha ko'rsatadi.
        if p >= 0.05:
            verdict = "teng"
        elif ours <= theirs:
            verdict = "yutdi"
        else:
            verdict = "YUTQAZDI"
        rows.append({
            "Function": idx[0], "Dimension": idx[1],
            "CBA-SHADE": ours, "Eng yaxshi raqib": best_rival, "Raqib": theirs,
            "Nisbat": (ours / theirs) if theirs > 0 else float("inf"),
            "p (Holm)": p, "Natija": verdict,
        })
    return pd.DataFrame(rows)


def markdown(df):
    out = []
    losses = df[df.Natija == "YUTQAZDI"].sort_values("Nisbat", ascending=False)
    ties = df[df.Natija == "teng"]
    wins = df[df.Natija == "yutdi"]

    out.append(f"CBA-SHADE {len(df)} instansiyadan **{len(wins)} tasida eng yaxshi**, "
               f"{len(ties)} tasida statistik jihatdan teng, "
               f"**{len(losses)} tasida yutqazdi**.\n")

    if len(losses):
        out.append("### Statistik jihatdan ahamiyatli mag'lubiyatlar\n")
        out.append("| Funksiya | D | CBA-SHADE | Yutgan algoritm | Uning natijasi | Nisbat | p (Holm) |")
        out.append("|---|---|---|---|---|---|---|")
        for _, r in losses.iterrows():
            out.append(f"| {r.Function} | {r.Dimension} | {r['CBA-SHADE']:.3e} | "
                       f"{r['Eng yaxshi raqib']} | {r.Raqib:.3e} | "
                       f"{r.Nisbat:.1f}x | {r['p (Holm)']:.3f} |")
        out.append("")
    else:
        out.append("Statistik jihatdan ahamiyatli mag'lubiyat yo'q.\n")

    if len(ties):
        out.append("### Teng chiqqan instansiyalar\n")
        out.append("Bu instansiyalarda farq Holm tuzatishidan keyin ahamiyatli emas "
                   "(`p >= 0.05`), ya'ni ustunlik da'vo qilib bo'lmaydi:\n")
        out.append("| Funksiya | D | CBA-SHADE | Eng yaqin raqib | Uning natijasi | p (Holm) |")
        out.append("|---|---|---|---|---|---|")
        for _, r in ties.iterrows():
            out.append(f"| {r.Function} | {r.Dimension} | {r['CBA-SHADE']:.3e} | "
                       f"{r['Eng yaxshi raqib']} | {r.Raqib:.3e} | {r['p (Holm)']:.3f} |")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    tables = sys.argv[1] if len(sys.argv) > 1 else "results/full/tables"
    df = analyse(*load(tables))
    df.to_csv("weaknesses.csv", index=False)
    print(markdown(df))
