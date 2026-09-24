#!/usr/bin/env python3
"""B1: ACE-SHADE parametrlarini CEC-2017 da sozlash.

Sozlash to'plami baholash to'plamidan (CEC-2022) BUTUNLAY ajratilgan.
Byudjet CEC-2017 ning o'z protokoli bo'yicha `10000 * D`.

Ikkita parametr sozlanadi:
    r_N   populyatsiya koeffitsienti, N_init = r_N * D
    eta   kredit yangilanishining o'rganish tezligi

Qolgan hamma narsa (N_min, H, arxiv, RSP, p chegaralari) manba
usullaridan o'zgarishsiz olinadi va sozlanmaydi.

Ishga tushirish:
    CFG=rn6_eta0.2 OUT_DIR=tune python3 tune_b1.py      # bitta konfiguratsiya
    MERGE=t1,t2,... OUT_DIR=results/tuning_v2 python3 tune_b1.py
"""
import os
import sys
import time
import itertools

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aceshade.core import Tracker
from aceshade.benchmarks import make_problem, CEC2017_TUNING_FUNCS
from aceshade.ace import ace_shade

SEED_BASE = 42
R_N_GRID = (4.0, 6.0, 8.0, 10.0)
ETA_GRID = (0.1, 0.2, 0.4)
DIMS = (10, 30)
FES_PER_DIM = 10_000                      # CEC-2017 rasmiy protokoli
PRECISION_FLOOR = 1e-8


def cfg_name(r_n, eta):
    return f"rn{r_n:g}_eta{eta:g}"


def run_one(r_n, eta, func, dim, run_id, budget):
    np.random.seed(SEED_BASE + 1000 * dim + run_id)
    obj, lb, ub, _ = make_problem(func, dim, 2017)
    tr = Tracker(obj, budget, 2)
    try:
        ace_shade(tr, dim, (lb, ub), budget, R_N=r_n, ETA=eta)
    except Exception as exc:
        print(f"[!] {cfg_name(r_n, eta)} {func} {dim}D run {run_id}: {exc}")
    return {"Config": cfg_name(r_n, eta), "r_N": r_n, "eta": eta,
            "Function": func, "Dimension": dim, "Run": run_id,
            "Best": float(tr.best_f)}


def analyse(df, out_dir):
    d = df.copy()
    d["Best"] = d["Best"].where(d["Best"] >= PRECISION_FLOOR, 0.0)
    piv = (d.groupby(["Function", "Dimension", "Config"])["Best"].median()
             .unstack("Config"))
    ranks = piv.rank(axis=1).mean().sort_values()
    os.makedirs(out_dir, exist_ok=True)
    d.to_csv(f"{out_dir}/tuning_raw.csv", index=False)
    piv.to_csv(f"{out_dir}/tuning_medians.csv")
    ranks.to_csv(f"{out_dir}/tuning_ranks.csv", header=["Average_Rank"])

    # O'lcham bo'yicha alohida: r_N o'lchamga bog'liq bo'lishi mumkin
    per_dim = {}
    for dim in sorted(d["Dimension"].unique()):
        sub = piv[piv.index.get_level_values("Dimension") == dim]
        per_dim[dim] = sub.rank(axis=1).mean().sort_values()
    pd.DataFrame(per_dim).to_csv(f"{out_dir}/tuning_ranks_per_dim.csv")

    print("\n=== O'rtacha rank (past = yaxshi) ===")
    print(ranks.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n=== O'lcham bo'yicha ===")
    print(pd.DataFrame(per_dim).to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\n[*] Eng yaxshi: {ranks.index[0]}")
    return ranks


def main():
    out_dir = os.environ.get("OUT_DIR", "results/tuning_v2")
    merge = [x.strip().rstrip("/") for x in os.environ.get("MERGE", "").split(",") if x.strip()]
    if merge:
        frames = [pd.read_csv(f"{d}/tuning_raw.csv") for d in merge
                  if os.path.exists(f"{d}/tuning_raw.csv")]
        if not frames:
            sys.exit("[!] bo'lak topilmadi")
        analyse(pd.concat(frames, ignore_index=True), out_dir)
        return

    runs = int(os.environ.get("RUNS", "10"))
    quick = os.environ.get("QUICK") == "1"
    only = os.environ.get("CFG", "")
    funcs = CEC2017_TUNING_FUNCS[:2] if quick else CEC2017_TUNING_FUNCS
    dims = (10,) if quick else DIMS

    grid = [(r, e) for r, e in itertools.product(R_N_GRID, ETA_GRID)
            if not only or cfg_name(r, e) == only]
    if not grid:
        sys.exit(f"[!] konfiguratsiya topilmadi: {only}")

    tasks = []
    for r_n, eta in grid:
        for func in funcs:
            for dim in dims:
                budget = FES_PER_DIM * dim // (20 if quick else 1)
                for r in range(runs):
                    tasks.append((r_n, eta, func, dim, r, budget))
    print(f"[*] {len(tasks)} vazifa | {len(grid)} konfiguratsiya | "
          f"{len(funcs)} funksiya | o'lchamlar {dims} | {runs} run")
    t0 = time.time()
    rows = Parallel(n_jobs=-1, verbose=5)(delayed(run_one)(*t) for t in tasks)
    print(f"[*] {time.time() - t0:.0f} s")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{out_dir}/tuning_raw.csv", index=False)
    if not only:
        analyse(pd.DataFrame(rows), out_dir)
    else:
        print(f"[*] bo'lak saqlandi: {out_dir}/tuning_raw.csv")


if __name__ == "__main__":
    main()
