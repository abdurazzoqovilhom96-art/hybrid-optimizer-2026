#!/usr/bin/env python3
"""ACE-SHADE eksperiment ishga tushirgichi.

Bo'lak rejimi (Actions shardi):
    FUNCS=F1,F2 DIMS=10 ALGS=ACE-SHADE,jSO RUNS=30 OUT_DIR=chunk python3 ace_shade.py

Birlashtirish va tahlil:
    MERGE=chunks/a,chunks/b OUT_DIR=results/v2 python3 ace_shade.py

Muhit o'zgaruvchilari:
    FUNCS    vergul bilan (standart: barcha 12 ta)
    DIMS     vergul bilan (standart: 10,20)
    ALGS     vergul bilan (standart: reestrdagi 8 ta)
    RUNS     mustaqil run soni (standart: 30)
    SUITE    ablation -> B4 variantlari; aks holda asosiy ro'yxat
    QUICK    1 -> byudjet 20x kichraytiriladi (tez tekshiruv)
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aceshade.core import Tracker
from aceshade.benchmarks import make_problem, CEC2022_BUDGET, CEC2022_FUNCS
from aceshade.registry import ALGORITHMS, ABLATION, TARGET
from aceshade.analysis import analyse

SEED_BASE = 42
N_POINTS = 100


def _env_list(name, default):
    raw = os.environ.get(name, "")
    return [s.strip() for s in raw.split(",") if s.strip()] or list(default)


def run_task(alg_name, alg, func, dim, run_id, budget):
    # Bir xil (dim, run) uchun barcha algoritmlarga bir xil seed:
    # taqqoslash juftlashtirilgan bo'ladi.
    np.random.seed(SEED_BASE + 1000 * dim + run_id)
    obj, lb, ub, _ = make_problem(func, dim, 2022)
    tr = Tracker(obj, budget, N_POINTS)
    try:
        alg(tr, dim, (lb, ub), budget)
    except Exception as exc:                      # bitta run butun ishni to'xtatmasin
        print(f"[!] {alg_name} {func} {dim}D run {run_id}: {type(exc).__name__}: {exc}")
    return ({"Algorithm": alg_name, "Function": func, "Dimension": dim,
             "Run": run_id, "Best": float(tr.best_f)},
            (alg_name, func, dim, run_id), tr.finalize())


def main():
    merge = [d.strip().rstrip("/") for d in os.environ.get("MERGE", "").split(",") if d.strip()]
    out_dir = os.environ.get("OUT_DIR", "results/v2")
    os.makedirs(out_dir, exist_ok=True)

    if merge:
        frames, curves = [], {}
        for d in merge:
            p = f"{d}/raw_data/full_raw_results.csv"
            if os.path.exists(p):
                frames.append(pd.read_csv(p))
            npz = f"{d}/raw_data/curves.npz"
            if os.path.exists(npz):
                z = np.load(npz, allow_pickle=True)
                for k in z.files:
                    a, f, dd, r = k.split("|")
                    curves.setdefault((a, f, int(dd)), []).append(z[k])
        if not frames:
            sys.exit("[!] birlashtirish uchun bo'lak topilmadi")
        df = pd.concat(frames, ignore_index=True)
        print(f"[*] {len(merge)} bo'lak, {len(df)} qator birlashtirildi")
        med = {k: np.nanmedian(np.vstack(v), axis=0) for k, v in curves.items()}
        analyse(df, TARGET, out_dir, curves={k: v for k, v in curves.items()})
        return

    registry = ABLATION if os.environ.get("SUITE") == "ablation" else ALGORITHMS
    funcs = _env_list("FUNCS", CEC2022_FUNCS)
    dims = [int(x) for x in _env_list("DIMS", (10, 20))]
    algs = _env_list("ALGS", registry.keys())
    runs = int(os.environ.get("RUNS", "30"))
    quick = os.environ.get("QUICK") == "1"

    tasks = []
    for func in funcs:
        for dim in dims:
            budget = CEC2022_BUDGET[dim] // (20 if quick else 1)
            for name in algs:
                if name not in registry:
                    sys.exit(f"[!] noma'lum algoritm: {name}")
                for r in range(runs):
                    tasks.append((name, registry[name], func, dim, r, budget))

    print(f"[*] {len(tasks)} ta vazifa | funksiyalar={funcs} o'lchamlar={dims} "
          f"algoritmlar={len(algs)} runlar={runs}{' (QUICK)' if quick else ''}")
    t0 = time.time()
    out = Parallel(n_jobs=-1, verbose=5)(delayed(run_task)(*t) for t in tasks)
    print(f"[*] {time.time() - t0:.0f} s")

    rows = [o[0] for o in out]
    os.makedirs(f"{out_dir}/raw_data", exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{out_dir}/raw_data/full_raw_results.csv", index=False)
    np.savez_compressed(f"{out_dir}/raw_data/curves.npz",
                        **{f"{a}|{f}|{d}|{r}": c for (a, f, d, r), c in
                           ((o[1], o[2]) for o in out)})
    print(f"[*] bo'lak saqlandi: {out_dir}/raw_data/")


if __name__ == "__main__":
    main()
