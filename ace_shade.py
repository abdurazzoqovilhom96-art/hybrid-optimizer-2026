#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ACE-SHADE: Adaptive Covariance-Ensemble SHADE.

Yagona fayl: benchmarklar, algoritm, raqobatchilar, sozlash, tahlil va
hisobot. Butun eksperiment bitta buyruq bilan boshidan oxirigacha o'tadi.

    python3 ace_shade.py                 # to'liq quvur
    STAGE=tune    python3 ace_shade.py   # faqat sozlash
    STAGE=run     python3 ace_shade.py   # faqat baholash
    STAGE=ablate  python3 ace_shade.py   # faqat ablatsiya
    STAGE=report  python3 ace_shade.py   # faqat tahlil va hisobot
    SELFTEST=1    python3 ace_shade.py   # birlik testlari

Shardlash (GitHub Actions): FUNCS / DIMS / ALGS / CFG / OUT_DIR / MERGE
muhit o'zgaruvchilari bilan. Hisoblash bir necha mashinaga taqsimlanadi,
mantiq esa shu bitta faylda qoladi.

ILMIY DIZAYN
------------
Ikki hissa:

  N1  TO'PLAMLI KOVARIATSIYA. Oldingi versiya har avlodda
      `cov(pop[:N/2])` olardi: `n = 3D` namunadan `D` o'lchamda
      baholangan kovariatsiyaning spektral xatosi `O(sqrt(D/n))`, ya'ni
      bu yerda ~58%. Kichik xos qiymatlarga mos yo'nalishlar amalda
      shovqin bo'lib qolardi - holbuki yomon shartlangan masalada aynan
      o'sha yo'nalishlar hal qiluvchi (o'lchangan: RotatedElliptic 30D
      da LSHADE-cnEpSin dan 949x mag'lubiyat).

      O'rniga CMA-ES uslubidagi to'plamli baholagich qo'yiladi: rank-1
      (evolyutsiya yo'li) + rank-mu yangilanish. Uning samarali namuna
      hajmi `O(D^2)` avlod tartibida o'sadi, shuning uchun o'lcham
      ortishi bilan buzilmaydi.

      Bitta `C` ikki iste'molchiga xizmat qiladi:
        (a) crossover eigen-bazisi (LSHADE-cnEpSin g'oyasi),
        (b) CMA taqsimotidan namuna olish tarmog'i (SPACMA g'oyasi).
      Alohida "quyruq" fazasi va uning tarqalish chegarasi olib
      tashlanadi - ulush uzluksiz moslashadi.

  N2  PARAMETR REJIMI ANSAMBLI. jSO va L-SHADE xotira rejimlari teskari
      bog'liq (o'lchangan, Schwefel 30D: jSO 1.18e+02, L-SHADE
      3.82e-04 - besh tartib farq). Qat'iy tanlov aralash to'plamda har
      doim suboptimal, shuning uchun ikkala bank olib boriladi va
      ulushi yaxshilanish MIQDORI krediti bilan moslashadi.

Saqlangan yadro: current-to-pbest-w/1 + arxiv (SHADE/L-SHADE), rank
asosidagi tanlov bosimi (LSHADE-RSP), chiziqli populyatsiya kamayishi
(L-SHADE), midpoint-target chegara tuzatishi.

METODOLOGIK QOIDALAR
--------------------
1. Sozlash CEC-2017 da, baholash CEC-2022 da. To'plamlar kesishmaydi.
2. Sozlash byudjeti baholash byudjeti bilan AYNAN bir xil rejimda.
3. Parametrlar sozlashdan keyin muzlatiladi va `frozen_params.json` ga
   yoziladi; baholash faqat o'sha fayldan o'qiydi.
4. Barcha algoritmlar bir xil: seed, chegara ishlovi, baholash
   hisoblagichi, to'xtash sharti.
5. Xato `< 1e-8` -> 0 (CEC konventsiyasi) faqat tahlilda; xom qiymatlar
   o'zgarishsiz saqlanadi.
6. Natijalar bezalmaydi: mag'lubiyat raqami bilan yoziladi.
"""

import inspect
import itertools
import json
import math
import os
import sys
import time
from functools import partial

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import friedmanchisquare, mannwhitneyu, norm

# ==============================================================================
# 0. TEZLASHTIRISH: opfunu ning sekin ildiz funksiyasi
# ==============================================================================
# `opfunu.utils.operator.katsuura_func` Python sikli bilan yozilgan va har
# qadamda 32 ta skalyar uchun `np.sum` chaqiradi: D=20 da 2235 us. U F7,
# F8, F11, F12 ichida ishlatiladi va butun eksperiment narxini belgilaydi.
# Quyidagi variant AYNAN bir xil matematikani numpy bilan bajaradi (33 us,
# 66x). Farq 1.6e-15, ya'ni qo'shmaqiymat yaxlitlash darajasida; SELFTEST
# buni 2000 tasodifiy nuqtada tekshiradi.
#
# `schaffer_f7_func` ham siklga ega, lekin atigi 23 us va uni
# vektorlashtirish yig'indi tartibi tufayli 2.6e-12 chetlanish beradi -
# tejash kichik, chetlanish bepul emas, shuning uchun tegilmaydi.

def _katsuura_fast(x):
    x = np.asarray(x, dtype=float).ravel()
    ndim = x.size
    p = 2.0 ** np.arange(1, 33)
    px = p * x[:, None]
    temp = (np.abs(px - np.round(px)) / p).sum(axis=1)
    idx = np.arange(1, ndim + 1)
    return (np.prod((1.0 + idx * temp) ** (10.0 / ndim ** 1.2)) - 1.0) * 10.0 / ndim ** 2


_KATSUURA_ORIGINAL = None


def _patch_opfunu():
    """Almashtirishni o'rnatadi. Bir necha marta chaqirilishi xavfsiz."""
    global _KATSUURA_ORIGINAL
    from opfunu.utils import operator
    if _KATSUURA_ORIGINAL is None:
        _KATSUURA_ORIGINAL = operator.katsuura_func
    operator.katsuura_func = _katsuura_fast


_patch_opfunu()

# ==============================================================================
# 1. BENCHMARKLAR
# ==============================================================================
# Ikkala to'plam ham `opfunu` orqali RASMIY siljitish va aylantirish
# ma'lumotlari bilan yuklanadi - sintetik generatsiya qilinmaydi.
# Optimumda har bir funksiya rasmiy `f_bias` ni qaytaradi, shuning uchun
# xato `f(x) - f_bias` ko'rinishida hisoblanadi (SELFTEST tekshiradi).

CEC2022_BUDGET = {10: 200_000, 20: 1_000_000}     # rasmiy protokol
CEC2022_DIMS = (10, 20)
CEC2022_FUNCS = tuple(f"F{i}" for i in range(1, 13))
CEC2022_RUNS = 30                                  # rasmiy protokol

CEC2022_CATEGORIES = {
    "bir ekstremumli": ("F1",),
    "asosiy ko'p ekstremumli": ("F2", "F3", "F4", "F5"),
    "gibrid": ("F6", "F7", "F8"),
    "kompozitsiya": ("F9", "F10", "F11", "F12"),
}

# Sozlash to'plami: CEC-2017. CEC-2022 bilan bitta ham umumiy funksiya
# yo'q. F2 rasmiy ravishda chiqarib tashlangan (beqaror xulq).
CEC2017_TUNING_FUNCS = ("F1", "F3", "F4", "F5", "F6", "F7", "F9", "F10", "F11", "F15")

PRECISION_FLOOR = float(os.environ.get("PRECISION_FLOOR", "1e-8"))
SEED_BASE = 42
N_CURVE_POINTS = 100


def make_problem(name, dim, year):
    """`(obj, lb, ub, f_bias)`. `obj(x)` XATO qiymatini qaytaradi."""
    if year == 2022:
        from opfunu.cec_based import cec2022 as mod
    elif year == 2017:
        from opfunu.cec_based import cec2017 as mod
    else:
        raise ValueError(f"qo'llab-quvvatlanmaydigan to'plam: {year}")
    prob = getattr(mod, f"{name}{year}")(ndim=dim)
    bias = float(prob.f_global)
    lb = np.asarray(prob.lb, dtype=float)
    ub = np.asarray(prob.ub, dtype=float)

    def obj(x):
        v = float(prob.evaluate(np.asarray(x, dtype=float))) - bias
        # Sonli qo'riqchi: NaN/inf optimizatsiyani buzmasligi kerak.
        return v if np.isfinite(v) else np.inf

    return obj, lb, ub, bias


def category_of(func_name):
    for cat, members in CEC2022_CATEGORIES.items():
        if func_name in members:
            return cat
    return "noma'lum"


# ==============================================================================
# 2. YADRO: baholash hisoblagichi va DE yordamchilari
# ==============================================================================

class Tracker:
    """Yagona FES hisoblagichi va best-so-far egri chizig'i.

    Barcha algoritmlar shu orqali baholaydi, shuning uchun byudjet hisobi
    ular orasida aynan bir xil. `max_fes` dan keyingi baholashlar
    hisobga olinmaydi (SELFTEST tekshiradi).
    """

    __slots__ = ("func", "max_fes", "fes", "best_f", "best_x",
                 "checkpoints", "curve", "k")

    def __init__(self, func, max_fes, n_points=N_CURVE_POINTS):
        self.func, self.max_fes = func, max_fes
        self.fes = 0
        self.best_f, self.best_x = np.inf, None
        self.checkpoints = np.linspace(max_fes / n_points, max_fes, n_points).astype(np.int64)
        self.curve = np.full(n_points, np.nan)
        self.k = 0

    def __call__(self, x):
        f = self.func(x)
        self.fes += 1
        if self.fes <= self.max_fes and f < self.best_f:
            self.best_f = f
            self.best_x = np.array(x, dtype=float).copy()
        while self.k < len(self.checkpoints) and self.fes >= self.checkpoints[self.k]:
            self.curve[self.k] = self.best_f
            self.k += 1
        return f

    def finalize(self):
        self.curve[self.k:] = self.best_f
        return self.curve


def sample_F(mu):
    """Cauchy(mu, 0.1) dan F; F <= 0 qayta tanlanadi, 1.0 bilan chegaralanadi."""
    F = mu + 0.1 * np.random.standard_cauchy(len(mu))
    bad = F <= 0
    guard = 0
    while np.any(bad) and guard < 100:
        F[bad] = mu[bad] + 0.1 * np.random.standard_cauchy(int(bad.sum()))
        bad = F <= 0
        guard += 1
    F[bad] = 0.5                      # patologik holat uchun zaxira
    return np.minimum(F, 1.0)


def lehmer(vals, w):
    """Vaznlangan Lehmer o'rtachasi (SHADE xotira yangilanishi)."""
    den = np.sum(w * vals)
    return np.sum(w * vals ** 2) / den if den > 0 else -1.0


def pick_r2(n_union, idx, r1):
    """r2 ni P u A dan tanlash: r2 != i va r2 != r1."""
    n = len(idx)
    r2 = np.random.randint(0, n_union, n)
    clash = (r2 == idx) | (r2 == r1)
    guard = 0
    while np.any(clash) and guard < 100:
        r2[clash] = np.random.randint(0, n_union, int(clash.sum()))
        clash = (r2 == idx) | (r2 == r1)
        guard += 1
    return r2


def midpoint(U, pop, lb, ub):
    """Midpoint-target chegara tuzatishi. Barcha algoritmlarda bir xil."""
    low, high = U < lb, U > ub
    U[low] = ((lb + pop) / 2.0)[low]
    U[high] = ((ub + pop) / 2.0)[high]
    return U


def arch_push(archive, losers, arc_max):
    if arc_max <= 0:
        return archive
    archive = np.vstack([archive, losers]) if len(archive) else losers.copy()
    if len(archive) > arc_max:
        archive = archive[np.random.choice(len(archive), arc_max, replace=False)]
    return archive


def rsp_ranks(pop_size, idx, k_rsp):
    """Rank asosidagi tanlov bosimi (LSHADE-RSP): r1 indekslari."""
    rk = k_rsp * (pop_size - idx) / pop_size + 1.0
    pr = rk / rk.sum()
    r1 = np.random.choice(pop_size, pop_size, p=pr)
    same = r1 == idx
    guard = 0
    while np.any(same) and guard < 100:
        r1[same] = np.random.choice(pop_size, int(same.sum()), p=pr)
        same = r1 == idx
        guard += 1
    return r1


# ==============================================================================
# 3. ACE-SHADE
# ==============================================================================

def _credit_share(groups, gain, gain_total, cred, eta, lo=0.02, hi=0.98):
    """Yaxshilanish MIQDORI krediti (FIR). Uchala moslashuvchan ulush uchun
    bir xil qoida: guruhning umumiy yutuqdagi ulushi uning hajm ulushiga
    bo'linadi, so'ng eksponensial o'rtachalanadi.

    `groups` - ikkita mantiqiy niqob; qaytadigan qiymat ikkinchisining ulushi.
    """
    for j, mask in enumerate(groups):
        share = float(mask.mean())
        if share > 0:
            fir = (gain[mask].sum() / gain_total / share) if gain_total > 0 else 0.0
            cred[j] = (1.0 - eta) * cred[j] + eta * fir
    s = cred.sum()
    if not np.isfinite(s) or s <= 0:
        return 0.5
    return float(np.clip(cred[1] / s, lo, hi))


def ace_shade(obj_func, dim, bounds, max_fes,
              R_N=10.0, ETA=0.1,
              N_MIN=4, H_SIZE=6, ARC_RATE=2.6, K_RSP=3.0,
              P_MAX=0.25, P_MIN_RATE=0.125, P_CMA_CAP=0.5,
              USE_COV=True, USE_ENSEMBLE=True, CMU_SOURCE="accepted",
              on_generation=None):
    """ACE-SHADE.

    `R_N` va `ETA` sozlanadi (CEC-2017 da), qolgani manba usullaridan
    o'zgarishsiz olinadi. `USE_COV` / `USE_ENSEMBLE` / `CMU_SOURCE`
    ablatsiya uchun. `on_generation` - diagnostika uchun kuzatuv nuqtasi,
    qidiruvga ta'sir qilmaydi.
    """
    lb, ub = bounds
    span = float(np.mean(ub - lb))
    sigma_floor = 1e-14 * span
    N_init = max(N_MIN * 2, int(round(R_N * dim)))
    pop_size = N_init

    pop = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.array([obj_func(ind) for ind in pop], dtype=float)
    fes = pop_size

    # --- N2: ikki xotira banki ------------------------------------------------
    # 0-bank (jSO):     M_F=0.3, M_CR=0.8, doimiy terminal katak, F cheklovlari.
    # 1-bank (L-SHADE): M_F=M_CR=0.5, barcha kataklar yangilanadi, cheklovsiz.
    M_F = np.stack([np.full(H_SIZE, 0.3), np.full(H_SIZE, 0.5)])
    M_CR = np.stack([np.full(H_SIZE, 0.8), np.full(H_SIZE, 0.5)])
    M_F[0, -1], M_CR[0, -1] = 0.9, 0.9
    n_upd = (H_SIZE - 1, H_SIZE)
    k_mem = [0, 0]
    p_bank, bank_credit = 0.5, np.array([0.5, 0.5])

    # --- N1: to'plamli kovariatsiya ------------------------------------------
    C = np.eye(dim)
    B = np.eye(dim)
    d_eig = np.ones(dim)
    p_c = np.zeros(dim)
    m = pop.mean(axis=0)
    eigeneval = 0.0
    p_eig, eig_credit = 0.5, np.array([0.5, 0.5])
    p_cma, cma_credit = 0.1, np.array([0.9, 0.1])

    archive = np.empty((0, dim))

    while fes < max_fes:
        t = fes / max_fes
        order = np.argsort(fitness)
        pop, fitness = pop[order], fitness[order]
        idx = np.arange(pop_size)

        # CMA konstantalari joriy populyatsiya hajmiga qarab
        mu = max(2, pop_size // 2)
        w_mu = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w_mu /= w_mu.sum()
        mu_eff = 1.0 / np.sum(w_mu ** 2)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2.0 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Qadam hajmi populyatsiya tarqalishidan, CSA emas: avlodlarning bir
        # qismi DE mutatsiyasidan keladi, CSA esa Gauss namunasini faraz
        # qiladi va aralash manbada beqaror bo'ladi.
        sigma = max(float(np.mean(np.std(pop, axis=0))), sigma_floor)

        # Dangasa xos yoyilma: O(D^3) narx O(D^2) ga amortizatsiya qilinadi.
        if USE_COV and (fes - eigeneval) > 1.0 / (10.0 * dim * (c_1 + c_mu)):
            eigeneval = fes
            C = (C + C.T) / 2.0
            if np.all(np.isfinite(C)):
                try:
                    vals, vecs = np.linalg.eigh(C)
                    vmax = float(vals.max())
                    if np.isfinite(vmax) and vmax > 0:
                        vals = np.maximum(vals, 1e-20 * vmax)
                        B, d_eig = vecs, np.sqrt(vals)
                    else:
                        C, B, d_eig = np.eye(dim), np.eye(dim), np.ones(dim)
                except np.linalg.LinAlgError:
                    C, B, d_eig = np.eye(dim), np.eye(dim), np.ones(dim)
            else:
                C, B, d_eig = np.eye(dim), np.eye(dim), np.ones(dim)

        # --- Parametrlar: bank tanlovi + success-history ----------------------
        pb = p_bank if USE_ENSEMBLE else 1.0       # ansamblsiz: faqat 0-bank
        bank = (np.random.rand(pop_size) >= pb).astype(np.int64)
        r = np.random.randint(0, H_SIZE, pop_size)
        mu_CR, mu_F = M_CR[bank, r], M_F[bank, r]
        CR = np.clip(np.random.normal(mu_CR, 0.1), 0.0, 1.0)
        CR[mu_CR < 0] = 0.0                        # L-SHADE terminal qiymati
        F = sample_F(mu_F)
        jso = bank == 0
        if np.any(jso) and t < 0.6:
            F[jso] = np.minimum(F[jso], 0.7)       # jSO cheklovi
        Fw = F.copy()
        if np.any(jso):
            Fw[jso] *= (0.7 if t < 0.2 else 0.8 if t < 0.4 else 1.2)

        # --- Tarmoq taqsimoti -------------------------------------------------
        use_cma = (np.random.rand(pop_size) < p_cma) if USE_COV \
            else np.zeros(pop_size, dtype=bool)

        # --- DE tarmog'i: current-to-pbest-w/1 + arxiv ------------------------
        p_num = max(2, int(round((P_MAX - (P_MAX - P_MIN_RATE) * t) * pop_size)))
        p_num = min(p_num, pop_size)
        x_pbest = pop[np.random.randint(0, p_num, pop_size)]
        r1 = rsp_ranks(pop_size, idx, K_RSP)
        union_pop = np.vstack([pop, archive]) if len(archive) else pop
        r2 = pick_r2(len(union_pop), idx, r1)
        V = pop + Fw[:, None] * (x_pbest - pop) + F[:, None] * (pop[r1] - union_pop[r2])

        cross = np.random.rand(pop_size, dim) < CR[:, None]
        cross[idx, np.random.randint(0, dim, pop_size)] = True   # kamida bitta o'lcham
        U = np.where(cross, V, pop)

        # (a) eigen-bazisda crossover. Satr vektorlar uchun `x @ B` - bu
        #     `B^T x` proyeksiyasi, `@ B.T` esa teskari almashtirish.
        use_eig = np.zeros(pop_size, dtype=bool)
        if USE_COV:
            use_eig = np.random.rand(pop_size) < p_eig
            sel = use_eig & ~use_cma
            if np.any(sel):
                U[sel] = np.where(cross[sel], V[sel] @ B, pop[sel] @ B) @ B.T

        # (b) CMA taqsimotidan namuna
        if np.any(use_cma):
            z = np.random.randn(int(use_cma.sum()), dim)
            U[use_cma] = m + sigma * (z * d_eig) @ B.T

        U = midpoint(U, pop, lb, ub)
        if not np.all(np.isfinite(U)):
            U = np.where(np.isfinite(U), U, pop)   # sonli qo'riqchi

        n_eval = min(pop_size, max_fes - fes)
        fit_U = np.full(pop_size, np.inf)
        for i in range(n_eval):
            fit_U[i] = obj_func(U[i])
        fes += n_eval

        improved = fit_U < fitness
        accept = fit_U <= fitness
        gain = np.where(improved, np.maximum(fitness - fit_U, 0.0), 0.0)
        gain = np.where(np.isfinite(gain), gain, 0.0)
        gain_total = float(gain.sum())

        # --- Xotira yangilanishi: har bank alohida ----------------------------
        if np.any(improved):
            archive = arch_push(archive, pop[improved], int(round(ARC_RATE * pop_size)))
            for b in (0, 1):
                mb = improved & (bank == b)
                if not np.any(mb):
                    continue
                df = fitness[mb] - fit_U[mb]
                s_df = df.sum()
                if not np.isfinite(s_df) or s_df <= 0:
                    continue
                ww = df / s_df
                mf = lehmer(F[mb], ww)
                mcr = -1.0 if (M_CR[b, k_mem[b]] == -1 or np.sum(ww * CR[mb]) == 0) \
                    else lehmer(CR[mb], ww)
                if b == 1:                          # L-SHADE: to'g'ridan almashtirish
                    M_F[b, k_mem[b]], M_CR[b, k_mem[b]] = mf, mcr
                else:                               # jSO: avvalgisi bilan o'rtachalash
                    M_F[b, k_mem[b]] = (M_F[b, k_mem[b]] + mf) / 2.0
                    M_CR[b, k_mem[b]] = -1.0 if mcr == -1 else (M_CR[b, k_mem[b]] + mcr) / 2.0
                k_mem[b] = (k_mem[b] + 1) % n_upd[b]

        # --- Birlashgan kredit arbitraji --------------------------------------
        if USE_ENSEMBLE:
            p_bank = 1.0 - _credit_share([bank == 0, bank == 1], gain, gain_total,
                                         bank_credit, ETA)
        if USE_COV:
            p_eig = _credit_share([~use_eig, use_eig], gain, gain_total, eig_credit, ETA)
            p_cma = min(_credit_share([~use_cma, use_cma], gain, gain_total,
                                      cma_credit, ETA), P_CMA_CAP)

        # --- N1: kovariatsiyani yangilash -------------------------------------
        if USE_COV:
            if CMU_SOURCE == "accepted":
                src = np.where(accept & np.isfinite(fit_U))[0]
                cand, cfit = (U[src], fit_U[src]) if len(src) else (None, None)
            else:
                cand, cfit = pop, fitness
            if cand is not None and len(cand) >= 2:
                k = min(mu, len(cand))
                best = np.argsort(cfit)[:k]
                wk = np.log(k + 0.5) - np.log(np.arange(1, k + 1))
                wk /= wk.sum()
                m_old = m
                m = wk @ cand[best]
                y_w = (m - m_old) / sigma
                p_c = (1 - c_c) * p_c + math.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
                Y = (cand[best] - m_old) / sigma
                C_new = ((1 - c_1 - c_mu) * C
                         + c_1 * np.outer(p_c, p_c)
                         + c_mu * (Y.T * wk) @ Y)
                if np.all(np.isfinite(C_new)):
                    C = C_new

        pop[accept], fitness[accept] = U[accept], fit_U[accept]

        # --- Chiziqli populyatsiya kamayishi ----------------------------------
        new_size = max(N_MIN, int(round(N_init + (N_MIN - N_init) * fes / max_fes)))
        if new_size < pop_size:
            keep = np.argsort(fitness)[:new_size]
            pop, fitness = pop[keep], fitness[keep]
            pop_size = new_size
        arc_max = int(round(ARC_RATE * pop_size))
        if len(archive) > arc_max:
            archive = archive[np.random.choice(len(archive), arc_max, replace=False)]

        if on_generation is not None:
            on_generation({"fes": fes, "pop_size": pop_size, "sigma": sigma,
                           "p_bank": p_bank, "p_eig": p_eig, "p_cma": p_cma,
                           "best": float(np.min(fitness))})


# ==============================================================================
# 4. RAQOBATCHILAR
# ==============================================================================
# Ular ikki guruhga bo'linadi va HISOBOTDA HAM SHUNDAY AJRATILADI:
#
#   (a) TASDIQLANGAN TO'RTLIK - L-SHADE, jSO, LSHADE-cnEpSin, CMA-ES.
#       Bular oldingi versiyada (V1) ishlatilgan va o'sha yerda sinovdan
#       o'tgan. Har biri ACE-SHADE yaxshilashni da'vo qilayotgan aniq
#       narsaning manbai: cnEpSin va CMA-ES kovariatsiya da'vosiga,
#       jSO va L-SHADE esa rejim ansambli da'vosiga qarshi turadi.
#       ASOSIY DA'VO shu to'rtlikka nisbatan qo'yiladi.
#
#   (b) QAYTA AMALGA OSHIRILGANLAR - LSHADE-RSP, LSHADE-SPACMA,
#       NL-SHADE-RSP. Bu muhitda asl maqolalarga va mualliflar kodiga
#       kirish yo'q (tarmoq siyosati bloklaydi), shuning uchun ular
#       e'lon qilingan MEXANIZM ta'rifi bo'yicha yozilgan. Ba'zi
#       ikkilamchi tanlovlar (populyatsiya hajmi formulasi, konstantalar)
#       tasdiqlanmagan va kodda `# NOANIQ:` bilan belgilangan.
#       IKKILAMCHI DA'VO faqat shu guruhga nisbatan va cheklov ochiq
#       aytilgan holda qo'yiladi.
#
# Bu ajratish ataylab: qayta amalga oshirilgan raqib e'lon qilingan
# natijasidan past ishlasa, unga nisbatan g'alaba hech narsani
# isbotlamaydi. Tasdiqlangan to'rtlikka nisbatan g'alaba esa isbotlaydi.

def cma_core(obj_func, dim, bounds, fes, max_fes, xmean, sigma, restart="uniform"):
    """(mu/mu_w, lambda)-CMA-ES yadrosi (Hansen & Ostermeier, 2001).
    Ham mustaqil raqobatchi, ham CBA-SHADE ning yakuniy aniqlashtirishi uchun.
    restart="uniform" - tarqalish yo'qolganda butun sohadan qayta boshlaydi
    (mustaqil CMA-ES); restart="local" - eng yaxshi nuqta atrofida, kichraygan
    qadam bilan qayta boshlaydi (lokal aniqlashtirish rejimi)."""
    lb, ub = bounds
    lam = 4 + int(3 * math.log(dim))
    mu = lam // 2
    w = math.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / np.sum(w ** 2)
    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, math.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
    chiN = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
    pc, psig = np.zeros(dim), np.zeros(dim)
    B, D, C, invsqrtC = np.eye(dim), np.ones(dim), np.eye(dim), np.eye(dim)
    gen, eigeneval, restarts = 0, fes, 0
    sigma0 = sigma
    x_best, f_best = xmean.copy(), np.inf
    f_scale = 1.0                                    # jarima miqyosi (moslashuvchan)
    while fes < max_fes:
        gen += 1
        Y = np.random.randn(lam, dim) @ (B * D).T
        X = xmean + sigma * Y
        Xc = np.clip(X, lb, ub)
        # Chegaradan chiqish jarimasi: masofa qidiruv oralig'iga normallashtiriladi
        # va joriy avlodning fitness tarqalishiga (IQR) moslanadi, aks holda
        # jarima funksiya miqyosiga bog'liq holda yo juda kuchli, yo ta'sirsiz bo'ladi.
        pen = np.sum(((X - Xc) / (ub - lb)) ** 2, axis=1)
        f_raw = np.full(lam, np.inf)
        for i in range(lam):
            if fes >= max_fes:
                break
            f_raw[i] = obj_func(Xc[i])
            fes += 1
        f = f_raw + f_scale * pen
        ok = f_raw[np.isfinite(f_raw)]
        if ok.size >= 2:
            iqr = float(np.percentile(ok, 75) - np.percentile(ok, 25))
            if iqr > 0:
                f_scale = 0.5 * f_scale + 0.5 * iqr
        srt = np.argsort(f)
        if f_raw[srt[0]] < f_best:
            f_best, x_best = f_raw[srt[0]], Xc[srt[0]].copy()
        Yo = Y[srt[:mu]]
        xmean = xmean + sigma * (w @ Yo)
        psig = (1 - cs) * psig + math.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ (w @ Yo))
        hsig = np.linalg.norm(psig) / math.sqrt(1 - (1 - cs) ** (2 * gen)) / chiN < 1.4 + 2 / (dim + 1)
        pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * (w @ Yo)
        C = ((1 - c1 - cmu) * C
             + c1 * (np.outer(pc, pc) + (not hsig) * cc * (2 - cc) * C)
             + cmu * (Yo.T * w) @ Yo)
        sigma *= math.exp((cs / damps) * (np.linalg.norm(psig) / chiN - 1))
        sigma = min(sigma, 1e3 * float(np.max(ub - lb)))
        degenerate = False
        if fes - eigeneval > lam / (c1 + cmu) / dim / 10:
            eigeneval = fes
            C = np.triu(C) + np.triu(C, 1).T
            try:
                Dv, B = np.linalg.eigh(C)
            except np.linalg.LinAlgError:
                degenerate = True
            else:
                if np.min(Dv) <= 0:
                    degenerate = True
                else:
                    D = np.sqrt(Dv)
                    invsqrtC = B @ np.diag(1.0 / D) @ B.T
        if degenerate or not np.isfinite(sigma) or sigma < 1e-16:
            restarts += 1
            if restarts > 100:
                return
            if restart == "local":
                xmean = x_best.copy()
                sigma = sigma0 * (0.5 ** restarts)
                if sigma < 1e-14:
                    return
            else:
                xmean = lb + np.random.rand(dim) * (ub - lb)
                sigma = 0.3 * float(np.mean(ub - lb))
            pc, psig = np.zeros(dim), np.zeros(dim)
            B, D, C, invsqrtC = np.eye(dim), np.ones(dim), np.eye(dim), np.eye(dim)
            gen, eigeneval = 0, fes

def baseline_LSHADE(obj_func, dim, bounds, max_fes):
    """Tanabe & Fukunaga (CEC 2014). SHADE + chiziqli populyatsiya kamayishi."""
    lb, ub = bounds
    N_init, N_min, H, p_rate, arc_rate = int(18 * dim), 4, 6, 0.11, 2.6
    pop_size = N_init
    pop = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.array([obj_func(x) for x in pop])
    fes = pop_size
    M_F, M_CR, k = np.full(H, 0.5), np.full(H, 0.5), 0
    archive = np.empty((0, dim))
    while fes < max_fes:
        order = np.argsort(fitness)
        pop, fitness = pop[order], fitness[order]
        idx = np.arange(pop_size)
        r = np.random.randint(0, H, pop_size)
        CR = np.clip(np.random.normal(M_CR[r], 0.1), 0.0, 1.0)
        CR[M_CR[r] < 0] = 0.0
        F = sample_F(M_F[r])
        p_num = max(2, int(round(p_rate * pop_size)))
        pbest = pop[np.random.randint(0, p_num, pop_size)]
        union = np.vstack([pop, archive]) if len(archive) else pop
        r1 = (idx + np.random.randint(1, pop_size, pop_size)) % pop_size
        r2 = pick_r2(len(union), idx, r1)
        Fc = F[:, None]
        V = pop + Fc * (pbest - pop) + Fc * (pop[r1] - union[r2])
        cross = np.random.rand(pop_size, dim) < CR[:, None]
        cross[idx, np.random.randint(0, dim, pop_size)] = True
        U = midpoint(np.where(cross, V, pop), pop, lb, ub)
        n_eval = min(pop_size, max_fes - fes)
        fit_U = np.full(pop_size, np.inf)
        for i in range(n_eval):
            fit_U[i] = obj_func(U[i])
        fes += n_eval
        imp = fit_U < fitness
        if np.any(imp):
            df = fitness[imp] - fit_U[imp]
            w = df / df.sum()
            archive = arch_push(archive, pop[imp], int(round(arc_rate * pop_size)))
            M_F[k] = lehmer(F[imp], w)
            M_CR[k] = -1.0 if (M_CR[k] == -1 or np.sum(w * CR[imp]) == 0) else lehmer(CR[imp], w)
            k = (k + 1) % H
        acc = fit_U <= fitness
        pop[acc], fitness[acc] = U[acc], fit_U[acc]
        new_size = max(N_min, int(round(N_init + (N_min - N_init) * fes / max_fes)))
        if new_size < pop_size:
            keep = np.argsort(fitness)[:new_size]
            pop, fitness, pop_size = pop[keep], fitness[keep], new_size
            archive = archive[:int(round(arc_rate * pop_size))]

def baseline_jSO(obj_func, dim, bounds, max_fes):
    """Brest, Maucec & Boskovic (CEC 2017). iL-SHADE ning takomillashgan varianti."""
    lb, ub = bounds
    N_init = max(10, int(round(25 * math.log(dim) * math.sqrt(dim))))
    N_min, H, arc_rate = 4, 5, 1.0
    pop_size = N_init
    pop = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.array([obj_func(x) for x in pop])
    fes = pop_size
    M_F, M_CR = np.full(H, 0.3), np.full(H, 0.8)
    M_F[-1], M_CR[-1] = 0.9, 0.9
    k = 0
    archive = np.empty((0, dim))
    while fes < max_fes:
        t = fes / max_fes
        order = np.argsort(fitness)
        pop, fitness = pop[order], fitness[order]
        idx = np.arange(pop_size)
        r = np.random.randint(0, H, pop_size)
        CR = np.clip(np.random.normal(M_CR[r], 0.1), 0.0, 1.0)
        CR[M_CR[r] < 0] = 0.0
        if t < 0.25:
            CR = np.maximum(CR, 0.7)
        elif t < 0.5:
            CR = np.maximum(CR, 0.6)
        F = sample_F(M_F[r])
        if t < 0.6:
            F = np.minimum(F, 0.7)
        Fw = F * (0.7 if t < 0.2 else 0.8 if t < 0.4 else 1.2)
        p_num = max(2, int(round(0.25 * (1.0 - 0.5 * t) * pop_size)))
        pbest = pop[np.random.randint(0, p_num, pop_size)]
        union = np.vstack([pop, archive]) if len(archive) else pop
        r1 = (idx + np.random.randint(1, pop_size, pop_size)) % pop_size
        r2 = pick_r2(len(union), idx, r1)
        Fc, Fwc = F[:, None], Fw[:, None]
        V = pop + Fwc * (pbest - pop) + Fc * (pop[r1] - union[r2])
        cross = np.random.rand(pop_size, dim) < CR[:, None]
        cross[idx, np.random.randint(0, dim, pop_size)] = True
        U = midpoint(np.where(cross, V, pop), pop, lb, ub)
        n_eval = min(pop_size, max_fes - fes)
        fit_U = np.full(pop_size, np.inf)
        for i in range(n_eval):
            fit_U[i] = obj_func(U[i])
        fes += n_eval
        imp = fit_U < fitness
        if np.any(imp):
            df = fitness[imp] - fit_U[imp]
            w = df / df.sum()
            archive = arch_push(archive, pop[imp], int(round(arc_rate * pop_size)))
            mf = lehmer(F[imp], w)
            mcr = -1.0 if (M_CR[k] == -1 or np.sum(w * CR[imp]) == 0) else lehmer(CR[imp], w)
            M_F[k] = (M_F[k] + mf) / 2.0
            M_CR[k] = -1.0 if mcr == -1 else (M_CR[k] + mcr) / 2.0
            k = (k + 1) % (H - 1)
        acc = fit_U <= fitness
        pop[acc], fitness[acc] = U[acc], fit_U[acc]
        new_size = max(N_min, int(round(N_init + (N_min - N_init) * fes / max_fes)))
        if new_size < pop_size:
            keep = np.argsort(fitness)[:new_size]
            pop, fitness, pop_size = pop[keep], fitness[keep], new_size
            archive = archive[:int(round(arc_rate * pop_size))]

def baseline_LSHADE_cnEpSin(obj_func, dim, bounds, max_fes):
    """Awad, Ali, Suganthan & Reynolds (CEC 2017). L-SHADE + ansambl sinusoidal
    F moslashuvi + kovariatsiya bazisidagi crossover."""
    lb, ub = bounds
    N_init, N_min, H, p_rate, arc_rate = int(18 * dim), 4, 5, 0.11, 1.4
    LP, ps, pc = 20, 0.5, 0.4
    pop_size = N_init
    pop = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.array([obj_func(x) for x in pop])
    fes = pop_size
    M_F, M_CR, M_freq, k = np.full(H, 0.5), np.full(H, 0.5), np.full(H, 0.5), 0
    archive = np.empty((0, dim))
    g, G_max = 0, max(1, int(max_fes / max(1, N_init)))
    hist1, hist2 = [], []
    while fes < max_fes:
        g += 1
        order = np.argsort(fitness)
        pop, fitness = pop[order], fitness[order]
        idx = np.arange(pop_size)
        r = np.random.randint(0, H, pop_size)
        CR = np.clip(np.random.normal(M_CR[r], 0.1), 0.0, 1.0)
        CR[M_CR[r] < 0] = 0.0
        cfg = np.zeros(pop_size, dtype=int)
        freq_i = None
        if fes < max_fes / 2:                       # birinchi yarmi: ansambl sinusoidal
            if len(hist1) >= LP:
                s1 = sum(a for a, _ in hist1[-LP:]) / max(1e-12, sum(b for _, b in hist1[-LP:]))
                s2 = sum(a for a, _ in hist2[-LP:]) / max(1e-12, sum(b for _, b in hist2[-LP:]))
                p1 = (s1 + 0.01) / (s1 + s2 + 0.02)
            else:
                p1 = 0.5
            cfg = (np.random.rand(pop_size) >= p1).astype(int)
            freq_i = M_freq[r] + 0.1 * np.random.standard_cauchy(pop_size)
            freq_i = np.where(freq_i <= 0, 0.5, np.minimum(freq_i, 1.0))
            F1 = 0.5 * (np.sin(2 * np.pi * 0.5 * g) * (G_max - g) / G_max + 1.0)
            F2 = 0.5 * (np.sin(2 * np.pi * freq_i * g) * g / G_max + 1.0)
            F = np.clip(np.where(cfg == 0, F1, F2), 0.05, 1.0)
        else:
            F = sample_F(M_F[r])
        p_num = max(2, int(round(p_rate * pop_size)))
        pbest = pop[np.random.randint(0, p_num, pop_size)]
        union = np.vstack([pop, archive]) if len(archive) else pop
        r1 = (idx + np.random.randint(1, pop_size, pop_size)) % pop_size
        r2 = pick_r2(len(union), idx, r1)
        Fc = F[:, None]
        V = pop + Fc * (pbest - pop) + Fc * (pop[r1] - union[r2])
        cross = np.random.rand(pop_size, dim) < CR[:, None]
        cross[idx, np.random.randint(0, dim, pop_size)] = True
        U = np.where(cross, V, pop)
        if np.random.rand() < pc:                   # kovariatsiya bazisida crossover
            C = np.cov(pop[:max(2, int(round(ps * pop_size)))], rowvar=False)
            if np.all(np.isfinite(C)):
                try:
                    _, Bc = np.linalg.eigh(C)
                    U = np.where(cross, V @ Bc, pop @ Bc) @ Bc.T
                except np.linalg.LinAlgError:
                    pass
        U = midpoint(U, pop, lb, ub)
        n_eval = min(pop_size, max_fes - fes)
        fit_U = np.full(pop_size, np.inf)
        for i in range(n_eval):
            fit_U[i] = obj_func(U[i])
        fes += n_eval
        imp = fit_U < fitness
        hist1.append((int(np.sum(imp & (cfg == 0))), int(np.sum(cfg == 0))))
        hist2.append((int(np.sum(imp & (cfg == 1))), int(np.sum(cfg == 1))))
        if np.any(imp):
            df = fitness[imp] - fit_U[imp]
            w = df / df.sum()
            archive = arch_push(archive, pop[imp], int(round(arc_rate * pop_size)))
            M_F[k] = lehmer(F[imp], w)
            M_CR[k] = -1.0 if (M_CR[k] == -1 or np.sum(w * CR[imp]) == 0) else lehmer(CR[imp], w)
            if freq_i is not None and np.any(imp & (cfg == 1)):
                wf = df[cfg[imp] == 1]
                if wf.sum() > 0:
                    M_freq[k] = lehmer(freq_i[imp][cfg[imp] == 1], wf / wf.sum())
            k = (k + 1) % H
        acc = fit_U <= fitness
        pop[acc], fitness[acc] = U[acc], fit_U[acc]
        new_size = max(N_min, int(round(N_init + (N_min - N_init) * fes / max_fes)))
        if new_size < pop_size:
            keep = np.argsort(fitness)[:new_size]
            pop, fitness, pop_size = pop[keep], fitness[keep], new_size
            archive = archive[:int(round(arc_rate * pop_size))]

def baseline_CMAES(obj_func, dim, bounds, max_fes):
    """Hansen & Ostermeier (2001), tarqalish yo'qolganda qayta ishga tushadi."""
    lb, ub = bounds
    x0 = lb + np.random.rand(dim) * (ub - lb)
    cma_core(obj_func, dim, bounds, 0, max_fes, x0,
              0.3 * float(np.mean(ub - lb)), restart="uniform")

def baseline_LSHADE_RSP(obj_func, dim, bounds, max_fes):
    """Stanovov, Akhmedova & Semenkin (CEC 2018).

    Asosiy hissa: r1 va r2 ni RANK asosidagi ehtimollik bilan tanlash
    (rank-based selective pressure), ya'ni yaxshiroq individlar donor
    sifatida ko'proq tanlanadi. Qolgani jSO ga yaqin: F/CR cheklovlari,
    Fw vaznlash, LPSR.
    """
    lb, ub = bounds
    N_init, N_min, H, arc_rate = int(18 * dim), 4, 5, 2.6   # NOANIQ: N_init formulasi
    k_rsp = 3.0                                             # NOANIQ: maqolada kG
    pop_size = N_init
    pop = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.array([obj_func(x) for x in pop])
    fes = pop_size
    M_F, M_CR = np.full(H, 0.3), np.full(H, 0.8)
    M_F[-1], M_CR[-1] = 0.9, 0.9
    k = 0
    archive = np.empty((0, dim))
    while fes < max_fes:
        t = fes / max_fes
        order = np.argsort(fitness)
        pop, fitness = pop[order], fitness[order]
        idx = np.arange(pop_size)
        r = np.random.randint(0, H, pop_size)
        CR = np.clip(np.random.normal(M_CR[r], 0.1), 0.0, 1.0)
        CR[M_CR[r] < 0] = 0.0
        if t < 0.25:
            CR = np.maximum(CR, 0.7)
        elif t < 0.5:
            CR = np.maximum(CR, 0.6)
        F = sample_F(M_F[r])
        if t < 0.6:
            F = np.minimum(F, 0.7)
        Fw = F * (0.7 if t < 0.2 else 0.8 if t < 0.4 else 1.2)
        p_num = max(2, int(round(0.25 * (1.0 - 0.5 * t) * pop_size)))
        pbest = pop[np.random.randint(0, p_num, pop_size)]
        r1 = rsp_ranks(pop_size, idx, k_rsp)                # <- RSP hissasi
        union = np.vstack([pop, archive]) if len(archive) else pop
        r2 = pick_r2(len(union), idx, r1)
        V = pop + Fw[:, None] * (pbest - pop) + F[:, None] * (pop[r1] - union[r2])
        cross = np.random.rand(pop_size, dim) < CR[:, None]
        cross[idx, np.random.randint(0, dim, pop_size)] = True
        U = midpoint(np.where(cross, V, pop), pop, lb, ub)
        n_eval = min(pop_size, max_fes - fes)
        fit_U = np.full(pop_size, np.inf)
        for i in range(n_eval):
            fit_U[i] = obj_func(U[i])
        fes += n_eval
        imp = fit_U < fitness
        if np.any(imp):
            df = fitness[imp] - fit_U[imp]
            w = df / df.sum()
            archive = arch_push(archive, pop[imp], int(round(arc_rate * pop_size)))
            mf = lehmer(F[imp], w)
            mcr = -1.0 if (M_CR[k] == -1 or np.sum(w * CR[imp]) == 0) else lehmer(CR[imp], w)
            M_F[k] = (M_F[k] + mf) / 2.0
            M_CR[k] = -1.0 if mcr == -1 else (M_CR[k] + mcr) / 2.0
            k = (k + 1) % (H - 1)
        acc = fit_U <= fitness
        pop[acc], fitness[acc] = U[acc], fit_U[acc]
        new_size = max(N_min, int(round(N_init + (N_min - N_init) * fes / max_fes)))
        if new_size < pop_size:
            keep = np.argsort(fitness)[:new_size]
            pop, fitness, pop_size = pop[keep], fitness[keep], new_size
            archive = archive[:int(round(arc_rate * pop_size))]

def baseline_LSHADE_SPACMA(obj_func, dim, bounds, max_fes):
    """Mohamed, Hadi, Fahmy & Nasr (CEC 2017).

    Asosiy hissa: populyatsiya ikki sinfga bo'linadi - birinchi sinf
    L-SHADE mutatsiyasi bilan, ikkinchi sinf CMA-ES taqsimotidan namuna
    bilan ishlaydi. Sinf ehtimolligi (FCP) muvaffaqiyat asosida
    moslashadi. Bu ACE-SHADE ning eng yaqin gibrid raqibi: farq shundaki,
    SPACMA da CMA holati alohida olib boriladi va aralashtirish
    muvaffaqiyat SONIGA, bizda esa yaxshilanish MIQDORIGA tayanadi.
    """
    lb, ub = bounds
    N_init, N_min, H, arc_rate = int(18 * dim), 4, 6, 2.6
    fcp, fcp_lr = 0.5, 0.8                       # NOANIQ: o'rganish tezligi
    pop_size = N_init
    pop = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.array([obj_func(x) for x in pop])
    fes = pop_size
    M_F, M_CR, k = np.full(H, 0.5), np.full(H, 0.5), 0
    archive = np.empty((0, dim))
    # CMA holati (soddalashtirilgan: populyatsiya tarqalishi va kovariatsiyasi)
    xmean = pop.mean(axis=0)
    C = np.eye(dim)
    B, d_eig = np.eye(dim), np.ones(dim)
    while fes < max_fes:
        order = np.argsort(fitness)
        pop, fitness = pop[order], fitness[order]
        idx = np.arange(pop_size)
        sigma = max(float(np.mean(np.std(pop, axis=0))), 1e-14)
        try:
            vals, B = np.linalg.eigh((C + C.T) / 2.0)
            d_eig = np.sqrt(np.maximum(vals, 1e-20 * max(vals.max(), 1e-300)))
        except np.linalg.LinAlgError:
            B, d_eig = np.eye(dim), np.ones(dim)
        r = np.random.randint(0, H, pop_size)
        CR = np.clip(np.random.normal(M_CR[r], 0.1), 0.0, 1.0)
        CR[M_CR[r] < 0] = 0.0
        F = sample_F(M_F[r])
        cls2 = np.random.rand(pop_size) >= fcp          # True -> CMA sinfi
        p_num = max(2, int(round(0.11 * pop_size)))
        pbest = pop[np.random.randint(0, p_num, pop_size)]
        union = np.vstack([pop, archive]) if len(archive) else pop
        r1 = (idx + np.random.randint(1, pop_size, pop_size)) % pop_size
        r2 = pick_r2(len(union), idx, r1)
        V = pop + F[:, None] * (pbest - pop) + F[:, None] * (pop[r1] - union[r2])
        cross = np.random.rand(pop_size, dim) < CR[:, None]
        cross[idx, np.random.randint(0, dim, pop_size)] = True
        U = np.where(cross, V, pop)
        if np.any(cls2):
            z = np.random.randn(int(cls2.sum()), dim)
            U[cls2] = xmean + sigma * (z * d_eig) @ B.T
        U = midpoint(U, pop, lb, ub)
        n_eval = min(pop_size, max_fes - fes)
        fit_U = np.full(pop_size, np.inf)
        for i in range(n_eval):
            fit_U[i] = obj_func(U[i])
        fes += n_eval
        imp = fit_U < fitness
        # FCP: muvaffaqiyat SONI bo'yicha (ACE-SHADE dan farqi shu)
        s1, s2 = imp[~cls2].sum(), imp[cls2].sum()
        if s1 + s2 > 0:
            fcp = float(np.clip(fcp_lr * fcp + (1 - fcp_lr) * s1 / (s1 + s2), 0.2, 0.8))
        if np.any(imp):
            df = fitness[imp] - fit_U[imp]
            w = df / df.sum()
            archive = arch_push(archive, pop[imp], int(round(arc_rate * pop_size)))
            M_F[k] = lehmer(F[imp], w)
            M_CR[k] = -1.0 if (M_CR[k] == -1 or np.sum(w * CR[imp]) == 0) else lehmer(CR[imp], w)
            k = (k + 1) % H
        acc = fit_U <= fitness
        pop[acc], fitness[acc] = U[acc], fit_U[acc]
        # CMA holatini yangilash (eng yaxshi yarim bo'yicha)
        mu = max(2, pop_size // 2)
        best = np.argsort(fitness)[:mu]
        w_mu = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w_mu /= w_mu.sum()
        xold = xmean
        xmean = w_mu @ pop[best]
        Y = (pop[best] - xold) / sigma
        c_mu = min(0.9, 2.0 / ((dim + 2) ** 2))
        C = (1 - c_mu) * C + c_mu * (Y.T * w_mu) @ Y
        new_size = max(N_min, int(round(N_init + (N_min - N_init) * fes / max_fes)))
        if new_size < pop_size:
            keep = np.argsort(fitness)[:new_size]
            pop, fitness, pop_size = pop[keep], fitness[keep], new_size
            archive = archive[:int(round(arc_rate * pop_size))]

def baseline_NL_SHADE_RSP(obj_func, dim, bounds, max_fes):
    """Stanovov, Akhmedova & Semenkin (CEC 2021 g'olibi).

    LSHADE-RSP ustiga uchta qo'shimcha:
      (i)   NOCHIZIQLI populyatsiya kamayishi (LPSR o'rniga);
      (ii)  arxivdan foydalanish ehtimolligining moslashuvi;
      (iii) binomial va eksponensial crossover o'rtasida moslashuvchan tanlov.
    """
    lb, ub = bounds
    N_init, N_min, H, arc_rate = int(30 * dim), 4, 20, 2.0   # NOANIQ: N_init, H
    k_rsp = 3.0
    pop_size = N_init
    pop = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.array([obj_func(x) for x in pop])
    fes = pop_size
    M_F, M_CR, k = np.full(H, 0.2), np.full(H, 0.2), 0
    archive = np.empty((0, dim))
    p_arch = 0.5            # arxivdan foydalanish ehtimolligi (moslashadi)
    p_bin = 0.5             # binomial crossover ehtimolligi (moslashadi)
    while fes < max_fes:
        t = fes / max_fes
        order = np.argsort(fitness)
        pop, fitness = pop[order], fitness[order]
        idx = np.arange(pop_size)
        r = np.random.randint(0, H, pop_size)
        CR = np.clip(np.random.normal(M_CR[r], 0.1), 0.0, 1.0)
        F = sample_F(M_F[r])
        p_num = max(2, int(round(0.17 * (1.0 - 0.5 * t) * pop_size)))
        pbest = pop[np.random.randint(0, p_num, pop_size)]
        r1 = rsp_ranks(pop_size, idx, k_rsp)
        use_arch = (np.random.rand(pop_size) < p_arch) & (len(archive) > 0)
        union = np.vstack([pop, archive]) if len(archive) else pop
        r2 = pick_r2(len(union) if use_arch.any() else pop_size, idx, r1)
        donor2 = union[r2] if use_arch.any() else pop[r2 % pop_size]
        V = pop + F[:, None] * (pbest - pop) + F[:, None] * (pop[r1] - donor2)
        # Moslashuvchan crossover: binomial yoki eksponensial
        binm = np.random.rand(pop_size) < p_bin
        cross = np.zeros((pop_size, dim), dtype=bool)
        cross[binm] = np.random.rand(int(binm.sum()), dim) < CR[binm, None]
        if np.any(~binm):
            ii = np.where(~binm)[0]
            starts = np.random.randint(0, dim, len(ii))
            L = np.minimum(dim, 1 + np.random.geometric(1 - CR[ii] * 0.99, len(ii)))
            for j, s, ln in zip(ii, starts, L):
                cross[j, (s + np.arange(ln)) % dim] = True
        cross[idx, np.random.randint(0, dim, pop_size)] = True
        U = midpoint(np.where(cross, V, pop), pop, lb, ub)
        n_eval = min(pop_size, max_fes - fes)
        fit_U = np.full(pop_size, np.inf)
        for i in range(n_eval):
            fit_U[i] = obj_func(U[i])
        fes += n_eval
        imp = fit_U < fitness
        # (ii) arxiv va (iii) crossover ehtimolliklarini moslashtirish
        for mask, cur, lo, hi in [(use_arch, "arch", 0.1, 0.9), (binm, "bin", 0.1, 0.9)]:
            a, b = imp[mask].sum(), imp[~mask].sum()
            if a + b > 0:
                val = float(np.clip(0.9 * (p_arch if cur == "arch" else p_bin)
                                    + 0.1 * a / (a + b), lo, hi))
                if cur == "arch":
                    p_arch = val
                else:
                    p_bin = val
        if np.any(imp):
            df = fitness[imp] - fit_U[imp]
            w = df / df.sum()
            archive = arch_push(archive, pop[imp], int(round(arc_rate * pop_size)))
            M_F[k] = 0.5 * M_F[k] + 0.5 * lehmer(F[imp], w)
            M_CR[k] = 0.5 * M_CR[k] + 0.5 * lehmer(CR[imp], w)
            k = (k + 1) % H
        acc = fit_U <= fitness
        pop[acc], fitness[acc] = U[acc], fit_U[acc]
        # (i) NOCHIZIQLI populyatsiya kamayishi
        new_size = max(N_min, int(round((N_min - N_init) * (t ** (1.0 - t)) + N_init)))
        if new_size < pop_size:
            keep = np.argsort(fitness)[:new_size]
            pop, fitness, pop_size = pop[keep], fitness[keep], new_size
            archive = archive[:int(round(arc_rate * pop_size))]

# ==============================================================================
# 5. REESTR
# ==============================================================================

TARGET = "ACE-SHADE"

# Tasdiqlangan to'rtlik: asosiy da'vo shularga nisbatan.
PRIMARY_RIVALS = ("LSHADE-cnEpSin", "jSO", "L-SHADE", "CMA-ES")
# Qayta amalga oshirilganlar: ikkilamchi da'vo, cheklov bilan.
SECONDARY_RIVALS = ("NL-SHADE-RSP", "LSHADE-SPACMA", "LSHADE-RSP")

ALGORITHMS = {
    "ACE-SHADE": ace_shade,
    "NL-SHADE-RSP": baseline_NL_SHADE_RSP,
    "LSHADE-SPACMA": baseline_LSHADE_SPACMA,
    "LSHADE-cnEpSin": baseline_LSHADE_cnEpSin,
    "LSHADE-RSP": baseline_LSHADE_RSP,
    "jSO": baseline_jSO,
    "L-SHADE": baseline_LSHADE,
    "CMA-ES": baseline_CMAES,
}

ABLATION = {
    "ACE-SHADE": ace_shade,
    "N1-off": partial(ace_shade, USE_COV=False),
    "N2-off": partial(ace_shade, USE_ENSEMBLE=False),
    "cmu-population": partial(ace_shade, CMU_SOURCE="population"),
}

FROZEN_PATH = "frozen_params.json"
DEFAULT_PARAMS = {"R_N": 10.0, "ETA": 0.1}


def load_frozen():
    """Muzlatilgan parametrlarni o'qiydi. Yo'q bo'lsa standart qiymat."""
    if os.path.exists(FROZEN_PATH):
        with open(FROZEN_PATH) as fh:
            p = json.load(fh)
        return {"R_N": float(p["R_N"]), "ETA": float(p["ETA"])}
    print(f"[!] {FROZEN_PATH} topilmadi - standart qiymatlar ishlatiladi")
    return dict(DEFAULT_PARAMS)


def bind(alg, params):
    """Muzlatilgan parametrlarni algoritmga biriktiradi.

    Parametrlar faqat ularni QABUL QILADIGAN algoritmga uzatiladi -
    ya'ni ACE-SHADE va uning ablatsiya variantlariga. Raqobatchilar o'z
    e'lon qilingan qiymatlarida qoladi.

    Imzo tekshiriladi, `TypeError` ushlanmaydi: aks holda algoritm
    ichidagi boshqa `TypeError` jimgina yutilib, run parametrsiz qayta
    ishlagan bo'lardi va buni hech kim sezmasdi.
    """
    try:
        accepted = set(inspect.signature(alg).parameters)
    except (TypeError, ValueError):
        return alg
    use = {k: v for k, v in params.items() if k in accepted}
    if not use:
        return alg

    def wrapped(tracker, dim, bounds, max_fes):
        return alg(tracker, dim, bounds, max_fes, **use)
    return wrapped


# ==============================================================================
# 6. SOZLASH (CEC-2017, baholash byudjetida)
# ==============================================================================

R_N_GRID = (6.0, 8.0, 10.0, 12.0, 14.0, 18.0)
ETA_GRID = (0.1, 0.2)
TUNE_DIMS = (10, 20)
TUNE_RUNS = int(os.environ.get("TUNE_RUNS", "10"))


def cfg_name(r_n, eta):
    return f"rn{r_n:g}_eta{eta:g}"


def _tune_one(r_n, eta, func, dim, run_id, budget):
    np.random.seed(SEED_BASE + 1000 * dim + run_id)
    obj, lb, ub, _ = make_problem(func, dim, 2017)
    tr = Tracker(obj, budget, 2)
    try:
        ace_shade(tr, dim, (lb, ub), budget, R_N=r_n, ETA=eta)
    except Exception as exc:
        print(f"[!] {cfg_name(r_n, eta)} {func} {dim}D run {run_id}: "
              f"{type(exc).__name__}: {exc}")
    return {"Config": cfg_name(r_n, eta), "r_N": r_n, "eta": eta,
            "Function": func, "Dimension": dim, "Run": run_id,
            "Best": float(tr.best_f)}


def stage_tune(out_dir, only_cfg="", quick=False):
    """Sozlash. `only_cfg` berilsa bitta konfiguratsiya (shard rejimi)."""
    grid = [(r, e) for r, e in itertools.product(R_N_GRID, ETA_GRID)
            if not only_cfg or cfg_name(r, e) == only_cfg]
    if not grid:
        sys.exit(f"[!] konfiguratsiya topilmadi: {only_cfg}")
    funcs = CEC2017_TUNING_FUNCS[:2] if quick else CEC2017_TUNING_FUNCS
    dims = (10,) if quick else TUNE_DIMS
    runs = 2 if quick else TUNE_RUNS

    tasks = [(r, e, f, d, i, CEC2022_BUDGET[d] // (50 if quick else 1))
             for r, e in grid for f in funcs for d in dims for i in range(runs)]
    print(f"[*] SOZLASH: {len(tasks)} vazifa | {len(grid)} konfiguratsiya "
          f"| {len(funcs)} funksiya | o'lchamlar {dims} | {runs} run")
    t0 = time.time()
    rows = Parallel(n_jobs=-1, verbose=5)(delayed(_tune_one)(*t) for t in tasks)
    print(f"[*] {time.time() - t0:.0f} s")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{out_dir}/tuning_raw.csv", index=False)
    return pd.DataFrame(rows)


def freeze_from_tuning(df, out_dir):
    """Sozlash natijasidan eng yaxshi konfiguratsiyani tanlab MUZLATADI.

    Tanlov qoidasi oldindan belgilangan va natijaga qarab o'zgartirilmaydi:
      1. Har bir (funksiya, o'lcham) instansiyasida medianalar bo'yicha rank;
      2. O'lchamlar bo'ylab o'rtacha rank eng past bo'lgan konfiguratsiya;
      3. Teng bo'lsa kichikroq `R_N` (soddaroq model afzal).
    """
    d = df.copy()
    d["Best"] = d["Best"].where(d["Best"] >= PRECISION_FLOOR, 0.0)
    piv = d.groupby(["Function", "Dimension", "Config"])["Best"].median().unstack("Config")
    ranks = piv.rank(axis=1).mean()
    per_dim = pd.DataFrame({
        dim: piv[piv.index.get_level_values("Dimension") == dim].rank(axis=1).mean()
        for dim in sorted(d["Dimension"].unique())})

    os.makedirs(out_dir, exist_ok=True)
    d.to_csv(f"{out_dir}/tuning_raw.csv", index=False)
    piv.to_csv(f"{out_dir}/tuning_medians.csv")
    ranks.sort_values().to_csv(f"{out_dir}/tuning_ranks.csv", header=["Average_Rank"])
    per_dim.to_csv(f"{out_dir}/tuning_ranks_per_dim.csv")

    best = sorted(ranks.index, key=lambda c: (ranks[c], float(c.split("_")[0][2:])))[0]
    r_n = float(best.split("_")[0][2:])
    eta = float(best.split("_")[1][3:])
    frozen = {"R_N": r_n, "ETA": eta, "selected_config": best,
              "average_rank": float(ranks[best]),
              "grid_R_N": list(R_N_GRID), "grid_ETA": list(ETA_GRID),
              "tuning_suite": "CEC-2017", "tuning_funcs": list(CEC2017_TUNING_FUNCS),
              "tuning_dims": list(TUNE_DIMS), "tuning_runs": TUNE_RUNS,
              "budget_regime": {str(k): v for k, v in CEC2022_BUDGET.items()},
              "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(FROZEN_PATH, "w") as fh:
        json.dump(frozen, fh, indent=2)

    print("\n=== SOZLASH RANKLARI (past = yaxshi) ===")
    print(ranks.sort_values().to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n=== O'LCHAM BO'YICHA ===")
    print(per_dim.to_string(float_format=lambda v: f"{v:.3f}"))
    at_edge = r_n in (R_N_GRID[0], R_N_GRID[-1])
    print(f"\n[*] MUZLATILDI: R_N={r_n:g}, ETA={eta:g}  ({best}, rank {ranks[best]:.3f})")
    if at_edge:
        print("[!] OGOHLANTIRISH: tanlangan R_N panjara chekkasida - "
              "haqiqiy optimum undan tashqarida bo'lishi mumkin. "
              "Bu hisobotda cheklov sifatida qayd etiladi.")
    frozen["r_n_at_grid_edge"] = bool(at_edge)
    with open(FROZEN_PATH, "w") as fh:
        json.dump(frozen, fh, indent=2)
    return frozen


# ==============================================================================
# 7. BAHOLASH (CEC-2022)
# ==============================================================================

def _run_one(alg_name, alg, func, dim, run_id, budget):
    # Bir xil (o'lcham, run) uchun barcha algoritmlarga bir xil seed:
    # taqqoslash juftlashtirilgan bo'ladi va boshlang'ich holat farqi
    # natijaga ta'sir qilmaydi.
    np.random.seed(SEED_BASE + 1000 * dim + run_id)
    obj, lb, ub, _ = make_problem(func, dim, 2022)
    tr = Tracker(obj, budget, N_CURVE_POINTS)
    try:
        alg(tr, dim, (lb, ub), budget)
    except Exception as exc:                      # bitta run butun ishni to'xtatmasin
        print(f"[!] {alg_name} {func} {dim}D run {run_id}: "
              f"{type(exc).__name__}: {exc}")
    return ({"Algorithm": alg_name, "Function": func, "Dimension": dim,
             "Run": run_id, "Best": float(tr.best_f)},
            f"{alg_name}|{func}|{dim}|{run_id}", tr.finalize())


def stage_run(out_dir, registry, params, funcs, dims, algs, runs, quick=False):
    tasks = []
    for func in funcs:
        for dim in dims:
            budget = CEC2022_BUDGET[dim] // (50 if quick else 1)
            for name in algs:
                if name not in registry:
                    sys.exit(f"[!] noma'lum algoritm: {name}")
                bound = bind(registry[name], params)
                for r in range(runs):
                    tasks.append((name, bound, func, dim, r, budget))
    print(f"[*] BAHOLASH: {len(tasks)} vazifa | funksiyalar {len(funcs)} "
          f"| o'lchamlar {dims} | algoritmlar {len(algs)} | {runs} run")
    t0 = time.time()
    out = Parallel(n_jobs=-1, verbose=5)(delayed(_run_one)(*t) for t in tasks)
    print(f"[*] {time.time() - t0:.0f} s")

    os.makedirs(f"{out_dir}/raw_data", exist_ok=True)
    pd.DataFrame([o[0] for o in out]).to_csv(
        f"{out_dir}/raw_data/full_raw_results.csv", index=False)
    np.savez_compressed(f"{out_dir}/raw_data/curves.npz",
                        **{o[1]: o[2] for o in out})
    print(f"[*] saqlandi: {out_dir}/raw_data/")


# ==============================================================================
# 8. TAHLIL
# ==============================================================================

def holm(pvals):
    """Holm-Bonferroni tuzatishi (monoton, chegaralangan)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, p[i] * (n - rank)))
        adj[i] = run
    return adj


def posthoc(pivot, target):
    """Friedman post-hoc, nazorat = target."""
    algs = list(pivot.columns)
    k, N = len(algs), len(pivot)
    avg = pivot.rank(axis=1).mean()
    se = math.sqrt(k * (k + 1) / (6.0 * N))
    rows = []
    for a in algs:
        if a == target:
            continue
        diff = avg[a] - avg[target]
        z = diff / se
        rows.append({"Algorithm": a, "Group": "asosiy" if a in PRIMARY_RIVALS
                     else "ikkilamchi", "Average_Rank": avg[a],
                     "Rank_diff_vs_control": diff, "z": z,
                     "p_unadjusted": 2 * (1 - norm.cdf(abs(z)))})
    df = pd.DataFrame(rows)
    df["p_Holm"] = holm(df["p_unadjusted"].values)
    df["significant_0.05"] = df["p_Holm"] < 0.05
    return df.sort_values("p_Holm").reset_index(drop=True)


def win_tie_loss(df, target, algs, funcs, dims):
    """Instansiya bo'yicha Mann-Whitney + Holm."""
    rivals = [a for a in algs if a != target]
    pvals, wtl = [], {c: {"+": 0, "=": 0, "-": 0} for c in rivals}
    for f in funcs:
        for d in dims:
            sel = (df["Function"] == f) & (df["Dimension"] == d)
            vt = df[sel & (df["Algorithm"] == target)]["Best"].values
            if len(vt) == 0:
                continue
            raw_p, better = [], []
            for c in rivals:
                vc = df[sel & (df["Algorithm"] == c)]["Best"].values
                try:
                    _, p = mannwhitneyu(vt, vc, alternative="two-sided")
                except ValueError:
                    p = 1.0
                raw_p.append(1.0 if (p is None or np.isnan(p)) else float(p))
                better.append(np.median(vt) < np.median(vc))
            adj = holm(raw_p)
            row = {"Function": f, "Dimension": d}
            for c, p, b in zip(rivals, adj, better):
                row[c] = p
                wtl[c]["=" if p >= 0.05 else ("+" if b else "-")] += 1
            pvals.append(row)
    return pd.DataFrame(pvals), pd.DataFrame(wtl).T[["+", "=", "-"]]


def category_tables(df, pivot, target, out_dir):
    """Kategoriya x raqib bo'yicha to'liq tahlil."""
    algs = list(pivot.columns)
    rivals = [a for a in algs if a != target]
    dims = sorted(df["Dimension"].unique())
    all_funcs = set(pivot.index.get_level_values("Function"))

    rows = []
    for cat, members in CEC2022_CATEGORIES.items():
        sub = pivot[pivot.index.get_level_values("Function").isin(members)]
        if len(sub):
            rows.append({"Category": cat, "n_instances": len(sub),
                         **sub.rank(axis=1).mean().to_dict()})
    pd.DataFrame(rows).to_csv(f"{out_dir}/tables/category_ranks.csv", index=False)

    rows = []
    for cat, members in CEC2022_CATEGORIES.items():
        for d in dims:
            sub = pivot[pivot.index.get_level_values("Function").isin(members)
                        & (pivot.index.get_level_values("Dimension") == d)]
            if len(sub):
                rows.append({"Category": cat, "Dimension": d, "n_instances": len(sub),
                             **sub.rank(axis=1).mean().to_dict()})
    pd.DataFrame(rows).to_csv(f"{out_dir}/tables/dimension_category.csv", index=False)

    lines = ["# Kategoriya bo'yicha tahlil", "",
             f"Nazorat: **{target}**. `YUTDIK` faqat Holm tuzatilgan "
             "`p < 0.05` bo'lganda - o'rtacha kichikroq bo'lishi yetarli emas.",
             "", "Raqiblar ikki guruhga ajratilgan: **asosiy** "
             f"({', '.join(PRIMARY_RIVALS)}) - tasdiqlangan amalga oshirish; "
             f"**ikkilamchi** ({', '.join(SECONDARY_RIVALS)}) - mexanizm "
             "ta'rifi bo'yicha qayta yozilgan, cheklov hisobotda.", ""]
    for cat, members in CEC2022_CATEGORIES.items():
        funcs = [f for f in members if f in all_funcs]
        if not funcs:
            continue
        sub = pivot[pivot.index.get_level_values("Function").isin(funcs)]
        ranks = sub.rank(axis=1).mean()
        _, wtl = win_tie_loss(df[df["Function"].isin(funcs)], target, algs, funcs, dims)
        wtl.to_csv(f"{out_dir}/tables/category_wtl_{cat.replace(' ', '_')}.csv")

        lines += [f"## {cat} ({', '.join(funcs)})", "",
                  f"Instansiyalar: {len(sub)}. {target} o'rtacha ranki: "
                  f"**{ranks[target]:.2f}**", "",
                  "| Raqib | Guruh | + | = | − | Bizning rank | Uning ranki | Holat |",
                  "|---|---|---|---|---|---|---|---|"]
        for c in rivals:
            w, t_, l = int(wtl.loc[c, "+"]), int(wtl.loc[c, "="]), int(wtl.loc[c, "-"])
            status = "**YUTDIK**" if w > l else ("**YUTQAZDIK**" if l > w else "teng")
            grp = "asosiy" if c in PRIMARY_RIVALS else "ikkilamchi"
            lines.append(f"| {c} | {grp} | {w} | {t_} | {l} | {ranks[target]:.2f} "
                         f"| {ranks[c]:.2f} | {status} |")
        lines += ["", "| Instansiya | " + " | ".join(algs) + " |",
                  "|---|" + "---|" * len(algs)]
        for ix in sub.index:
            row = sub.loc[ix]
            best = row.min()
            cells = []
            for a in algs:
                txt = "0" if row[a] <= 0 else f"{row[a]:.2e}"
                cells.append(f"**{txt}**" if row[a] <= best + 1e-300 else txt)
            lines.append(f"| {ix[0]} {ix[1]}D | " + " | ".join(cells) + " |")
        lines.append("")

    with open(f"{out_dir}/tables/category_summary.md", "w") as fh:
        fh.write("\n".join(lines))


def analyse(df_raw, target, out_dir, curves=None):
    for sub in ("tables", "figures", "raw_data"):
        os.makedirs(f"{out_dir}/{sub}", exist_ok=True)
    df_raw.to_csv(f"{out_dir}/raw_data/full_raw_results.csv", index=False)

    df = df_raw.copy()
    n_floor = int((df["Best"] < PRECISION_FLOOR).sum())
    df["Best"] = df["Best"].where(df["Best"] >= PRECISION_FLOOR, 0.0)
    print(f"[*] aniqlik chegarasi {PRECISION_FLOOR:g}: {n_floor} natija 0 ga tenglashtirildi")

    present = list(df["Algorithm"].unique())
    algs = [target] + [a for a in ALGORITHMS if a in present and a != target]
    algs += [a for a in present if a not in algs]
    funcs = sorted(df["Function"].unique(), key=lambda s: int(s[1:]))
    dims = sorted(df["Dimension"].unique())

    summ = df.groupby(["Function", "Dimension", "Algorithm"])["Best"] \
             .agg(["mean", "std"]).reset_index()
    summ.to_csv(f"{out_dir}/tables/summary_mean_std.csv", index=False)
    pivot = summ.pivot(index=["Function", "Dimension"], columns="Algorithm",
                       values="mean")[algs].dropna()
    pivot.to_csv(f"{out_dir}/tables/summary_results.csv")
    try:
        pivot.to_latex(f"{out_dir}/tables/summary_table.tex", float_format="%.2e")
    except Exception as exc:
        print(f"[!] LaTeX jadval yozilmadi: {exc}")

    chi2, p = friedmanchisquare(*[pivot[a].values for a in algs])
    avg = pivot.rank(axis=1).mean().sort_values()
    with open(f"{out_dir}/tables/friedman_test.txt", "w") as fh:
        fh.write(f"Friedman chi2 = {chi2:.4f}, p = {p:.4e}\n"
                 f"instansiyalar = {len(pivot)}, algoritmlar = {len(algs)}\n\n"
                 + avg.to_string())
    avg.rename("Average_Rank").rename_axis("Algorithm").reset_index() \
       .to_csv(f"{out_dir}/tables/overall_average_ranks.csv", index=False)

    ph = posthoc(pivot, target)
    ph.to_csv(f"{out_dir}/tables/friedman_posthoc.csv", index=False)
    pv, wtl = win_tie_loss(df, target, algs, funcs, dims)
    pv.to_csv(f"{out_dir}/tables/wilcoxon_pvalues.csv", index=False)
    wtl.to_csv(f"{out_dir}/tables/win_tie_loss.csv")
    category_tables(df, pivot, target, out_dir)

    if curves:
        _plot(curves, out_dir, algs)
    print(f"[*] tahlil tayyor: {out_dir}/tables/")
    return {"pivot": pivot, "ranks": avg, "posthoc": ph, "wtl": wtl,
            "chi2": float(chi2), "p": float(p), "n_instances": len(pivot),
            "algs": algs, "n_floor": n_floor}


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
            plt.semilogy(np.linspace(0, 1, len(med)), np.maximum(med, 1e-12),
                         label=a, lw=1.3)
        plt.xlabel("byudjet ulushi"); plt.ylabel("xato (median)")
        plt.title(f"{f}, D={d}"); plt.grid(alpha=.3); plt.legend(fontsize=7)
        plt.tight_layout(); plt.savefig(f"{out_dir}/figures/Conv_{f}_{d}D.png", dpi=110)
        plt.close()


# ==============================================================================
# 9. HISOBOT
# ==============================================================================

def write_report(res, out_dir, doc_path="docs/RESULTS_V2.md", ablation=None,
                 n_runs=None):
    """Natijalardan `docs/RESULTS_V2.md` ni avtomatik yozadi.

    Hisobot natijaga qarab yumshatilmaydi: post-hoc ahamiyatli bo'lmasa
    shunday yoziladi, mag'lubiyatlar raqami bilan keltiriladi.
    """
    pivot, ranks, ph, wtl = res["pivot"], res["ranks"], res["posthoc"], res["wtl"]
    frozen = load_frozen()
    fro = {}
    if os.path.exists(FROZEN_PATH):
        with open(FROZEN_PATH) as fh:
            fro = json.load(fh)

    our_rank = ranks[TARGET]
    beaten_primary = [r["Algorithm"] for _, r in ph.iterrows()
                      if r["Algorithm"] in PRIMARY_RIVALS and r["significant_0.05"]
                      and r["Rank_diff_vs_control"] > 0]
    lost_primary = [r["Algorithm"] for _, r in ph.iterrows()
                    if r["Algorithm"] in PRIMARY_RIVALS and r["significant_0.05"]
                    and r["Rank_diff_vs_control"] < 0]
    best_alg = ranks.index[0]

    L = [f"# ACE-SHADE: CEC-2022 natijalari", "",
         f"Avtomatik yaratilgan: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}. "
         f"Xom ma'lumot `{out_dir}/raw_data/`, jadvallar `{out_dir}/tables/`, "
         f"grafiklar `{out_dir}/figures/`.", "",
         "## 1. Xulosa", ""]

    if best_alg == TARGET and beaten_primary:
        L.append(f"ACE-SHADE eng yaxshi o'rtacha rankka ega (**{our_rank:.2f}**) va "
                 f"post-hoc testda quyidagi asosiy raqiblardan statistik jihatdan "
                 f"ustun: **{', '.join(beaten_primary)}**.")
    elif best_alg == TARGET:
        L.append(f"ACE-SHADE eng yaxshi o'rtacha rankka ega (**{our_rank:.2f}**), "
                 f"lekin Holm tuzatilgan post-hoc testda asosiy raqiblarning "
                 f"**hech biridan** statistik jihatdan ustun emas. Bu natija "
                 f"yumshatilmaydi: rank ustunligi ahamiyatlilikni almashtirmaydi.")
    else:
        L.append(f"**ACE-SHADE eng yaxshi emas.** Eng yaxshi o'rtacha rank: "
                 f"{best_alg} ({ranks[best_alg]:.2f}); ACE-SHADE {our_rank:.2f}.")
    if lost_primary:
        L.append(f"\nAsosiy raqiblardan **statistik jihatdan yutqazdik**: "
                 f"{', '.join(lost_primary)}.")

    L += ["", "## 2. Protokol", "",
          f"- Baholash to'plami: **CEC-2022**, {len(pivot)} instansiya "
          f"({len(set(pivot.index.get_level_values('Function')))} funksiya x "
          f"{len(set(pivot.index.get_level_values('Dimension')))} o'lcham)",
          "- Byudjet: rasmiy protokol - "
          + ", ".join(f"D={d} -> {CEC2022_BUDGET[d]:,}"
                      for d in sorted(set(pivot.index.get_level_values("Dimension"))))
          + " FES",
          f"- Mustaqil runlar: {n_runs if n_runs is not None else CEC2022_RUNS}",
          f"- Seed sxemasi: `42 + 1000*D + run`; berilgan (o'lcham, run) da "
          f"barcha algoritmlar bir xil boshlang'ich holatdan boshlaydi",
          f"- Aniqlik chegarasi: `{PRECISION_FLOOR:g}` (CEC konventsiyasi); "
          f"{res['n_floor']} natija 0 ga tenglashtirildi, xom qiymatlar saqlandi",
          "", "### 2.1 Sozlash", "",
          f"- Sozlash to'plami: **CEC-2017** ({len(CEC2017_TUNING_FUNCS)} funksiya) - "
          f"CEC-2022 bilan bitta ham umumiy funksiya yo'q",
          f"- Byudjet rejimi baholash bilan **aynan bir xil**",
          f"- Sozlangan parametrlar: `R_N = {frozen['R_N']:g}`, `ETA = {frozen['ETA']:g}`"
          + (f" ({fro.get('selected_config', '')})" if fro else ""),
          "- Qolgan barcha parametrlar manba usullari e'lon qilgan qiymatda", ""]
    if fro.get("r_n_at_grid_edge"):
        L += ["> **Cheklov.** Tanlangan `R_N` sozlash panjarasining chekkasida. "
              "Haqiqiy optimum undan tashqarida bo'lishi mumkin.", ""]

    L += ["## 3. Raqiblar va da'volarning ajratilishi", "",
          "Raqiblar ikki guruhga bo'lingan va da'volar ham shunga mos ajratilgan.",
          "",
          f"**Asosiy guruh** ({', '.join(PRIMARY_RIVALS)}) - oldingi versiyada "
          "sinovdan o'tgan amalga oshirishlar. Har biri ACE-SHADE yaxshilashni "
          "da'vo qilayotgan aniq narsaning manbai: LSHADE-cnEpSin va CMA-ES "
          "kovariatsiya da'vosiga, jSO va L-SHADE rejim ansambli da'vosiga "
          "qarshi turadi. **Asosiy da'vo shu guruhga nisbatan qo'yiladi.**", "",
          f"**Ikkilamchi guruh** ({', '.join(SECONDARY_RIVALS)}) - e'lon qilingan "
          "mexanizm ta'rifi bo'yicha qayta yozilgan. Ish muhitida asl "
          "maqolalarga va mualliflar kodiga kirish bo'lmagan, shuning uchun "
          "ba'zi ikkilamchi tanlovlar (populyatsiya hajmi formulasi, ayrim "
          "konstantalar) tasdiqlanmagan. **Bu guruhga nisbatan g'alaba kuchsiz "
          "dalil**: agar bizning variantimiz e'lon qilingan natijadan past "
          "ishlasa, taqqoslash ularning foydasiga emas.", "",
          "## 4. Umumiy ranklar va Friedman testi", "",
          f"```\nFriedman chi2 = {res['chi2']:.4f},  p = {res['p']:.4e}"
          f"\ninstansiyalar = {res['n_instances']},  algoritmlar = {len(res['algs'])}\n```",
          "", "| Algoritm | Guruh | O'rtacha rank |", "|---|---|---|"]
    for a in ranks.index:
        grp = "**taklif**" if a == TARGET else ("asosiy" if a in PRIMARY_RIVALS
                                                else "ikkilamchi")
        mark = "**" if a == TARGET else ""
        L.append(f"| {mark}{a}{mark} | {grp} | {mark}{ranks[a]:.2f}{mark} |")

    L += ["", "## 5. Post-hoc taqqoslash (nazorat: ACE-SHADE)", "",
          "| Raqib | Guruh | Rank farqi | z | p (tuzatilmagan) | p (Holm) | Ahamiyatli? |",
          "|---|---|---|---|---|---|---|"]
    for _, r in ph.iterrows():
        L.append(f"| {r['Algorithm']} | {r['Group']} | {r['Rank_diff_vs_control']:+.2f} "
                 f"| {r['z']:.2f} | {r['p_unadjusted']:.3e} | {r['p_Holm']:.4f} "
                 f"| {'**ha**' if r['significant_0.05'] else 'yo`q'} |")

    L += ["", "## 6. Instansiya bo'yicha g'alaba/teng/mag'lubiyat", "",
          "Mann-Whitney U + Holm tuzatishi, `alpha = 0.05`.", "",
          "| Raqib | Guruh | + | = | − |", "|---|---|---|---|---|"]
    for c in wtl.index:
        grp = "asosiy" if c in PRIMARY_RIVALS else "ikkilamchi"
        L.append(f"| {c} | {grp} | {int(wtl.loc[c, '+'])} | {int(wtl.loc[c, '='])} "
                 f"| {int(wtl.loc[c, '-'])} |")

    L += ["", "## 7. Kategoriya bo'yicha tahlil", "",
          f"To'liq jadvallar: `{out_dir}/tables/category_summary.md`. "
          f"Kategoriya ranklari: `category_ranks.csv`, o'lcham kesimi: "
          f"`dimension_category.csv`.", ""]
    cr = pd.read_csv(f"{out_dir}/tables/category_ranks.csv")
    if len(cr):
        cols = [c for c in cr.columns if c not in ("Category", "n_instances")]
        L += ["| Kategoriya | n | " + " | ".join(cols) + " |",
              "|---|---|" + "---|" * len(cols)]
        for _, r in cr.iterrows():
            cells = [f"**{r[c]:.2f}**" if c == TARGET else f"{r[c]:.2f}" for c in cols]
            L.append(f"| {r['Category']} | {int(r['n_instances'])} | "
                     + " | ".join(cells) + " |")

    if ablation is not None and len(ablation):
        L += ["", "## 8. Ablatsiya", "",
              "Har bir hissa alohida o'chirilgan holda o'lchandi.", "",
              "| Variant | O'rtacha rank |", "|---|---|"]
        for a in ablation.index:
            L.append(f"| {a} | {ablation[a]:.2f} |")
        L.append("")
        L.append("`N1-off` - to'plamli kovariatsiya o'chirilgan (namunaviy "
                 "kovariatsiyaga qaytish). `N2-off` - rejim ansambli "
                 "o'chirilgan (faqat jSO banki). `cmu-population` - rank-mu "
                 "yangilanishi qabul qilingan avlodlar o'rniga butun "
                 "populyatsiyadan olinadi.")

    L += ["", "## 9. Cheklovlar", "",
          "1. Ikkilamchi guruhdagi uchta raqib qayta amalga oshirilgan va asl "
          "manbalar bilan tasdiqlanmagan (3-bo'limga qarang).",
          "2. Sozlash 10 ta CEC-2017 funksiyasida, 10 run bilan o'tkazilgan - "
          "bu sozlash shovqinini butunlay yo'qotmaydi.",
          "3. Hisoblash murakkabligi (T0/T1/T2) bu ishda o'lchanmagan.",
          "4. Shovqinli va dinamik masalalar CEC-2022 da yo'q, shuning uchun "
          "oldingi versiyada aniqlangan shovqin zaifligi bu yerda "
          "tekshirilmaydi va ochiq qoladi.", ""]
    if fro.get("r_n_at_grid_edge"):
        L.append("5. Sozlangan `R_N` panjara chekkasida (2.1-bo'limga qarang).")

    os.makedirs(os.path.dirname(doc_path), exist_ok=True)
    with open(doc_path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[*] hisobot yozildi: {doc_path}")


# ==============================================================================
# 10. BIRLIK TESTLARI
# ==============================================================================

def selftest():
    """Sonli va protokol tekshiruvlari. O'tmasa eksperiment boshlanmaydi."""
    DIM, FES = 10, 4000
    obj, lb, ub, _ = make_problem("F1", DIM, 2022)
    ok, fail = 0, []

    def check(name, cond, info=""):
        nonlocal ok
        if cond:
            ok += 1
            print(f"  O'TDI  {name:36s} {info}")
        else:
            fail.append(name)
            print(f"  XATO   {name:36s} {info}")

    # 1. Tezlashtirilgan katsuura asl variant bilan bir xil
    rng = np.random.default_rng(12345)
    worst = 0.0
    for ndim in (2, 5, 10, 20, 30):
        for _ in range(400):
            x = rng.uniform(-100, 100, ndim)
            a, b = _KATSUURA_ORIGINAL(x), _katsuura_fast(x)
            worst = max(worst, abs(a - b) / max(abs(a), 1e-300))
    check("katsuura aynan bir xil", worst < 1e-12, f"farq {worst:.2e}")

    # 2. Optimumda xato nol (ikkala to'plam)
    from opfunu.cec_based import cec2022, cec2017
    w = 0.0
    for fi in range(1, 13):
        for d in (10, 20):
            o, _, _, _ = make_problem(f"F{fi}", d, 2022)
            w = max(w, abs(o(getattr(cec2022, f"F{fi}2022")(ndim=d).x_global)))
    for nm in CEC2017_TUNING_FUNCS[:3]:
        o, _, _, _ = make_problem(nm, 10, 2017)
        w = max(w, abs(o(getattr(cec2017, f"{nm}2017")(ndim=10).x_global)))
    check("optimumda xato 0", w < 1e-6, f"eng katta {w:.2e}")

    # 3. Byudjetdan oshmaslik
    over = {}
    for name, alg in ALGORITHMS.items():
        np.random.seed(11)
        tr = Tracker(obj, FES)
        alg(tr, DIM, (lb, ub), FES)
        if tr.fes > FES:
            over[name] = tr.fes
    check("byudjet oshmaydi", not over, str(over) if over else f"8 algoritm, {FES} FES")

    # 4. Chegara saqlanishi
    bad = {}
    for name, alg in ALGORITHMS.items():
        cnt = {"n": 0}

        def guard(x, _f=obj, _c=cnt):
            a = np.asarray(x, dtype=float)
            if np.any(a < lb - 1e-9) or np.any(a > ub + 1e-9):
                _c["n"] += 1
            return _f(a)
        np.random.seed(5)
        alg(Tracker(guard, FES), DIM, (lb, ub), FES)
        if cnt["n"]:
            bad[name] = cnt["n"]
    check("chegara saqlanadi", not bad, str(bad) if bad else "8 algoritm")

    # 5. Seed takrorlanishi
    diff = {}
    for name, alg in ALGORITHMS.items():
        vals = []
        for _ in range(2):
            np.random.seed(77)
            tr = Tracker(obj, FES)
            alg(tr, DIM, (lb, ub), FES)
            vals.append(tr.best_f)
        if vals[0] != vals[1]:
            diff[name] = vals
    check("seed takrorlanadi", not diff, str(diff) if diff else "8 algoritm")

    # 6. midpoint har doim chegara ichiga qaytaradi
    rng = np.random.default_rng(0)
    lo, hi = np.full(8, -100.0), np.full(8, 100.0)
    good = True
    for _ in range(500):
        pp = rng.uniform(-100, 100, (16, 8))
        UU = midpoint(rng.uniform(-400, 400, (16, 8)), pp, lo, hi)
        good &= bool(np.all(UU >= lo - 1e-9) and np.all(UU <= hi + 1e-9))
    check("midpoint tuzatadi", good, "500 sinov")

    # 7. Kovariatsiya musbat yarim aniq; LPSR monoton va N_min da to'xtaydi
    log = []
    np.random.seed(9)
    ace_shade(Tracker(obj, FES), DIM, (lb, ub), FES, on_generation=log.append)
    sizes = [g["pop_size"] for g in log]
    shares_ok = all(0.0 <= g[k] <= 1.0 for g in log for k in ("p_bank", "p_eig", "p_cma"))
    check("LPSR va moslashuv chegarada",
          bool(log) and min(sizes) >= 4 and sizes == sorted(sizes, reverse=True)
          and shares_ok and all(g["sigma"] > 0 for g in log),
          f"N: {max(sizes)} -> {min(sizes)}, {len(log)} avlod")

    # 8. Ablatsiya variantlari ishlaydi va farq qiladi
    outs = {}
    for name, alg in ABLATION.items():
        np.random.seed(21)
        tr = Tracker(obj, FES)
        alg(tr, DIM, (lb, ub), FES)
        outs[name] = tr.best_f
    check("ablatsiya variantlari farq qiladi", len(set(outs.values())) > 1,
          ", ".join(f"{k}={v:.1e}" for k, v in outs.items()))

    # 9. Muzlatilgan parametrlar faqat ACE-SHADE ga uzatiladi
    b_ace = bind(ace_shade, {"R_N": 12.0, "ETA": 0.2})
    b_jso = bind(baseline_jSO, {"R_N": 12.0, "ETA": 0.2})
    check("parametrlar to'g'ri biriktiriladi",
          b_ace is not ace_shade and b_jso is baseline_jSO, "imzo tekshiruvi")

    print(f"\n{ok}/{ok + len(fail)} test o'tdi")
    return not fail


# ==============================================================================
# 11. QUVUR
# ==============================================================================

def _env_list(name, default):
    raw = os.environ.get(name, "")
    return [s.strip() for s in raw.split(",") if s.strip()] or list(default)


def _load_chunks(dirs):
    """Bo'laklarni birlashtiradi: xom natijalar va yaqinlashish egrilari."""
    frames, curves = [], {}
    for d in dirs:
        p = f"{d}/raw_data/full_raw_results.csv"
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
        npz = f"{d}/raw_data/curves.npz"
        if os.path.exists(npz):
            with np.load(npz, allow_pickle=False) as z:
                for k in z.files:
                    a, f, dd, _ = k.split("|")
                    curves.setdefault((a, f, int(dd)), []).append(z[k])
    return frames, curves


def main():
    stage = os.environ.get("STAGE", "all").lower()
    out_dir = os.environ.get("OUT_DIR", "results/v2")
    quick = os.environ.get("QUICK") == "1"
    merge = [x.strip().rstrip("/") for x in os.environ.get("MERGE", "").split(",") if x.strip()]

    if os.environ.get("SELFTEST") == "1" or stage == "selftest":
        sys.exit(0 if selftest() else 1)

    # --- Birlashtirish + tahlil + hisobot ------------------------------------
    if merge or stage in ("analyse", "report"):
        dirs = merge or [out_dir]
        frames, curves = _load_chunks(dirs)
        if not frames:
            sys.exit(f"[!] birlashtirish uchun bo'lak topilmadi: {dirs}")
        df = pd.concat(frames, ignore_index=True)
        print(f"[*] {len(dirs)} bo'lak, {len(df)} qator")
        abl_dirs = [d for d in dirs if "ablat" in d.lower()]
        main_df = df[df["Algorithm"].isin(ALGORITHMS)] if not abl_dirs else df
        res = analyse(main_df, TARGET, out_dir, curves=curves)

        abl_ranks = None
        abl_path = os.environ.get("ABLATION_DIR", "results/ablation_v2")
        if os.path.exists(f"{abl_path}/raw_data/full_raw_results.csv"):
            ad = pd.read_csv(f"{abl_path}/raw_data/full_raw_results.csv")
            ad["Best"] = ad["Best"].where(ad["Best"] >= PRECISION_FLOOR, 0.0)
            ap = (ad.groupby(["Function", "Dimension", "Algorithm"])["Best"]
                    .mean().unstack("Algorithm").dropna())
            if len(ap):
                abl_ranks = ap.rank(axis=1).mean().sort_values()
        n_runs = int(main_df.groupby(["Algorithm", "Function", "Dimension"])
                     .size().max())
        write_report(res, out_dir, ablation=abl_ranks, n_runs=n_runs)
        return

    # --- Sozlash --------------------------------------------------------------
    if stage in ("all", "tune"):
        cfg = os.environ.get("CFG", "")
        tdir = os.environ.get("TUNE_DIR", "results/tuning_v2")
        if os.environ.get("TUNE_MERGE"):
            dirs = [x.strip().rstrip("/") for x in os.environ["TUNE_MERGE"].split(",") if x.strip()]
            frames = [pd.read_csv(f"{d}/tuning_raw.csv") for d in dirs
                      if os.path.exists(f"{d}/tuning_raw.csv")]
            if not frames:
                sys.exit("[!] sozlash bo'lagi topilmadi")
            freeze_from_tuning(pd.concat(frames, ignore_index=True), tdir)
        else:
            df = stage_tune(tdir, only_cfg=cfg, quick=quick)
            if not cfg:
                freeze_from_tuning(df, tdir)
        if stage == "tune":
            return

    # --- Baholash / ablatsiya -------------------------------------------------
    params = load_frozen()
    print(f"[*] muzlatilgan parametrlar: R_N={params['R_N']:g}, ETA={params['ETA']:g}")

    if stage in ("all", "run", "ablate"):
        is_abl = stage == "ablate" or os.environ.get("SUITE") == "ablation"
        registry = ABLATION if is_abl else ALGORITHMS
        funcs = _env_list("FUNCS", CEC2022_FUNCS)
        dims = [int(x) for x in _env_list("DIMS", CEC2022_DIMS)]
        algs = _env_list("ALGS", registry.keys())
        runs = int(os.environ.get("RUNS", str(CEC2022_RUNS)))
        target_dir = out_dir if not is_abl else os.environ.get(
            "OUT_DIR", "results/ablation_v2")
        stage_run(target_dir, registry, params, funcs, dims, algs, runs, quick)
        if stage in ("run", "ablate"):
            return

    # --- To'liq quvurda tahlil ham shu yerda ---------------------------------
    frames, curves = _load_chunks([out_dir])
    if frames:
        df_all = pd.concat(frames, ignore_index=True)
        res = analyse(df_all, TARGET, out_dir, curves=curves)
        n_runs = int(df_all.groupby(["Algorithm", "Function", "Dimension"]).size().max())
        write_report(res, out_dir, n_runs=n_runs)


if __name__ == "__main__":
    main()
