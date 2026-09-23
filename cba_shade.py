import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import mannwhitneyu, friedmanchisquare, norm
import math
import os
import time
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
# CBA-SHADE - ALGORITM TARKIBI
# ------------------------------------------------------------------------------
# Algoritm nomi bitta joyda (ALGO_NAME) belgilanadi. Muqobil variantlar:
#   "CBA-SHADE"  - Covariance-Basis Adaptive SHADE (standart)
#   "LSHADE-CBA" - L-SHADE oilasining nomlash an'anasiga mos variant
#   "TEMOA-CB"   - muallifning avvalgi TEMOA brendini saqlagan variant
#
# V10 (gibrid portfel) dagi kuchsiz qismlar va ularning o'rniga qo'yilgan
# komponentlar. Har bir almashtirish 12 ta funksiya x 30/50 D ustida
# o'tkazilgan ablatsiya tajribasi bilan asoslangan (izoh: docs/ABLATION.md).
#
#  V10 (kuchsiz)                          ->  CBA-SHADE (kuchli, manba)
#  ------------------------------------------------------------------------------
#  Metafora asosidagi operator portfeli:  -> Portfel butunlay olib tashlandi.
#  GWO lider-DE, WOA spirali, HHO Levy       Ablatsiya: portfelni o'chirib faqat
#  operatorlari + muvaffaqiyat ULUSHIga      current-to-pbest-w/1 qoldirilganda
#  asoslangan probability matching           Schwefel 30D xatosi 2428 -> 0.0 ga
#                                            tushdi. Uchala metafora operatori
#                                            ham ko'p ekstremumli relyefda
#                                            zarar keltirar edi.
#
#  Tasodifiy (bir tekis) r1 tanlovi       -> Rank asosidagi tanlov bosimi (RSP),
#                                            LSHADE-RSP (Stanovov va b., 2018)
#
#  P_EIG ni muvaffaqiyat ULUSHI bo'yicha  -> Kovariatsiya bilan boshqariladigan
#  [0.1, 0.9] oralig'ida moslashtirish       bazis moslashuvi: separabellik
#  (separabel masalalarda o'chira olmaydi)   ko'rsatkichi rho (korrelyatsiya
#                                            matritsasining diagonaldan tashqari
#                                            o'rtacha moduli) boshlang'ich
#                                            taqsimotni beradi, kredit esa
#                                            yaxshilanish MIQDORIga asoslanadi
#                                            (FIR krediti, Fialho va b., 2010).
#                                            Oraliq [0.02, 0.98] - ya'ni bazis
#                                            to'liq o'chishi ham mumkin.
#
#  Oxirgi 5% byudjet: (1+1)-ES, 1/5       -> CMA-ES bilan yakuniy aniqlashtirish
#  qoidasi (ablatsiya: natijaga TA'SIRI      (LSHADE-SPACMA / EBOwithCMAR uslubi).
#  UMUMAN YO'Q edi, byudjet isrof bo'lardi)
#
#  N_init = 6*D, POP_MAX = 500 cheklovi   -> N_init = 18*D (L-SHADE standarti),
#                                            sun'iy cheklovsiz
#
#  Vaznsiz Lehmer mean                       (saqlandi) |df| bilan vaznlangan
#                                            Lehmer mean + jSO F/CR cheklovlari
#  Midpoint-target chegara                   (saqlandi, L-SHADE)
#  LPSR                                      (saqlandi, L-SHADE)
# ==============================================================================

ALGO_NAME = "CBA-SHADE"

# ==============================================================================
# 1. MATEMATIK TEST FUNKSIYALARI (12 ta benchmark)
# ==============================================================================
def ackley(x):
    dim = len(x)
    return -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x**2)/dim)) - np.exp(np.sum(np.cos(2*np.pi*x))/dim) + 20.0 + np.e

def bent_cigar(x):
    return x[0]**2 + 1e6 * np.sum(x[1:]**2)

def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def griewank(x):
    dim = len(x)
    return np.sum(x**2)/4000.0 - np.prod(np.cos(x / np.sqrt(np.arange(1, dim+1)))) + 1.0

def schwefel(x):
    dim = len(x)
    return 418.9829 * dim - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def levy(x):
    w = 1.0 + (x - 1.0) / 4.0
    term1 = (np.sin(np.pi * w[0]))**2
    term3 = (w[-1] - 1.0)**2 * (1.0 + (np.sin(2 * np.pi * w[-1]))**2)
    term2 = np.sum((w[:-1] - 1.0)**2 * (1.0 + 10.0 * (np.sin(np.pi * w[:-1] + 1.0))**2))
    return term1 + term2 + term3

def zakharov(x):
    dim = len(x)
    i = np.arange(1, dim+1)
    s1 = np.sum(x**2)
    s2 = np.sum(0.5 * i * x)
    return s1 + s2**2 + s2**4

def elliptic(x):
    dim = len(x)
    weights = 10**(6 * np.linspace(0, 1, dim))
    return np.sum(weights * (x**2))

def rastrigin(x):
    return np.sum(x**2 - 10.0 * np.cos(2 * np.pi * x) + 10.0)

def sphere(x):
    return np.sum(x**2)

def composition(x):
    w1, w2 = 0.5, 0.5
    return w1 * ackley(x) + w2 * np.sum(x**2)

# (bazaviy funksiya, lb, ub, optimumdagi z qiymati, turi)
# turi: "shift" - siljitiladi, "rotate" - siljitiladi + aylantiriladi,
#       "flip" - koordinata ishoralari tasodifiy almashtiriladi (Schwefel optimumi
#       diagonaldan chiqariladi va chegara ichida qoladi), "noisy", "dynamic"
PROBLEMS = {
    "Ackley":          (ackley,      -32,   32,  0.0, "shift"),
    "BentCigar":       (bent_cigar,  -100,  100, 0.0, "shift"),
    "Composition":     (composition, -100,  100, 0.0, "shift"),
    "DynamicSphere":   (sphere,      -100,  100, 0.0, "dynamic"),
    "Griewank":        (griewank,    -600,  600, 0.0, "shift"),
    "Levy":            (levy,        -10,   10,  1.0, "shift"),
    "NoisyRastrigin":  (rastrigin,   -5.12, 5.12, 0.0, "noisy"),
    "NoisySphere":     (sphere,      -100,  100, 0.0, "noisy"),
    "Rosenbrock":      (rosenbrock,  -30,   30,  1.0, "shift"),
    "RotatedElliptic": (elliptic,    -100,  100, 0.0, "rotate"),
    "Schwefel":        (schwefel,    -500,  500, 0.0, "flip"),
    "Zakharov":        (zakharov,    -10,   10,  0.0, "shift"),
}
funcs = PROBLEMS

# True: CEC uslubida siljitilgan optimum (koordinata boshiga og'ishni yo'qotadi)
SHIFTED = True
SHIFT_SEED = 20260922
NOISE_AMP = 0.5

def make_problem(name, dim):
    """(obj_func, true_func, lb, ub) qaytaradi. true_func - shovqinsiz qiymat
    (noisy uchun), dynamic uchun None."""
    base, lb, ub, z_opt, kind = PROBLEMS[name]
    rng = np.random.default_rng(SHIFT_SEED + 1000 * dim + list(PROBLEMS).index(name))
    use_shift = SHIFTED and kind != "flip"
    o = rng.uniform(0.8 * lb, 0.8 * ub, dim) if use_shift else np.zeros(dim)
    offset = z_opt if use_shift else 0.0
    signs = rng.choice([-1.0, 1.0], dim) if (SHIFTED and kind == "flip") else np.ones(dim)
    M = None
    if kind == "rotate":
        q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
        M = q * np.sign(np.diag(r))

    def transform(x):
        z = signs * (x - o)
        if M is not None:
            z = M @ z
        return z + offset

    def true_func(x):
        return base(transform(x))

    if kind == "noisy":
        def obj(x):
            return base(transform(x)) + np.random.uniform(-NOISE_AMP, NOISE_AMP)
        return obj, true_func, lb, ub

    if kind == "dynamic":
        counter = [0]
        def obj(x):
            counter[0] += 1
            drift = 5.0 * np.sin(counter[0] / 2000.0)   # optimum FES bo'yicha siljiydi
            return base(transform(x) - drift)
        return obj, None, lb, ub

    return true_func, true_func, lb, ub

class Tracker:
    """Barcha algoritmlar uchun yagona FES hisoblagichi va best-so-far egri chizig'i."""
    def __init__(self, func, max_fes, n_points):
        self.func, self.max_fes = func, max_fes
        self.fes = 0
        self.best_f, self.best_x = np.inf, None
        self.checkpoints = np.linspace(max_fes / n_points, max_fes, n_points).astype(int)
        self.curve = np.full(n_points, np.nan)
        self.k = 0

    def __call__(self, x):
        f = self.func(x)
        self.fes += 1
        if self.fes <= self.max_fes and f < self.best_f:
            self.best_f, self.best_x = f, np.array(x, dtype=float).copy()
        while self.k < len(self.checkpoints) and self.fes >= self.checkpoints[self.k]:
            self.curve[self.k] = self.best_f
            self.k += 1
        return f

    def finalize(self):
        self.curve[self.k:] = self.best_f
        return self.curve

# ==============================================================================
# 2. CBA-SHADE VA RAQOBATCHILAR
# ==============================================================================
# ---- DE oilasi uchun umumiy yordamchi funksiyalar ----------------------------
def _sample_F(mu):
    """Cauchy(mu, 0.1) dan F; F<=0 bo'lsa qayta tanlanadi, yuqoridan 1.0 bilan chegaralanadi.
    mu - har bir individ uchun xotiradan olingan o'rtacha qiymatlar vektori."""
    F = mu + 0.1 * np.random.standard_cauchy(len(mu))
    bad = F <= 0
    while np.any(bad):
        F[bad] = mu[bad] + 0.1 * np.random.standard_cauchy(int(bad.sum()))
        bad = F <= 0
    return np.minimum(F, 1.0)

def _lehmer(vals, w):
    """Vaznlangan Lehmer o'rtachasi (SHADE xotira yangilanishi)."""
    den = np.sum(w * vals)
    return np.sum(w * vals ** 2) / den if den > 0 else -1.0

def _pick_r2(n_union, idx, r1):
    """r2 ni P u A dan tanlash: r2 != i va r2 != r1."""
    n = len(idx)
    r2 = np.random.randint(0, n_union, n)
    clash = (r2 == idx) | (r2 == r1)
    while np.any(clash):
        r2[clash] = np.random.randint(0, n_union, clash.sum())
        clash = (r2 == idx) | (r2 == r1)
    return r2

def _midpoint(U, pop, lb, ub):
    """Midpoint-target chegara tuzatishi (L-SHADE)."""
    low, high = U < lb, U > ub
    U[low] = ((lb + pop) / 2.0)[low]
    U[high] = ((ub + pop) / 2.0)[high]
    return U

def _arch_push(archive, losers, arc_max):
    if arc_max <= 0:
        return archive
    archive = np.vstack([archive, losers]) if len(archive) else losers.copy()
    if len(archive) > arc_max:
        archive = archive[np.random.choice(len(archive), arc_max, replace=False)]
    return archive

def _cma_core(obj_func, dim, bounds, fes, max_fes, xmean, sigma, restart="uniform"):
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

# ------------------------------- ASOSIY ALGORITM ------------------------------
def CBA_SHADE(obj_func, dim, bounds, max_fes, POP_FACTOR=6, N_MIN=4, H_SIZE=6,
              ARC_RATE=2.6, RSP=True, K_RSP=3.0, EIG=True, EIG_FREE=True,
              EIG_PRIOR=True, EIG_LR=0.2, P_MAX=0.25, P_MIN_RATE=0.125,
              JSO_F=True, MEM_INIT="ensemble", TAIL="cma", TAIL_FRAC=0.05,
              TAIL_DIV=1e-3):
    """Covariance-Basis Adaptive SHADE.

    Yadro: current-to-pbest-w/1 + arxiv, muvaffaqiyat tarixi bilan F/CR
    moslashuvi, chiziqli populyatsiya kamayishi (LPSR), midpoint-target chegara.
    Yangi komponentlar:
      (i)   rank asosidagi tanlov bosimi (RSP) - LSHADE-RSP;
      (ii)  kovariatsiya bilan boshqariladigan crossover bazisi moslashuvi;
      (iii) CMA-ES bilan yakuniy lokal aniqlashtirish.
    Parametrlar faqat alohida sozlash to'plamida (10D) tanlangan va muzlatilgan.
    """
    lb, ub = bounds
    N_init = max(40, int(round(POP_FACTOR * dim)))
    pop_size = N_init
    tail_start = int((1.0 - TAIL_FRAC) * max_fes) if TAIL != "none" else max_fes
    span = float(np.mean(ub - lb))
    do_tail = False

    pop = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.array([obj_func(ind) for ind in pop])
    fes = pop_size

    # --- Ikki parametr rejimi (bank) ansambli --------------------------------
    # 0-bank (jSO): M_F=0.3, M_CR=0.8, doimiy terminal katak, F cheklovlari va
    #   Fw vazni - yuqori CR ga moyil, bog'langan/aylantirilgan relyef uchun mos.
    # 1-bank (L-SHADE): M_F=M_CR=0.5, barcha kataklar yangilanadi, F cheklovsiz,
    #   Fw = F - past CR ga yo'l ochadi, separabel relyef uchun mos.
    # Ablatsiya (30D, 15 run) ikkalasining teskari bog'liqligini ko'rsatdi:
    #   Schwefel   (separabel)  jSO 1.18e+02  |  L-SHADE 3.82e-04
    #   RotEllipt. (aylantir.)  jSO 1.41e+02  |  L-SHADE 3.89e+03
    # Shuning uchun rejim qat'iy tanlanmaydi: bank ehtimolligi p_eig bilan bir
    # xil mexanizm orqali - yaxshilanish MIQDORI krediti va rho prior - moslashadi.
    M_F = np.stack([np.full(H_SIZE, 0.3), np.full(H_SIZE, 0.5)])
    M_CR = np.stack([np.full(H_SIZE, 0.8), np.full(H_SIZE, 0.5)])
    M_F[0, -1], M_CR[0, -1] = 0.9, 0.9
    n_upd = (H_SIZE - 1, H_SIZE)
    k_mem = [0, 0]
    p_bank = 0.5                           # 0-bank (jSO) dan foydalanish ehtimolligi
    bank_credit = np.array([0.5, 0.5])
    p_bank_fixed = {"jso": 1.0, "lshade": 0.0}.get(MEM_INIT)
    archive = np.empty((0, dim))
    p_eig = 0.5                            # eigen-bazisda crossover ehtimolligi
    eig_credit = np.array([0.5, 0.5])      # [koordinata bazisi, eigen bazisi]
    B = np.eye(dim)

    while fes < max_fes:
        t = fes / max_fes
        order = np.argsort(fitness)
        pop, fitness = pop[order], fitness[order]
        idx = np.arange(pop_size)

        # --- Kovariatsiya bazisi + separabellik ko'rsatkichi rho --------------
        rho = 0.0
        if EIG:
            C = np.cov(pop[:max(2, pop_size // 2)], rowvar=False)
            if np.all(np.isfinite(C)):
                try:
                    _, B = np.linalg.eigh(C)
                except np.linalg.LinAlgError:
                    pass
                if EIG_PRIOR:
                    sd = np.sqrt(np.clip(np.diag(C), 1e-300, None))
                    R = C / np.outer(sd, sd)
                    off = np.abs(R[~np.eye(dim, dtype=bool)])
                    off = off[np.isfinite(off)]
                    rho = float(np.mean(off)) if off.size else 0.0

        # --- Parametrlar: success-history + jSO cheklovlari -------------------
        pb = p_bank if p_bank_fixed is None else p_bank_fixed
        if EIG_PRIOR and p_bank_fixed is None and t < 0.1:
            pb = 0.5 * p_bank + 0.5 * min(1.0, 2.0 * rho)
        bank = (np.random.rand(pop_size) >= pb).astype(int)   # 0 = jSO, 1 = L-SHADE
        r = np.random.randint(0, H_SIZE, pop_size)
        mu_CR, mu_F = M_CR[bank, r], M_F[bank, r]
        CR = np.clip(np.random.normal(mu_CR, 0.1), 0.0, 1.0)
        CR[mu_CR < 0] = 0.0                # L-SHADE terminal qiymati
        F = _sample_F(mu_F)
        jso = bank == 0
        if JSO_F and t < 0.6:
            F[jso] = np.minimum(F[jso], 0.7)
        Fw = F.copy()
        if JSO_F:
            Fw[jso] *= (0.7 if t < 0.2 else 0.8 if t < 0.4 else 1.2)
        Fc, Fwc = F[:, None], Fw[:, None]

        # --- Donorlar: pbest + rank asosidagi tanlov bosimi (RSP) ------------
        p_num = max(2, int(round((P_MAX - (P_MAX - P_MIN_RATE) * t) * pop_size)))
        x_pbest = pop[np.random.randint(0, p_num, pop_size)]
        if RSP:
            rk = K_RSP * (pop_size - idx) / pop_size + 1.0
            pr = rk / rk.sum()
            r1 = np.random.choice(pop_size, pop_size, p=pr)
            same = r1 == idx
            while np.any(same):
                r1[same] = np.random.choice(pop_size, same.sum(), p=pr)
                same = r1 == idx
        else:
            r1 = (idx + np.random.randint(1, pop_size, pop_size)) % pop_size
        union_pop = np.vstack([pop, archive]) if len(archive) else pop
        r2 = _pick_r2(len(union_pop), idx, r1)
        V = pop + Fwc * (x_pbest - pop) + Fc * (pop[r1] - union_pop[r2])

        # --- Crossover: koordinata bazisi yoki eigen bazisi -------------------
        cross = np.random.rand(pop_size, dim) < CR[:, None]
        cross[idx, np.random.randint(0, dim, pop_size)] = True
        U = np.where(cross, V, pop)
        use_eig = np.zeros(pop_size, dtype=bool)
        if EIG:
            pe = p_eig
            if EIG_PRIOR and t < 0.1:      # boshlanishda rho prior sifatida
                pe = 0.5 * p_eig + 0.5 * min(1.0, 3.0 * rho)
            use_eig = np.random.rand(pop_size) < pe
            if np.any(use_eig):
                U[use_eig] = np.where(cross[use_eig], V[use_eig] @ B,
                                      pop[use_eig] @ B) @ B.T

        U = _midpoint(U, pop, lb, ub)

        n_eval = min(pop_size, max_fes - fes)
        fit_U = np.full(pop_size, np.inf)
        for i in range(n_eval):
            fit_U[i] = obj_func(U[i])
        fes += n_eval

        improved = fit_U < fitness
        accept = fit_U <= fitness

        gain = np.where(improved, np.maximum(fitness - fit_U, 0.0), 0.0)
        gain_total = gain.sum()
        if np.any(improved):
            archive = _arch_push(archive, pop[improved], int(round(ARC_RATE * pop_size)))
            for b in (0, 1):               # har bir bank o'z xotirasini yangilaydi
                mb = improved & (bank == b)
                if not np.any(mb):
                    continue
                df = fitness[mb] - fit_U[mb]
                w = df / df.sum()
                mf = _lehmer(F[mb], w)
                mcr = -1.0 if (M_CR[b, k_mem[b]] == -1 or np.sum(w * CR[mb]) == 0) \
                           else _lehmer(CR[mb], w)
                if b == 1:                 # L-SHADE: to'g'ridan-to'g'ri almashtirish
                    M_F[b, k_mem[b]] = mf
                    M_CR[b, k_mem[b]] = mcr
                else:                      # jSO/iL-SHADE: avvalgisi bilan o'rtachalash
                    M_F[b, k_mem[b]] = (M_F[b, k_mem[b]] + mf) / 2.0
                    M_CR[b, k_mem[b]] = -1.0 if mcr == -1 else (M_CR[b, k_mem[b]] + mcr) / 2.0
                k_mem[b] = (k_mem[b] + 1) % n_upd[b]

        # --- Bank tanlovini moslashtirish (bazis bilan bir xil kredit qoidasi) --
        if p_bank_fixed is None:
            for b in (0, 1):
                mk = bank == b
                share = mk.mean()
                if share > 0:
                    fir = (gain[mk].sum() / gain_total / share) if gain_total > 0 else 0.0
                    bank_credit[b] = (1.0 - EIG_LR) * bank_credit[b] + EIG_LR * fir
            s_bank = bank_credit.sum()
            if s_bank > 0:
                p_bank = float(np.clip(bank_credit[0] / s_bank, 0.02, 0.98))

        # --- Bazis moslashuvi: kredit = yaxshilanish MIQDORI (FIR krediti) ---
        if EIG:
            for j, mk in enumerate([~use_eig, use_eig]):
                share = mk.mean()
                if share > 0:
                    fir = (gain[mk].sum() / gain_total / share) if gain_total > 0 else 0.0
                    eig_credit[j] = (1.0 - EIG_LR) * eig_credit[j] + EIG_LR * fir
            s_cred = eig_credit.sum()
            if s_cred > 0:
                lo, hi = (0.02, 0.98) if EIG_FREE else (0.1, 0.9)
                p_eig = float(np.clip(eig_credit[1] / s_cred, lo, hi))

        pop[accept], fitness[accept] = U[accept], fit_U[accept]

        # --- LPSR + arxiv hajmi ----------------------------------------------
        new_size = max(N_MIN, int(round(N_init + (N_MIN - N_init) * fes / max_fes)))
        if new_size < pop_size:
            keep = np.argsort(fitness)[:new_size]
            pop, fitness = pop[keep], fitness[keep]
            pop_size = new_size
        arc_max = int(round(ARC_RATE * pop_size))
        if len(archive) > arc_max:
            archive = archive[np.random.choice(len(archive), arc_max, replace=False)]

        # --- Yakuniy aniqlashtirishga o'tish SHARTI --------------------------
        # CMA-ES quyrug'i faqat populyatsiya yaqinlashganda foyda beradi: bunda
        # 4 ta individdan iborat DE populyatsiyasidan ko'ra lokal kovariatsiya
        # modeli ancha tezroq aniqlashtiradi. Populyatsiya hali tarqoq bo'lsa
        # (ko'p ekstremumli yoki shovqinli relyef) quyruq zarar qiladi -
        # ablatsiya: NoisyRastrigin 30D da 1.55 -> 14.0. Shuning uchun quyruq
        # tarqalish TAIL_DIV dan past tushgandagina ishga tushadi.
        if TAIL == "cma" and fes >= tail_start:
            if float(np.mean(np.std(pop, axis=0))) / span < TAIL_DIV:
                do_tail = True
                break                      # qolgan byudjetni CMA-ES oladi
            # aks holda DE sikli byudjet oxirigacha davom etadi

    # --- Yakuniy aniqlashtirish: CMA-ES (LSHADE-SPACMA / EBOwithCMAR uslubi) --
    if do_tail and fes < max_fes:
        best_i = np.argmin(fitness)
        sigma0 = float(np.mean(np.std(pop, axis=0))) + 1e-12 * span
        _cma_core(obj_func, dim, bounds, fes, max_fes, pop[best_i].copy(),
                  sigma0, restart="local")

# --------------- RAQOBATCHILAR: ZAMONAVIY DE / ES OILASI ----------------------
# Bu to'plam CBA-SHADE ning bevosita "ota-onalari"ni o'z ichiga oladi: yadro
# L-SHADE va jSO dan, eigen-crossover g'oyasi LSHADE-cnEpSin dan, yakuniy
# aniqlashtirish CMA-ES dan olingan. Ularsiz gibridning hissasi isbotlanmaydi.
def baseline_SHADE(obj_func, dim, bounds, max_fes):
    """Tanabe & Fukunaga (CEC 2013). Success-history DE, LPSR yo'q."""
    lb, ub = bounds
    pop_size, H, p_rate, arc_rate = 100, 100, 0.1, 2.0
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
        F = _sample_F(M_F[r])
        p_i = np.maximum(2, (np.random.uniform(2.0 / pop_size, p_rate, pop_size) * pop_size).astype(int))
        pbest = pop[(np.random.rand(pop_size) * p_i).astype(int)]
        union = np.vstack([pop, archive]) if len(archive) else pop
        r1 = (idx + np.random.randint(1, pop_size, pop_size)) % pop_size
        r2 = _pick_r2(len(union), idx, r1)
        Fc = F[:, None]
        V = pop + Fc * (pbest - pop) + Fc * (pop[r1] - union[r2])
        cross = np.random.rand(pop_size, dim) < CR[:, None]
        cross[idx, np.random.randint(0, dim, pop_size)] = True
        U = _midpoint(np.where(cross, V, pop), pop, lb, ub)
        n_eval = min(pop_size, max_fes - fes)
        fit_U = np.full(pop_size, np.inf)
        for i in range(n_eval):
            fit_U[i] = obj_func(U[i])
        fes += n_eval
        imp = fit_U < fitness
        if np.any(imp):
            df = fitness[imp] - fit_U[imp]
            w = df / df.sum()
            archive = _arch_push(archive, pop[imp], int(round(arc_rate * pop_size)))
            M_F[k] = _lehmer(F[imp], w)
            M_CR[k] = -1.0 if (M_CR[k] == -1 or np.sum(w * CR[imp]) == 0) else _lehmer(CR[imp], w)
            k = (k + 1) % H
        acc = fit_U <= fitness
        pop[acc], fitness[acc] = U[acc], fit_U[acc]

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
        F = _sample_F(M_F[r])
        p_num = max(2, int(round(p_rate * pop_size)))
        pbest = pop[np.random.randint(0, p_num, pop_size)]
        union = np.vstack([pop, archive]) if len(archive) else pop
        r1 = (idx + np.random.randint(1, pop_size, pop_size)) % pop_size
        r2 = _pick_r2(len(union), idx, r1)
        Fc = F[:, None]
        V = pop + Fc * (pbest - pop) + Fc * (pop[r1] - union[r2])
        cross = np.random.rand(pop_size, dim) < CR[:, None]
        cross[idx, np.random.randint(0, dim, pop_size)] = True
        U = _midpoint(np.where(cross, V, pop), pop, lb, ub)
        n_eval = min(pop_size, max_fes - fes)
        fit_U = np.full(pop_size, np.inf)
        for i in range(n_eval):
            fit_U[i] = obj_func(U[i])
        fes += n_eval
        imp = fit_U < fitness
        if np.any(imp):
            df = fitness[imp] - fit_U[imp]
            w = df / df.sum()
            archive = _arch_push(archive, pop[imp], int(round(arc_rate * pop_size)))
            M_F[k] = _lehmer(F[imp], w)
            M_CR[k] = -1.0 if (M_CR[k] == -1 or np.sum(w * CR[imp]) == 0) else _lehmer(CR[imp], w)
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
        F = _sample_F(M_F[r])
        if t < 0.6:
            F = np.minimum(F, 0.7)
        Fw = F * (0.7 if t < 0.2 else 0.8 if t < 0.4 else 1.2)
        p_num = max(2, int(round(0.25 * (1.0 - 0.5 * t) * pop_size)))
        pbest = pop[np.random.randint(0, p_num, pop_size)]
        union = np.vstack([pop, archive]) if len(archive) else pop
        r1 = (idx + np.random.randint(1, pop_size, pop_size)) % pop_size
        r2 = _pick_r2(len(union), idx, r1)
        Fc, Fwc = F[:, None], Fw[:, None]
        V = pop + Fwc * (pbest - pop) + Fc * (pop[r1] - union[r2])
        cross = np.random.rand(pop_size, dim) < CR[:, None]
        cross[idx, np.random.randint(0, dim, pop_size)] = True
        U = _midpoint(np.where(cross, V, pop), pop, lb, ub)
        n_eval = min(pop_size, max_fes - fes)
        fit_U = np.full(pop_size, np.inf)
        for i in range(n_eval):
            fit_U[i] = obj_func(U[i])
        fes += n_eval
        imp = fit_U < fitness
        if np.any(imp):
            df = fitness[imp] - fit_U[imp]
            w = df / df.sum()
            archive = _arch_push(archive, pop[imp], int(round(arc_rate * pop_size)))
            mf = _lehmer(F[imp], w)
            mcr = -1.0 if (M_CR[k] == -1 or np.sum(w * CR[imp]) == 0) else _lehmer(CR[imp], w)
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
            F = _sample_F(M_F[r])
        p_num = max(2, int(round(p_rate * pop_size)))
        pbest = pop[np.random.randint(0, p_num, pop_size)]
        union = np.vstack([pop, archive]) if len(archive) else pop
        r1 = (idx + np.random.randint(1, pop_size, pop_size)) % pop_size
        r2 = _pick_r2(len(union), idx, r1)
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
        U = _midpoint(U, pop, lb, ub)
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
            archive = _arch_push(archive, pop[imp], int(round(arc_rate * pop_size)))
            M_F[k] = _lehmer(F[imp], w)
            M_CR[k] = -1.0 if (M_CR[k] == -1 or np.sum(w * CR[imp]) == 0) else _lehmer(CR[imp], w)
            if freq_i is not None and np.any(imp & (cfg == 1)):
                wf = df[cfg[imp] == 1]
                if wf.sum() > 0:
                    M_freq[k] = _lehmer(freq_i[imp][cfg[imp] == 1], wf / wf.sum())
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
    _cma_core(obj_func, dim, bounds, 0, max_fes, x0,
              0.3 * float(np.mean(ub - lb)), restart="uniform")

# ------------- RAQOBATCHILAR: KLASSIK / METAFORA ASOSIDAGI TO'PLAM ------------
# Eski adabiyot bilan bog'lash uchun saqlangan; INCLUDE_CLASSIC bilan yoqiladi.
# Eslatma: konvergensiya Tracker orqali yoziladi; gbest qiymati qayta hisoblanmaydi.
def baseline_DE(obj_func, dim, bounds, max_fes):
    """DE/rand/1/bin (Storn & Price, 1997), F=0.5, CR=0.8, NP=30."""
    pop_size, (lb, ub) = 30, bounds
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([obj_func(i) for i in pop])
    fes = pop_size
    while fes < max_fes:
        for i in range(pop_size):
            if fes >= max_fes: break
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + 0.5 * (b - c), lb, ub)
            cross = np.random.rand(dim) < 0.8
            if not np.any(cross): cross[np.random.randint(0, dim)] = True
            trial = np.where(cross, mutant, pop[i])
            f_trial = obj_func(trial)
            fes += 1
            if f_trial < fitness[i]: pop[i], fitness[i] = trial, f_trial

def baseline_GWO(obj_func, dim, bounds, max_fes):
    pop_size, (lb, ub) = 30, bounds
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([obj_func(ind) for ind in pop])
    fes = pop_size
    while fes < max_fes:
        s_idx = np.argsort(fitness)
        pop, fitness = pop[s_idx], fitness[s_idx]
        a = 2.0 - fes * (2.0 / max_fes)
        for i in range(pop_size):
            if fes >= max_fes: break
            X_new = np.zeros(dim)
            for lead in [pop[0], pop[1], pop[2]]:
                A, C = 2 * a * np.random.rand(dim) - a, 2 * np.random.rand(dim)
                X_new += lead - A * np.abs(C * lead - pop[i])
            step = np.clip(X_new / 3.0, lb, ub)
            pop[i], fitness[i] = step, obj_func(step)
            fes += 1

def baseline_WOA(obj_func, dim, bounds, max_fes):
    pop_size, (lb, ub) = 30, bounds
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([obj_func(ind) for ind in pop])
    fes = pop_size
    while fes < max_fes:
        X_best = pop[np.argmin(fitness)].copy()
        a, a2 = 2.0 - fes * (2.0 / max_fes), -1.0 + fes * (-1.0 / max_fes)
        for i in range(pop_size):
            if fes >= max_fes: break
            A, C = 2 * a * np.random.rand() - a, 2 * np.random.rand()
            l = (a2 - 1) * np.random.rand() + 1
            if np.random.rand() < 0.5:
                if abs(A) >= 1:
                    X_r = pop[np.random.randint(0, pop_size)]
                    step = X_r - A * np.abs(C * X_r - pop[i])
                else: step = X_best - A * np.abs(C * X_best - pop[i])
            else:
                step = np.abs(X_best - pop[i]) * np.exp(l) * np.cos(2 * np.pi * l) + X_best
            step = np.clip(step, lb, ub)
            pop[i], fitness[i] = step, obj_func(step)
            fes += 1

def baseline_PSO(obj_func, dim, bounds, max_fes):
    pop_size, (lb, ub) = 30, bounds
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    vel = np.zeros_like(pop)
    pbest, pbest_fit = pop.copy(), np.array([obj_func(ind) for ind in pop])
    g = np.argmin(pbest_fit)
    gbest, gbest_fit = pbest[g].copy(), pbest_fit[g]
    fes = pop_size
    while fes < max_fes:
        w = 0.9 - (0.5) * (fes / max_fes)
        for i in range(pop_size):
            if fes >= max_fes: break
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            vel[i] = w * vel[i] + 2.0 * r1 * (pbest[i] - pop[i]) + 2.0 * r2 * (gbest - pop[i])
            pop[i] = np.clip(pop[i] + vel[i], lb, ub)
            fit = obj_func(pop[i])
            fes += 1
            if fit < pbest_fit[i]:
                pbest[i], pbest_fit[i] = pop[i].copy(), fit
                if fit < gbest_fit: gbest, gbest_fit = pop[i].copy(), fit

def baseline_SCA(obj_func, dim, bounds, max_fes):
    pop_size, (lb, ub) = 30, bounds
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([obj_func(ind) for ind in pop])
    X_best, X_best_fit = pop[np.argmin(fitness)].copy(), np.min(fitness)
    fes = pop_size
    while fes < max_fes:
        r1 = 2.0 - fes * (2.0 / max_fes)
        for i in range(pop_size):
            if fes >= max_fes: break
            r2, r3, r4 = 2 * np.pi * np.random.rand(dim), 2 * np.random.rand(dim), np.random.rand(dim)
            mask = r4 < 0.5
            step = np.zeros(dim)
            step[mask] = pop[i][mask] + r1 * np.sin(r2[mask]) * np.abs(r3[mask] * X_best[mask] - pop[i][mask])
            step[~mask] = pop[i][~mask] + r1 * np.cos(r2[~mask]) * np.abs(r3[~mask] * X_best[~mask] - pop[i][~mask])
            step = np.clip(step, lb, ub)
            fit = obj_func(step)
            fes += 1
            if fit < fitness[i]:
                pop[i], fitness[i] = step, fit
                if fit < X_best_fit: X_best, X_best_fit = step.copy(), fit

def baseline_HHO(obj_func, dim, bounds, max_fes):
    pop_size, (lb, ub) = 30, bounds
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([obj_func(ind) for ind in pop])
    X_rabbit, rabbit_fit = pop[np.argmin(fitness)].copy(), np.min(fitness)
    fes = pop_size
    while fes < max_fes:
        E1 = 2 * (1 - (fes / max_fes))
        for i in range(pop_size):
            if fes >= max_fes: break
            E = E1 * (2 * np.random.rand() - 1)
            if abs(E) >= 1:
                if np.random.rand() >= 0.5:
                    X_rand = pop[np.random.randint(0, pop_size)]
                    step = X_rand - np.random.rand() * abs(X_rand - 2 * np.random.rand() * pop[i])
                else:
                    step = (X_rabbit - np.mean(pop, axis=0)) - np.random.rand() * (lb + np.random.rand() * (ub - lb))
            else:
                r = np.random.rand()
                if r >= 0.5 and abs(E) >= 0.5: step = (X_rabbit - pop[i]) - E * abs(2 * (1 - np.random.rand()) * X_rabbit - pop[i])
                elif r >= 0.5 and abs(E) < 0.5: step = X_rabbit - E * abs(X_rabbit - pop[i])
                elif r < 0.5 and abs(E) >= 0.5: step = X_rabbit - E * abs(2 * (1 - np.random.rand()) * X_rabbit - pop[i])
                else: step = X_rabbit - E * abs(2 * (1 - np.random.rand()) * X_rabbit - np.mean(pop, axis=0))
            step = np.clip(step, lb, ub)
            fit = obj_func(step)
            fes += 1
            if fit < fitness[i]:
                pop[i], fitness[i] = step, fit
                if fit < rabbit_fit: X_rabbit, rabbit_fit = step.copy(), fit

# ------------------------------ ALGORITMLAR RO'YXATI --------------------------
# INCLUDE_CLASSIC=True bo'lsa metafora asosidagi eski to'plam ham qo'shiladi
# (maqolaning "klassik algoritmlar bilan taqqoslash" jadvali uchun).
INCLUDE_CLASSIC = os.environ.get("INCLUDE_CLASSIC", "0") == "1"

MODERN = {
    "L-SHADE":        baseline_LSHADE,
    "jSO":            baseline_jSO,
    "LSHADE-cnEpSin": baseline_LSHADE_cnEpSin,
    "SHADE":          baseline_SHADE,
    "CMA-ES":         baseline_CMAES,
    "DE":             baseline_DE,
}
CLASSIC = {
    "GWO": baseline_GWO, "WOA": baseline_WOA,
    "PSO": baseline_PSO, "SCA": baseline_SCA, "HHO": baseline_HHO,
}

algorithms = {ALGO_NAME: CBA_SHADE}
algorithms.update(MODERN)
if INCLUDE_CLASSIC:
    algorithms.update(CLASSIC)
target = ALGO_NAME

# ==============================================================================
# 3. YUKORI TEZLIKDAGI PARALLEL EKSPERIMENT
# ==============================================================================
# To'liq protokol: 12 funksiya x {30, 50, 100} D x 30 run. Bu Colab kabi
# 2 yadroli muhitda bir necha soat oladi, shuning uchun tajribani bo'laklab
# ishlatish mumkin (quyidagi muhit o'zgaruvchilari), keyin MERGE bilan
# barcha bo'laklarni birlashtirib yagona statistika va grafiklar olinadi.
#
#   QUICK=1            10D x 3 run smoke-test
#   DIMS=30            faqat 30D (vergul bilan: DIMS=30,50)
#   RUNS=30            run soni
#   OUT_DIR=nom        chiqish papkasi nomi
#   MERGE=d1,d2        avvalgi bo'laklarning papkalarini qo'shib tahlil qilish
#   ANALYZE_ONLY=1     eksperimentni o'tkazmay, faqat MERGE dagi tayyor
#                      bo'laklardan jadval, statistika va grafiklarni qurish
#   PRECISION_FLOOR=0  CEC aniqlik chegarasini o'chirish (standart 1e-8)
#   INCLUDE_CLASSIC=1  metafora asosidagi eski algoritmlarni ham qo'shish
QUICK = os.environ.get("QUICK", os.environ.get("TEMOA_QUICK", "0")) == "1"
dimensions = [10] if QUICK else [30, 50, 100]
runs = 3 if QUICK else 30
if os.environ.get("DIMS"):
    dimensions = [int(x) for x in os.environ["DIMS"].split(",")]
if os.environ.get("RUNS"):
    runs = int(os.environ["RUNS"])
FES_PER_DIM = 3000
N_POINTS = 100
SEED_BASE = 42
MERGE_DIRS = [d.strip().rstrip("/") for d in os.environ.get("MERGE", "").split(",") if d.strip()]
ANALYZE_ONLY = os.environ.get("ANALYZE_ONLY") == "1"
# CEC musobaqalari konventsiyasi: 1e-8 dan kichik xato 0 deb qabul qilinadi.
# Busiz ranklar mashina aniqligidagi ma'nosiz farqlar bilan buziladi - masalan
# 1.5e-32 va 4.1e-31 "g'alaba/mag'lubiyat" sifatida sanaladi, holbuki ikkalasi
# ham aniq yechim. Xom qiymatlar full_raw_results.csv da o'zgarishsiz saqlanadi.
PRECISION_FLOOR = float(os.environ.get("PRECISION_FLOOR", "1e-8"))
_slug = ALGO_NAME.lower().replace("-", "_")
_tag = "_" + "_".join(str(d) for d in dimensions) + "D" if os.environ.get("DIMS") else ""
out_dir = os.environ.get("OUT_DIR") or (f"{_slug}_quicktest{_tag}" if QUICK
                                        else f"{_slug}_outputs{_tag}")

def run_task(alg_name, func_name, dim, run_id):
    # Bir xil (dim, run) uchun hamma algoritmlarga bir xil seed
    np.random.seed(SEED_BASE + 1000 * dim + run_id)
    obj, true_obj, lb, ub = make_problem(func_name, dim)
    max_fes = FES_PER_DIM * dim
    tracker = Tracker(obj, max_fes, N_POINTS)
    algorithms[alg_name](tracker, dim, (lb, ub), max_fes)
    curve = tracker.finalize()
    final = true_obj(tracker.best_x) if true_obj is not None else tracker.best_f
    return alg_name, func_name, dim, run_id, curve, final

def complexity_analysis(dims, func_name="Rosenbrock", n_fes=200000, reps=5):
    """CEC uslubidagi hisoblash murakkabligi jadvali (T0/T1/T2).

    T0 - standart arifmetik siklning vaqti (mashinaga bog'liq mos yozuvlar);
    T1 - n_fes ta funksiya chaqiruvining vaqti;
    T2 - algoritmning o'sha byudjetdagi o'rtacha vaqti (reps ta takror).
    (T2 - T1) / T0 - algoritmning o'zi qo'shadigan, mashinadan mustaqil yuk.
    """
    t0_start = time.time()
    x = 0.55
    for _ in range(1000000):
        x = x + x; x = x / 2.0; x = x * x; x = math.sqrt(x)
        x = math.log(x) if x > 0 else 0.55
        x = math.exp(x); x = x / (x + 2.0)
    T0 = time.time() - t0_start

    rows = []
    for d in dims:
        obj, _, lb, ub = make_problem(func_name, d)
        probe = lb + np.random.rand(n_fes, d) * (ub - lb)
        t1 = time.time()
        for i in range(n_fes):
            obj(probe[i])
        T1 = time.time() - t1

        t2 = []
        for rep in range(reps):
            np.random.seed(SEED_BASE + rep)
            tracker = Tracker(make_problem(func_name, d)[0], n_fes, 2)
            start = time.time()
            algorithms[target](tracker, d, (lb, ub), n_fes)
            t2.append(time.time() - start)
        T2 = float(np.mean(t2))
        rows.append({"Dimension": d, "T0": T0, "T1": T1, "T2_hat": T2,
                     "(T2_hat-T1)/T0": (T2 - T1) / T0})
    return pd.DataFrame(rows)

def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj

if __name__ == "__main__":
    for d in ["tables", "figures", "raw_data"]: os.makedirs(f"{out_dir}/{d}", exist_ok=True)

    if ANALYZE_ONLY:
        if not MERGE_DIRS:
            raise SystemExit("[!] ANALYZE_ONLY=1 uchun MERGE=... ko'rsatilishi shart")
        print(f"[*] ANALYZE_ONLY: eksperiment o'tkazilmaydi, MERGE dan tahlil qilinadi")
        results_raw = []
    else:
        tasks = [(a, f, d, r) for a in algorithms for f in funcs for d in dimensions for r in range(runs)]
        print(f"[*] {ALGO_NAME} experiment. Total tasks: {len(tasks)} "
              f"(QUICK={QUICK}, SHIFTED={SHIFTED}, CLASSIC={INCLUDE_CLASSIC})")
        start_time = time.time()
        results_raw = Parallel(n_jobs=-1, backend="loky", verbose=5)(
            delayed(run_task)(a, f, d, r) for (a, f, d, r) in tasks)
        print(f"[*] Execution completed in {time.time() - start_time:.2f} seconds.")

    # ==========================================================================
    # 4. MA'LUMOTLARNI EKSPORT QILISH
    # ==========================================================================
    summary_data = []
    curves = {f: {d: {a: [] for a in algorithms} for d in dimensions} for f in funcs}
    for alg_name, func_name, dim, run_id, curve, final in results_raw:
        summary_data.append({"Algorithm": alg_name, "Function": func_name, "Dimension": dim,
                             "Run": run_id, "Best": final})
        curves[func_name][dim][alg_name].append(curve)

    df_raw = pd.DataFrame(summary_data,
                          columns=["Algorithm", "Function", "Dimension", "Run", "Best"])
    curve_store = {f"{f}|{d}|{a}": np.array(curves[f][d][a])
                   for f in funcs for d in dimensions for a in algorithms
                   if curves[f][d][a]}

    # Avvalgi bo'laklarni qo'shish (Colab uchun: har bir o'lcham alohida sessiyada)
    for mdir in MERGE_DIRS:
        m_csv = f"{mdir}/raw_data/full_raw_results.csv"
        if not os.path.exists(m_csv):
            print(f"[!] MERGE: {m_csv} topilmadi, o'tkazib yuborildi")
            continue
        df_raw = pd.concat([df_raw, pd.read_csv(m_csv)], ignore_index=True)
        m_npz = f"{mdir}/raw_data/curves.npz"
        if os.path.exists(m_npz):
            with np.load(m_npz) as z:
                curve_store.update({k: z[k] for k in z.files})
        print(f"[*] MERGE: {mdir} qo'shildi")
    df_raw = df_raw.drop_duplicates(subset=["Algorithm", "Function", "Dimension", "Run"])
    # MERGE dan keyin (ayniqsa ANALYZE_ONLY da) ustunlar object turida qolishi
    # mumkin - statistik testlar buni qabul qilmaydi.
    df_raw["Best"] = pd.to_numeric(df_raw["Best"], errors="coerce")
    for col in ("Dimension", "Run"):
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce").astype("int64")
    df_raw = df_raw.dropna(subset=["Best"])
    if df_raw.empty:
        raise SystemExit("[!] Tahlil qilinadigan natija yo'q (MERGE papkalarini tekshiring)")
    dimensions = sorted(df_raw["Dimension"].unique())
    alg_names = [target] + [a for a in df_raw["Algorithm"].unique() if a != target]

    df_raw.to_csv(f"{out_dir}/raw_data/full_raw_results.csv", index=False)
    np.savez_compressed(f"{out_dir}/raw_data/curves.npz", **curve_store)

    if PRECISION_FLOOR > 0:
        n_floor = int((df_raw["Best"] < PRECISION_FLOOR).sum())
        df_raw["Best"] = df_raw["Best"].where(df_raw["Best"] >= PRECISION_FLOOR, 0.0)
        print(f"[*] CEC aniqlik chegarasi {PRECISION_FLOOR:g}: {n_floor} ta natija "
              f"0 ga tenglashtirildi (xom qiymatlar raw_data/ da saqlandi)")

    df_summary = df_raw.groupby(["Function", "Dimension", "Algorithm"])["Best"].agg(["mean", "std"]).reset_index()
    df_summary.to_csv(f"{out_dir}/tables/summary_mean_std.csv", index=False)
    df_pivot = df_summary.pivot(index=["Function", "Dimension"], columns="Algorithm", values="mean")
    df_pivot = df_pivot[alg_names]
    n_before = len(df_pivot)
    df_pivot = df_pivot.dropna()
    if len(df_pivot) < n_before:
        print(f"[!] {n_before - len(df_pivot)} ta instansiya to'liq emas (ba'zi "
              f"algoritmlar ishlatilmagan) - rank tahlilidan chiqarildi")
    df_pivot.to_csv(f"{out_dir}/tables/summary_results.csv")
    try:
        df_pivot.to_latex(f"{out_dir}/tables/summary_table.tex", float_format="%.2e")
    except ImportError as e:
        print(f"[!] LaTeX jadval yozilmadi: {e}")

    # Friedman testi + o'rtacha rank
    ranks = df_pivot.rank(axis=1, method="average")
    avg_ranks = ranks.mean(axis=0).sort_values().reset_index()
    avg_ranks.columns = ["Algorithm", "Average_Rank"]
    avg_ranks.to_csv(f"{out_dir}/tables/overall_average_ranks.csv", index=False)
    fr_stat, fr_p = friedmanchisquare(*[df_pivot[a].values for a in alg_names])
    with open(f"{out_dir}/tables/friedman_test.txt", "w") as fh:
        fh.write(f"Friedman chi2 = {fr_stat:.4f}, p = {fr_p:.4e}\n")
        fh.write(avg_ranks.to_string(index=False))
    print(avg_ranks.to_string(index=False))
    print(f"[*] Friedman chi2 = {fr_stat:.4f}, p = {fr_p:.4e}")

    # Friedman post-hoc: nazorat algoritmi = ALGO_NAME (Bonferroni-Dunn + Holm).
    # Q1 darajadagi jurnallar aynan shu tahlilni talab qiladi: Friedman testining
    # o'zi faqat "algoritmlar farq qiladi" deydi, ustunlikni post-hoc isbotlaydi.
    k_alg, n_inst = df_pivot.shape[1], df_pivot.shape[0]
    R = avg_ranks.set_index("Algorithm")["Average_Rank"]
    SE = math.sqrt(k_alg * (k_alg + 1) / (6.0 * n_inst))
    others = [a for a in R.index if a != target]
    z_raw = [(R[a] - R[target]) / SE for a in others]
    p_raw = [2.0 * (1.0 - norm.cdf(abs(z))) for z in z_raw]
    p_holm = holm(p_raw)
    df_post = pd.DataFrame({
        "Algorithm": others,
        "Average_Rank": [R[a] for a in others],
        "Rank_diff_vs_control": [R[a] - R[target] for a in others],
        "z": z_raw, "p_unadjusted": p_raw, "p_Holm": p_holm,
        "significant_0.05": [p < 0.05 for p in p_holm],
    }).sort_values("Average_Rank")
    df_post.to_csv(f"{out_dir}/tables/friedman_posthoc.csv", index=False)
    print(f"[*] Friedman post-hoc (nazorat = {target}, SE = {SE:.4f}):")
    print(df_post.to_string(index=False))

    # Wilcoxon rank-sum (Mann-Whitney U) + Holm tuzatishi, +/=/- hisoblari
    competitors = [a for a in alg_names if a != target]
    pvals, wtl = [], {c: {"+": 0, "=": 0, "-": 0} for c in competitors}
    for f in funcs:
        for d in dimensions:
            sel = (df_raw["Function"] == f) & (df_raw["Dimension"] == d)
            v_target = df_raw[sel & (df_raw["Algorithm"] == target)]["Best"].values
            raw_p, signs = [], []
            for c in competitors:
                v_comp = df_raw[sel & (df_raw["Algorithm"] == c)]["Best"].values
                try:
                    _, p = mannwhitneyu(v_target, v_comp, alternative="two-sided")
                except ValueError:
                    p = 1.0
                raw_p.append(1.0 if np.isnan(p) else p)
                signs.append(np.median(v_target) < np.median(v_comp))
            adj = holm(raw_p)
            row = {"Function": f, "Dimension": d}
            for c, p, better in zip(competitors, adj, signs):
                row[c] = p
                wtl[c]["=" if p >= 0.05 else ("+" if better else "-")] += 1
            pvals.append(row)
    pd.DataFrame(pvals).to_csv(f"{out_dir}/tables/wilcoxon_pvalues.csv", index=False)
    df_wtl = pd.DataFrame(wtl).T[["+", "=", "-"]]
    df_wtl.to_csv(f"{out_dir}/tables/win_tie_loss.csv")
    print(f"[*] {ALGO_NAME} vs raqiblar (+ yutdi / = teng / - yutqazdi):")
    print(df_wtl.to_string())

    # Konvergensiya grafiklari (X o'qi: FES)
    x_fes = lambda d: np.linspace(FES_PER_DIM * d / N_POINTS, FES_PER_DIM * d, N_POINTS)
    for f in funcs:
        offset = NOISE_AMP if PROBLEMS[f][4] == "noisy" else 0.0
        for d in dimensions:
            plt.figure(figsize=(10, 6))
            for a in alg_names:
                key = f"{f}|{d}|{a}"
                if key not in curve_store or len(curve_store[key]) == 0:
                    continue
                avg_conv = np.mean(curve_store[key], axis=0) + offset
                plt.plot(x_fes(d), np.maximum(avg_conv, 1e-300), label=a,
                         linewidth=2.5 if a == target else 1.0)
            plt.yscale("log")
            plt.title(f"Convergence on {f} ({d}D)")
            plt.xlabel("Function evaluations (FES)")
            plt.ylabel("Best-so-far error" + (" (+noise offset)" if offset else ""))
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{out_dir}/figures/Conv_{f}_{d}D.png", dpi=300)
            plt.close()

    # CEC uslubidagi hisoblash murakkabligi jadvali (COMPLEXITY=1 bilan yoqiladi)
    if os.environ.get("COMPLEXITY") == "1":
        print("[*] Hisoblash murakkabligi o'lchanmoqda (T0/T1/T2)...")
        df_cx = complexity_analysis(dimensions)
        df_cx.to_csv(f"{out_dir}/tables/complexity.csv", index=False)
        print(df_cx.to_string(index=False))

    print(f"[*] RESULTS EXPORTED to {out_dir}/")
