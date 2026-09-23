import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import mannwhitneyu, friedmanchisquare
import math
import os
import time
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
# TEMOA_V10_HYBRID - GIBRID ALGORITM TARKIBI
# ------------------------------------------------------------------------------
# V9 dagi kuchsiz qismlar va ularning o'rniga qo'yilgan kuchli komponentlar:
#
#  V9 (kuchsiz)                          ->  V10 (kuchli, manba)
#  ------------------------------------------------------------------------------
#  |E|>=1 bo'lsagina DE (t>0.5 da o'chadi) -> Adaptiv operator portfeli: har bir
#                                            operator muvaffaqiyat ulushiga qarab
#                                            ehtimollik oladi (Adaptive Operator
#                                            Selection, probability matching)
#  current-to-lead/1 (X_lead)             -> current-to-pbest-w/1 + arxiv
#                                            (L-SHADE / jSO yadrosi)
#  GWO encircling |C*X_alpha - x|         -> Translyatsiyaga invariant lider
#  (koordinata boshiga og'ish)               yo'naltirishi: x + F(X_lead - x) + F*diff
#  WOA spiral X_lead atrofida             -> WOA spirali x_pbest atrofida (diversity)
#  Yo'q                                   -> HHO Levy-flight "rapid dive"
#                                            (og'ir dumli sakrashlar, lokal
#                                            minimumdan chiqish)
#  Faqat binomial crossover               -> Adaptiv eigen-crossover (kovariatsiya
#                                            bazisida, LSHADE-cnEpSin); binomial va
#                                            eigen o'rtasidagi ulush muvaffaqiyatga
#                                            qarab moslashadi (separabel va
#                                            aylantirilgan masalalar uchun)
#  Vaznsiz Lehmer mean                    -> |df| bilan vaznlangan Lehmer mean +
#                                            jSO F/CR cheklovlari va xotira
#  Reflect + clip                         -> Midpoint-target chegara (L-SHADE)
#  (ABC "scout" sinovdan o'tkazildi, 10D sozlashda zarar qilgani uchun standart
#   holatda o'chiq: STAG_LIMIT=None)
#  Yo'q                                  -> Oxirgi 5% byudjet: (1+1)-ES, 1/5 qoidasi
#                                            (Rechenberg) bilan lokal aniqlashtirish
#  LPSR (saqlangan)                       -> LPSR (L-SHADE)
# ==============================================================================

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
# 2. TEMOA_V10_HYBRID VA RAQOBATCHILAR
# ==============================================================================
def levy_flight(shape, beta=1.5):
    # Mantegna algoritmi
    sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
               (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0.0, sigma_u, shape)
    v = np.random.normal(0.0, 1.0, shape)
    return u / np.abs(v) ** (1 / beta)

def TEMOA_V10_HYBRID(obj_func, dim, bounds, max_fes, POP_FACTOR=6, POP_MAX=500, P_MIN=0.05,
                     P_EIG=0.4, ADAPT_EIG=True, ARC_RATE=1.0, STAG_LIMIT=None, LS_FRACTION=0.05,
                     CR_FLOOR=True):
    # Parametrlar faqat alohida sozlash to'plamida (10D) tanlanadi va keyin muzlatiladi
    lb, ub = bounds
    N_init = int(np.clip(round(POP_FACTOR * dim), 40, POP_MAX))
    N_min = 4
    H_SIZE = 6
    K_OPS = 4            # 0: pbest-DE, 1: lider-DE (GWO), 2: spiral (WOA), 3: Levy (HHO)
    ls_start = int((1.0 - LS_FRACTION) * max_fes)

    pop_size = N_init
    pop = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.array([obj_func(ind) for ind in pop])
    fes = pop_size

    M_F = np.full(H_SIZE, 0.3)
    M_CR = np.full(H_SIZE, 0.8)
    M_F[-1], M_CR[-1] = 0.9, 0.9      # jSO: oxirgi xotira katagi doimiy
    k_mem = 0
    archive = np.empty((0, dim))
    op_quality = np.full(K_OPS, 0.5)
    op_prob = np.full(K_OPS, 1.0 / K_OPS)
    eig_quality = np.array([0.5, 0.5])      # [binomial, eigen] muvaffaqiyat ulushi
    stag = np.zeros(pop_size, dtype=int)
    B = np.eye(dim)
    idx = np.arange(pop_size)

    while fes < ls_start:
        t = fes / max_fes
        order = np.argsort(fitness)
        pop, fitness, stag = pop[order], fitness[order], stag[order]
        idx = np.arange(pop_size)

        # Eigen bazis: eng yaxshi yarmining kovariatsiyasi
        C = np.cov(pop[:max(2, pop_size // 2)], rowvar=False)
        if np.all(np.isfinite(C)):
            _, B = np.linalg.eigh(C)

        # Parametrlar: success-history + jSO cheklovlari
        r = np.random.randint(0, H_SIZE, pop_size)
        CR = np.clip(np.random.normal(M_CR[r], 0.1), 0.0, 1.0)
        if CR_FLOOR:
            if t < 0.25: CR = np.maximum(CR, 0.7)
            elif t < 0.5: CR = np.maximum(CR, 0.6)
        F = M_F[r] + 0.1 * np.random.standard_cauchy(pop_size)
        bad = F <= 0
        while np.any(bad):
            F[bad] = M_F[r[bad]] + 0.1 * np.random.standard_cauchy(bad.sum())
            bad = F <= 0
        F = np.minimum(F, 1.0)
        if t < 0.6: F = np.minimum(F, 0.7)
        Fw = F * (0.7 if t < 0.2 else 0.8 if t < 0.4 else 1.2)
        Fc, Fwc = F[:, None], Fw[:, None]

        # Donorlar
        p_num = max(2, int(round((0.25 - 0.20 * t) * pop_size)))
        x_pbest = pop[np.random.randint(0, p_num, pop_size)]
        r1 = (idx + np.random.randint(1, pop_size, pop_size)) % pop_size
        union_pop = np.vstack([pop, archive]) if len(archive) else pop
        r2 = np.random.randint(0, len(union_pop), pop_size)
        clash = (r2 == idx) | (r2 == r1)
        while np.any(clash):
            r2[clash] = np.random.randint(0, len(union_pop), clash.sum())
            clash = (r2 == idx) | (r2 == r1)
        diff = pop[r1] - union_pop[r2]
        X_lead = np.array([0.5, 0.3, 0.2]) @ pop[:3]

        # Operator portfeli
        ops = np.random.choice(K_OPS, pop_size, p=op_prob)
        V = np.empty_like(pop)
        m = ops == 0
        V[m] = pop[m] + Fwc[m] * (x_pbest[m] - pop[m]) + Fc[m] * diff[m]
        m = ops == 1
        V[m] = pop[m] + Fc[m] * (X_lead - pop[m]) + Fc[m] * diff[m]
        m = ops == 2
        l = np.random.uniform(-1.0, 1.0, (m.sum(), 1))
        V[m] = x_pbest[m] + np.abs(x_pbest[m] - pop[m]) * np.exp(l) * np.cos(2.0 * np.pi * l)
        m = ops == 3
        L = np.clip(levy_flight((m.sum(), dim)), -5.0, 5.0)
        V[m] = pop[m] + Fc[m] * (x_pbest[m] - pop[m]) + Fc[m] * L * diff[m]

        # Crossover: binomial yoki eigen-bazisda binomial
        cross = np.random.rand(pop_size, dim) < CR[:, None]
        cross[idx, np.random.randint(0, dim, pop_size)] = True
        U = np.where(cross, V, pop)
        use_eig = np.random.rand(pop_size) < P_EIG
        if np.any(use_eig):
            xe, ve = pop[use_eig] @ B, V[use_eig] @ B
            U[use_eig] = np.where(cross[use_eig], ve, xe) @ B.T

        # Midpoint-target chegara
        low, high = U < lb, U > ub
        U[low] = ((lb + pop) / 2.0)[low]
        U[high] = ((ub + pop) / 2.0)[high]

        n_eval = min(pop_size, max_fes - fes)
        fit_U = np.full(pop_size, np.inf)
        for i in range(n_eval):
            fit_U[i] = obj_func(U[i])
        fes += n_eval

        improved = fit_U < fitness
        accept = fit_U <= fitness

        if np.any(improved):
            archive = np.vstack([archive, pop[improved]])
            df = fitness[improved] - fit_U[improved]
            w = df / df.sum()
            S_CR = CR[improved]
            den = np.sum(w * S_CR)
            mcr = np.sum(w * S_CR**2) / den if den > 0 else 0.0
            uses_F = ops[improved] != 2
            if np.any(uses_F):
                wf = df[uses_F] / df[uses_F].sum()
                S_F = F[improved][uses_F]
                mf = np.sum(wf * S_F**2) / np.sum(wf * S_F)
                M_F[k_mem] = (M_F[k_mem] + mf) / 2.0
            M_CR[k_mem] = (M_CR[k_mem] + mcr) / 2.0
            k_mem = (k_mem + 1) % (H_SIZE - 1)

        # Operator sifati (probability matching)
        for k in range(K_OPS):
            mk = ops == k
            if np.any(mk):
                op_quality[k] = 0.7 * op_quality[k] + 0.3 * improved[mk].mean()
        q_sum = op_quality.sum()
        op_prob = P_MIN + (1.0 - K_OPS * P_MIN) * (op_quality / q_sum if q_sum > 0 else np.full(K_OPS, 1.0 / K_OPS))
        if ADAPT_EIG:
            for k, mk in enumerate([~use_eig, use_eig]):
                if np.any(mk):
                    eig_quality[k] = 0.7 * eig_quality[k] + 0.3 * improved[mk].mean()
            e_sum = eig_quality.sum()
            P_EIG = float(np.clip(eig_quality[1] / e_sum, 0.1, 0.9)) if e_sum > 0 else 0.5

        pop[accept], fitness[accept] = U[accept], fit_U[accept]
        stag[improved] = 0
        stag[~improved] += 1

        # ABC scout (ixtiyoriy, standart holatda o'chiq: 10D sozlashda zarar qildi)
        best_i = np.argmin(fitness)
        scouts = np.where(stag > STAG_LIMIT)[0] if STAG_LIMIT is not None else []
        for i in scouts:
            if i == best_i or fes >= ls_start: continue
            if t < 0.5:
                x_new = lb + np.random.rand(dim) * (ub - lb)
            else:
                x_new = np.clip(pop[best_i] + 0.1 * (1.0 - t) * (ub - lb) * np.random.randn(dim), lb, ub)
            pop[i], fitness[i], stag[i] = x_new, obj_func(x_new), 0
            fes += 1

        # LPSR + arxiv hajmi
        new_size = max(N_min, int(round(N_init + (N_min - N_init) * fes / max_fes)))
        if new_size < pop_size:
            keep = np.argsort(fitness)[:new_size]
            pop, fitness, stag = pop[keep], fitness[keep], stag[keep]
            pop_size = new_size
        arc_max = int(round(ARC_RATE * pop_size))
        if len(archive) > arc_max:
            archive = archive[np.random.choice(len(archive), arc_max, replace=False)]

    # (1+1)-ES lokal aniqlashtirish, 1/5 muvaffaqiyat qoidasi
    best_i = np.argmin(fitness)
    x_best, f_best = pop[best_i].copy(), fitness[best_i]
    sigma = np.std(pop, axis=0) + 1e-8 * (ub - lb)
    s = 1.0
    while fes < max_fes:
        y = np.clip(x_best + s * sigma * np.random.randn(dim), lb, ub)
        fy = obj_func(y)
        fes += 1
        if fy <= f_best:
            x_best, f_best = y, fy
            s *= math.exp(0.8)
        else:
            s *= math.exp(-0.2)

# ----------------- BASELINE ALGORITHMS (ORIGINAL VERSIYALAR) -----------------
# Eslatma: konvergensiya Tracker orqali yoziladi; gbest qiymati qayta hisoblanmaydi
# (avval hisobga olinmagan qo'shimcha FES chaqiruvlari olib tashlandi).
def baseline_DE(obj_func, dim, bounds, max_fes):
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

algorithms = {
    "TEMOA_V10_HYBRID": TEMOA_V10_HYBRID,
    "DE": baseline_DE, "GWO": baseline_GWO, "WOA": baseline_WOA,
    "PSO": baseline_PSO, "SCA": baseline_SCA, "HHO": baseline_HHO
}
target = "TEMOA_V10_HYBRID"

# ==============================================================================
# 3. YUKORI TEZLIKDAGI PARALLEL EKSPERIMENT
# ==============================================================================
QUICK = os.environ.get("TEMOA_QUICK") == "1"     # smoke test: TEMOA_QUICK=1
dimensions = [10] if QUICK else [30, 50, 100]
runs = 3 if QUICK else 30
FES_PER_DIM = 3000
N_POINTS = 100
SEED_BASE = 42
out_dir = "temoa_v10_quicktest" if QUICK else "temoa_v10_outputs"

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

def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj

if __name__ == "__main__":
    for d in ["tables", "figures", "raw_data"]: os.makedirs(f"{out_dir}/{d}", exist_ok=True)

    tasks = [(a, f, d, r) for a in algorithms for f in funcs for d in dimensions for r in range(runs)]
    print(f"[*] TEMOA_V10_HYBRID experiment. Total tasks: {len(tasks)} (QUICK={QUICK}, SHIFTED={SHIFTED})")
    start_time = time.time()
    results_raw = Parallel(n_jobs=-1, backend="loky")(delayed(run_task)(a, f, d, r) for (a, f, d, r) in tasks)
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

    df_raw = pd.DataFrame(summary_data)
    df_raw.to_csv(f"{out_dir}/raw_data/full_raw_results.csv", index=False)

    df_summary = df_raw.groupby(["Function", "Dimension", "Algorithm"])["Best"].agg(["mean", "std"]).reset_index()
    df_summary.to_csv(f"{out_dir}/tables/summary_mean_std.csv", index=False)
    df_pivot = df_summary.pivot(index=["Function", "Dimension"], columns="Algorithm", values="mean")
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
    fr_stat, fr_p = friedmanchisquare(*[df_pivot[a].values for a in algorithms])
    with open(f"{out_dir}/tables/friedman_test.txt", "w") as fh:
        fh.write(f"Friedman chi2 = {fr_stat:.4f}, p = {fr_p:.4e}\n")
        fh.write(avg_ranks.to_string(index=False))
    print(avg_ranks.to_string(index=False))
    print(f"[*] Friedman chi2 = {fr_stat:.4f}, p = {fr_p:.4e}")

    # Wilcoxon rank-sum (Mann-Whitney U) + Holm tuzatishi, +/=/- hisoblari
    competitors = [a for a in algorithms if a != target]
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
    print("[*] TEMOA vs raqiblar (+ yutdi / = teng / - yutqazdi):")
    print(df_wtl.to_string())

    # Konvergensiya grafiklari (X o'qi: FES)
    x_fes = lambda d: np.linspace(FES_PER_DIM * d / N_POINTS, FES_PER_DIM * d, N_POINTS)
    for f in funcs:
        offset = NOISE_AMP if PROBLEMS[f][4] == "noisy" else 0.0
        for d in dimensions:
            plt.figure(figsize=(10, 6))
            for a in algorithms:
                avg_conv = np.mean(curves[f][d][a], axis=0) + offset
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

    print(f"[*] RESULTS EXPORTED to {out_dir}/")
