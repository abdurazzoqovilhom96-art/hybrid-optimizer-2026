"""V1 dan ko'chirilgan raqobatchilar: L-SHADE, jSO, LSHADE-cnEpSin, CMA-ES.

Bu to'rttasi CBA-SHADE (V1) eksperimentida ishlatilgan va o'sha yerda
sinovdan o'tgan. Kod dasturiy ravishda ko'chirilgan (qo'lda emas),
shuning uchun transkripsiya xatosi yo'q. Yagona o'zgarish - yordamchi
funksiya nomlari yangi modulga moslashtirilgan.

Yangi raqobatchilar (LSHADE-RSP, LSHADE-SPACMA, NL-SHADE-RSP)
`aceshade/rivals_new.py` da.
"""
import math
import numpy as np

from .core import sample_F, lehmer, pick_r2, midpoint, arch_push


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

# ------------------------------- ASOSIY ALGORITM ------------------------------

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

# ------------- RAQOBATCHILAR: KLASSIK / METAFORA ASOSIDAGI TO'PLAM ------------
# Eski adabiyot bilan bog'lash uchun saqlangan; INCLUDE_CLASSIC bilan yoqiladi.
# Eslatma: konvergensiya Tracker orqali yoziladi; gbest qiymati qayta hisoblanmaydi.
