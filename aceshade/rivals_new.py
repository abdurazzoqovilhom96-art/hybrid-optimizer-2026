"""Yangi raqobatchilar: LSHADE-RSP, LSHADE-SPACMA, NL-SHADE-RSP.

HALOL OGOHLANTIRISH. Bu muhitda asl maqolalarga yoki mualliflar kodiga
kirish yo'q (tarmoq siyosati bloklaydi). Quyidagi amalga oshirishlar
har bir usulning E'LON QILINGAN MEXANIZMI ta'rifiga asoslanadi, lekin
ba'zi ikkilamchi tanlovlar (populyatsiya hajmi formulasi, ba'zi
konstantalar) aniq tasdiqlanmagan. Har bir shunday joy `# NOANIQ:` izohi
bilan belgilangan.

Bu `docs/RESULTS_V2.md` da cheklov sifatida ochiq yoziladi. Agar bizning
varianti e'lon qilingan natijalardan sezilarli past ishlasa, taqqoslash
shubhali bo'ladi va shu holat hisobotda ko'rsatiladi.
"""
import math
import numpy as np

from .core import sample_F, lehmer, pick_r2, midpoint, arch_push, rsp_ranks
from .baselines import cma_core


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
