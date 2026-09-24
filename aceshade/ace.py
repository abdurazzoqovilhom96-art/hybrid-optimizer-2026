"""ACE-SHADE: Adaptive Covariance-Ensemble SHADE.

Ikki hissa (`docs/ACE_SHADE_PROMPT.md` 4-bo'lim):

  N1  To'plamli kovariatsiya. V1 har avlodda `cov(pop[:N/2])` olardi:
      `n = 3D` namunadan `D` o'lchamda, spektral xato `O(sqrt(D/n)) ~ 58%`.
      Kichik xos qiymatlarga mos yo'nalishlar shovqin bo'lib qolardi -
      aynan yomon shartlangan masalada hal qiluvchi yo'nalishlar.
      O'rniga CMA uslubidagi to'plamli baholagich: samarali namuna hajmi
      `O(D^2)` avlod, shuning uchun o'lcham bilan buzilmaydi.

      Bitta `C` ikki iste'molchiga xizmat qiladi:
        (a) crossover eigen-bazisi (LSHADE-cnEpSin g'oyasi),
        (b) CMA taqsimotidan namuna olish tarmog'i (SPACMA g'oyasi).
      Alohida "quyruq" fazasi yo'q - ulush uzluksiz moslashadi.

  N2  Parametr rejimi ansambli. jSO va L-SHADE xotira rejimlari teskari
      bog'liq (V1 o'lchovi, Schwefel 30D: jSO 1.18e+02, L-SHADE 3.82e-04).
      Qat'iy tanlov aralash to'plamda har doim suboptimal, shuning uchun
      ikkala bank olib boriladi va ulushi yaxshilanish MIQDORI krediti
      bilan moslashadi.

Saqlangan yadro: current-to-pbest-w/1 + arxiv (SHADE/L-SHADE), RSP
(LSHADE-RSP), LPSR (L-SHADE), midpoint-target chegara.
"""
import numpy as np

from .core import sample_F, lehmer, pick_r2, midpoint, arch_push, rsp_ranks

NAME = "ACE-SHADE"


def _credit(groups, gain, gain_total, cred, eta, n):
    """Yaxshilanish MIQDORI krediti (FIR). Uchala ulush uchun bir xil qoida."""
    for j, mask in enumerate(groups):
        share = mask.mean()
        if share > 0:
            fir = (gain[mask].sum() / gain_total / share) if gain_total > 0 else 0.0
            cred[j] = (1.0 - eta) * cred[j] + eta * fir
    s = cred.sum()
    return float(np.clip(cred[1] / s, 0.02, 0.98)) if s > 0 else 0.5


def ace_shade(obj_func, dim, bounds, max_fes,
              R_N=6.0, ETA=0.2, N_MIN=4, H_SIZE=6, ARC_RATE=2.6,
              K_RSP=3.0, P_MAX=0.25, P_MIN_RATE=0.125,
              USE_COV=True, USE_ENSEMBLE=True, CMU_SOURCE="accepted",
              on_generation=None):
    """ACE-SHADE. `USE_COV`/`USE_ENSEMBLE` ablatsiya uchun (B4).

    `on_generation(state)` - ixtiyoriy kuzatuv nuqtasi (diagnostika va
    testlar uchun). Algoritm xulqiga ta'sir qilmaydi.
    """
    lb, ub = bounds
    span = float(np.mean(ub - lb))
    N_init = max(40, int(round(R_N * dim)))
    pop_size = N_init

    pop = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.array([obj_func(ind) for ind in pop])
    fes = pop_size

    # --- N2: ikki xotira banki ------------------------------------------------
    # 0-bank (jSO): M_F=0.3, M_CR=0.8, doimiy terminal katak, F cheklovlari.
    # 1-bank (L-SHADE): M_F=M_CR=0.5, barcha kataklar yangilanadi, F cheklovsiz.
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
    eigeneval = 0
    p_eig, eig_credit = 0.5, np.array([0.5, 0.5])
    p_cma, cma_credit = 0.1, np.array([0.9, 0.1])

    archive = np.empty((0, dim))

    while fes < max_fes:
        t = fes / max_fes
        order = np.argsort(fitness)
        pop, fitness = pop[order], fitness[order]
        idx = np.arange(pop_size)

        mu = max(2, pop_size // 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2.0 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Qadam hajmi populyatsiya tarqalishidan (CSA emas: avlodlarning bir
        # qismi DE mutatsiyasidan keladi, CSA esa Gauss namunasini faraz qiladi).
        sigma = max(float(np.mean(np.std(pop, axis=0))), 1e-14 * span)

        # Dangasa xos yoyilma: O(D^3) -> O(D^2) amortizatsiya
        if USE_COV and fes - eigeneval > 1.0 / (10.0 * dim * (c_1 + c_mu)):
            eigeneval = fes
            C = (C + C.T) / 2.0
            try:
                vals, B = np.linalg.eigh(C)
                vals = np.maximum(vals, 1e-20 * max(vals.max(), 1e-300))
                d_eig = np.sqrt(vals)
            except np.linalg.LinAlgError:
                B, d_eig = np.eye(dim), np.ones(dim)

        # --- Parametrlar: bank tanlovi + success-history ----------------------
        pb = p_bank if USE_ENSEMBLE else 1.0          # ansamblsiz: faqat jSO banki
        bank = (np.random.rand(pop_size) >= pb).astype(int)
        r = np.random.randint(0, H_SIZE, pop_size)
        mu_CR, mu_F = M_CR[bank, r], M_F[bank, r]
        CR = np.clip(np.random.normal(mu_CR, 0.1), 0.0, 1.0)
        CR[mu_CR < 0] = 0.0
        F = sample_F(mu_F)
        jso = bank == 0
        if np.any(jso) and t < 0.6:
            F[jso] = np.minimum(F[jso], 0.7)
        Fw = F.copy()
        Fw[jso] *= (0.7 if t < 0.2 else 0.8 if t < 0.4 else 1.2)

        # --- Tarmoq taqsimoti: DE yoki CMA namunasi ---------------------------
        use_cma = (np.random.rand(pop_size) < p_cma) if USE_COV else np.zeros(pop_size, bool)

        # --- DE tarmog'i: current-to-pbest-w/1 + arxiv ------------------------
        p_num = max(2, int(round((P_MAX - (P_MAX - P_MIN_RATE) * t) * pop_size)))
        x_pbest = pop[np.random.randint(0, p_num, pop_size)]
        r1 = rsp_ranks(pop_size, idx, K_RSP)
        union_pop = np.vstack([pop, archive]) if len(archive) else pop
        r2 = pick_r2(len(union_pop), idx, r1)
        V = pop + Fw[:, None] * (x_pbest - pop) + F[:, None] * (pop[r1] - union_pop[r2])

        cross = np.random.rand(pop_size, dim) < CR[:, None]
        cross[idx, np.random.randint(0, dim, pop_size)] = True
        U = np.where(cross, V, pop)

        # (a) eigen-bazisda crossover
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

        n_eval = min(pop_size, max_fes - fes)
        fit_U = np.full(pop_size, np.inf)
        for i in range(n_eval):
            fit_U[i] = obj_func(U[i])
        fes += n_eval

        improved = fit_U < fitness
        accept = fit_U <= fitness
        gain = np.where(improved, np.maximum(fitness - fit_U, 0.0), 0.0)
        gain_total = gain.sum()

        # --- Xotira yangilanishi (har bank alohida) ---------------------------
        if np.any(improved):
            archive = arch_push(archive, pop[improved], int(round(ARC_RATE * pop_size)))
            for b in (0, 1):
                mb = improved & (bank == b)
                if not np.any(mb):
                    continue
                df = fitness[mb] - fit_U[mb]
                ww = df / df.sum()
                mf = lehmer(F[mb], ww)
                mcr = -1.0 if (M_CR[b, k_mem[b]] == -1 or np.sum(ww * CR[mb]) == 0) \
                           else lehmer(CR[mb], ww)
                if b == 1:
                    M_F[b, k_mem[b]], M_CR[b, k_mem[b]] = mf, mcr
                else:
                    M_F[b, k_mem[b]] = (M_F[b, k_mem[b]] + mf) / 2.0
                    M_CR[b, k_mem[b]] = -1.0 if mcr == -1 else (M_CR[b, k_mem[b]] + mcr) / 2.0
                k_mem[b] = (k_mem[b] + 1) % n_upd[b]

        # --- Birlashgan kredit arbitraji (N2 mexanizmi) -----------------------
        if USE_ENSEMBLE:
            p_bank = 1.0 - _credit([bank == 0, bank == 1], gain, gain_total,
                                   bank_credit, ETA, pop_size)
        if USE_COV:
            p_eig = _credit([~use_eig, use_eig], gain, gain_total, eig_credit, ETA, pop_size)
            p_cma = _credit([~use_cma, use_cma], gain, gain_total, cma_credit, ETA, pop_size)
            p_cma = min(p_cma, 0.5)          # DE yadrosi asosiy bo'lib qoladi

        # --- N1: kovariatsiyani yangilash -------------------------------------
        if USE_COV:
            src = np.where(accept)[0] if CMU_SOURCE == "accepted" else np.arange(pop_size)
            if len(src) >= 2:
                cand = U[src] if CMU_SOURCE == "accepted" else pop
                cf = fit_U[src] if CMU_SOURCE == "accepted" else fitness
                k = min(mu, len(src))
                best = np.argsort(cf)[:k]
                wk = np.log(k + 0.5) - np.log(np.arange(1, k + 1))
                wk /= wk.sum()
                m_old = m
                m = wk @ cand[best]
                y_w = (m - m_old) / sigma
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
                Y = (cand[best] - m_old) / sigma
                C = ((1 - c_1 - c_mu) * C
                     + c_1 * np.outer(p_c, p_c)
                     + c_mu * (Y.T * wk) @ Y)

        pop[accept], fitness[accept] = U[accept], fit_U[accept]

        # --- LPSR -------------------------------------------------------------
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
                           "best": float(fitness.min())})
