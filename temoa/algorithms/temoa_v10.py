"""TEMOA_V10_HYBRID -- the algorithm under study, unchanged.

This is a faithful port of ``TEMOA_V10_HYBRID`` from ``hybrid 2026.py``. The only
change is that the global NumPy RNG is replaced by an injected
``numpy.random.Generator``, so runs are reproducible from a recorded seed and
independent of process scheduling. Every operator, constant and update rule is
identical to the original; results from this module reproduce the original
algorithm's behaviour.

Structure (as described by the original author's header comment):

  operator portfolio, selected by probability matching over success rates
    0  current-to-pbest-w/1 with archive   (L-SHADE / jSO core)
    1  translation-invariant leader step   (GWO-inspired, X_lead = weighted top-3)
    2  logarithmic spiral around x_pbest   (WOA-inspired)
    3  Levy-flight jump                    (HHO-inspired)
  crossover : binomial, or binomial in the covariance eigenbasis (LSHADE-cnEpSin)
  parameters: success-history memory with jSO's F/CR constraints
  bounds    : midpoint-target repair (L-SHADE)
  population: linear population size reduction (LPSR)
  finish    : (1+1)-ES with Rechenberg's 1/5 rule on the last LS_FRACTION of budget
"""

from __future__ import annotations

import math

import numpy as np

from .common import levy_flight


def TEMOA_V10_HYBRID(obj_func, dim, bounds, max_fes, rng,
                     POP_FACTOR=6, POP_MAX=500, P_MIN=0.05, P_EIG=0.4,
                     ADAPT_EIG=True, ARC_RATE=1.0, STAG_LIMIT=None,
                     LS_FRACTION=0.05, CR_FLOOR=True, OPS=(0, 1, 2, 3), logger=None):
    """``OPS`` selects which operators take part; the default is all four, which
    reproduces the original exactly. Any subset may be passed to ablate the
    portfolio (e.g. ``OPS=(0,)`` leaves only the jSO-style pbest mutation)."""
    lb, ub = bounds
    OPS = tuple(OPS)
    N_init = int(np.clip(round(POP_FACTOR * dim), 40, POP_MAX))
    N_min = 4
    H_SIZE = 6
    K_OPS = len(OPS)
    ls_start = int((1.0 - LS_FRACTION) * max_fes)

    pop_size = N_init
    pop = lb + rng.random((pop_size, dim)) * (ub - lb)
    fitness = np.array([obj_func(ind) for ind in pop])
    fes = pop_size

    M_F = np.full(H_SIZE, 0.3)
    M_CR = np.full(H_SIZE, 0.8)
    M_F[-1], M_CR[-1] = 0.9, 0.9          # jSO: last memory cell is frozen
    k_mem = 0
    archive = np.empty((0, dim))
    op_quality = np.full(K_OPS, 0.5)
    op_prob = np.full(K_OPS, 1.0 / K_OPS)
    eig_quality = np.array([0.5, 0.5])
    stag = np.zeros(pop_size, dtype=int)
    B = np.eye(dim)

    while fes < ls_start:
        t = fes / max_fes
        order = np.argsort(fitness)
        pop, fitness, stag = pop[order], fitness[order], stag[order]
        idx = np.arange(pop_size)

        C = np.cov(pop[:max(2, pop_size // 2)], rowvar=False)
        if np.all(np.isfinite(C)):
            _, B = np.linalg.eigh(C)

        r = rng.integers(0, H_SIZE, pop_size)
        CR = np.clip(rng.normal(M_CR[r], 0.1), 0.0, 1.0)
        if CR_FLOOR:
            if t < 0.25:
                CR = np.maximum(CR, 0.7)
            elif t < 0.5:
                CR = np.maximum(CR, 0.6)
        F = M_F[r] + 0.1 * rng.standard_cauchy(pop_size)
        bad = F <= 0
        while np.any(bad):
            F[bad] = M_F[r[bad]] + 0.1 * rng.standard_cauchy(int(bad.sum()))
            bad = F <= 0
        F = np.minimum(F, 1.0)
        if t < 0.6:
            F = np.minimum(F, 0.7)
        Fw = F * (0.7 if t < 0.2 else 0.8 if t < 0.4 else 1.2)
        Fc, Fwc = F[:, None], Fw[:, None]

        p_num = max(2, int(round((0.25 - 0.20 * t) * pop_size)))
        x_pbest = pop[rng.integers(0, p_num, pop_size)]
        r1 = (idx + rng.integers(1, pop_size, pop_size)) % pop_size
        union_pop = np.vstack([pop, archive]) if len(archive) else pop
        r2 = rng.integers(0, len(union_pop), pop_size)
        clash = (r2 == idx) | (r2 == r1)
        while np.any(clash):
            r2[clash] = rng.integers(0, len(union_pop), int(clash.sum()))
            clash = (r2 == idx) | (r2 == r1)
        diff = pop[r1] - union_pop[r2]
        X_lead = np.array([0.5, 0.3, 0.2]) @ pop[:3]

        ops = np.asarray(OPS)[rng.choice(K_OPS, pop_size, p=op_prob)]
        V = np.empty_like(pop)
        m = ops == 0
        V[m] = pop[m] + Fwc[m] * (x_pbest[m] - pop[m]) + Fc[m] * diff[m]
        m = ops == 1
        V[m] = pop[m] + Fc[m] * (X_lead - pop[m]) + Fc[m] * diff[m]
        m = ops == 2
        l = rng.uniform(-1.0, 1.0, (int(m.sum()), 1))
        V[m] = x_pbest[m] + np.abs(x_pbest[m] - pop[m]) * np.exp(l) * np.cos(2.0 * np.pi * l)
        m = ops == 3
        L = np.clip(levy_flight(rng, (int(m.sum()), dim)), -5.0, 5.0)
        V[m] = pop[m] + Fc[m] * (x_pbest[m] - pop[m]) + Fc[m] * L * diff[m]

        cross = rng.random((pop_size, dim)) < CR[:, None]
        cross[idx, rng.integers(0, dim, pop_size)] = True
        U = np.where(cross, V, pop)
        use_eig = rng.random(pop_size) < P_EIG
        if np.any(use_eig):
            xe, ve = pop[use_eig] @ B, V[use_eig] @ B
            U[use_eig] = np.where(cross[use_eig], ve, xe) @ B.T

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

        for slot, k in enumerate(OPS):
            mk = ops == k
            if np.any(mk):
                op_quality[slot] = 0.7 * op_quality[slot] + 0.3 * improved[mk].mean()
        q_sum = op_quality.sum()
        op_prob = P_MIN + (1.0 - K_OPS * P_MIN) * (
            op_quality / q_sum if q_sum > 0 else np.full(K_OPS, 1.0 / K_OPS))
        if ADAPT_EIG:
            for k, mk in enumerate([~use_eig, use_eig]):
                if np.any(mk):
                    eig_quality[k] = 0.7 * eig_quality[k] + 0.3 * improved[mk].mean()
            e_sum = eig_quality.sum()
            P_EIG = float(np.clip(eig_quality[1] / e_sum, 0.1, 0.9)) if e_sum > 0 else 0.5

        if logger is not None:
            # diagnostics only; never affects the search
            rate, gain = {}, {}
            for k in OPS:
                mk = ops == k
                rate[k] = float(improved[mk].mean()) if np.any(mk) else 0.0
                g = (fitness[mk & improved] - fit_U[mk & improved])
                gain[k] = float(g.sum()) if g.size else 0.0
            logger.append({"fes": fes, "t": t, "op_prob": op_prob.copy(),
                           "succ_rate": rate, "total_gain": gain,
                           "best": float(fitness.min())})

        pop[accept], fitness[accept] = U[accept], fit_U[accept]
        stag[improved] = 0
        stag[~improved] += 1

        best_i = np.argmin(fitness)
        scouts = np.where(stag > STAG_LIMIT)[0] if STAG_LIMIT is not None else []
        for i in scouts:
            if i == best_i or fes >= ls_start:
                continue
            if t < 0.5:
                x_new = lb + rng.random(dim) * (ub - lb)
            else:
                x_new = np.clip(pop[best_i] + 0.1 * (1.0 - t) * (ub - lb) * rng.standard_normal(dim), lb, ub)
            pop[i], fitness[i], stag[i] = x_new, obj_func(x_new), 0
            fes += 1

        new_size = max(N_min, int(round(N_init + (N_min - N_init) * fes / max_fes)))
        if new_size < pop_size:
            keep = np.argsort(fitness)[:new_size]
            pop, fitness, stag = pop[keep], fitness[keep], stag[keep]
            pop_size = new_size
        arc_max = int(round(ARC_RATE * pop_size))
        if len(archive) > arc_max:
            archive = archive[rng.choice(len(archive), arc_max, replace=False)]

    # (1+1)-ES refinement with Rechenberg's 1/5 rule
    best_i = np.argmin(fitness)
    x_best, f_best = pop[best_i].copy(), fitness[best_i]
    sigma = np.std(pop, axis=0) + 1e-8 * (ub - lb)
    s = 1.0
    while fes < max_fes:
        y = np.clip(x_best + s * sigma * rng.standard_normal(dim), lb, ub)
        fy = obj_func(y)
        fes += 1
        if fy <= f_best:
            x_best, f_best = y, fy
            s *= math.exp(0.8)
        else:
            s *= math.exp(-0.2)
