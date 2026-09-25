"""Modern adaptive DE baselines -- the comparison the original study omits.

The original compares TEMOA_V10_HYBRID against DE/rand/1/bin (1995) and five
swarm metaheuristics (2014-2019). It does not compare against any modern
adaptive DE, even though its own header comment names L-SHADE and jSO as the
sources of its core. That makes the headline claim untestable: beating
DE/rand/1/bin does not show that the *hybridisation* contributes anything.

These two implementations close that gap.

``LSHADE``
    Tanabe & Fukunaga, "Improving the search performance of SHADE using linear
    population size reduction", CEC 2014. N_init = 18*D, H = 6, p = 0.11,
    archive rate 2.6, current-to-pbest/1 with archive, binomial crossover,
    midpoint boundary repair, LPSR, weighted Lehmer means with the terminal-CR
    rule.

``jSO``
    Brest et al., "Single objective real-parameter optimization: algorithm jSO",
    CEC 2017. This is the tightest possible comparator: TEMOA's operator 0, its
    F/CR constraint schedule, its weighted F, its frozen last memory cell, its
    p-schedule and its boundary repair are all jSO. TEMOA = jSO + three extra
    operators + eigen-crossover + a (1+1)-ES tail. Any gap between TEMOA and jSO
    is attributable to exactly those additions.
"""

from __future__ import annotations

import numpy as np

TERMINAL = -1.0     # L-SHADE's "CR memory is dead" marker


def _sample_F(rng, M_F, r, n):
    F = M_F[r] + 0.1 * rng.standard_cauchy(n)
    bad = F <= 0
    while np.any(bad):
        F[bad] = M_F[r[bad]] + 0.1 * rng.standard_cauchy(int(bad.sum()))
        bad = F <= 0
    return np.minimum(F, 1.0)


def _distinct_r1_r2(rng, pop_size, n_union, idx):
    r1 = (idx + rng.integers(1, pop_size, pop_size)) % pop_size
    r2 = rng.integers(0, n_union, pop_size)
    clash = (r2 == idx) | (r2 == r1)
    guard = 0
    while np.any(clash) and guard < 100:
        r2[clash] = rng.integers(0, n_union, int(clash.sum()))
        clash = (r2 == idx) | (r2 == r1)
        guard += 1
    return r1, r2


def LSHADE(obj_func, dim, bounds, max_fes, rng, POP_FACTOR=18, ARC_RATE=2.6, P_BEST=0.11):
    lb, ub = bounds
    N_init = int(POP_FACTOR * dim)
    N_min, H = 4, 6
    pop_size = N_init
    pop = lb + rng.random((pop_size, dim)) * (ub - lb)
    fitness = np.array([obj_func(ind) for ind in pop])
    fes = pop_size

    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k_mem = 0
    archive = np.empty((0, dim))

    while fes < max_fes:
        order = np.argsort(fitness)
        pop, fitness = pop[order], fitness[order]
        idx = np.arange(pop_size)

        r = rng.integers(0, H, pop_size)
        CR = np.where(M_CR[r] == TERMINAL, 0.0,
                      np.clip(rng.normal(np.where(M_CR[r] == TERMINAL, 0.0, M_CR[r]), 0.1), 0.0, 1.0))
        F = _sample_F(rng, M_F, r, pop_size)
        Fc = F[:, None]

        p_num = max(2, int(round(P_BEST * pop_size)))
        x_pbest = pop[rng.integers(0, p_num, pop_size)]
        union_pop = np.vstack([pop, archive]) if len(archive) else pop
        r1, r2 = _distinct_r1_r2(rng, pop_size, len(union_pop), idx)

        V = pop + Fc * (x_pbest - pop) + Fc * (pop[r1] - union_pop[r2])

        cross = rng.random((pop_size, dim)) < CR[:, None]
        cross[idx, rng.integers(0, dim, pop_size)] = True
        U = np.where(cross, V, pop)

        low, high = U < lb, U > ub
        U[low] = ((lb + pop) / 2.0)[low]
        U[high] = ((ub + pop) / 2.0)[high]

        n_eval = min(pop_size, max_fes - fes)
        fit_U = np.full(pop_size, np.inf)
        for i in range(n_eval):
            fit_U[i] = obj_func(U[i])
        fes += n_eval

        improved = fit_U < fitness
        if np.any(improved):
            archive = np.vstack([archive, pop[improved]])
            df = fitness[improved] - fit_U[improved]
            w = df / df.sum()
            S_CR, S_F = CR[improved], F[improved]
            if M_CR[k_mem] == TERMINAL or S_CR.max() == 0.0:
                M_CR[k_mem] = TERMINAL
            else:
                M_CR[k_mem] = np.sum(w * S_CR**2) / np.sum(w * S_CR)
            M_F[k_mem] = np.sum(w * S_F**2) / np.sum(w * S_F)
            k_mem = (k_mem + 1) % H

        accept = fit_U <= fitness
        pop[accept], fitness[accept] = U[accept], fit_U[accept]

        new_size = max(N_min, int(round(N_init + (N_min - N_init) * fes / max_fes)))
        if new_size < pop_size:
            keep = np.argsort(fitness)[:new_size]
            pop, fitness = pop[keep], fitness[keep]
            pop_size = new_size
        arc_max = int(round(ARC_RATE * pop_size))
        if len(archive) > arc_max:
            archive = archive[rng.choice(len(archive), arc_max, replace=False)]


def jSO(obj_func, dim, bounds, max_fes, rng, ARC_RATE=1.0):
    lb, ub = bounds
    N_init = int(round(25.0 * np.log(dim) * np.sqrt(dim)))
    N_min, H = 4, 5
    pop_size = N_init
    pop = lb + rng.random((pop_size, dim)) * (ub - lb)
    fitness = np.array([obj_func(ind) for ind in pop])
    fes = pop_size

    M_F = np.full(H, 0.3)
    M_CR = np.full(H, 0.8)
    M_F[-1], M_CR[-1] = 0.9, 0.9      # frozen last cell
    k_mem = 0
    archive = np.empty((0, dim))
    p_max, p_min = 0.25, 0.125

    while fes < max_fes:
        t = fes / max_fes
        order = np.argsort(fitness)
        pop, fitness = pop[order], fitness[order]
        idx = np.arange(pop_size)

        r = rng.integers(0, H, pop_size)
        CR = np.clip(rng.normal(M_CR[r], 0.1), 0.0, 1.0)
        if t < 0.25:
            CR = np.maximum(CR, 0.7)
        elif t < 0.5:
            CR = np.maximum(CR, 0.6)
        F = _sample_F(rng, M_F, r, pop_size)
        if t < 0.6:
            F = np.minimum(F, 0.7)
        Fw = F * (0.7 if t < 0.2 else 0.8 if t < 0.4 else 1.2)
        Fc, Fwc = F[:, None], Fw[:, None]

        p = p_max - (p_max - p_min) * t
        p_num = max(2, int(round(p * pop_size)))
        x_pbest = pop[rng.integers(0, p_num, pop_size)]
        union_pop = np.vstack([pop, archive]) if len(archive) else pop
        r1, r2 = _distinct_r1_r2(rng, pop_size, len(union_pop), idx)

        V = pop + Fwc * (x_pbest - pop) + Fc * (pop[r1] - union_pop[r2])

        cross = rng.random((pop_size, dim)) < CR[:, None]
        cross[idx, rng.integers(0, dim, pop_size)] = True
        U = np.where(cross, V, pop)

        low, high = U < lb, U > ub
        U[low] = ((lb + pop) / 2.0)[low]
        U[high] = ((ub + pop) / 2.0)[high]

        n_eval = min(pop_size, max_fes - fes)
        fit_U = np.full(pop_size, np.inf)
        for i in range(n_eval):
            fit_U[i] = obj_func(U[i])
        fes += n_eval

        improved = fit_U < fitness
        if np.any(improved):
            archive = np.vstack([archive, pop[improved]])
            df = fitness[improved] - fit_U[improved]
            w = df / df.sum()
            S_CR, S_F = CR[improved], F[improved]
            den = np.sum(w * S_CR)
            mcr = np.sum(w * S_CR**2) / den if den > 0 else 0.0
            M_CR[k_mem] = (M_CR[k_mem] + mcr) / 2.0
            M_F[k_mem] = (M_F[k_mem] + np.sum(w * S_F**2) / np.sum(w * S_F)) / 2.0
            k_mem = (k_mem + 1) % (H - 1)

        accept = fit_U <= fitness
        pop[accept], fitness[accept] = U[accept], fit_U[accept]

        new_size = max(N_min, int(round(N_init + (N_min - N_init) * fes / max_fes)))
        if new_size < pop_size:
            keep = np.argsort(fitness)[:new_size]
            pop, fitness = pop[keep], fitness[keep]
            pop_size = new_size
        arc_max = int(round(ARC_RATE * pop_size))
        if len(archive) > arc_max:
            archive = archive[rng.choice(len(archive), arc_max, replace=False)]
