"""L-SHADE-DGR: L-SHADE with a Diversity Guard and Restart.

The name declares the lineage on purpose. This is an L-SHADE/jSO variant, and a
reviewer should learn that from the title rather than discover it -- the 2025
surveys that found 26 published metaheuristics structurally identical to earlier
ones are a direct warning against names that hide their ancestry.

WHAT IS INHERITED, AND FROM WHOM
--------------------------------
Nothing here is presented as new except where stated.

    current-to-pbest-w/1 with archive, success-history F/CR   L-SHADE, jSO
    linear population size reduction (LPSR)                   L-SHADE
    eigenbasis crossover                                      LSHADE-cnEpSin,
                                                              EA4eig, L-SRTDE
    Levy-flight operator                                      HHO
    IPOP-style restart                                        Auger & Hansen 2005

The eigen crossover in particular is **not ours**. EA4eig, the CEC'2022 winner,
is itself an operator portfolio with an eigenbasis crossover, which makes it the
structurally closest rival to this algorithm and the reason "portfolio plus eigen
crossover" is never claimed as a contribution here.

WHAT MEASUREMENT CHANGED
------------------------
Every deviation from the original hybrid came from an ablation, not from taste.

  removed  WOA spiral operator   dominated on every function tested: adding it to
                                 the pbest mutation was worse than the pbest
                                 mutation alone, everywhere
  removed  (1+1)-ES tail         no measurable contribution; on two functions the
                                 results were bit-identical with and without it,
                                 so 5% of the budget returned to the main loop
  removed  noise-robust credit   the theory held -- credit-set contamination fell
                                 8-9x -- but the final error did not move. The
                                 F/CR memory is not what limits performance under
                                 noise; selection error is. Kept in temoa_v12.py
                                 as a recorded negative result
  gated    eigen crossover       only when the covariance has more samples than
                                 dimensions. LPSR drives the population below D
                                 long before the end, and a 100x100 covariance
                                 estimated from 25 points is mostly noise
  lowered  P_MIN 0.05 -> 0.02    a 0.05 floor guarantees every operator 5% of
                                 every generation for the whole run, so a
                                 dominated operator can never be switched off
  added    diversity guard       an operator that collapses the population earns
                                 the largest immediate credit while causing the
                                 collapse that loses the run; instrumentation
                                 measured 66% of all fitness gain going to such
                                 an operator on Schwefel. No credit scheme built
                                 on immediate improvement can see this, so the
                                 guard measures diversity directly
  added    restart               the CEC'2017 gate measured this family weakest
                                 on the composition class (3.95 against
                                 BIPOP-CMA-ES's 3.70), which is where restarts
                                 decide the outcome

ON NORMALISATION. DE/rand/1/bin is invariant under any diagonal affine change of
variables: the mutation is built from differences, which scale with the
coordinates, and binomial crossover acts coordinate-wise. So normalising groups
of variables to a common box buys this algorithm nothing. It is standard
preparation, stated for reproducibility, and is not a contribution.
"""

from __future__ import annotations

import numpy as np

from .common import levy_flight


def LSHADE_DGR(obj_func, dim, bounds, max_fes, rng,
               POP_FACTOR=12, POP_MAX=500, P_MIN=0.02, P_EIG=0.4,
               ADAPT_EIG=True, ARC_RATE=1.0, CR_FLOOR=True,
               OPS=(0, 1, 3), EIGEN_GATE=True,
               DIV_GUARD=True, DIV_THRESH=1e-4, DIV_BOOST=0.5,
               RESTART=True, RESTART_BUDGET_FRAC=0.2, RESTART_GROWTH=2.0,
               logger=None):
    lb, ub = bounds
    OPS = tuple(OPS)
    N_min, H_SIZE = 4, 6
    K_OPS = len(OPS)
    LEVY_SLOT = OPS.index(3) if 3 in OPS else None
    span = float(ub - lb)

    N_init = int(np.clip(round(POP_FACTOR * dim), 40, POP_MAX))
    fes = 0

    def evaluate(x):
        nonlocal fes
        fes += 1
        return obj_func(x)

    # -- one epoch per (re)start ------------------------------------------------
    while fes < max_fes:
        pop_size = N_init
        pop = lb + rng.random((pop_size, dim)) * (ub - lb)
        fitness = np.empty(pop_size)
        for i in range(pop_size):
            if fes >= max_fes:
                return
            fitness[i] = evaluate(pop[i])

        M_F = np.full(H_SIZE, 0.3)
        M_CR = np.full(H_SIZE, 0.8)
        M_F[-1], M_CR[-1] = 0.9, 0.9          # jSO: last memory cell frozen
        k_mem = 0
        archive = np.empty((0, dim))
        op_quality = np.full(K_OPS, 0.5)
        op_prob = np.full(K_OPS, 1.0 / K_OPS)
        eig_quality = np.array([0.5, 0.5])
        B = np.eye(dim)
        epoch_fes0 = fes
        stalled = 0

        while fes < max_fes:
            t = fes / max_fes
            order = np.argsort(fitness)
            pop, fitness = pop[order], fitness[order]
            idx = np.arange(pop_size)

            # eigenbasis only where the covariance is actually estimable
            n_samples = max(2, pop_size // 2)
            eig_ok = (not EIGEN_GATE) or (n_samples > dim)
            if eig_ok:
                C = np.cov(pop[:n_samples], rowvar=False)
                if np.all(np.isfinite(C)):
                    _, B = np.linalg.eigh(C)

            # parameters: success-history with jSO's constraint schedule
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

            p_num = max(2, int(round((0.25 - 0.125 * t) * pop_size)))
            x_pbest = pop[rng.integers(0, p_num, pop_size)]
            r1 = (idx + rng.integers(1, pop_size, pop_size)) % pop_size
            union_pop = np.vstack([pop, archive]) if len(archive) else pop
            r2 = rng.integers(0, len(union_pop), pop_size)
            clash = (r2 == idx) | (r2 == r1)
            guard = 0
            while np.any(clash) and guard < 100:
                r2[clash] = rng.integers(0, len(union_pop), int(clash.sum()))
                clash = (r2 == idx) | (r2 == r1)
                guard += 1
            diff = pop[r1] - union_pop[r2]
            X_lead = np.array([0.5, 0.3, 0.2]) @ pop[:3]

            # -- diversity guard ------------------------------------------------
            # Credit assignment cannot see a population collapse, because the
            # operator that causes it is also the one producing the largest
            # immediate gains. Measuring the spread directly can.
            diversity = float(np.mean(np.std(pop, axis=0))) / span
            probs = op_prob
            guard_active = False
            if DIV_GUARD and LEVY_SLOT is not None and K_OPS > 1:
                guard_active = diversity < DIV_THRESH and t < 0.95
                if guard_active:
                    probs = np.full(K_OPS, (1.0 - DIV_BOOST) / (K_OPS - 1))
                    probs[LEVY_SLOT] = DIV_BOOST

            ops = np.asarray(OPS)[rng.choice(K_OPS, pop_size, p=probs)]
            V = np.empty_like(pop)
            m = ops == 0                       # current-to-pbest-w/1 (jSO core)
            if np.any(m):
                V[m] = pop[m] + Fwc[m] * (x_pbest[m] - pop[m]) + Fc[m] * diff[m]
            m = ops == 1                       # translation-invariant leader step
            if np.any(m):
                V[m] = pop[m] + Fc[m] * (X_lead - pop[m]) + Fc[m] * diff[m]
            m = ops == 3                       # Levy-flight escape
            if np.any(m):
                L = np.clip(levy_flight(rng, (int(m.sum()), dim)), -5.0, 5.0)
                V[m] = pop[m] + Fc[m] * (x_pbest[m] - pop[m]) + Fc[m] * L * diff[m]

            cross = rng.random((pop_size, dim)) < CR[:, None]
            cross[idx, rng.integers(0, dim, pop_size)] = True
            U = np.where(cross, V, pop)
            if eig_ok:
                use_eig = rng.random(pop_size) < P_EIG
                if np.any(use_eig):
                    xe, ve = pop[use_eig] @ B, V[use_eig] @ B
                    U[use_eig] = np.where(cross[use_eig], ve, xe) @ B.T
            else:
                use_eig = np.zeros(pop_size, dtype=bool)

            low, high = U < lb, U > ub         # midpoint-target repair (L-SHADE)
            U[low] = ((lb + pop) / 2.0)[low]
            U[high] = ((ub + pop) / 2.0)[high]

            n_eval = min(pop_size, max_fes - fes)
            fit_U = np.full(pop_size, np.inf)
            for i in range(n_eval):
                fit_U[i] = evaluate(U[i])

            improved = fit_U < fitness
            accept = fit_U <= fitness

            if np.any(improved):
                archive = np.vstack([archive, pop[improved]])
                df = fitness[improved] - fit_U[improved]
                w = df / df.sum()
                S_CR = CR[improved]
                den = np.sum(w * S_CR)
                mcr = np.sum(w * S_CR**2) / den if den > 0 else 0.0
                S_F = F[improved]
                dF = np.sum(w * S_F)
                if dF > 0:
                    M_F[k_mem] = (M_F[k_mem] + np.sum(w * S_F**2) / dF) / 2.0
                M_CR[k_mem] = (M_CR[k_mem] + mcr) / 2.0
                k_mem = (k_mem + 1) % (H_SIZE - 1)

            if logger is not None:
                logger.append({"fes": fes, "t": t, "diversity": diversity,
                               "guard": guard_active, "op_prob": op_prob.copy(),
                               "eig_ok": eig_ok, "pop_size": pop_size,
                               "best": float(min(fitness.min(), fit_U.min()))})

            # credit for the operator portfolio; frozen while the guard overrides
            # the distribution, so forced Levy trials do not distort the estimates
            if not guard_active:
                for slot, k in enumerate(OPS):
                    mk = ops == k
                    if np.any(mk):
                        op_quality[slot] = 0.7 * op_quality[slot] + 0.3 * improved[mk].mean()
                q_sum = op_quality.sum()
                op_prob = P_MIN + (1.0 - K_OPS * P_MIN) * (
                    op_quality / q_sum if q_sum > 0 else np.full(K_OPS, 1.0 / K_OPS))
                if ADAPT_EIG and eig_ok:
                    for k, mk in enumerate([~use_eig, use_eig]):
                        if np.any(mk):
                            eig_quality[k] = 0.7 * eig_quality[k] + 0.3 * improved[mk].mean()
                    e_sum = eig_quality.sum()
                    P_EIG = float(np.clip(eig_quality[1] / e_sum, 0.1, 0.9)) if e_sum > 0 else 0.5

            pop[accept], fitness[accept] = U[accept], fit_U[accept]
            stalled = 0 if np.any(improved) else stalled + 1

            # LPSR measured over this epoch's own share of the budget
            epoch_span = max(1, max_fes - epoch_fes0)
            frac = (fes - epoch_fes0) / epoch_span
            new_size = max(N_min, int(round(N_init + (N_min - N_init) * frac)))
            if new_size < pop_size:
                keep = np.argsort(fitness)[:new_size]
                pop, fitness = pop[keep], fitness[keep]
                pop_size = new_size
            arc_max = int(round(ARC_RATE * pop_size))
            if len(archive) > arc_max:
                archive = archive[rng.choice(len(archive), arc_max, replace=False)]

            if RESTART:
                budget_left = (max_fes - fes) / max_fes
                collapsed = diversity < DIV_THRESH or pop_size <= N_min
                if collapsed and stalled > 0 and budget_left > RESTART_BUDGET_FRAC:
                    N_init = int(min(POP_MAX, round(N_init * RESTART_GROWTH)))
                    break                      # outer loop reinitialises
