"""TEMOA_V12 -- V11 plus noise-robust credit assignment and a restart.

V11 was a repair driven by measurement. V12 adds one mechanism that is meant to
be a contribution, and one that is a known remedy for a measured weakness. They
are kept separate on purpose, and each is a flag so the ablation can price it.

THE MECHANISM (the contribution)
--------------------------------
Derived in ``temoa/noise.py``. In short: SHADE-family algorithms weight their
F/CR memory update by the *observed* improvement ``delta~``, and under noise the
trials that were accepted because the noise draw was large carry the largest
weights. Once true improvements fall to the order of ``sigma``, the memory is
driven by noise rather than by which parameters worked.

Four switches, each ablatable:

``N0`` ``NOISE_M``       duplicate evaluations per generation, feeding an online
                         estimate of the noise scale (``m=5, alpha=0.15``;
                         see NoiseEstimator for why those values)
``N1`` ``CREDIT_TAU``    a trial enters the *credit set* only if
                         ``delta~ > tau * sigma_hat * sqrt(2)``. At tau=1 this
                         bounds the false-inclusion rate at ``1 - Phi(1) = 0.159``
``N2`` ``RANK_WEIGHTS``  log-rank weights instead of ``w propto delta~``, so a
                         single noise-inflated gap cannot dominate the mean
``N3`` ``REEVAL_R/Q``    re-evaluate the ``r`` largest-gap members of the credit
                         set ``q`` times to sharpen their weights

**Selection is untouched.** The population still accepts on ``fit_U <= fitness``
exactly as V11 does. Only the *credit set* -- which trials inform the F/CR memory
-- is filtered. This is what separates the mechanism from UH-CMA-ES and friends,
which stabilise the ranking that drives selection. Decoupling the two means the
search keeps its selection pressure while the parameter adaptation stops being
fed noise.

Every extra evaluation from N0 and N3 goes through the tracker and is charged to
the budget. A test enforces it: an algorithm that quietly buys evaluations would
win for the wrong reason.

THE KNOWN REMEDY (not a contribution, cited)
--------------------------------------------
The CEC'2017 gate measured V11 weakest on the composition class, F21-F30, where
BIPOP-CMA-ES leads (3.70 against V11's 3.95) -- the class that needs global
restarts. ``RESTART`` adds an IPOP-style restart (Auger and Hansen, 2005):
on trigger the population is reinitialised uniformly and ``N_init`` doubles.
This is a textbook mechanism used as such.

DEFAULTS. Inherited from V11 where V11 tuned them at 10D. The noise parameters
come from the simulation documented in ``NoiseEstimator``. ``CREDIT_TAU`` and the
re-evaluation counts are to be tuned on the noise study's own tuning split and
then frozen, not chosen from the reported results.
"""

from __future__ import annotations

import numpy as np

from ..noise import NoiseEstimator
from .common import levy_flight


def TEMOA_V12(obj_func, dim, bounds, max_fes, rng,
              POP_FACTOR=12, POP_MAX=500, P_MIN=0.02, P_EIG=0.4,
              ADAPT_EIG=True, ARC_RATE=1.0, CR_FLOOR=True,
              OPS=(0, 1, 3), EIGEN_GATE=True,
              DIV_GUARD=True, DIV_THRESH=1e-4, DIV_BOOST=0.5,
              # --- N0..N3: the mechanism -------------------------------------
              NOISE_M=5, NOISE_ALPHA=0.15, CREDIT_TAU=1.0,
              RANK_WEIGHTS=True, REEVAL_R=0, REEVAL_Q=1,
              # --- known remedy ----------------------------------------------
              RESTART=True, RESTART_BUDGET_FRAC=0.2, RESTART_GROWTH=2.0,
              logger=None):
    lb, ub = bounds
    OPS = tuple(OPS)
    N_min, H_SIZE = 4, 6
    K_OPS = len(OPS)
    LEVY_SLOT = OPS.index(3) if 3 in OPS else None
    span = float(ub - lb)

    N_init = int(np.clip(round(POP_FACTOR * dim), 40, POP_MAX))
    noise = NoiseEstimator(m=NOISE_M, alpha=NOISE_ALPHA)
    fes = 0

    def evaluate(x):
        nonlocal fes
        fes += 1
        return obj_func(x)

    # ---- one epoch = one (re)start -------------------------------------------
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
        M_F[-1], M_CR[-1] = 0.9, 0.9
        k_mem = 0
        archive = np.empty((0, dim))
        op_quality = np.full(K_OPS, 0.5)
        op_prob = np.full(K_OPS, 1.0 / K_OPS)
        eig_quality = np.array([0.5, 0.5])
        B = np.eye(dim)
        guard_active = False
        epoch_fes0 = fes
        stalled = 0

        while fes < max_fes:
            t = fes / max_fes
            order = np.argsort(fitness)
            pop, fitness = pop[order], fitness[order]
            idx = np.arange(pop_size)

            # -- eigenbasis, only where the covariance is estimable -------------
            n_samples = max(2, pop_size // 2)
            eig_ok = (not EIGEN_GATE) or (n_samples > dim)
            if eig_ok:
                C = np.cov(pop[:n_samples], rowvar=False)
                if np.all(np.isfinite(C)):
                    _, B = np.linalg.eigh(C)

            # -- parameters (jSO schedule) --------------------------------------
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
            m = ops == 0
            if np.any(m):
                V[m] = pop[m] + Fwc[m] * (x_pbest[m] - pop[m]) + Fc[m] * diff[m]
            m = ops == 1
            if np.any(m):
                V[m] = pop[m] + Fc[m] * (X_lead - pop[m]) + Fc[m] * diff[m]
            m = ops == 2
            if np.any(m):
                l = rng.uniform(-1.0, 1.0, (int(m.sum()), 1))
                V[m] = x_pbest[m] + np.abs(x_pbest[m] - pop[m]) * np.exp(l) * np.cos(2.0 * np.pi * l)
            m = ops == 3
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

            low, high = U < lb, U > ub
            U[low] = ((lb + pop) / 2.0)[low]
            U[high] = ((ub + pop) / 2.0)[high]

            n_eval = min(pop_size, max_fes - fes)
            fit_U = np.full(pop_size, np.inf)
            for i in range(n_eval):
                fit_U[i] = evaluate(U[i])

            # -- N0: estimate the noise scale ----------------------------------
            # The pair must be two independent draws at the SAME point, from a
            # point that has not been through selection. Using a stored parent
            # fitness is wrong: it was accepted partly because its noise draw was
            # favourable, so the difference has a non-zero mean and inflated
            # spread. Measured, that bias made sigma_hat about 2.7x too large,
            # which over-filtered the credit set. Trial vectors are untouched by
            # selection, so they give an unbiased pair. Only the first value is
            # used for selection; the second feeds the estimator alone.
            if NOISE_M > 0:
                who = rng.choice(n_eval, size=min(NOISE_M, n_eval), replace=False) \
                    if n_eval > 0 else np.empty(0, dtype=int)
                diffs = []
                for i in who:
                    if fes >= max_fes:
                        break
                    diffs.append(fit_U[i] - evaluate(U[i]))
                if diffs:
                    noise.update_from_pairs(diffs)

            improved = fit_U < fitness
            accept = fit_U <= fitness          # selection: unchanged from V11
            df = fitness - fit_U               # observed gap delta~

            # -- N1: filter the CREDIT set, not the selection -------------------
            credit = improved.copy()
            if CREDIT_TAU > 0 and noise.sigma > 0:
                credit &= df > CREDIT_TAU * noise.gap_sigma

            # -- N3: sharpen the weights of the largest-gap credited trials -----
            if REEVAL_R > 0 and REEVAL_Q > 1 and np.any(credit):
                cand = np.where(credit)[0]
                cand = cand[np.argsort(-df[cand])][:REEVAL_R]
                for i in cand:
                    if fes + (REEVAL_Q - 1) >= max_fes:
                        break
                    vals = [fit_U[i]] + [evaluate(U[i]) for _ in range(REEVAL_Q - 1)]
                    fit_U[i] = float(np.mean(vals))
                    df[i] = fitness[i] - fit_U[i]
                credit &= df > 0

            if np.any(credit):
                archive = np.vstack([archive, pop[credit]])
                d = df[credit]
                if RANK_WEIGHTS:
                    # log-rank weights: bounded influence, so one noise-inflated
                    # gap cannot take over the Lehmer mean
                    n_c = d.size
                    rank = np.empty(n_c)
                    rank[np.argsort(-d)] = np.arange(1, n_c + 1)
                    w = np.log(n_c + 0.5) - np.log(rank)
                    w = np.maximum(w, 0.0)
                    w = w / w.sum() if w.sum() > 0 else np.full(n_c, 1.0 / n_c)
                else:
                    w = d / d.sum()

                S_CR = CR[credit]
                den = np.sum(w * S_CR)
                mcr = np.sum(w * S_CR**2) / den if den > 0 else 0.0
                uses_F = ops[credit] != 2
                if np.any(uses_F):
                    wf = w[uses_F]
                    wf = wf / wf.sum() if wf.sum() > 0 else wf
                    S_F = F[credit][uses_F]
                    dF = np.sum(wf * S_F)
                    if dF > 0:
                        M_F[k_mem] = (M_F[k_mem] + np.sum(wf * S_F**2) / dF) / 2.0
                M_CR[k_mem] = (M_CR[k_mem] + mcr) / 2.0
                k_mem = (k_mem + 1) % (H_SIZE - 1)

            if logger is not None:
                logger.append({"fes": fes, "t": t, "sigma_hat": noise.sigma,
                               "accepted": accept.copy(), "credit": credit.copy(),
                               "obs_gap": df.copy(), "U": U.copy(), "parent": pop.copy(),
                               "M_F": M_F.copy(), "M_CR": M_CR.copy(),
                               "best": float(min(fitness.min(), fit_U.min()))})

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

            # LPSR, measured over this epoch's own budget
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

            # -- restart trigger (IPOP-style, cited) ----------------------------
            if RESTART:
                budget_left = (max_fes - fes) / max_fes
                collapsed = diversity < DIV_THRESH or pop_size <= N_min
                if collapsed and stalled > 0 and budget_left > RESTART_BUDGET_FRAC:
                    N_init = int(min(POP_MAX, round(N_init * RESTART_GROWTH)))
                    break          # leave the epoch; outer loop reinitialises
