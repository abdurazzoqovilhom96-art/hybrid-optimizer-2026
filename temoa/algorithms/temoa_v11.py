"""TEMOA_V11 -- a repair of V10 driven by measurement, not by adding novelty.

Every change below is a response to something that was measured on the 30D
rotated suite (8 runs, median error). Nothing is included because it sounds
good; two ideas that sounded good were tested and dropped.

WHAT THE ABLATION SHOWED (30D, median error, V10 with operator subsets)

    configuration          Schwefel  NoisyRastrigin  RotatedElliptic  Rosenbrock
    FULL (0,1,2,3)         1.86e+03     3.53e+01         1.49e+04       1.22e+01
    op0 only               5.92e+01     1.51e+01         2.19e+02       1.33e+01
    op0 + op1 (leader)     1.18e+03     2.04e+01         1.32e+04       8.55e+00
    op0 + op2 (spiral)     2.38e+03     4.18e+01         9.97e+02       1.85e+01
    op0 + op3 (Levy)       2.79e-09     1.98e+01         1.39e+03       1.34e+01
    FULL, no ES tail       1.86e+03     3.53e+01         1.48e+04       1.21e+01

1. **Operator 2 (the WOA spiral) is dominated.** Adding it to operator 0 is
   worse than operator 0 alone on every function tested. It is removed.

2. **The (1+1)-ES tail contributes nothing.** V10 and V10-without-the-tail agree
   to within noise everywhere, and are bit-identical on two functions. It spends
   5% of the budget (4500 FES at 30D) for no measurable return. Removed; the
   budget returns to the main loop.

3. **The eigen-crossover is decisive where it applies, and only there.** On
   RotatedElliptic, operator 0 alone scores 2.19e+02 with the eigenbasis and
   1.35e+04 without it -- a factor of 62. But V10 builds the basis from
   ``cov(pop[:N/2])``, and LPSR drives N below the dimension well before the end:
   at 100D with 90% of the budget spent, a 100x100 covariance is estimated from
   25 points. The basis is then mostly noise. V11 gates the eigen-crossover on
   having strictly more samples than dimensions.

4. **The adaptive operator selection cannot fix this, and a better credit
   assignment would make it worse.** Instrumenting V10 on Schwefel: the spiral
   receives 33-64% of the trials and produces **66% of all fitness improvement**,
   while Levy is squeezed to 8.6%. The spiral genuinely earns its credit in the
   short term -- it descends fast by collapsing the population onto x_pbest --
   and it is exactly that collapse that costs six orders of magnitude at the end.
   So the first fix considered here, replacing success-frequency credit with
   DeltaF-magnitude credit, was **tested and rejected**: it would have given the
   spiral *more* probability, not less. Any credit scheme built on immediate
   improvement is blind to this, because the damage is delayed and shared.

   The response is therefore not a cleverer credit formula but an explicit
   diversity guard, which measures the thing the credit cannot see.

WHAT V11 CHANGES

  a. ``OPS = (0, 1, 3)``     -- the dominated spiral is gone.
  b. ``LS_FRACTION = 0.0``   -- no ES tail; 5% of the budget returns to search.
  c. eigen-crossover gated on ``n_samples > dim``.
  d. ``P_MIN = 0.02``        -- a dominated operator can actually be switched off
                                (V10's 0.05 floor guarantees every operator 5% of
                                every generation for the whole run).
  e. **diversity guard**     -- when the normalised population diversity falls
     below ``DIV_THRESH`` and the budget is not yet spent, the operator
     distribution is overridden in favour of the Levy operator until diversity
     recovers. This is the escape mechanism the ablation identified (op0+op3
     solves Schwefel to 2.8e-09) and which the credit assignment reliably
     starves.

PARAMETER TUNING. ``POP_FACTOR``, ``DIV_THRESH`` and ``DIV_BOOST`` were selected
by a 3 x 3 grid at **10D** over three functions chosen to cover the landscape
types (Ackley, Schwefel, RotatedElliptic), 10 runs each, scored by the mean of
log10(median error). The winner was POP_FACTOR = 12, DIV_THRESH = 1e-4,
DIV_BOOST = 0.5. Those values are then **frozen** for every dimension and every
function reported. No parameter was chosen by looking at 30D/50D/100D results.
Note that V10's POP_FACTOR = 6 is itself badly miscalibrated above 10D: at 30D,
raising V10's population from 6D to 18D changes RotatedElliptic from 1.45e+04 to
2.19 (a factor of 6600) while costing a factor of 21 on Rosenbrock. A large part
of what the original study reports as hybridisation benefit is population size.

HONEST SCOPE. V11 is expected to help where V10's collapse was diagnosed:
multimodal landscapes and ill-conditioned rotated ones. It is not expected to
dominate jSO or L-SHADE everywhere, and on a valley problem such as Rosenbrock
the removal of operator 1's advantage may cost a little. See the report for the
measured outcome, including where it loses.
"""

from __future__ import annotations

import numpy as np

from .common import levy_flight


def TEMOA_V11(obj_func, dim, bounds, max_fes, rng,
              POP_FACTOR=12, POP_MAX=500, P_MIN=0.02, P_EIG=0.4,
              ADAPT_EIG=True, ARC_RATE=1.0, CR_FLOOR=True,
              OPS=(0, 1, 3), EIGEN_GATE=True,
              DIV_GUARD=True, DIV_THRESH=1e-4, DIV_BOOST=0.5):
    lb, ub = bounds
    OPS = tuple(OPS)
    N_init = int(np.clip(round(POP_FACTOR * dim), 40, POP_MAX))
    N_min = 4
    H_SIZE = 6
    K_OPS = len(OPS)
    LEVY_SLOT = OPS.index(3) if 3 in OPS else None
    span = float(np.max(ub - lb)) if np.ndim(ub) else float(ub - lb)

    pop_size = N_init
    pop = lb + rng.random((pop_size, dim)) * (ub - lb)
    fitness = np.array([obj_func(ind) for ind in pop])
    fes = pop_size

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

    while fes < max_fes:
        t = fes / max_fes
        order = np.argsort(fitness)
        pop, fitness = pop[order], fitness[order]
        idx = np.arange(pop_size)

        # -- eigenbasis, only when the covariance is actually estimable --------
        n_samples = max(2, pop_size // 2)
        eig_ok = (not EIGEN_GATE) or (n_samples > dim)
        if eig_ok:
            C = np.cov(pop[:n_samples], rowvar=False)
            if np.all(np.isfinite(C)):
                _, B = np.linalg.eigh(C)

        # -- parameters (jSO schedule, unchanged from V10) ---------------------
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

        # -- diversity guard ---------------------------------------------------
        # Normalised mean per-coordinate spread. The credit assignment cannot see
        # population collapse, because the operator that causes it is also the one
        # producing the largest immediate gains. This can.
        diversity = float(np.mean(np.std(pop, axis=0))) / span
        probs = op_prob
        if DIV_GUARD and LEVY_SLOT is not None:
            guard_active = diversity < DIV_THRESH and t < 0.95
            if guard_active:
                probs = np.full(K_OPS, (1.0 - DIV_BOOST) / max(1, K_OPS - 1))
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
                M_F[k_mem] = (M_F[k_mem] + np.sum(wf * S_F**2) / np.sum(wf * S_F)) / 2.0
            M_CR[k_mem] = (M_CR[k_mem] + mcr) / 2.0
            k_mem = (k_mem + 1) % (H_SIZE - 1)

        # Credit assignment is left as success-frequency probability matching.
        # It is not the bottleneck (see the module docstring): a DeltaF-weighted
        # version was measured and would reward the harmful operator more.
        # While the diversity guard is overriding the distribution, credit is not
        # updated, so the forced Levy trials do not distort the estimates.
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

        new_size = max(N_min, int(round(N_init + (N_min - N_init) * fes / max_fes)))
        if new_size < pop_size:
            keep = np.argsort(fitness)[:new_size]
            pop, fitness = pop[keep], fitness[keep]
            pop_size = new_size
        arc_max = int(round(ARC_RATE * pop_size))
        if len(archive) > arc_max:
            archive = archive[rng.choice(len(archive), arc_max, replace=False)]
