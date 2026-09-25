"""Comparison baselines.

Two of the original baselines are configured below their published form, which
inflates the reported margin. Both the original and the corrected version are
provided here, so the report can quantify how much of the original claim came
from the baseline configuration rather than from the hybrid.

``PSO_orig``  c1 = c2 = 2.0, w: 0.9 -> 0.4, no velocity clamp, no constriction.
              This sits outside the Poli/Trelea/Clerc stability region: the
              second-order recurrence governing each coordinate has a spectral
              radius > 1 for w >= 0.5 at c = 2.0, so the swarm diverges and is
              held together only by the box clamp. Particles then spend the run
              bouncing off the bounds.
``PSO``       Clerc-Kennedy constriction: chi = 0.7298, c1 = c2 = 2.05, with the
              usual Vmax = 0.5*(ub-lb) clamp. The standard, convergent form.

``HHO_orig``  The original study's HHO with the Levy-flight "progressive rapid
              dive" removed -- that is, all four exploitation branches collapsed
              to plain besiege steps. TEMOA explicitly borrows and credits HHO's
              Levy rapid dive as one of its four operators, so comparing against
              an HHO stripped of exactly that component is self-serving.
``HHO``       Heidari et al. (2019) as published, rapid dives included. The dive
              branches cost two extra evaluations each; they are charged.

DE, GWO, WOA and SCA are faithful ports of the original file. Note that its SCA
adds a greedy selection step that the published SCA does not have; that makes
the baseline *stronger*, so it is kept.

All are population 30, as in the original study.
"""

from __future__ import annotations

import numpy as np

from .common import levy_flight

POP = 30


def baseline_DE(obj_func, dim, bounds, max_fes, rng):
    """DE/rand/1/bin, F = 0.5, CR = 0.8, NP = 30."""
    lb, ub = bounds
    pop = rng.uniform(lb, ub, (POP, dim))
    fitness = np.array([obj_func(i) for i in pop])
    fes = POP
    while fes < max_fes:
        for i in range(POP):
            if fes >= max_fes:
                break
            choices = rng.choice(np.delete(np.arange(POP), i), 3, replace=False)
            a, b, c = pop[choices]
            mutant = np.clip(a + 0.5 * (b - c), lb, ub)
            cross = rng.random(dim) < 0.8
            if not np.any(cross):
                cross[rng.integers(0, dim)] = True
            trial = np.where(cross, mutant, pop[i])
            f_trial = obj_func(trial)
            fes += 1
            if f_trial < fitness[i]:
                pop[i], fitness[i] = trial, f_trial


def baseline_GWO(obj_func, dim, bounds, max_fes, rng):
    lb, ub = bounds
    pop = rng.uniform(lb, ub, (POP, dim))
    fitness = np.array([obj_func(ind) for ind in pop])
    fes = POP
    while fes < max_fes:
        s_idx = np.argsort(fitness)
        pop, fitness = pop[s_idx], fitness[s_idx]
        a = 2.0 - fes * (2.0 / max_fes)
        for i in range(POP):
            if fes >= max_fes:
                break
            X_new = np.zeros(dim)
            for lead in (pop[0], pop[1], pop[2]):
                A = 2 * a * rng.random(dim) - a
                C = 2 * rng.random(dim)
                X_new += lead - A * np.abs(C * lead - pop[i])
            step = np.clip(X_new / 3.0, lb, ub)
            pop[i], fitness[i] = step, obj_func(step)
            fes += 1


def baseline_WOA(obj_func, dim, bounds, max_fes, rng):
    lb, ub = bounds
    pop = rng.uniform(lb, ub, (POP, dim))
    fitness = np.array([obj_func(ind) for ind in pop])
    fes = POP
    while fes < max_fes:
        X_best = pop[np.argmin(fitness)].copy()
        a = 2.0 - fes * (2.0 / max_fes)
        a2 = -1.0 + fes * (-1.0 / max_fes)
        for i in range(POP):
            if fes >= max_fes:
                break
            A = 2 * a * rng.random() - a
            C = 2 * rng.random()
            l = (a2 - 1) * rng.random() + 1
            if rng.random() < 0.5:
                if abs(A) >= 1:
                    X_r = pop[rng.integers(0, POP)]
                    step = X_r - A * np.abs(C * X_r - pop[i])
                else:
                    step = X_best - A * np.abs(C * X_best - pop[i])
            else:
                step = np.abs(X_best - pop[i]) * np.exp(l) * np.cos(2 * np.pi * l) + X_best
            step = np.clip(step, lb, ub)
            pop[i], fitness[i] = step, obj_func(step)
            fes += 1


def _pso(obj_func, dim, bounds, max_fes, rng, constriction: bool):
    lb, ub = bounds
    pop = rng.uniform(lb, ub, (POP, dim))
    vel = np.zeros_like(pop)
    pbest = pop.copy()
    pbest_fit = np.array([obj_func(ind) for ind in pop])
    g = np.argmin(pbest_fit)
    gbest, gbest_fit = pbest[g].copy(), pbest_fit[g]
    fes = POP
    vmax = 0.5 * (ub - lb)
    while fes < max_fes:
        w = 0.9 - 0.5 * (fes / max_fes)
        for i in range(POP):
            if fes >= max_fes:
                break
            r1, r2 = rng.random(dim), rng.random(dim)
            if constriction:
                chi, c1, c2 = 0.7298, 2.05, 2.05
                vel[i] = chi * (vel[i] + c1 * r1 * (pbest[i] - pop[i])
                                + c2 * r2 * (gbest - pop[i]))
                vel[i] = np.clip(vel[i], -vmax, vmax)
            else:
                vel[i] = w * vel[i] + 2.0 * r1 * (pbest[i] - pop[i]) + 2.0 * r2 * (gbest - pop[i])
            pop[i] = np.clip(pop[i] + vel[i], lb, ub)
            fit = obj_func(pop[i])
            fes += 1
            if fit < pbest_fit[i]:
                pbest[i], pbest_fit[i] = pop[i].copy(), fit
                if fit < gbest_fit:
                    gbest, gbest_fit = pop[i].copy(), fit


def baseline_PSO(obj_func, dim, bounds, max_fes, rng):
    """Clerc-Kennedy constriction PSO (convergent, standard)."""
    _pso(obj_func, dim, bounds, max_fes, rng, constriction=True)


def baseline_PSO_orig(obj_func, dim, bounds, max_fes, rng):
    """The original study's PSO: c=2.0, no clamp, no constriction (divergent)."""
    _pso(obj_func, dim, bounds, max_fes, rng, constriction=False)


def baseline_SCA(obj_func, dim, bounds, max_fes, rng):
    lb, ub = bounds
    pop = rng.uniform(lb, ub, (POP, dim))
    fitness = np.array([obj_func(ind) for ind in pop])
    X_best, X_best_fit = pop[np.argmin(fitness)].copy(), float(np.min(fitness))
    fes = POP
    while fes < max_fes:
        r1 = 2.0 - fes * (2.0 / max_fes)
        for i in range(POP):
            if fes >= max_fes:
                break
            r2 = 2 * np.pi * rng.random(dim)
            r3 = 2 * rng.random(dim)
            r4 = rng.random(dim)
            mask = r4 < 0.5
            step = np.empty(dim)
            step[mask] = pop[i][mask] + r1 * np.sin(r2[mask]) * np.abs(r3[mask] * X_best[mask] - pop[i][mask])
            step[~mask] = pop[i][~mask] + r1 * np.cos(r2[~mask]) * np.abs(r3[~mask] * X_best[~mask] - pop[i][~mask])
            step = np.clip(step, lb, ub)
            fit = obj_func(step)
            fes += 1
            if fit < fitness[i]:
                pop[i], fitness[i] = step, fit
                if fit < X_best_fit:
                    X_best, X_best_fit = step.copy(), fit


def _hho(obj_func, dim, bounds, max_fes, rng, rapid_dive: bool):
    lb, ub = bounds
    pop = rng.uniform(lb, ub, (POP, dim))
    fitness = np.array([obj_func(ind) for ind in pop])
    b = np.argmin(fitness)
    X_rabbit, rabbit_fit = pop[b].copy(), float(fitness[b])
    fes = POP
    while fes < max_fes:
        E1 = 2 * (1 - fes / max_fes)
        X_m = np.mean(pop, axis=0)
        for i in range(POP):
            if fes >= max_fes:
                break
            E = E1 * (2 * rng.random() - 1)
            J = 2 * (1 - rng.random())          # random jump strength
            if abs(E) >= 1:                      # exploration
                if rng.random() >= 0.5:
                    X_rand = pop[rng.integers(0, POP)]
                    step = X_rand - rng.random() * np.abs(X_rand - 2 * rng.random() * pop[i])
                else:
                    step = (X_rabbit - X_m) - rng.random() * (lb + rng.random() * (ub - lb))
                step = np.clip(step, lb, ub)
                fit = obj_func(step)
                fes += 1
            else:                                # exploitation
                r = rng.random()
                if r >= 0.5 and abs(E) >= 0.5:               # soft besiege
                    step = (X_rabbit - pop[i]) - E * np.abs(J * X_rabbit - pop[i])
                    step = np.clip(step, lb, ub)
                    fit = obj_func(step)
                    fes += 1
                elif r >= 0.5:                               # hard besiege
                    step = X_rabbit - E * np.abs(X_rabbit - pop[i])
                    step = np.clip(step, lb, ub)
                    fit = obj_func(step)
                    fes += 1
                elif not rapid_dive:
                    # the original study's simplification: no Levy dive
                    ref = pop[i] if abs(E) >= 0.5 else X_m
                    step = X_rabbit - E * np.abs(J * X_rabbit - ref)
                    step = np.clip(step, lb, ub)
                    fit = obj_func(step)
                    fes += 1
                else:
                    # progressive rapid dives (Heidari et al. 2019, eq. 10-12)
                    ref = pop[i] if abs(E) >= 0.5 else X_m
                    Y = np.clip(X_rabbit - E * np.abs(J * X_rabbit - ref), lb, ub)
                    fY = obj_func(Y)
                    fes += 1
                    if fY < fitness[i]:
                        step, fit = Y, fY
                    elif fes < max_fes:
                        S = rng.random(dim)
                        Z = np.clip(Y + S * levy_flight(rng, dim), lb, ub)
                        fZ = obj_func(Z)
                        fes += 1
                        step, fit = (Z, fZ) if fZ < fitness[i] else (pop[i], fitness[i])
                    else:
                        step, fit = pop[i], fitness[i]
            if fit < fitness[i]:
                pop[i], fitness[i] = step, fit
                if fit < rabbit_fit:
                    X_rabbit, rabbit_fit = np.asarray(step).copy(), fit


def baseline_HHO(obj_func, dim, bounds, max_fes, rng):
    """HHO as published, including the Levy-flight progressive rapid dives."""
    _hho(obj_func, dim, bounds, max_fes, rng, rapid_dive=True)


def baseline_HHO_orig(obj_func, dim, bounds, max_fes, rng):
    """The original study's HHO, with the Levy rapid dive removed."""
    _hho(obj_func, dim, bounds, max_fes, rng, rapid_dive=False)
