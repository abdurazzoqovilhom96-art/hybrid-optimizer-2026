"""Barcha algoritmlar uchun umumiy yadro: FES hisoblagichi va DE yordamchilari.

Bu yerdagi kod CBA-SHADE (V1) dan o'zgarishsiz ko'chirilgan va V1 da
sinovdan o'tgan. Yagona hisoblagich barcha algoritmlar uchun bir xil
byudjet hisobini kafolatlaydi.
"""
import numpy as np


class Tracker:
    """Yagona FES hisoblagichi va best-so-far egri chizig'i."""

    def __init__(self, func, max_fes, n_points=100):
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


def sample_F(mu):
    """Cauchy(mu, 0.1) dan F; F <= 0 qayta tanlanadi, yuqoridan 1.0 bilan chegaralanadi."""
    F = mu + 0.1 * np.random.standard_cauchy(len(mu))
    bad = F <= 0
    while np.any(bad):
        F[bad] = mu[bad] + 0.1 * np.random.standard_cauchy(int(bad.sum()))
        bad = F <= 0
    return np.minimum(F, 1.0)


def lehmer(vals, w):
    """Vaznlangan Lehmer o'rtachasi (SHADE xotira yangilanishi)."""
    den = np.sum(w * vals)
    return np.sum(w * vals ** 2) / den if den > 0 else -1.0


def pick_r2(n_union, idx, r1):
    """r2 ni P u A dan tanlash: r2 != i va r2 != r1."""
    n = len(idx)
    r2 = np.random.randint(0, n_union, n)
    clash = (r2 == idx) | (r2 == r1)
    while np.any(clash):
        r2[clash] = np.random.randint(0, n_union, int(clash.sum()))
        clash = (r2 == idx) | (r2 == r1)
    return r2


def midpoint(U, pop, lb, ub):
    """Midpoint-target chegara tuzatishi (L-SHADE). Barcha algoritmlarda bir xil."""
    low, high = U < lb, U > ub
    U[low] = ((lb + pop) / 2.0)[low]
    U[high] = ((ub + pop) / 2.0)[high]
    return U


def arch_push(archive, losers, arc_max):
    if arc_max <= 0:
        return archive
    archive = np.vstack([archive, losers]) if len(archive) else losers.copy()
    if len(archive) > arc_max:
        archive = archive[np.random.choice(len(archive), arc_max, replace=False)]
    return archive


def rsp_ranks(pop_size, idx, k_rsp):
    """Rank asosidagi tanlov bosimi (LSHADE-RSP): r1 uchun indekslar."""
    rk = k_rsp * (pop_size - idx) / pop_size + 1.0
    pr = rk / rk.sum()
    r1 = np.random.choice(pop_size, pop_size, p=pr)
    same = r1 == idx
    while np.any(same):
        r1[same] = np.random.choice(pop_size, int(same.sum()), p=pr)
        same = r1 == idx
    return r1
