"""Algorithm implementations. Every algorithm has the signature

    alg(tracker, dim, bounds, max_fes, rng) -> None

and reports results only through ``tracker``. ``rng`` is a
``numpy.random.Generator``; no algorithm touches the global RNG.
"""
