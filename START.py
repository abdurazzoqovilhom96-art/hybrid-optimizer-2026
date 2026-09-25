"""START.py -- the single entry point. Run this and nothing else.

    python START.py              # check, test, then run the gate and analyse it
    python START.py --check      # dependencies and tests only (~6 minutes)
    python START.py --jobs 20    # set the worker count explicitly

It refuses to start the long run until every test passes, because a benchmark
that is wrong is worse than no benchmark.

WHAT IT RUNS, AND WHY THAT AND NOT MORE
---------------------------------------
The default is the **decision gate**: CEC'2017 at D=10, 29 functions, 51 runs,
100 000 evaluations each -- 1.18e9 evaluations, about 2.5 hours on 20 workers.
That is the run that answers the only question worth answering right now: where,
if anywhere, is this algorithm actually competitive against CMA-ES and the
adaptive-DE lineage it was built from.

Higher dimensions are deliberately NOT the default. Measured from this machine's
throughput, D=30 is a further 11 hours and D=50 about 27 hours on 20 workers.
Running them before the gate has been read would spend days to answer a question
the gate already answers. When you want them:

    python START.py --dims 10 30            # after reading the gate result

Everything is resumable. Interrupt with Ctrl+C and run the same command again;
finished runs are skipped and the results are bit-identical to an uninterrupted
run (there is a test for that).
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

REQUIRED = [
    ("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"),
    ("joblib", "joblib"), ("matplotlib", "matplotlib"),
    ("opfunu", "opfunu"), ("cma", "cma"),
]

TEST_FILES = [
    ("tests/test_all.py", "regression: problems, tracker, statistics, reproducibility"),
    ("tests/test_suites.py", "CEC suites: function numbering, optima, protocols"),
    ("tests/test_registry.py", "the line-up: eight rivals, no weak baselines, stable seeds"),
    ("tests/test_competitor_validation.py", "the competitors really are what they claim"),
]

SLOW_TEST = ("tests/test_port_fidelity.py", "ported V10 matches the original file")


def rule(title=""):
    width = min(shutil.get_terminal_size((80, 20)).columns, 78)
    print("-" * width if not title else f"--- {title} " + "-" * max(0, width - len(title) - 5))


def run(cmd, **kw):
    """Run a subprocess, streaming its output. Returns the exit code."""
    return subprocess.run(cmd, cwd=ROOT, **kw).returncode


def check_python():
    if sys.version_info < (3, 9):
        print(f"[X] Python {sys.version_info.major}.{sys.version_info.minor} is too old; 3.9+ required.")
        return False
    print(f"[ok] Python {platform.python_version()} ({platform.system()} {platform.machine()})")
    return True


def check_deps(auto_install=True):
    missing = []
    for import_name, pip_name in REQUIRED:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)
    if not missing:
        print(f"[ok] all {len(REQUIRED)} dependencies present")
        return True
    print(f"[!] missing: {', '.join(missing)}")
    if not auto_install:
        print(f"    install with: {PY} -m pip install -r requirements.txt")
        return False
    print("    installing from requirements.txt ...")
    if run([PY, "-m", "pip", "install", "-q", "-r", "requirements.txt"]) != 0:
        print("[X] installation failed. Install manually and re-run.")
        return False
    for import_name, pip_name in REQUIRED:
        try:
            __import__(import_name)
        except ImportError:
            print(f"[X] {pip_name} still not importable after installation.")
            return False
    print("[ok] dependencies installed")
    return True


def run_tests(include_slow=False):
    ok = True
    tests = list(TEST_FILES) + ([SLOW_TEST] if include_slow else [])
    for path, what in tests:
        print(f"\n> {path} -- {what}")
        code = run([PY, path])
        if code != 0:
            print(f"[X] FAILED: {path}")
            ok = False
        else:
            print(f"[ok] {path}")
    return ok


def gate(dims, runs, jobs, out, resume, smoke):
    cmd = [PY, "run_all.py", "--suite", "cec2017", "--jobs", str(jobs), "--out", out]
    if smoke:
        cmd.append("--smoke")
    else:
        cmd += ["--dims", *map(str, dims)]
        if runs:
            cmd += ["--runs", str(runs)]
    if resume:
        cmd.append("--resume")
    return run(cmd)


def analyse(out):
    # The control is read from the registry, never typed here: a name that
    # drifts out of sync would silently analyse the wrong algorithm.
    from temoa.registry import TARGET
    code = run([PY, "analyze.py", "--out", out, "--control", TARGET])
    if code == 0:
        run([PY, "analyze.py", "--out", out, "--control", "TEMOA_V10"])
    return code


def default_jobs():
    n = os.cpu_count() or 4
    return max(1, int(n * 0.8))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Single entry point for the TEMOA benchmark study.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("WHAT IT RUNS")[0])
    ap.add_argument("--check", action="store_true",
                    help="dependencies and tests only, then stop")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end run instead of the gate (~2 min)")
    ap.add_argument("--dims", type=int, nargs="+", default=[10],
                    help="dimensions (default 10 = the gate). 30 adds ~11 h, 50 ~27 h")
    ap.add_argument("--runs", type=int, default=None,
                    help="override the competition's 51 runs (not comparable with published tables)")
    ap.add_argument("--jobs", type=int, default=default_jobs())
    ap.add_argument("--out", default="results_cec2017")
    ap.add_argument("--no-resume", action="store_true", help="start over instead of resuming")
    ap.add_argument("--full-tests", action="store_true",
                    help="also run the slow port-fidelity check (~4 min)")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    print("=" * 78)
    print("  TEMOA benchmark study -- decision gate")
    print("=" * 78)

    rule("1/4  environment")
    if not check_python() or not check_deps():
        return 1
    print(f"[ok] workers: {args.jobs} of {os.cpu_count()} logical cores")

    rule("2/4  tests")
    if not run_tests(include_slow=args.full_tests):
        print("\n[X] Tests failed. The experiment will NOT start.")
        print("    A benchmark that is wrong is worse than no benchmark.")
        return 1
    print("\n[ok] all tests passed")

    if args.check:
        print(f"\nChecks finished in {time.perf_counter() - t0:.0f}s. "
              f"Run without --check to start the experiment.")
        return 0

    rule("3/4  experiment")
    if args.smoke:
        print("smoke run: smallest dimension, 3 runs, tiny budget (~2 min)")
    else:
        big = [d for d in args.dims if d >= 30]
        print(f"CEC'2017, dimensions {args.dims}, "
              f"{args.runs or 51} runs, 10000*D evaluations per run")
        print("estimated: ~2.5 h for D=10 on 20 workers"
              + (f"; plus ~11 h per dimension for {big}" if big else ""))
        print("resumable: interrupt with Ctrl+C and re-run the same command")
        if not args.yes and sys.stdin.isatty():
            try:
                if input("\nstart? [y/N] ").strip().lower() not in ("y", "yes"):
                    print("cancelled.")
                    return 0
            except (EOFError, KeyboardInterrupt):
                print("\ncancelled.")
                return 0

    try:
        code = gate(args.dims, args.runs, args.jobs, args.out,
                    resume=not args.no_resume, smoke=args.smoke)
    except KeyboardInterrupt:
        print("\n[!] interrupted. Re-run the same command to resume where it stopped.")
        return 130
    if code != 0:
        print("[X] the experiment did not finish cleanly.")
        return code

    rule("4/4  analysis")
    out = args.out + ("_smoke" if args.smoke else "")
    if analyse(out) != 0:
        print("[X] analysis failed, but the raw results are safe in "
              f"{out}/raw/results.csv -- re-run: {PY} analyze.py --out {out}")
        return 1

    rule()
    print(f"DONE in {(time.perf_counter() - t0) / 60:.1f} min")
    print(f"  tables  : {out}/tables")
    print(f"  figures : {out}/figures")
    print(f"  raw     : {out}/raw/results.csv")
    print(f"  manifest: {out}/manifest.json  (versions, seeds, exact protocol)")
    print("\nRead reports/TAHLIL_UZ.md and reports/PRIOR_ART.md before drawing any")
    print("conclusion from these numbers.")
    return 0


if __name__ == "__main__":      # required on Windows: loky spawns processes
    sys.exit(main())
