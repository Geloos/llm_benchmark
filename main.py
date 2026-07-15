#!/usr/bin/env python3
"""
main.py

One entry point for the whole pipeline. Runs, in order:
  1. scripts/inject_prompts.py   attack_logs + injections -> attack_logs_injected/
  2. scripts/run_benchmark.py    attack_logs_injected     -> results/

How to run it:
  python main.py                 # inject, then benchmark (all defaults)
  python main.py --skip-inject   # skip step 1, only run the benchmark
  python main.py --skip-benchmark  # only (re)build attack_logs_injected/

  Any extra args after the known flags are forwarded to run_benchmark.py, e.g.
      python main.py --models llama3.1:8b --injections T0 T1
      python main.py --skip-inject --logs drupal --exclude-injections S_

Notes:
  - Paths are resolved relative to this file, so it works from any cwd.
  - If step 1 fails, step 2 does not run.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
INJECT = SCRIPTS / "inject_prompts.py"
BENCHMARK = SCRIPTS / "run_benchmark.py"


def run_step(name: str, script: Path, extra_args: list[str]) -> int:
    """Run one script with the shared cwd (project root) so its default relative
    paths (attack_logs, attack_logs_injected, results, ...) resolve correctly."""
    cmd = [sys.executable, str(script), *extra_args]
    print(f"\n=== {name} ===")
    print("  " + " ".join(cmd) + "\n")
    return subprocess.run(cmd, cwd=ROOT).returncode


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the injection pipeline: inject prompts, then benchmark.",
        epilog="Unknown args are forwarded to scripts/run_benchmark.py.",
    )
    ap.add_argument("--skip-inject", action="store_true",
                    help="skip step 1 (reuse the existing attack_logs_injected/)")
    ap.add_argument("--skip-benchmark", action="store_true",
                    help="skip step 2 (only rebuild attack_logs_injected/)")
    args, benchmark_args = ap.parse_known_args()

    if not args.skip_inject:
        rc = run_step("STEP 1  inject prompts", INJECT, [])
        if rc != 0:
            print(f"\ninject step failed (exit {rc}); not running the benchmark.")
            return rc
    else:
        print("\n=== STEP 1  inject prompts -> skipped ===")

    if not args.skip_benchmark:
        rc = run_step("STEP 2  run benchmark", BENCHMARK, benchmark_args)
        if rc != 0:
            print(f"\nbenchmark step failed (exit {rc}).")
            return rc
    else:
        print("\n=== STEP 2  run benchmark -> skipped ===")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
