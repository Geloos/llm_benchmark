#!/usr/bin/env python3
"""
main.py

One entry point for the whole pipeline. Runs, in order:
  1. scripts/inject_prompts.py     attack_logs + injections -> attack_logs_injected/
  1b. scripts/check_context.py     attack_logs_injected     -> analysis/context_check.csv
  2. scripts/run_benchmark.py      attack_logs_injected     -> results/
  3. scripts/summarize_results.py  results                  -> analysis/

Step 1b is a preflight: it proves each model really ingests a whole injected log instead
of a silently truncated one, and aborts the run if it does not.

How to run it:
  python main.py                 # inject, check, benchmark, then summarize (all defaults)
  python main.py --skip-inject   # skip step 1, only check + benchmark + summary
  python main.py --skip-check    # trust num_ctx, go straight to the benchmark
  python main.py --skip-benchmark  # only (re)build attack_logs_injected/, then summarize
  python main.py --skip-summary  # inject + benchmark, no analysis
  python main.py --skip-inject --skip-benchmark  # only re-run the analysis (step 3)

  Any extra args after the known flags are forwarded to run_benchmark.py, e.g.
      python main.py --models llama3.1:8b --injections DO_ PH_
      python main.py --skip-inject --logs drupal --exclude-injections AF_ SPLIT_

Notes:
  - Paths are resolved relative to this file, so it works from any cwd.
  - If step 1 fails, steps 2 and 3 do not run. The summary (step 3) runs independently of
    the inject/benchmark skips, so you can re-summarize existing results on their own.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
INJECT = SCRIPTS / "inject_prompts.py"
CHECK = SCRIPTS / "check_context.py"
BENCHMARK = SCRIPTS / "run_benchmark.py"
SUMMARIZE = SCRIPTS / "summarize_results.py"


def run_step(name: str, script: Path, extra_args: list[str]) -> int:
    """Run one script with the shared cwd (project root) so its default relative
    paths (attack_logs, attack_logs_injected, results, ...) resolve correctly."""
    cmd = [sys.executable, str(script), *extra_args]
    print(f"\n=== {name} ===")
    print("  " + " ".join(cmd) + "\n")
    return subprocess.run(cmd, cwd=ROOT).returncode


def forward_flags(extra_args: list[str], keep: set[str]) -> list[str]:
    """Pass only the flags a step actually accepts. The benchmark takes filters the
    context preflight knows nothing about (--injections, --exclude-logs, ...), so hand it
    just the ones it shares: which models, which logs, which window."""
    out, taking = [], False
    for token in extra_args:
        if token.startswith("--"):
            taking = token.split("=", 1)[0] in keep
        if taking:
            out.append(token)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the injection pipeline: inject prompts, then benchmark.",
        epilog="Unknown args are forwarded to scripts/run_benchmark.py.",
    )
    ap.add_argument("--skip-inject", action="store_true",
                    help="skip step 1 (reuse the existing attack_logs_injected/)")
    ap.add_argument("--skip-check", action="store_true",
                    help="skip the context preflight (step 1b)")
    ap.add_argument("--skip-benchmark", action="store_true",
                    help="skip step 2 (only rebuild attack_logs_injected/)")
    ap.add_argument("--skip-summary", action="store_true",
                    help="skip step 3 (do not (re)build analysis/)")
    args, benchmark_args = ap.parse_known_args()

    if not args.skip_inject:
        rc = run_step("STEP 1  inject prompts", INJECT, [])
        if rc != 0:
            print(f"\ninject step failed (exit {rc}); not running the benchmark.")
            return rc
    else:
        print("\n=== STEP 1  inject prompts -> skipped ===")

    if not args.skip_benchmark and not args.skip_check:
        rc = run_step("STEP 1b context preflight", CHECK,
                      forward_flags(benchmark_args, {"--models", "--logs", "--num-ctx"}))
        if rc != 0:
            print(f"\ncontext preflight failed (exit {rc}): at least one model would be "
                  f"fed a truncated log, which makes its verdict meaningless. Fix the "
                  f"window or the model list, or re-run with --skip-check to override.")
            return rc
    else:
        print("\n=== STEP 1b context preflight -> skipped ===")

    if not args.skip_benchmark:
        rc = run_step("STEP 2  run benchmark", BENCHMARK, benchmark_args)
        if rc != 0:
            print(f"\nbenchmark step failed (exit {rc}).")
            return rc
    else:
        print("\n=== STEP 2  run benchmark -> skipped ===")

    if not args.skip_summary:
        rc = run_step("STEP 3  summarize results", SUMMARIZE, [])
        if rc != 0:
            print(f"\nsummary step failed (exit {rc}).")
            return rc
    else:
        print("\n=== STEP 3  summarize results -> skipped ===")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
