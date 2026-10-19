#!/usr/bin/env python3
"""
main.py

What it does:
  Runs the whole pipeline for one or more lanes: inject -> context preflight, then per
  temperature: benchmark the injected logs -> summarize -> charts, then a cross-lane
  comparison chart per temperature.

How to run it:
  python main.py                     # the plain corpus (attack_logs/)
  python main.py --corpus hexa       # the hex corpus (attack_logs_hexa/)
  python main.py --corpus both       # plain + hexa, then the comparison chart
  python main.py --temperatures low  # only level low (default: low medium)

  Stages are skipped with --skip-inject / --skip-check / --skip-benchmark /
  --skip-summary / --skip-charts / --skip-compare, and they compose. Any extra
  args are forwarded to run_benchmark.py:
      python main.py --models llama3.1:8b --injections DO_ PH_

What it outputs:
  Per lane: attack_logs_injected*/, then one temp_<t>/ folder per temperature under
  results*/ and analysis*/ (verdicts.csv,
  verdicts_by_injection.csv, summary.jsonl, report.md, reasoning_report.md at temp_medium,
  charts/), plus analysis*/context_check.csv. And per temperature the cross-lane chart
  analysis/temp_<t>/charts/plain_vs_hexa.png.

  Sampling, temperatures, seeds and the model set per temperature live in
  scripts/experiment.py. Every model sends its full recommended sampling; the levels are
  low / medium = 0.0 (greedy) / recommended, per model. Two experiments
  share the run, every series repeated 5 times (seeds 42..46, one seed_<n>/ folder each)
  and reported as mean +- std:
    temperature  every level x gpt-oss@low, llama3.1, gemma3;
    reasoning    medium x gpt-oss@low / @medium / @high. medium x gpt-oss@low is the
                 temperature experiment's run too, so only @medium and @high are extra.
  Narrow the
  seeds with a forwarded --seeds, or change the count with --runs:
      python main.py --temperatures medium --seeds 42 43 44
      python main.py --temperatures medium --runs 10

"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
import experiment  

INJECT = SCRIPTS / "inject_prompts.py"
CHECK = SCRIPTS / "check_context.py"
BENCHMARK = SCRIPTS / "run_benchmark.py"
SUMMARIZE = SCRIPTS / "summarize_results.py"
CHARTS = SCRIPTS / "plot_results.py"

CORPORA = {
    "plain": {"attack_dir": "attack_logs",
              "injected": "attack_logs_injected",
              "results": "results",
              "analysis": "analysis"},
    "hexa": {"attack_dir": "attack_logs_hexa",
             "injected": "attack_logs_injected_hexa",
             "results": "results_hexa",
             "analysis": "analysis_hexa"},
}

LANE_SETS = {
    "plain": ("plain",),
    "hexa": ("hexa",),
    "both": ("plain", "hexa"),
}


def run_step(name: str, script: Path, extra_args: list[str]) -> int:
    cmd = [sys.executable, str(script), *extra_args]
    print(f"\n=== {name} ===")
    print("  " + " ".join(cmd) + "\n")
    return subprocess.run(cmd, cwd=ROOT).returncode


def forward_flags(extra_args: list[str], keep: set[str]) -> list[str]:
    out, taking = [], False
    for token in extra_args:
        if token.startswith("--"):
            taking = token.split("=", 1)[0] in keep
        if taking:
            out.append(token)
    return out


def run_lane(lane: str, args, benchmark_args: list[str], labelled: bool) -> int:
    paths = CORPORA[lane]
    tag = f"[{lane}] " if labelled else ""

    if not args.skip_inject:
        rc = run_step(f"{tag}STEP 1  inject prompts", INJECT,
                      ["--attack-dir", paths["attack_dir"], "--out-dir", paths["injected"]])
        if rc != 0:
            print(f"\ninject step failed (exit {rc}); not running the benchmark.")
            return rc
    else:
        print(f"\n=== {tag}STEP 1  inject prompts -> skipped ===")

    if not args.skip_benchmark and not args.skip_check:
        rc = run_step(f"{tag}STEP 1b context preflight", CHECK,
                      ["--input-root", paths["injected"],
                       "--out", f"{paths['analysis']}/context_check.csv"]
                      + forward_flags(benchmark_args, {"--models", "--logs", "--num-ctx"}))
        if rc != 0:
            print(f"\ncontext preflight failed (exit {rc}): at least one model would be "
                  f"fed a truncated log, which makes its verdict meaningless. Fix the "
                  f"window or the model list, or re-run with --skip-check to override.")
            return rc
    else:
        print(f"\n=== {tag}STEP 1b context preflight -> skipped ===")

    for temp in args.temperatures:
        rc = run_temperature(lane, temp, args, benchmark_args, tag)
        if rc != 0:
            return rc
    return 0


def run_temperature(lane: str, temp: str, args, benchmark_args: list[str], tag: str) -> int:
    paths = CORPORA[lane]
    tag = f"{tag}[temp_{temp}] "
    temp_args = ["--temperature", temp]

    if not args.skip_benchmark:
        rc = run_step(f"{tag}STEP 2  run benchmark", BENCHMARK,
                      ["--input-root", paths["injected"],
                       "--results-root", paths["results"]] + temp_args + benchmark_args)
        if rc != 0:
            print(f"\nbenchmark step failed (exit {rc}).")
            return rc
    else:
        print(f"\n=== {tag}STEP 2  run benchmark -> skipped ===")

    if not args.skip_summary:
        rc = run_step(f"{tag}STEP 3  summarize results", SUMMARIZE,
                      ["--results-root", paths["results"],
                       "--out-dir", paths["analysis"], "--lane", lane] + temp_args)
        if rc != 0:
            print(f"\nsummary step failed (exit {rc}).")
            return rc
    else:
        print(f"\n=== {tag}STEP 3  summarize results -> skipped ===")

    if not args.skip_charts:
        rc = run_step(f"{tag}STEP 4  plot charts", CHARTS,
                      ["--analysis-dir", str(experiment.temp_dir(paths["analysis"], temp))])
        if rc != 0:
            print(f"\nchart step failed (exit {rc}).")
            return rc
    else:
        print(f"\n=== {tag}STEP 4  plot charts -> skipped ===")

    return 0


def run_compare(args) -> int:
    if args.skip_compare:
        print("\n=== STEP 5  cross-lane comparison -> skipped ===")
        return 0

    for temp in args.temperatures:
        def analysis(lane):
            return experiment.temp_dir(CORPORA[lane]["analysis"], temp)

        have = [lane for lane in LANE_SETS["both"]
                if (ROOT / analysis(lane) / "verdicts.csv").is_file()]
        if len(have) < 2:
            print(f"\n=== [temp_{temp}] STEP 5  cross-lane comparison -> skipped ===")
            missing = [lane for lane in ("plain", "hexa") if lane not in have]
            print(f"  need two lanes with a verdicts.csv, have {len(have)} -- run "
                  f"{' and '.join(missing)} first "
                  f"(python main.py --corpus {missing[0]})")
            continue

        baseline, others = have[0], have[1:]
        rc = run_step(f"[temp_{temp}] STEP 5  cross-lane comparison ({' vs '.join(have)})",
                      CHARTS,
                      ["--analysis-dir", str(analysis(baseline)),
                       "--compare", *[str(analysis(lane)) for lane in others],
                       "--lane-labels", *have])
        if rc != 0:
            return rc
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the injection pipeline: inject prompts, then benchmark.",
        epilog="Unknown args are forwarded to scripts/run_benchmark.py.",
    )
    ap.add_argument("--corpus", choices=tuple(LANE_SETS), default="plain",
                    help="which lane(s) to run: plain (attack_logs, the default); hexa "
                         "(attack_logs_hexa, auditd fields hex-encoded); both (plain+hexa)")
    ap.add_argument("--temperatures", nargs="+", choices=experiment.TEMPERATURE_LEVELS,
                    default=list(experiment.TEMPERATURE_LEVELS),
                    help="which temperature levels to run, each into its own temp_<t>/ "
                         "folder: low, medium = 0.0 (greedy) / each model's recommended "
                         "(default: both)")
    ap.add_argument("--skip-inject", action="store_true",
                    help="skip step 1 (reuse the existing injected corpus)")
    ap.add_argument("--skip-check", action="store_true",
                    help="skip the context preflight (step 1b)")
    ap.add_argument("--skip-benchmark", action="store_true",
                    help="skip step 2 (only rebuild the injected corpus)")
    ap.add_argument("--skip-summary", action="store_true",
                    help="skip step 3 (do not (re)build the analysis dir)")
    ap.add_argument("--skip-charts", action="store_true",
                    help="skip step 4 (do not (re)draw the per-lane charts)")
    ap.add_argument("--skip-compare", action="store_true",
                    help="skip step 5 (the cross-lane comparison charts)")
    args, benchmark_args = ap.parse_known_args()

    lanes = LANE_SETS[args.corpus]
    for lane in lanes:
        rc = run_lane(lane, args, benchmark_args, labelled=len(lanes) > 1)
        if rc != 0:
            return rc

    rc = run_compare(args)
    if rc != 0:
        print(f"\ncomparison step failed (exit {rc}).")
        return rc

    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
