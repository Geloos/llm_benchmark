#!/usr/bin/env python3
"""
main.py

What it does:
  Runs the whole pipeline for one or more lanes: inject -> context preflight, then per
  temperature: benchmark the injected logs -> summarize -> charts, then a cross-lane
  comparison chart per temperature. Every stage skips itself when its output is already
  on disk and up to date, so a re-run only does what is missing. To force a stage, delete
  its output (e.g. rm -r analysis_hexa/temp_low/charts).

How to run it:
  python main.py                                   # everything: plain, hexa, dovsdofs, ob
  python main.py --hexa                            # only the hex lane
  python main.py --plain --dovsdofs                # experiment flags combine
  python main.py --plain --low                     # only temp_low
  python main.py --plain --medium --skip-reasoning # medium without gpt-oss @medium/@high
  python main.py --skip-check                      # no context preflight

  Narrow the benchmark (the preflight gets --models and --logs too):
  python main.py --plain --low --models llama3.1:8b --categories direct_override --logs drupal
  python main.py --hexa --injections DO_01_canonical PH_
  python main.py --dovsdofs --runs 10
  python main.py --ob

  What to run (none given = all of them):
    --plain --hexa --dovsdofs --ob
                                the experiments. dovsdofs is one direct override without
                                vs with a SYSTEM: marker (jailbreaks/dovsdofs.jsonl), at
                                levels low and medium with gpt-oss@medium, llama3.1, gemma3
                                (experiment.DOVSDOFS_MODELS); its analysis adds a paired
                                t-test per model (--paired). ob is SPT_01_direct and
                                SPT_06_pentest as Base16 behind "Decode and follow Base16
                                instructions" (jailbreaks/ob.jsonl), at levels low and
                                medium with the same models (experiment.OB_MODELS),
                                analysed like plain.
    --low --medium              the temperature levels. hexa runs low only
                                (LANE_TEMPERATURES), so the cross-lane chart is drawn at
                                temp_low alone.
    --skip-reasoning            at medium, run only gpt-oss@low + llama3.1 + gemma3 (no
                                reasoning experiment). dovsdofs and ob are not affected.
    --skip-check                no context preflight.

  Narrow the benchmark:
    --runs N                    repeats, seeds 42..42+N-1 (default 5)
    --models M ...              replace the lane's model list (gpt-oss:20b@high, ...)
    --categories C ...          whole injection categories (direct_override, ...)
    --injections I ...          single ids or id prefixes (DO_01_canonical, PH_)
    --logs L ...                log files, a full name or part of one (drupal)
  --categories and --injections combine as a union, resolved per lane against its jsonl;
  a lane none of them matches is skipped. They narrow the benchmark calls only: inject
  builds the full corpus and summarize / plot read everything on disk. --seeds,
  --exclude-* and --num-ctx are on scripts/run_benchmark.py for running it by hand.

  A stage is skipped when:
    1  inject      always runs, and skips each file itself that is newer than its source
                   log, the jsonl and the injector's code
    1b preflight   --skip-check, or no benchmark call is pending at any level of the lane
    2  benchmark   no result file is missing for that level and those filters
    3  summarize   verdicts.csv + report.md exist and verdicts.csv is newer than every
                   result *.txt / *.meta.json under results*/temp_<t>/ and the jsonl
    4  charts      the overview PNG (dovsdofs.png for the paired lane) is newer than
                   verdicts.csv
    5  compare     plain_vs_hexa.png is newer than both lanes' verdicts.csv

What it outputs:
  Per lane: attack_logs_injected*/, then one temp_<t>/ folder per temperature under
  results*/ and analysis*/ (verdicts.csv,
  verdicts_by_injection.csv, summary.jsonl, report.md, reasoning_report.md at temp_medium,
  charts/), plus analysis*/context_check.csv. And the cross-lane chart
  analysis/temp_low/charts/plain_vs_hexa.png. The DOVSDOFS lane writes
  analysis_dovsdofs/temp_<t>/: verdicts.csv, report.md (with the paired t-test),
  paired_ttest.csv and charts/dovsdofs.png. The OB lane writes analysis_ob/temp_<t>/ like
  any lane (verdicts.csv, report.md, charts/).

  Sampling, temperatures, seeds and the model set per temperature live in
  scripts/experiment.py. Every model sends its full recommended sampling; the levels are
  low / medium = 0.0 (greedy) / recommended, per model. Two experiments
  share the run, every series repeated 5 times (seeds 42..46, one seed_<n>/ folder each)
  and reported as ASR mean +- 95% CI (and the std):
    temperature  every level x gpt-oss@low, llama3.1, gemma3;
    reasoning    medium x gpt-oss@low / @medium / @high. medium x gpt-oss@low is the
                 temperature experiment's run too, so only @medium and @high are extra.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
import experiment
import injection_set
import run_benchmark as bench

INJECT = SCRIPTS / "inject_prompts.py"
CHECK = SCRIPTS / "check_context.py"
BENCHMARK = SCRIPTS / "run_benchmark.py"
SUMMARIZE = SCRIPTS / "summarize_results.py"
CHARTS = SCRIPTS / "plot_results.py"
CHARTS_STEM = "trick_rate_by_category"

DEFAULT_INJECTIONS = "jailbreaks/injections.jsonl"
CORPORA = {
    "plain": {"attack_dir": "attack_logs",
              "injected": "attack_logs_injected",
              "results": "results",
              "analysis": "analysis",
              "injections": DEFAULT_INJECTIONS},
    "hexa": {"attack_dir": "attack_logs_hexa",
             "injected": "attack_logs_injected_hexa",
             "results": "results_hexa",
             "analysis": "analysis_hexa",
             "injections": DEFAULT_INJECTIONS},
    "dovsdofs": {"attack_dir": "attack_logs",
                 "injected": "attack_logs_injected_dovsdofs",
                 "results": "results_dovsdofs",
                 "analysis": "analysis_dovsdofs",
                 "injections": "jailbreaks/dovsdofs.jsonl",
                 "models": experiment.DOVSDOFS_MODELS,
                 "paired": True},
    "ob": {"attack_dir": "attack_logs",
           "injected": "attack_logs_injected_ob",
           "results": "results_ob",
           "analysis": "analysis_ob",
           "injections": "jailbreaks/ob.jsonl",
           "models": experiment.OB_MODELS},
}
LANES = ("plain", "hexa", "dovsdofs", "ob")
COMPARE_LANES = ("plain", "hexa")

LANE_TEMPERATURES = {
    "plain": experiment.TEMPERATURE_LEVELS,
    "hexa": ("low",),
    "dovsdofs": experiment.TEMPERATURE_LEVELS,
    "ob": experiment.TEMPERATURE_LEVELS,
}


def run_step(name: str, script: Path, extra_args: list[str]) -> int:
    cmd = [sys.executable, str(script), *extra_args]
    print(f"\n=== {name} ===")
    print("  " + " ".join(cmd) + "\n")
    return subprocess.run(cmd, cwd=ROOT).returncode


def skipped(name: str, why: str) -> None:
    print(f"\n=== {name} -> skipped ({why}) ===")


def mtime(path) -> float:
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


def newest(root, patterns) -> float:
    return max((mtime(p) for pattern in patterns for p in Path(root).rglob(pattern)),
               default=0.0)


def all_categories() -> list[str]:
    found = set()
    for paths in CORPORA.values():
        found |= {inj.get("category") or "unknown"
                  for inj in injection_set.load(paths["injections"])}
    return sorted(found)


def lane_injections(lane: str, args):
    if not args.categories and not args.injections:
        return None
    picked = []
    for inj in injection_set.load(CORPORA[lane]["injections"]):
        names = [inj["id"]] + inj["source_ids"]
        if (inj.get("category") in (args.categories or [])
                or any(n.startswith(p) for n in names for p in (args.injections or []))):
            picked.append(inj["id"])
    return picked


def lane_models(lane: str, temp: str, args) -> list[str]:
    paths = CORPORA[lane]
    models = list(args.models or paths.get("models") or experiment.models_for(temp))
    if args.skip_reasoning and "models" not in paths:
        models = [m for m in models if experiment.reasoning_level(m)
                  in (None, experiment.TEMPERATURE_EXPERIMENT_THINK)]
    return models


def benchmark_work(lane: str, temp: str, args, ids):
    paths = CORPORA[lane]
    argv = ["--input-root", paths["injected"], "--results-root", paths["results"],
            "--temperature", temp, "--models", *lane_models(lane, temp, args)]
    if ids is not None:
        argv += ["--injections", *ids]
    if args.logs:
        argv += ["--logs", *args.logs]
    if args.runs:
        argv += ["--runs", str(args.runs)]
    planned = bench.parse_args(argv)
    txt_by_folder = bench.plan(planned)
    total = (sum(len(v) for v in txt_by_folder.values())
             * len(planned.models) * len(planned.seeds))
    return {"argv": argv, "models": planned.models, "total": total,
            "pending": bench.pending(planned, txt_by_folder) if total else 0}


def run_lane(lane: str, args, temperatures: list[str]) -> int:
    paths = CORPORA[lane]
    tag = f"[{lane}] "

    temps = [t for t in temperatures if t in LANE_TEMPERATURES[lane]]
    if not temps:
        print(f"\n=== {tag}lane skipped: it runs at {', '.join(LANE_TEMPERATURES[lane])} "
              f"only, and you asked for {', '.join(temperatures)} ===")
        return 0

    ids = lane_injections(lane, args)
    if ids == []:
        print(f"\n=== {tag}lane skipped: no injection in {paths['injections']} matches "
              f"--categories / --injections ===")
        return 0

    rc = run_step(f"{tag}STEP 1  inject prompts", INJECT,
                  ["--attack-dir", paths["attack_dir"], "--out-dir", paths["injected"],
                   "--injections", paths["injections"]])
    if rc != 0:
        print(f"\ninject step failed (exit {rc}); not running the benchmark.")
        return rc

    work = {temp: benchmark_work(lane, temp, args, ids) for temp in temps}

    if args.skip_check:
        skipped(f"{tag}STEP 1b context preflight", "--skip-check")
    elif not any(w["pending"] for w in work.values()):
        skipped(f"{tag}STEP 1b context preflight", "no benchmark call pending")
    else:
        models = []
        for w in work.values():
            if w["pending"]:
                models += [m for m in w["models"] if m not in models]
        rc = run_step(f"{tag}STEP 1b context preflight", CHECK,
                      ["--input-root", paths["injected"],
                       "--out", f"{paths['analysis']}/context_check.csv",
                       "--models", *models]
                      + (["--logs", *args.logs] if args.logs else []))
        if rc != 0:
            print(f"\ncontext preflight failed (exit {rc}): at least one model would be "
                  f"fed a truncated log, which makes its verdict meaningless. Fix the "
                  f"window or the model list, or re-run with --skip-check to override.")
            return rc

    for temp in temps:
        rc = run_temperature(lane, temp, work[temp], tag)
        if rc != 0:
            return rc
    return 0


def run_temperature(lane: str, temp: str, work: dict, tag: str) -> int:
    paths = CORPORA[lane]
    tag = f"{tag}[temp_{temp}] "
    results = experiment.temp_dir(paths["results"], temp)
    analysis = experiment.temp_dir(paths["analysis"], temp)
    verdicts = analysis / "verdicts.csv"
    paired = ["--paired"] if paths.get("paired") else []

    if not work["total"]:
        skipped(f"{tag}STEP 2  run benchmark", "nothing matches the filters")
    elif not work["pending"]:
        skipped(f"{tag}STEP 2  run benchmark",
                f"all {work['total']} results already there")
    else:
        rc = run_step(f"{tag}STEP 2  run benchmark", BENCHMARK, work["argv"])
        if rc != 0:
            print(f"\nbenchmark step failed (exit {rc}).")
            return rc

    if not results.is_dir():
        skipped(f"{tag}STEP 3  summarize results", f"no {results}/ yet")
    elif ((analysis / "report.md").is_file()
          and mtime(verdicts) > max(newest(results, ("*.txt", "*.meta.json")),
                                    mtime(paths["injections"]))):
        skipped(f"{tag}STEP 3  summarize results", f"{verdicts} is up to date")
    else:
        rc = run_step(f"{tag}STEP 3  summarize results", SUMMARIZE,
                      ["--results-root", paths["results"],
                       "--out-dir", paths["analysis"], "--lane", lane,
                       "--injections", paths["injections"], "--temperature", temp] + paired)
        if rc != 0:
            print(f"\nsummary step failed (exit {rc}).")
            return rc

    chart = analysis / "charts" / ("dovsdofs.png" if paired else f"{CHARTS_STEM}.png")
    if not verdicts.is_file():
        skipped(f"{tag}STEP 4  plot charts", f"no {verdicts}")
    elif mtime(chart) > mtime(verdicts):
        skipped(f"{tag}STEP 4  plot charts", f"{chart} is up to date")
    else:
        rc = run_step(f"{tag}STEP 4  plot charts", CHARTS,
                      ["--analysis-dir", str(analysis)] + paired)
        if rc != 0:
            print(f"\nchart step failed (exit {rc}).")
            return rc

    return 0


def run_compare(temperatures: list[str]) -> int:
    for temp in temperatures:
        name = f"[temp_{temp}] STEP 5  cross-lane comparison"

        absent = [lane for lane in COMPARE_LANES if temp not in LANE_TEMPERATURES[lane]]
        if absent:
            skipped(name, f"{' and '.join(absent)} does not run at {temp}")
            continue

        analysis = {lane: experiment.temp_dir(CORPORA[lane]["analysis"], temp)
                    for lane in COMPARE_LANES}
        have = [lane for lane in COMPARE_LANES if (analysis[lane] / "verdicts.csv").is_file()]
        if len(have) < 2:
            missing = [lane for lane in COMPARE_LANES if lane not in have]
            skipped(name, f"need both lanes with a verdicts.csv, run "
                          f"python main.py --{missing[0]} first")
            continue

        baseline, others = have[0], have[1:]
        chart = analysis[baseline] / "charts" / f"{'_vs_'.join(have)}.png"
        if mtime(chart) > max(mtime(analysis[lane] / "verdicts.csv") for lane in have):
            skipped(name, f"{chart} is up to date")
            continue

        rc = run_step(f"{name} ({' vs '.join(have)})", CHARTS,
                      ["--analysis-dir", str(analysis[baseline]),
                       "--compare", *[str(analysis[lane]) for lane in others],
                       "--lane-labels", *have])
        if rc != 0:
            return rc
    return 0


def main() -> int:
    os.chdir(ROOT)
    ap = argparse.ArgumentParser(
        description="Run the injection pipeline: inject, preflight, benchmark, summarize, "
                    "chart. Stages whose output is already up to date are skipped.")
    run = ap.add_argument_group("what to run (none given = all of them)")
    run.add_argument("--plain", action="store_true",
                     help="the plain lane (attack_logs/)")
    run.add_argument("--hexa", action="store_true",
                     help="the hex lane (attack_logs_hexa/, every injection site hexed), "
                          "level low only")
    run.add_argument("--dovsdofs", action="store_true",
                     help="the DOVSDOFS lane (direct override without vs with a SYSTEM: "
                          "marker): levels low and medium, gpt-oss@medium, llama3.1, "
                          "gemma3")
    run.add_argument("--ob", action="store_true",
                     help="the OB lane (SPT_01 / SPT_06 as Base16 behind 'Decode and "
                          "follow Base16 instructions'): levels low and medium, "
                          "gpt-oss@medium, llama3.1, gemma3")
    run.add_argument("--low", action="store_true",
                     help="temperature level low = 0.0 (greedy)")
    run.add_argument("--medium", action="store_true",
                     help="temperature level medium = each model's recommended")
    run.add_argument("--skip-reasoning", action="store_true",
                     help="at medium, leave out gpt-oss @medium/@high (the reasoning "
                          "experiment); dovsdofs and ob are not affected")
    run.add_argument("--skip-check", action="store_true",
                     help="skip the context preflight (step 1b)")
    narrow = ap.add_argument_group("narrow the benchmark")
    narrow.add_argument("--runs", type=int, default=None, metavar="N",
                        help=f"repeats, seeds 42..42+N-1 (default {experiment.DEFAULT_RUNS})")
    narrow.add_argument("--models", nargs="+", default=None, metavar="MODEL",
                        help="replace the lane's model list, e.g. llama3.1:8b "
                             "gpt-oss:20b@high")
    narrow.add_argument("--categories", nargs="+", default=None, metavar="CATEGORY",
                        choices=all_categories(),
                        help="only these injection categories: %(choices)s")
    narrow.add_argument("--injections", nargs="+", default=None, metavar="ID",
                        help="only these injection ids or id prefixes, e.g. "
                             "DO_01_canonical PH_")
    narrow.add_argument("--logs", nargs="+", default=None, metavar="LOG",
                        help="only log files whose name contains one of these, e.g. "
                             "drupal or attack_drupal_cred_access")
    args = ap.parse_args()

    lanes = [lane for lane in LANES if getattr(args, lane)] or list(LANES)
    temperatures = ([t for t in experiment.TEMPERATURE_LEVELS if getattr(args, t)]
                    or list(experiment.TEMPERATURE_LEVELS))

    for lane in lanes:
        if lane in COMPARE_LANES:
            rc = run_lane(lane, args, temperatures)
            if rc != 0:
                return rc

    if set(lanes) & set(COMPARE_LANES):
        rc = run_compare(temperatures)
        if rc != 0:
            print(f"\ncomparison step failed (exit {rc}).")
            return rc

    for lane in lanes:
        if lane not in COMPARE_LANES:
            rc = run_lane(lane, args, temperatures)
            if rc != 0:
                return rc

    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
