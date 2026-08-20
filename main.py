#!/usr/bin/env python3
"""
main.py

One entry point for the whole pipeline. Runs, in order:
  1. scripts/inject_prompts.py     <attack_dir> + injections -> <injected>/
  1b. scripts/check_context.py     <injected>                -> <analysis>/context_check.csv
  2. scripts/run_benchmark.py      <injected>                -> <results>/
  3. scripts/summarize_results.py  <results>                 -> <analysis>/
  4. scripts/plot_results.py       <analysis>/verdicts.csv   -> <analysis>/charts/
  5. scripts/plot_results.py       every lane's verdicts.csv -> analysis/charts/<lanes>.png

Step 1b is a preflight: it proves each model really ingests a whole injected log instead
of a silently truncated one, and aborts the run if it does not.

Steps 1-4 run once per LANE. A lane is a corpus plus what gets hexed into its auditd
fields; the three differ only in what the classifier sees in those fields:

  plain    attack_logs/       marker typed as one underscored token, so auditd left it
                              quoted:  a3="Ignore_all_previous_instructions_..."
  hexa     attack_logs_hexa/  marker typed WITH SPACES, so auditd hex-encoded it and
                              dropped the quotes.  The hex carries the PAYLOAD -- prose,
                              spaces and all:  a3=<hex of "Ignore all previous ...">
  control  attack_logs_hexa/  same files, same hex encoding, but the hex carries the TAG --
                              byte-identical text to the plain lane:
                              a3=<hex of "Ignore_all_previous_instructions_...">

Why the control lane exists: plain and hexa differ in TWO things at once -- the encoding
AND the wording, because a quoted auditd field cannot hold a space, so the plain lane can
only ever carry the underscored tag. A gap between them is therefore not attributable to
hex alone. The control pins the wording and varies only the encoding:

  plain vs control  -> the effect of HEX ENCODING (same text either side)
  control vs hexa   -> the effect of WORDING     (same encoding either side)

The control is synthetic -- auditd would never hex a space-free token -- so it is an
attribution aid, not a scenario a real attacker produces. Free-text surfaces (Apache UA,
sshd username, mysql) carry identical text in all three lanes; only auditd fields differ.

Step 5 charts every lane that has results against the others, and needs at least two.

How to run it:
  python main.py                 # plain lane: inject, check, benchmark, summarize, chart
  python main.py --corpus hexa   # the same four steps over the hex-encoded corpus
  python main.py --corpus both   # plain + hexa back to back, then the comparison chart
  python main.py --corpus all    # plain + control + hexa, then the three-lane chart
  python main.py --corpus control  # the encoding control on its own
  python main.py --skip-inject   # skip step 1, only check + benchmark + summary + charts
  python main.py --skip-check    # trust num_ctx, go straight to the benchmark
  python main.py --skip-benchmark  # only (re)build the injected corpus, then summarize
  python main.py --skip-summary  # inject + benchmark, no analysis
  python main.py --skip-charts   # skip step 4 (e.g. matplotlib is not installed)
  python main.py --skip-compare  # skip step 5
  python main.py --skip-inject --skip-benchmark  # only re-run the analysis (steps 3 + 4)

  Any extra args after the known flags are forwarded to run_benchmark.py, e.g.
      python main.py --models llama3.1:8b --injections DO_ PH_
      python main.py --corpus hexa --logs drupal --exclude-injections SPT_ SPLIT_

Notes:
  - Paths are resolved relative to this file, so it works from any cwd.
  - The lane's roots are prepended to each step's args, so a root passed by hand on the
    command line still wins (argparse takes the last occurrence).
  - If step 1 fails, the later steps do not run. The summary (step 3) and the charts
    (step 4) run independently of the inject/benchmark skips, so you can re-summarize
    existing results on their own.
  - Step 4 runs last within a lane and needs matplotlib, so a missing dependency can never
    cost a benchmark run -- everything else is already on disk by the time it is reached.
  - Step 5 is skipped with a note (not an error) unless at least two lanes have a
    verdicts.csv, so a single-lane run stays quiet until there is something to compare. It
    picks up whichever lanes are on disk, so running --corpus control later folds it into
    the chart without re-running anything else.
"""

from __future__ import annotations

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
CHARTS = SCRIPTS / "plot_results.py"

# One entry per corpus: the four roots that step 1-4 read and write. Everything else about
# the two lanes is identical -- same injections.jsonl, same models, same scoring.
CORPORA = {
    "plain": {"attack_dir": "attack_logs",
              "injected": "attack_logs_injected",
              "results": "results",
              "analysis": "analysis",
              "hex_variant": "payload"},
    "hexa": {"attack_dir": "attack_logs_hexa",
             "injected": "attack_logs_injected_hexa",
             "results": "results_hexa",
             "analysis": "analysis_hexa",
             "hex_variant": "payload"},
    # Same source files as hexa, same hex encoding -- but the hex carries the underscored
    # `tag`, i.e. byte-identical text to what the plain lane's quoted fields carry. auditd
    # would never hex a space-free token, so this lane is synthetic: it exists only so the
    # encoding effect can be told apart from the wording effect.
    "control": {"attack_dir": "attack_logs_hexa",
                "injected": "attack_logs_injected_control",
                "results": "results_control",
                "analysis": "analysis_control",
                "hex_variant": "tag"},
}

# Which lanes each --corpus choice runs, in the order they should be charted.
LANE_SETS = {
    "plain": ("plain",),
    "hexa": ("hexa",),
    "control": ("control",),
    "both": ("plain", "hexa"),
    "all": ("plain", "control", "hexa"),
}


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


def run_lane(lane: str, args, benchmark_args: list[str], labelled: bool) -> int:
    """Steps 1-4 for one corpus. `labelled` tags each step header with the lane name, which
    only earns its keep when more than one lane is running."""
    paths = CORPORA[lane]
    tag = f"[{lane}] " if labelled else ""

    if not args.skip_inject:
        rc = run_step(f"{tag}STEP 1  inject prompts", INJECT,
                      ["--attack-dir", paths["attack_dir"], "--out-dir", paths["injected"],
                       "--hex-variant", paths["hex_variant"]])
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

    if not args.skip_benchmark:
        rc = run_step(f"{tag}STEP 2  run benchmark", BENCHMARK,
                      ["--input-root", paths["injected"],
                       "--results-root", paths["results"]] + benchmark_args)
        if rc != 0:
            print(f"\nbenchmark step failed (exit {rc}).")
            return rc
    else:
        print(f"\n=== {tag}STEP 2  run benchmark -> skipped ===")

    if not args.skip_summary:
        rc = run_step(f"{tag}STEP 3  summarize results", SUMMARIZE,
                      ["--results-root", paths["results"], "--out-dir", paths["analysis"]])
        if rc != 0:
            print(f"\nsummary step failed (exit {rc}).")
            return rc
    else:
        print(f"\n=== {tag}STEP 3  summarize results -> skipped ===")

    if not args.skip_charts:
        rc = run_step(f"{tag}STEP 4  plot charts", CHARTS,
                      ["--analysis-dir", paths["analysis"]])
        if rc != 0:
            print(f"\nchart step failed (exit {rc}).")
            return rc
    else:
        print(f"\n=== {tag}STEP 4  plot charts -> skipped ===")

    return 0


def run_compare(args) -> int:
    """Step 5: one chart reading BOTH lanes' verdicts.csv, so the effect of the hex
    encoding is readable in a single picture instead of by diffing two report.md files.
    Skipped with a note, not an error, when only one lane has been run."""
    if args.skip_compare:
        print("\n=== STEP 5  plain vs hexa comparison -> skipped ===")
        return 0

    # Chart every lane that has results, in LANE_SETS["all"] order: plain, control, hexa.
    # The control lane is optional -- without it this is the plain-vs-hexa pair, and adding
    # it later folds it into the chart without re-running anything else.
    have = [lane for lane in LANE_SETS["all"]
            if (ROOT / CORPORA[lane]["analysis"] / "verdicts.csv").is_file()]
    if len(have) < 2:
        print("\n=== STEP 5  cross-lane comparison -> skipped ===")
        missing = [lane for lane in ("plain", "hexa") if lane not in have]
        print(f"  need two lanes with a verdicts.csv, have {len(have)} -- run "
              f"{' and '.join(missing)} first "
              f"(python main.py --corpus {missing[0]})")
        return 0

    baseline, others = have[0], have[1:]
    return run_step(f"STEP 5  cross-lane comparison ({' vs '.join(have)})", CHARTS,
                    ["--analysis-dir", CORPORA[baseline]["analysis"],
                     "--compare", *[CORPORA[lane]["analysis"] for lane in others],
                     "--lane-labels", *have])


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the injection pipeline: inject prompts, then benchmark.",
        epilog="Unknown args are forwarded to scripts/run_benchmark.py.",
    )
    ap.add_argument("--corpus", choices=tuple(LANE_SETS), default="plain",
                    help="which lane(s) to run: plain (attack_logs, the default); hexa "
                         "(attack_logs_hexa, auditd fields hex-encoded); control (the same "
                         "hex corpus carrying the plain lane's text, to isolate the "
                         "encoding); both (plain+hexa); all (plain+control+hexa)")
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
                    help="skip step 5 (the plain-vs-hexa comparison chart)")
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
