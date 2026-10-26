#!/usr/bin/env python3
"""
compare_temperatures.py

What it does:
  Puts the temperature experiment's two levels side by side: the same models, logs,
  injections and seeds at level low (0.0, greedy) and medium (each model's recommended
  temperature), read from the two calls.jsonl files summarize_results.py wrote. Mean ASR
  only, no tests -- read the gaps by eye. Per model:

  1  the overall ASR at both levels (mean +- 95% CI across the seeds) and the change;
  2  the injections whose ASR moved the most;
  3  every category's ASR at both levels and the change.

  Only models present at both levels are compared (the temperature series: gpt-oss@low,
  llama3.1, gemma3). A model whose seeds differ between the levels is compared on the
  shared seeds, with a warning.

How to run it:
  python3 compare_temperatures.py --analysis-root analysis/analysis_plaintext \
      --out-dir analysis/analysis_temperature

  Reads <analysis-root>/temp_low/calls.jsonl and <analysis-root>/temp_medium/calls.jsonl.
  main.py runs it for the plain lane once both exist (step 5), then
  plot_results.py --compare-temperatures for the chart.

What it outputs (under --out-dir):
  temperature.csv         one row per model:
                          model,seeds,low_asr,low_low,low_high,medium_asr,medium_low,
                          medium_high,delta
                          low_/medium_low/high = each level's 95% t-interval over the
                          seeds (as in asr.csv); delta = medium_asr - low_asr.
  injection_shift.csv     one row per model x injection:
                          model,injection,category,low_asr,medium_asr,delta
  report.md               the three as tables.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import experiment
from summarize_results import fmt_ci, rounded, write_csv

LOW, MEDIUM = experiment.TEMPERATURE_LEVELS
TOP_MOVERS = 5
MODEL_FIELDS = ["model", "seeds", "low_asr", "low_low", "low_high", "medium_asr",
                "medium_low", "medium_high", "delta"]
SHIFT_FIELDS = ["model", "injection", "category", "low_asr", "medium_asr", "delta"]


def read_calls(path: Path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def grouped(rows, key_fn) -> dict:
    out = defaultdict(list)
    for r in rows:
        out[key_fn(r)].append((r["log"], r["seed"], r["tricked"]))
    return out


def compare_model(model: str, low_rows, medium_rows):
    low_seeds = {r["seed"] for r in low_rows}
    medium_seeds = {r["seed"] for r in medium_rows}
    seeds = low_seeds & medium_seeds
    if low_seeds != medium_seeds:
        print(f"WARNING: {model}: {LOW} has seeds {sorted(low_seeds)}, {MEDIUM} "
              f"{sorted(medium_seeds)} -- comparing the shared {sorted(seeds)} only")
    low_rows = [r for r in low_rows if r["seed"] in seeds]
    medium_rows = [r for r in medium_rows if r["seed"] in seeds]

    everything = lambda r: ""
    low_st = experiment.asr_stats(grouped(low_rows, everything)[""])
    medium_st = experiment.asr_stats(grouped(medium_rows, everything)[""])

    by_inj_low = grouped(low_rows, lambda r: r["injection"])
    by_inj_medium = grouped(medium_rows, lambda r: r["injection"])
    category_of = {r["injection"]: r["category"] for r in low_rows + medium_rows}
    shifts = []
    for inj in sorted(set(by_inj_low) & set(by_inj_medium)):
        a = experiment.asr_stats(by_inj_low[inj])["asr"]
        b = experiment.asr_stats(by_inj_medium[inj])["asr"]
        shifts.append({"model": model, "injection": inj, "category": category_of[inj],
                       "low_asr": round(a, 3), "medium_asr": round(b, 3),
                       "delta": round(b - a, 3)})

    by_cat_low = grouped(low_rows, lambda r: r["category"])
    by_cat_medium = grouped(medium_rows, lambda r: r["category"])
    categories = []
    for category in sorted(set(by_cat_low) & set(by_cat_medium)):
        a = experiment.asr_stats(by_cat_low[category])
        b = experiment.asr_stats(by_cat_medium[category])
        categories.append({"category": category, "_low": a, "_medium": b,
                           "delta": b["asr"] - a["asr"]})
    categories.sort(key=lambda c: (-abs(c["delta"]), c["category"]))

    return {
        "model": model, "seeds": len(seeds),
        "low_asr": round(low_st["asr"], 3), "low_low": rounded(low_st["low"]),
        "low_high": rounded(low_st["high"]),
        "medium_asr": round(medium_st["asr"], 3), "medium_low": rounded(medium_st["low"]),
        "medium_high": rounded(medium_st["high"]),
        "delta": round(medium_st["asr"] - low_st["asr"], 3),
        "_low": low_st, "_medium": medium_st, "_shifts": shifts, "_categories": categories,
    }


def movers(shifts) -> list:
    return sorted(shifts, key=lambda s: (-abs(s["delta"]), s["injection"]))[:TOP_MOVERS]


def report_lines(results) -> list:
    lines = [f"# Temperature: {LOW} (0.0, greedy) vs {MEDIUM} (recommended)", "",
             f"The same models, logs, injections and seeds at both levels, from "
             f"`temp_{LOW}/calls.jsonl` and `temp_{MEDIUM}/calls.jsonl`. Only the "
             f"temperature changes. ASR = mean +- 95% CI across the seeds, as in "
             f"`asr.csv`. The numbers are in `temperature.csv` and `injection_shift.csv`.",
             "",
             "## 1. Overall ASR", "",
             f"| model | {LOW} ASR +- 95% CI | {MEDIUM} ASR +- 95% CI | {MEDIUM} - {LOW} |",
             "|---|---:|---:|---:|"]
    for t in results:
        lines.append(f"| {t['model']} | {fmt_ci(t['_low'])} | {fmt_ci(t['_medium'])} | "
                     f"{t['delta']:+.2f} |")
    lines += ["", f"## 2. Biggest movers (top {TOP_MOVERS} by ASR change)", "",
              "Every injection is in `injection_shift.csv`.", ""]
    for t in results:
        lines += [f"**{t['model']}**", "",
                  f"| injection | category | {LOW} ASR | {MEDIUM} ASR | {MEDIUM} - {LOW} |",
                  "|---|---|---:|---:|---:|"]
        for s in movers(t["_shifts"]):
            lines.append(f"| {s['injection']} | {s['category']} | {s['low_asr']:.2f} | "
                         f"{s['medium_asr']:.2f} | {s['delta']:+.2f} |")
        lines.append("")
    lines += ["## 3. By category", "",
              "Each category's ASR at both levels, largest change first.", ""]
    for t in results:
        lines += [f"**{t['model']}**", "",
                  f"| category | {LOW} ASR | {MEDIUM} ASR | {MEDIUM} - {LOW} |",
                  "|---|---:|---:|---:|"]
        for c in t["_categories"]:
            lines.append(f"| {c['category']} | {fmt_ci(c['_low'])} | "
                         f"{fmt_ci(c['_medium'])} | {c['delta']:+.2f} |")
        lines.append("")
    return lines


def parse_args():
    ap = argparse.ArgumentParser(description="Temperature low vs medium per model, mean ASR.")
    ap.add_argument("--analysis-root", default="analysis/analysis_plaintext",
                    help="holds temp_low/calls.jsonl and temp_medium/calls.jsonl "
                         "(default: analysis/analysis_plaintext)")
    ap.add_argument("--out-dir", default="analysis/analysis_temperature",
                    help="where to write outputs (default: analysis/analysis_temperature)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    paths = {t: experiment.temp_dir(args.analysis_root, t) / "calls.jsonl"
             for t in (LOW, MEDIUM)}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        sys.exit(f"ERROR: {', '.join(missing)} not found -- summarize both levels first")
    rows = {t: read_calls(p) for t, p in paths.items()}

    models = {t: {r["model"] for r in rows[t]} for t in rows}
    for t, other in ((LOW, MEDIUM), (MEDIUM, LOW)):
        for m in sorted(models[t] - models[other]):
            print(f"note: {m} is at {t} only -- not compared")
    shared = sorted(models[LOW] & models[MEDIUM], key=experiment.model_sort_key)
    if not shared:
        sys.exit("ERROR: no model is present at both levels")

    results = [compare_model(m, [r for r in rows[LOW] if r["model"] == m],
                             [r for r in rows[MEDIUM] if r["model"] == m]) for m in shared]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv([{k: t[k] for k in MODEL_FIELDS} for t in results],
              out_dir / "temperature.csv", MODEL_FIELDS)
    write_csv([s for t in results for s in t["_shifts"]], out_dir / "injection_shift.csv",
              SHIFT_FIELDS)
    (out_dir / "report.md").write_text("\n".join(report_lines(results)), encoding="utf-8")

    print(f"{LOW} vs {MEDIUM}, mean ASR:")
    for t in results:
        print(f"  {t['model']:24} {LOW} {t['low_asr']:.2f}  {MEDIUM} {t['medium_asr']:.2f}  "
              f"change {t['delta']:+.2f}")
    print(f"  wrote {out_dir / 'temperature.csv'}, {out_dir / 'injection_shift.csv'}, "
          f"{out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
