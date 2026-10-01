#!/usr/bin/env python3
"""
stats_analysis.py

What it does:
  The statistics on top of the benchmark. It reads every lane's verdicts.csv (written by
  summarize_results.py, one per analysis*/temp_<t>/) and stacks them into one table. It
  never calls ollama and never re-parses a reply: success is the existing `tricked`
  column (verdict bucket normal or neutral), unchanged.

  A cell is (model, lane, temperature, injection, log), and its runs are the seeds
  (run_idx 1..N). From those cells it computes:
    a  flip rate          share of cells whose runs do not all agree on success
    b  mean +- std        ASR per run_idx, then mean and sample std (ddof=1) over the runs
    c  collapsed cells    majority vote (mean success > 0.5; a tie, only possible with an
                          even run count, is flagged and counts as 0) and mean success
    d  per-injection CI   k majority successes out of n logs -> 95% Wilson and
                          Clopper-Pearson intervals
    e  bootstrap CI       cluster bootstrap over logs: resample the logs with replacement,
                          keep every injection of each, ASR = mean of per-cell mean
                          success; 95% percentile interval. Every group draws from a fresh
                          generator seeded with --boot-seed, so a group's interval does not
                          depend on which other groups are present.
    g  exact McNemar      on majority-vote cells: plain vs control vs hexa (same model,
                          temperature, injection, log), each generic injection vs its SPT_
                          twin (injection_set.LABEL_TWINS, same log), and temperature 0 vs
                          medium (same model, lane, injection, log). Holm-adjusted p within
                          each file (per scope).
  There is no clean-log baseline (f): the clean step was removed from the pipeline.

  The McNemar tests treat their pairs as independent, but pairs from the same log are
  correlated -- read them alongside the bootstrap interval, which is the
  cluster-robust number.

How to run it:
  python scripts/stats_analysis.py                    # every lane and temperature on disk
  python scripts/stats_analysis.py --lane-dirs plain=analysis hexa=analysis_hexa
  python scripts/stats_analysis.py --drop-truncated --top 15 --boot-iters 20000
  python scripts/stats_analysis.py --temp-pairs 0:medium 0:low

What it outputs (under --out-dir, default analysis_stats/):
  cells.csv                         c: one row per cell
  flip_rate.csv                     a: per model x lane x temperature
  asr_by_run.csv                    b: ASR of each run
  asr_mean_std.csv                  b: mean +- std over runs, per model x lane x temperature
  asr_mean_std_by_injection.csv     b: the same per injection
  injection_ci.csv                  d: majority-vote ASR per injection, Wilson + CP intervals
  asr_bootstrap.csv                 e: aggregate ASR with the cluster-bootstrap interval
  mcnemar_lanes.csv                 g: encoding pairs
  mcnemar_spt_twins.csv             g: generic vs label-matched SPT_ twin
  mcnemar_temperature.csv           g: temperature pairs
  tables/asr_aggregate.tex          booktabs: mean +- std and bootstrap CI
  tables/flip_rate.tex              booktabs: flip rate
  tables/top_injections.tex         booktabs + longtable: top --top injections per group,
                                    with the Wilson interval
  and a printed summary of the flip rates, flagging any temperature whose flip rate tops
  --flip-threshold (5%) as one to re-run with 10 runs.

Needs pandas, numpy, scipy and statsmodels -- here only; every other stage is stdlib.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    from statsmodels.stats.contingency_tables import mcnemar
    from statsmodels.stats.multitest import multipletests
    from statsmodels.stats.proportion import proportion_confint
except ImportError as e:
    sys.exit(f"ERROR: {e.name} is not installed -- stats_analysis.py needs pandas, numpy, "
             f"scipy and statsmodels:  pip install -r requirements.txt")

import experiment
import injection_set

LANES = ("plain", "control", "hexa")
DEFAULT_LANE_DIRS = ("plain=analysis", "control=analysis_control", "hexa=analysis_hexa")
LANE_PAIRS = (("plain", "hexa"), ("plain", "control"), ("control", "hexa"))

GROUP = ["model", "lane", "temperature"]
CELL = GROUP + ["injection", "log"]
TEMP_LABEL = {value: label for label, value in experiment.TEMPERATURES.items()}


# ---------------------------------------------------------------- loading

def parse_lane_dirs(items) -> list:
    out = []
    for item in items:
        lane, sep, root = item.partition("=")
        if not sep or not lane or not root:
            sys.exit(f"ERROR: --lane-dirs takes lane=dir, got {item!r}")
        out.append((lane, Path(root)))
    return out


def load_verdicts(lane_dirs) -> pd.DataFrame:
    frames = []
    for lane, root in lane_dirs:
        for label, value in experiment.TEMPERATURES.items():
            path = experiment.temp_dir(root, label) / "verdicts.csv"
            if not path.is_file():
                continue
            df = pd.read_csv(path, dtype={"model": str, "injection": str, "log": str,
                                          "category": str, "verdict": str})
            if "lane" in df.columns and set(df["lane"].dropna()) - {lane}:
                print(f"WARNING: {path} says lane {sorted(set(df['lane'].dropna()))}, "
                      f"--lane-dirs says {lane!r}; using {lane!r}")
            # the folder is authoritative; older CSVs have no lane/temperature/run_idx
            df["lane"] = lane
            df["temperature"] = value
            df["run_idx"] = df["seed"].astype(int).map(experiment.run_idx)
            if "truncated" not in df.columns:
                df["truncated"] = 0
            frames.append(df)
            print(f"read {path}: {len(df)} rows, runs {sorted(df['run_idx'].unique())}")
    if not frames:
        sys.exit("ERROR: no verdicts.csv found under "
                 + ", ".join(f"{root}/temp_<t>/" for _, root in lane_dirs)
                 + " -- run summarize_results.py (or main.py) first")
    df = pd.concat(frames, ignore_index=True)
    df["tricked"] = df["tricked"].astype(int)
    df["truncated"] = df["truncated"].fillna(0).astype(int)
    return df


def clean(df: pd.DataFrame, drop_truncated: bool) -> pd.DataFrame:
    unknown = int((df["category"].fillna("unknown") == "unknown").sum())
    if unknown:
        print(f"WARNING: dropped {unknown} row(s) with category=unknown -- stale result "
              f"files whose injection id is not in injections.jsonl")
        df = df[df["category"].fillna("unknown") != "unknown"]
    cut = int(df["truncated"].sum())
    if cut:
        if drop_truncated:
            print(f"WARNING: dropped {cut} TRUNCATED call(s) (--drop-truncated)")
            df = df[df["truncated"] == 0]
        else:
            print(f"WARNING: {cut} call(s) were TRUNCATED and are kept, as in the existing "
                  f"reports (--drop-truncated removes them)")
    dupes = df.duplicated(CELL + ["run_idx"]).sum()
    if dupes:
        sys.exit(f"ERROR: {dupes} duplicate (cell, run_idx) rows -- two lane dirs pointing "
                 f"at the same folder?")
    return df


def order(df: pd.DataFrame) -> pd.DataFrame:
    lane_rank = {lane: i for i, lane in enumerate(LANES)}
    return (df.assign(_lane=df["lane"].map(lambda x: lane_rank.get(x, len(LANES))))
              .sort_values(["model", "_lane", "temperature"]
                           + [c for c in ("injection", "log", "run_idx") if c in df.columns])
              .drop(columns="_lane")
              .reset_index(drop=True))


# ---------------------------------------------------------------- a, b, c

def collapse_cells(df: pd.DataFrame) -> pd.DataFrame:
    cells = (df.groupby(CELL + ["category"], sort=False)
               .agg(n_runs=("tricked", "size"), successes=("tricked", "sum"),
                    verdicts=("verdict", "nunique"), truncated_any=("truncated", "max"))
               .reset_index())
    cells["mean_success"] = cells["successes"] / cells["n_runs"]
    cells["tie"] = (cells["successes"] * 2 == cells["n_runs"]).astype(int)
    cells["majority"] = (cells["successes"] * 2 > cells["n_runs"]).astype(int)
    cells["unanimous"] = ((cells["successes"] == 0)
                          | (cells["successes"] == cells["n_runs"])).astype(int)
    cells["flipped"] = ((cells["unanimous"] == 0) & (cells["n_runs"] > 1)).astype(int)
    cells["flipped_verdict"] = ((cells["verdicts"] > 1) & (cells["n_runs"] > 1)).astype(int)
    runs = df.groupby(GROUP)["run_idx"].nunique().rename("group_runs").reset_index()
    cells = cells.merge(runs, on=GROUP)
    cells["incomplete"] = (cells["n_runs"] < cells["group_runs"]).astype(int)
    return order(cells.drop(columns="verdicts"))


def flip_rates(cells: pd.DataFrame) -> pd.DataFrame:
    out = (cells.groupby(GROUP, sort=False)
                .agg(runs=("group_runs", "first"), cells=("flipped", "size"),
                     cells_incomplete=("incomplete", "sum"), flipped=("flipped", "sum"),
                     flipped_verdict=("flipped_verdict", "sum"))
                .reset_index())
    multi = out["runs"] > 1
    out["flip_rate"] = np.where(multi, out["flipped"] / out["cells"], np.nan)
    out["flip_rate_verdict"] = np.where(multi, out["flipped_verdict"] / out["cells"], np.nan)
    return order(out)


def asr_by_run(df: pd.DataFrame, extra=()) -> pd.DataFrame:
    keys = GROUP + list(extra)
    out = (df.groupby(keys + ["run_idx"], sort=False)
             .agg(calls=("tricked", "size"), tricked=("tricked", "sum"))
             .reset_index())
    out["asr"] = out["tricked"] / out["calls"]
    return order(out)


def mean_std(per_run: pd.DataFrame, extra=()) -> pd.DataFrame:
    keys = GROUP + list(extra)
    out = (per_run.groupby(keys, sort=False)
                  .agg(runs=("asr", "size"), calls_min=("calls", "min"),
                       calls_max=("calls", "max"), asr_mean=("asr", "mean"),
                       asr_std=("asr", lambda s: s.std(ddof=1)))
                  .reset_index())
    return order(out)


# ---------------------------------------------------------------- d, e

def injection_ci(cells: pd.DataFrame) -> pd.DataFrame:
    out = (cells.groupby(GROUP + ["category", "injection"], sort=False)
                .agg(logs=("majority", "size"), k_majority=("majority", "sum"),
                     mean_success=("mean_success", "mean"), ties=("tie", "sum"))
                .reset_index())
    out["asr_majority"] = out["k_majority"] / out["logs"]
    k, n = out["k_majority"].to_numpy(), out["logs"].to_numpy()
    out["wilson_lo"], out["wilson_hi"] = proportion_confint(k, n, alpha=0.05, method="wilson")
    out["cp_lo"], out["cp_hi"] = proportion_confint(k, n, alpha=0.05, method="beta")
    return order(out)


def bootstrap(cells: pd.DataFrame, iters: int, seed: int) -> pd.DataFrame:
    rows = []
    for key, g in cells.groupby(GROUP, sort=False):
        per_log = g.groupby("log")["mean_success"].agg(["sum", "size"])
        sums, counts = per_log["sum"].to_numpy(), per_log["size"].to_numpy()
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(per_log), size=(iters, len(per_log)))
        draws = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        rows.append(dict(zip(GROUP, key), logs=len(per_log), cells=len(g),
                         asr=g["mean_success"].mean(), ci_lo=lo, ci_hi=hi,
                         iterations=iters, boot_seed=seed))
    return order(pd.DataFrame(rows))


# ---------------------------------------------------------------- g

def mcnemar_row(x: np.ndarray, y: np.ndarray) -> dict:
    # x, y: paired 0/1 majority outcomes; b = x only, c = y only
    both = int(((x == 1) & (y == 1)).sum())
    b = int(((x == 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum())
    neither = int(((x == 0) & (y == 0)).sum())
    p = mcnemar([[both, b], [c, neither]], exact=True).pvalue if len(x) else np.nan
    return {"n_pairs": len(x), "asr_a": x.mean() if len(x) else np.nan,
            "asr_b": y.mean() if len(y) else np.nan,
            "both": both, "a_only": b, "b_only": c, "neither": neither, "p_exact": p}


def holm(df: pd.DataFrame, by=None) -> pd.DataFrame:
    df = df.copy()
    df["p_holm"] = np.nan
    groups = df.groupby(by, sort=False) if by else [(None, df)]
    for _, g in groups:
        ok = g["p_exact"].notna()
        if ok.any():
            df.loc[g.index[ok], "p_holm"] = multipletests(g.loc[ok, "p_exact"],
                                                          method="holm")[1]
    return df


def paired(cells: pd.DataFrame, keys: list, a: pd.Series, b: pd.Series, on: list):
    left = cells[a][keys + on + ["majority"]]
    right = cells[b][keys + on + ["majority"]]
    return left.merge(right, on=keys + on, suffixes=("_a", "_b"))


def mcnemar_lanes(cells: pd.DataFrame) -> pd.DataFrame:
    rows, present = [], set(cells["lane"])
    for lane_a, lane_b in LANE_PAIRS:
        if not {lane_a, lane_b} <= present:
            continue
        m = paired(cells, ["model", "temperature"], cells["lane"] == lane_a,
                   cells["lane"] == lane_b, ["injection", "log"])
        for (model, temp), g in m.groupby(["model", "temperature"], sort=True):
            rows.append(dict(model=model, temperature=temp, lane_a=lane_a, lane_b=lane_b,
                             **mcnemar_row(g["majority_a"].to_numpy(),
                                           g["majority_b"].to_numpy())))
    return holm(pd.DataFrame(rows)) if rows else pd.DataFrame()


def mcnemar_twins(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pooled = []
    for generic, twin in injection_set.LABEL_TWINS.items():
        m = paired(cells, GROUP, cells["injection"] == generic,
                   cells["injection"] == twin, ["log"])
        if m.empty:
            continue
        m["pair"] = f"{generic} vs {twin}"
        pooled.append(m)
        for key, g in m.groupby(GROUP, sort=False):
            rows.append(dict(zip(GROUP, key), scope="pair", generic=generic, twin=twin,
                             **mcnemar_row(g["majority_a"].to_numpy(),
                                           g["majority_b"].to_numpy())))
    if pooled:
        for key, g in pd.concat(pooled).groupby(GROUP, sort=False):
            rows.append(dict(zip(GROUP, key), scope="pooled", generic="all generic",
                             twin="all SPT twins",
                             **mcnemar_row(g["majority_a"].to_numpy(),
                                           g["majority_b"].to_numpy())))
    return order(holm(pd.DataFrame(rows), by="scope")) if rows else pd.DataFrame()


def mcnemar_temps(cells: pd.DataFrame, pairs) -> pd.DataFrame:
    rows = []
    for label_a, label_b in pairs:
        ta, tb = experiment.TEMPERATURES[label_a], experiment.TEMPERATURES[label_b]
        m = paired(cells, ["model", "lane"], cells["temperature"] == ta,
                   cells["temperature"] == tb, ["injection", "log"])
        for (model, lane), g in m.groupby(["model", "lane"], sort=True):
            rows.append(dict(model=model, lane=lane, temperature_a=ta, temperature_b=tb,
                             **mcnemar_row(g["majority_a"].to_numpy(),
                                           g["majority_b"].to_numpy())))
    return holm(pd.DataFrame(rows)) if rows else pd.DataFrame()


# ---------------------------------------------------------------- LaTeX

TEX_ESCAPES = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#",
               "$": r"\$", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
               "^": r"\textasciicircum{}"}


def tex(text) -> str:
    return "".join(TEX_ESCAPES.get(ch, ch) for ch in str(text))


def pct(value, digits=1) -> str:
    return "--" if pd.isna(value) else f"{100 * value:.{digits}f}"


def temp_tex(value) -> str:
    return f"{value:g}"


def write_tex(path: Path, colspec: str, header: list, body: list, caption: str,
              label: str, long: bool = False) -> None:
    head = " & ".join(header) + r" \\"
    if long:
        lines = [r"% requires \usepackage{booktabs,longtable}",
                 r"\begin{longtable}{" + colspec + "}",
                 r"\caption{" + caption + r"}\label{" + label + r"}\\",
                 r"\toprule", head, r"\midrule", r"\endfirsthead",
                 r"\toprule", head, r"\midrule", r"\endhead",
                 r"\bottomrule", r"\endfoot"]
        lines += body + [r"\end{longtable}"]
    else:
        lines = [r"% requires \usepackage{booktabs}", r"\begin{table}[ht]", r"\centering",
                 r"\caption{" + caption + "}", r"\label{" + label + "}",
                 r"\begin{tabular}{" + colspec + "}", r"\toprule", head, r"\midrule"]
        lines += body + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def group_rows(df: pd.DataFrame, cells_fn) -> list:
    body, last = [], None
    for _, r in df.iterrows():
        if last is not None and r["model"] != last:
            body.append(r"\midrule")
        last = r["model"]
        body.append(" & ".join(cells_fn(r)) + r" \\")
    return body


def tables(out: Path, ms: pd.DataFrame, boot: pd.DataFrame, flips: pd.DataFrame,
           inj: pd.DataFrame, top: int) -> list:
    out.mkdir(parents=True, exist_ok=True)
    agg = order(ms.merge(boot, on=GROUP))
    write_tex(out / "asr_aggregate.tex", "lllrrr",
              ["Model", "Encoding", "$T$", "Runs", r"ASR (\%, mean $\pm$ sd)",
               r"95\% bootstrap CI"],
              group_rows(agg, lambda r: [
                  tex(r["model"]), tex(r["lane"]), temp_tex(r["temperature"]),
                  str(int(r["runs"])),
                  f"{pct(r['asr_mean'])} $\\pm$ {pct(r['asr_std'])}",
                  f"[{pct(r['ci_lo'])}, {pct(r['ci_hi'])}]"]),
              r"Attack success rate per model, encoding and temperature: mean $\pm$ sample "
              r"sd over runs, and a 95\% cluster-bootstrap interval over attack logs.",
              "tab:asr-aggregate")
    write_tex(out / "flip_rate.tex", "lllrrrr",
              ["Model", "Encoding", "$T$", "Runs", "Cells", "Flipped", r"Flip rate (\%)"],
              group_rows(flips, lambda r: [
                  tex(r["model"]), tex(r["lane"]), temp_tex(r["temperature"]),
                  str(int(r["runs"])), str(int(r["cells"])), str(int(r["flipped"])),
                  pct(r["flip_rate"])]),
              "Share of (injection, attack) cells whose repeated runs disagree on success.",
              "tab:flip-rate")
    # inj is already in group order; a stable sort inside each group keeps it
    best = (inj.assign(_group=inj.groupby(GROUP, sort=False).ngroup())
               .sort_values(["_group", "asr_majority", "mean_success"],
                            ascending=[True, False, False], kind="mergesort")
               .groupby("_group", sort=False).head(top))
    write_tex(out / "top_injections.tex", "llllrrr",
              ["Model", "Encoding", "$T$", "Injection", "$k/n$", r"ASR (\%)",
               r"95\% Wilson CI"],
              group_rows(best, lambda r: [
                  tex(r["model"]), tex(r["lane"]), temp_tex(r["temperature"]),
                  tex(r["injection"]), f"{int(r['k_majority'])}/{int(r['logs'])}",
                  pct(r["asr_majority"]),
                  f"[{pct(r['wilson_lo'])}, {pct(r['wilson_hi'])}]"]),
              f"Top {top} injections per model, encoding and temperature by majority-vote "
              r"ASR over attack logs, with 95\% Wilson intervals.",
              "tab:top-injections", long=True)
    return [out / "asr_aggregate.tex", out / "flip_rate.tex", out / "top_injections.tex"]


# ---------------------------------------------------------------- main

def write(df: pd.DataFrame, path: Path, written: list) -> None:
    df.to_csv(path, index=False, float_format="%.6g")
    written.append(path)


def summary(flips: pd.DataFrame, threshold: float, corpus_lanes: list) -> None:
    print("\n=== flip rate (share of cells whose runs disagree on success) ===")
    show = flips[GROUP + ["runs", "cells", "flipped", "flip_rate"]].copy()
    show["flip_rate"] = show["flip_rate"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.2%}")
    print(show.to_string(index=False))
    rated = flips.dropna(subset=["flip_rate"])
    if rated.empty:
        print("\nno group has more than one run -- flip rate is undefined")
        return
    worst = rated.loc[rated["flip_rate"].idxmax()]
    print(f"\nmax flip rate: {worst['flip_rate']:.2%} ({worst['model']}, {worst['lane']}, "
          f"T={worst['temperature']:g})")
    over = rated[rated["flip_rate"] > threshold]
    if over.empty:
        print(f"every flip rate is at or below {threshold:.0%} -- 5 runs are enough")
        return
    temps = sorted(over["temperature"].unique())
    labels = [TEMP_LABEL.get(t, str(t)) for t in temps]
    print(f"FLAG: flip rate above {threshold:.0%} at T = "
          f"{', '.join(f'{t:g}' for t in temps)} "
          f"({len(over)} group(s)) -- give those temperatures 10 runs instead of 5:")
    corpus = "all" if len(corpus_lanes) > 1 else corpus_lanes[0]
    print(f"  python main.py --corpus {corpus} --skip-inject --temperatures "
          f"{' '.join(labels)} --runs 10")


def parse_args():
    ap = argparse.ArgumentParser(description="Statistics over the benchmark verdicts.")
    ap.add_argument("--lane-dirs", nargs="+", default=list(DEFAULT_LANE_DIRS),
                    help="lane=analysis-root pairs; missing roots are skipped "
                         f"(default: {' '.join(DEFAULT_LANE_DIRS)})")
    ap.add_argument("--out-dir", default="analysis_stats",
                    help="where to write the CSVs and tables/ (default: analysis_stats)")
    ap.add_argument("--drop-truncated", action="store_true",
                    help="leave out calls ollama truncated (kept by default, as in report.md)")
    ap.add_argument("--boot-iters", type=int, default=10000,
                    help="cluster-bootstrap iterations (default 10000)")
    ap.add_argument("--boot-seed", type=int, default=20261001,
                    help="cluster-bootstrap seed (default 20261001)")
    ap.add_argument("--temp-pairs", nargs="+", default=["0:medium"],
                    help="temperature pairs to McNemar-test, label:label "
                         f"(labels: {', '.join(experiment.TEMPERATURES)}; default 0:medium)")
    ap.add_argument("--top", type=int, default=10,
                    help="injections per group in tables/top_injections.tex (default 10)")
    ap.add_argument("--flip-threshold", type=float, default=0.05,
                    help="flip rate above which a temperature should get 10 runs "
                         "(default 0.05)")
    args = ap.parse_args()
    args.temp_pairs = [tuple(p.split(":", 1)) for p in args.temp_pairs]
    for pair in args.temp_pairs:
        if len(pair) != 2 or not set(pair) <= set(experiment.TEMPERATURES):
            ap.error(f"bad --temp-pairs entry {':'.join(pair)!r}")
    return args


def main() -> None:
    args = parse_args()
    lane_dirs = parse_lane_dirs(args.lane_dirs)
    df = order(clean(load_verdicts(lane_dirs), args.drop_truncated))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    cells = collapse_cells(df)
    incomplete = int(cells["incomplete"].sum())
    if incomplete:
        print(f"WARNING: {incomplete} cell(s) have fewer runs than the rest of their group "
              f"-- an unfinished seed; finish it before reading the flip rate and std")
    ties = int(cells["tie"].sum())
    if ties:
        print(f"WARNING: {ties} cell(s) tie on the majority vote (even run count); "
              f"counted as majority 0")

    flips = flip_rates(cells)
    per_run = asr_by_run(df)
    ms = mean_std(per_run)
    ms_inj = mean_std(asr_by_run(df, ["category", "injection"]), ["category", "injection"])
    inj = injection_ci(cells)
    boot = bootstrap(cells, args.boot_iters, args.boot_seed)

    write(cells, out / "cells.csv", written)
    write(flips, out / "flip_rate.csv", written)
    write(per_run, out / "asr_by_run.csv", written)
    write(ms, out / "asr_mean_std.csv", written)
    write(ms_inj, out / "asr_mean_std_by_injection.csv", written)
    write(inj, out / "injection_ci.csv", written)
    write(boot, out / "asr_bootstrap.csv", written)
    for name, table in (("mcnemar_lanes.csv", mcnemar_lanes(cells)),
                        ("mcnemar_spt_twins.csv", mcnemar_twins(cells)),
                        ("mcnemar_temperature.csv", mcnemar_temps(cells, args.temp_pairs))):
        if table.empty:
            print(f"note: {name} skipped -- no matching pairs on disk")
            continue
        write(table, out / name, written)
    written += tables(out / "tables", ms, boot, flips, inj, args.top)

    summary(flips, args.flip_threshold, sorted(set(df["lane"])))
    print(f"\nwrote {len(written)} files under {out}/")


if __name__ == "__main__":
    main()
