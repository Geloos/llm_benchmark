#!/usr/bin/env python3
"""
stats_analysis.py

What it does:
  The statistics on top of the benchmark. It reads every lane's verdicts.csv (written by
  summarize_results.py, one per analysis*/temp_<t>/) and stacks them into one table. It
  never calls ollama and never re-parses a reply: success is the existing `tricked`
  column (verdict bucket normal or neutral), unchanged.

  A cell is (model, lane, temperature, injection, log), and its runs are the seeds
  (run_idx 1..N). temperature is the level label (low / medium / high), not a number:
  each level is relative to the model's recommended temperature (experiment.py), so one
  level holds a different value per model -- asr_aggregate.csv carries both. From those cells it
  computes:
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
                          depend on which other groups are present. Drawn per model, per
                          model x category, and pooled over the models (overall, per
                          category, per injection). Pooled means the three temperature-
                          experiment series -- gpt-oss @low, llama3.1, gemma3 -- so
                          temp_medium's reasoning series do not count gpt-oss three times.
                          The per-injection pooled ASR is mean success, not d's majority vote.
    g  paired comparisons which test depends on how many pairs share a log.
                          Many per log -- plain vs control vs hexa and temperature low vs
                          medium vs high (same model, matched on injection and log), and
                          the SPT_ twins pooled over injection_set.LABEL_TWINS (matched on
                          log): the pairs are clustered, so the log is the unit (Miller
                          2024). d = mean success b - a per pair, summed per log; the mean
                          d gets a 95% paired cluster-bootstrap interval (resample logs,
                          both sides attached) and a cluster sign-flip p (flip the sign of
                          each log's summed d: exact over all 2^logs flips up to
                          SIGN_FLIP_EXACT logs, --boot-iters random flips above).
                          One per log -- a single generic injection vs its SPT_ twin: the
                          pairs are independent, so exact McNemar on majority-vote cells.
                          Holm-adjusted p within each file.
  There is no clean-log baseline (f): the clean step was removed from the pipeline.

  Every interval and test treats the attack logs as the independent units, and none uses
  a normal approximation: at 17 logs it is unreliable (Bowyer et al. 2025).

How to run it:
  python scripts/stats_analysis.py                    # every lane and temperature on disk
  python scripts/stats_analysis.py --lane-dirs plain=analysis hexa=analysis_hexa
  python scripts/stats_analysis.py --drop-truncated --boot-iters 20000
  python scripts/stats_analysis.py --temp-pairs low:medium low:high

What it outputs (under --out-dir, default analysis_stats/):
  cells.csv                         c: one row per cell
  flip_rate.csv                     a: per model x lane x temperature
  asr_by_run.csv                    b: ASR of each run
  asr_mean_std.csv                  b: mean +- std over runs, per model x lane x temperature
  asr_mean_std_by_injection.csv     b: the same per injection
  injection_ci.csv                  d: majority-vote ASR per injection, Wilson + CP intervals
  asr_bootstrap.csv                 e: aggregate ASR with the cluster-bootstrap interval
  asr_aggregate.csv                 b + e side by side per model x lane x temperature,
                                    with the model's numeric temperature_value
  asr_bootstrap_pooled.csv          e: the same, pooled over the models
  asr_bootstrap_by_category.csv     e: per model x category
  asr_bootstrap_by_category_pooled.csv   e: per category, pooled over the models
  asr_bootstrap_by_injection_pooled.csv  e: per injection, pooled over the models
  diff_lanes.csv                    g: encoding pairs, paired cluster bootstrap + sign-flip
  diff_temperature.csv              g: temperature pairs, the same
  diff_spt_twins.csv                g: generic vs SPT_ twin pooled over the pairs, the same
  mcnemar_spt_twins.csv             g: each generic vs its own SPT_ twin, exact McNemar
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
TEMP_RANK = {label: i for i, label in enumerate(experiment.TEMPERATURE_LEVELS)}


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
        for label in experiment.TEMPERATURE_LEVELS:
            path = experiment.temp_dir(root, label) / "verdicts.csv"
            if not path.is_file():
                continue
            df = pd.read_csv(path, dtype={"model": str, "injection": str, "log": str,
                                          "category": str, "verdict": str})
            if "lane" in df.columns and set(df["lane"].dropna()) - {lane}:
                print(f"WARNING: {path} says lane {sorted(set(df['lane'].dropna()))}, "
                      f"--lane-dirs says {lane!r}; using {lane!r}")
            # the folder is authoritative; older CSVs have no lane/temperature/run_idx.
            # temperature is the level label: the numeric value differs per model
            df["lane"] = lane
            df["temperature"] = label
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
    tail = [c for c in ("injection", "log", "run_idx") if c in df.columns]
    if "injection" not in df.columns and "category" in df.columns:
        tail = ["category"] + tail
    return (df.assign(_lane=df["lane"].map(lambda x: lane_rank.get(x, len(LANES))),
                      _temp=df["temperature"].map(lambda x: TEMP_RANK.get(x, len(TEMP_RANK))))
              .sort_values([c for c in ("model",) if c in df.columns] + ["_lane", "_temp"]
                           + tail)
              .drop(columns=["_lane", "_temp"])
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


def pooled_series(cells: pd.DataFrame) -> pd.DataFrame:
    # the series the model-pooled intervals average over: one per base model, gpt-oss at
    # the temperature experiment's reasoning level only -- temp_medium also holds the
    # reasoning experiment's @medium/@high, which would count gpt-oss three times
    level = cells["model"].map(experiment.reasoning_level)
    return cells[level.isna() | (level == experiment.TEMPERATURE_EXPERIMENT_THINK)]


def bootstrap(cells: pd.DataFrame, keys: list, iters: int, seed: int) -> pd.DataFrame:
    rows = []
    for key, g in cells.groupby(keys, sort=False):
        per_log = g.groupby("log")["mean_success"].agg(["sum", "size"])
        sums, counts = per_log["sum"].to_numpy(), per_log["size"].to_numpy()
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(per_log), size=(iters, len(per_log)))
        draws = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        row = dict(zip(keys, key))
        if "model" not in keys:
            row["models"] = " ".join(sorted(g["model"].unique()))
        rows.append(dict(row, logs=len(per_log), cells=len(g),
                         asr=g["mean_success"].mean(), ci_lo=lo, ci_hi=hi,
                         iterations=iters, boot_seed=seed))
    return order(pd.DataFrame(rows))


# ---------------------------------------------------------------- g

# up to this many logs the sign-flip test enumerates every flip (2^17 = 131072 at 17
# logs); above it, it draws --boot-iters random flips
SIGN_FLIP_EXACT = 20


def discordant(x: np.ndarray, y: np.ndarray) -> dict:
    # x, y: paired 0/1 majority outcomes; a_only = x only, b_only = y only
    return {"both": int(((x == 1) & (y == 1)).sum()),
            "a_only": int(((x == 1) & (y == 0)).sum()),
            "b_only": int(((x == 0) & (y == 1)).sum()),
            "neither": int(((x == 0) & (y == 0)).sum())}


def mcnemar_row(x: np.ndarray, y: np.ndarray) -> dict:
    counts = discordant(x, y)
    table = [[counts["both"], counts["a_only"]], [counts["b_only"], counts["neither"]]]
    p = mcnemar(table, exact=True).pvalue if len(x) else np.nan
    return {"n_pairs": len(x), "asr_a": x.mean() if len(x) else np.nan,
            "asr_b": y.mean() if len(y) else np.nan, **counts, "p_exact": p}


def sign_flip_p(sums: np.ndarray, iters: int, seed: int) -> tuple:
    # H0: no difference, so each log's summed d is as likely negative as positive.
    # p = share of sign assignments whose total lies at least as far from 0 as the observed
    observed = abs(sums.sum()) - 1e-9
    n = len(sums)
    if n <= SIGN_FLIP_EXACT:
        bits, hits, chunk = np.arange(n), 0, 1 << min(n, 16)
        for start in range(0, 1 << n, chunk):
            flips = 1 - 2 * ((np.arange(start, start + chunk)[:, None] >> bits) & 1)
            hits += int((np.abs(flips @ sums) >= observed).sum())
        return hits / (1 << n), "exact"
    flips = np.random.default_rng(seed).choice((-1, 1), size=(iters, n))
    hits = int((np.abs(flips @ sums) >= observed).sum())
    return (hits + 1) / (iters + 1), "monte carlo"


def diff_row(m: pd.DataFrame, iters: int, seed: int) -> dict:
    # m: the matched pairs of one comparison, several per log, so the log is the unit
    # (Miller 2024). d = mean success b - a per pair, summed per log; the bootstrap
    # resamples logs with both sides attached, the sign-flip test flips each log's sum
    d = m["mean_success_b"] - m["mean_success_a"]
    per_log = d.groupby(m["log"]).agg(["sum", "size"])
    sums, counts = per_log["sum"].to_numpy(), per_log["size"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(iters, len(sums)))
    draws = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p, method = sign_flip_p(sums, iters, seed)
    return {"pairs": len(m), "logs": len(sums),
            "asr_a": m["mean_success_a"].mean(), "asr_b": m["mean_success_b"].mean(),
            "diff": d.mean(), "ci_lo": lo, "ci_hi": hi,
            "excludes_zero": int(lo > 0 or hi < 0),
            **discordant(m["majority_a"].to_numpy(), m["majority_b"].to_numpy()),
            "p_signflip": p, "p_method": method, "iterations": iters, "boot_seed": seed}


def holm(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df["p_holm"] = np.nan
    ok = df[col].notna()
    if ok.any():
        df.loc[ok, "p_holm"] = multipletests(df.loc[ok, col], method="holm")[1]
    return df


def paired(cells: pd.DataFrame, keys: list, a: pd.Series, b: pd.Series, on: list):
    cols = keys + on + ["majority", "mean_success"]
    return cells[a][cols].merge(cells[b][cols], on=keys + on, suffixes=("_a", "_b"))


def diff_lanes(cells: pd.DataFrame, iters: int, seed: int) -> pd.DataFrame:
    rows, present = [], set(cells["lane"])
    for lane_a, lane_b in LANE_PAIRS:
        if not {lane_a, lane_b} <= present:
            continue
        m = paired(cells, ["model", "temperature"], cells["lane"] == lane_a,
                   cells["lane"] == lane_b, ["injection", "log"])
        for (model, temp), g in m.groupby(["model", "temperature"], sort=True):
            rows.append(dict(model=model, temperature=temp, lane_a=lane_a, lane_b=lane_b,
                             **diff_row(g, iters, seed)))
    return holm(pd.DataFrame(rows), "p_signflip") if rows else pd.DataFrame()


def diff_temps(cells: pd.DataFrame, pairs, iters: int, seed: int) -> pd.DataFrame:
    rows = []
    for label_a, label_b in pairs:
        m = paired(cells, ["model", "lane"], cells["temperature"] == label_a,
                   cells["temperature"] == label_b, ["injection", "log"])
        for (model, lane), g in m.groupby(["model", "lane"], sort=True):
            rows.append(dict(model=model, lane=lane, temperature_a=label_a,
                             temperature_b=label_b,
                             value_a=temp_value(model, label_a),
                             value_b=temp_value(model, label_b),
                             **diff_row(g, iters, seed)))
    return holm(pd.DataFrame(rows), "p_signflip") if rows else pd.DataFrame()


def twin_pairs(cells: pd.DataFrame) -> list:
    # each generic injection matched with its SPT_ twin on log: one pair per log
    out = []
    for generic, twin in injection_set.LABEL_TWINS.items():
        m = paired(cells, GROUP, cells["injection"] == generic,
                   cells["injection"] == twin, ["log"])
        if not m.empty:
            out.append(m.assign(generic=generic, twin=twin))
    return out


def mcnemar_twins(pairs: list) -> pd.DataFrame:
    # one pair per log, so the pairs are independent and exact McNemar holds
    rows = []
    for m in pairs:
        for key, g in m.groupby(GROUP, sort=False):
            rows.append(dict(zip(GROUP, key), generic=g["generic"].iloc[0],
                             twin=g["twin"].iloc[0],
                             **mcnemar_row(g["majority_a"].to_numpy(),
                                           g["majority_b"].to_numpy())))
    return order(holm(pd.DataFrame(rows), "p_exact")) if rows else pd.DataFrame()


def diff_twins(pairs: list, iters: int, seed: int) -> pd.DataFrame:
    # pooled over the twin pairs: one pair per twin per log, so clustered again
    if not pairs:
        return pd.DataFrame()
    rows = []
    for key, g in pd.concat(pairs).groupby(GROUP, sort=False):
        rows.append(dict(zip(GROUP, key), twins=" ".join(sorted(g["twin"].unique())),
                         **diff_row(g, iters, seed)))
    return order(holm(pd.DataFrame(rows), "p_signflip"))


# ---------------------------------------------------------------- temperature values

def temp_value(model: str, label: str):
    try:
        return experiment.temperature_for(model, label)
    except ValueError:
        return np.nan


def temp_text(model: str, label: str) -> str:
    # "high (0.9)": the level and this model's value at it
    value = temp_value(model, label)
    return label if pd.isna(value) else f"{label} ({value:g})"


def aggregate(ms: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    # b and e side by side per model x lane x temperature, with this model's numeric
    # temperature at the level
    out = order(ms.merge(boot, on=GROUP))
    out.insert(out.columns.get_loc("temperature") + 1, "temperature_value",
               [temp_value(m, t) for m, t in zip(out["model"], out["temperature"])])
    return out


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
          f"T={temp_text(worst['model'], worst['temperature'])})")
    over = rated[rated["flip_rate"] > threshold]
    if over.empty:
        print(f"every flip rate is at or below {threshold:.0%} -- 5 runs are enough")
        return
    labels = sorted(over["temperature"].unique(), key=lambda t: TEMP_RANK.get(t, len(TEMP_RANK)))
    print(f"FLAG: flip rate above {threshold:.0%} at T = {', '.join(labels)} "
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
                    help="where to write the CSVs (default: analysis_stats)")
    ap.add_argument("--drop-truncated", action="store_true",
                    help="leave out calls ollama truncated (kept by default, as in report.md)")
    ap.add_argument("--boot-iters", type=int, default=10000,
                    help="cluster-bootstrap iterations, and random sign flips past "
                         f"{SIGN_FLIP_EXACT} logs (default 10000)")
    ap.add_argument("--boot-seed", type=int, default=20261001,
                    help="cluster-bootstrap seed (default 20261001)")
    ap.add_argument("--temp-pairs", nargs="+", default=["low:medium", "medium:high"],
                    help="temperature pairs to compare, label:label "
                         f"(labels: {', '.join(experiment.TEMPERATURE_LEVELS)}; default "
                         "low:medium medium:high)")
    ap.add_argument("--flip-threshold", type=float, default=0.05,
                    help="flip rate above which a temperature should get 10 runs "
                         "(default 0.05)")
    args = ap.parse_args()
    args.temp_pairs = [tuple(p.split(":", 1)) for p in args.temp_pairs]
    for pair in args.temp_pairs:
        if len(pair) != 2 or not set(pair) <= set(experiment.TEMPERATURE_LEVELS):
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
    boot = bootstrap(cells, GROUP, args.boot_iters, args.boot_seed)
    pooled = pooled_series(cells)
    lane_temp = ["lane", "temperature"]
    boots = (("asr_bootstrap_pooled.csv", pooled, lane_temp),
             ("asr_bootstrap_by_category.csv", cells, GROUP + ["category"]),
             ("asr_bootstrap_by_category_pooled.csv", pooled, lane_temp + ["category"]),
             ("asr_bootstrap_by_injection_pooled.csv", pooled,
              lane_temp + ["category", "injection"]))

    write(cells, out / "cells.csv", written)
    write(flips, out / "flip_rate.csv", written)
    write(per_run, out / "asr_by_run.csv", written)
    write(ms, out / "asr_mean_std.csv", written)
    write(ms_inj, out / "asr_mean_std_by_injection.csv", written)
    write(inj, out / "injection_ci.csv", written)
    write(boot, out / "asr_bootstrap.csv", written)
    write(aggregate(ms, boot), out / "asr_aggregate.csv", written)
    for name, subset, keys in boots:
        write(bootstrap(subset, keys, args.boot_iters, args.boot_seed), out / name, written)
    iters, seed = args.boot_iters, args.boot_seed
    twins = twin_pairs(cells)
    for name, table in (("diff_lanes.csv", diff_lanes(cells, iters, seed)),
                        ("diff_temperature.csv",
                         diff_temps(cells, args.temp_pairs, iters, seed)),
                        ("diff_spt_twins.csv", diff_twins(twins, iters, seed)),
                        ("mcnemar_spt_twins.csv", mcnemar_twins(twins))):
        if table.empty:
            print(f"note: {name} skipped -- no matching pairs on disk")
            continue
        write(table, out / name, written)

    summary(flips, args.flip_threshold, sorted(set(df["lane"])))
    print(f"\nwrote {len(written)} files under {out}/")


if __name__ == "__main__":
    main()
