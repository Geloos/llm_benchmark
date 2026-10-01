#!/usr/bin/env python3
"""
experiment.py

What it does:
  The run settings several stages must agree on: the temperature grid and the folder
  each temperature writes to, which models run at which temperature, the seeds each
  temperature is repeated with, and how a model spec carries a gpt-oss reasoning level.
  Imported by run_benchmark.py, check_context.py, summarize_results.py and
  stats_analysis.py.

How to run it:
  Not a CLI. Import it:  import experiment;  experiment.models_for("0")

What it outputs:
  Nothing. The settings it holds:
      temperatures   0 -> 0.0, low -> 0.3, medium -> 0.7, one folder each:
                     <root>/temp_0, <root>/temp_low, <root>/temp_medium
      model specs    "gpt-oss:20b@high" = ollama model gpt-oss:20b sent with think="high".
      two experiments
                     main:      every temperature x gpt-oss@medium, llama3.1, gemma3, each
                                repeated DEFAULT_RUNS = 5 times (seeds 42..46).
                     reasoning: temperature 0 x gpt-oss@low / @medium / @high, run ONCE
                                (seed 42). @medium at seed 42 is the same call as the main
                                experiment's first run, so only @low and @high are extra.
                     So models_for("0") lists all five, and seeds_for_model() trims
                     @low / @high to seed 42.
      seeds          one folder each: <root>/temp_<t>/seed_<n>, so a main-experiment
                     result is reported as mean +- std across the runs. Temperature 0 is
                     repeated too: greedy decoding on a GPU is not guaranteed to be
                     deterministic, and its flip rate is what tests that. --runs N takes
                     seeds 42..42+N-1.
      run_idx        the 1-based position of a seed in that sequence (42 -> 1).
      result dirs    sanitize(spec): ':' and '/' -> '_', so "gpt-oss_20b@high"
"""

import re
from pathlib import Path

TEMPERATURES = {"0": 0.0, "low": 0.3, "medium": 0.7}

BASE_SEED = 42
DEFAULT_RUNS = 5
SEEDS = tuple(range(BASE_SEED, BASE_SEED + DEFAULT_RUNS))
SEED_RE = re.compile(r"^seed_(\d+)$")

REASONING_LEVELS = ("low", "medium", "high")
REASONING_RE = re.compile(r"@(%s)$" % "|".join(REASONING_LEVELS))
# the levels that exist only for the reasoning experiment, which runs once
REASONING_ONLY_LEVELS = ("low", "high")

GPT_OSS = "gpt-oss:20b"
OTHER_MODELS = ["llama3.1:8b", "gemma3:12b"]


def models_for(temp: str) -> list:
    if temp == "0":
        levels = REASONING_LEVELS
    else:
        levels = ("medium",)
    return ["%s@%s" % (GPT_OSS, level) for level in levels] + OTHER_MODELS


def all_models() -> list:
    out = []
    for temp in TEMPERATURES:
        out += [m for m in models_for(temp) if m not in out]
    return out


def parse_model(spec: str):
    base, sep, level = spec.partition("@")
    if not sep:
        return spec, None
    if level not in REASONING_LEVELS:
        raise ValueError("unknown reasoning level %r in %r (expected one of: %s)"
                         % (level, spec, ", ".join(REASONING_LEVELS)))
    return base, level


def reasoning_level(model_dir: str):
    m = REASONING_RE.search(model_dir)
    return m.group(1) if m else None


def sanitize(name: str) -> str:
    return re.sub(r"[:/]", "_", name)


def temp_dir(root, temp: str) -> Path:
    if temp not in TEMPERATURES:
        raise ValueError("unknown temperature %r (expected one of: %s)"
                         % (temp, ", ".join(TEMPERATURES)))
    return Path(root) / ("temp_%s" % temp)


def seeds_for(temp: str, runs: int = None) -> tuple:
    if temp not in TEMPERATURES:
        raise ValueError("unknown temperature %r" % temp)
    runs = DEFAULT_RUNS if runs is None else runs
    if runs < 1:
        raise ValueError("runs must be >= 1, got %d" % runs)
    return tuple(range(BASE_SEED, BASE_SEED + runs))


def runs_once(model: str) -> bool:
    # gpt-oss@low / @high belong to the reasoning experiment only; works on a spec
    # ("gpt-oss:20b@high") or a results dir name ("gpt-oss_20b@high")
    return reasoning_level(model) in REASONING_ONLY_LEVELS


def seeds_for_model(model: str, seeds) -> list:
    # the seeds this model actually runs at, out of the run's seed list
    if runs_once(model):
        return [s for s in seeds if s == BASE_SEED]
    return list(seeds)


def run_idx(seed: int) -> int:
    return seed - BASE_SEED + 1


def seed_dir(root, temp: str, seed: int) -> Path:
    return temp_dir(root, temp) / ("seed_%d" % seed)


def seed_dirs(root, temp: str) -> list:
    # every seed_<n>/ already on disk, by seed -- whichever seeds were actually run
    base = temp_dir(root, temp)
    if not base.is_dir():
        return []
    found = [(int(m.group(1)), p) for p in base.iterdir()
             if p.is_dir() for m in [SEED_RE.match(p.name)] if m]
    return sorted(found)
