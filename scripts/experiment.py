#!/usr/bin/env python3
"""
experiment.py

What it does:
  The run settings several stages must agree on: each model's recommended sampling, the
  two temperature levels derived from it and the folder each level writes to, which
  models run at which level, the seeds each level is repeated with, and how a model spec
  carries a gpt-oss reasoning level. Imported by run_benchmark.py, check_context.py,
  summarize_results.py and plot_results.py.

How to run it:
  Not a CLI. Import it:  import experiment;  experiment.options_for("llama3.1:8b", "medium", 42, 120000)

What it outputs:
  Nothing. The settings it holds:
      sampling       RECOMMENDED: one sampling config per model (temperature, top_p,
                     top_k, min_p, num_predict), the creators' published values. Every
                     key is sent in `options` on every call. repeat_penalty is deliberately
                     NOT sent: no creator publishes one, so it is left to the Modelfile or
                     ollama's default (1.1), as in the runs before RECOMMENDED existed.
                     run_config.json records the Modelfile's value. num_predict -1 = no
                     generation cap (no creator publishes one); num_ctx is the only limit.
      temperatures   two levels per model, relative to its recommended temperature:
                         low    = 0.0 (greedy)
                         medium = recommended         llama 0.6, gemma 1.0, gpt-oss 1.0
                     Only the temperature changes between levels. One folder each:
                     <root>/temp_low, <root>/temp_medium
      model specs    "gpt-oss:20b@high" = ollama model gpt-oss:20b sent with think="high".
      two experiments
                     temperature: every level x gpt-oss@low, llama3.1, gemma3.
                     reasoning:   medium (gpt-oss's recommended sampling) x
                                  gpt-oss@low / @medium / @high. medium x gpt-oss@low is the
                                  same call as the temperature experiment's, so it runs once.
                     Both repeat every series DEFAULT_RUNS = 5 times (seeds 42..46), so
                     models_for("medium") lists five series and models_for("low") three:
                     8 x 5 = 40 model runs per lane.
      seeds          one folder each: <root>/temp_<t>/seed_<n>. Level low is repeated too:
                     greedy decoding on a GPU is not guaranteed to be deterministic, and the
                     std across the runs is what shows that. --runs N takes seeds
                     42..42+N-1.
      run_idx        the 1-based position of a seed in that sequence (42 -> 1).
      ASR +- 95% CI  asr_stats() / mean_ci(): the ASR per seed (over the logs), its mean
                     across the seeds, the sample std, and the half-width of the two-sided
                     95% Student-t interval, t(0.975, n-1) * std / sqrt(n) -- 2.776 * std /
                     sqrt(5) at the default five seeds (T975 is a table for df 1..30, a
                     Cornish-Fisher expansion past it; stdlib has no t). It measures the
                     run-to-run variance of the LLM on these 17 logs, not how the result
                     generalises to other logs. Shared by summarize_results.py and
                     plot_results.py, so asr.csv, report.md and the whiskers agree.
      DOVSDOFS lane  DOVSDOFS_MODELS: gpt-oss@medium, llama3.1, gemma3, at both levels.
      OB lane        OB_MODELS: the same models, at both levels.
      result dirs    sanitize(spec): ':' and '/' -> '_', so "gpt-oss_20b@high"
"""

import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

RECOMMENDED = {
    "gpt-oss:20b": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "num_predict": -1},
    "gemma3:12b":  {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.0, "num_predict": -1},
    "llama3.1:8b": {"temperature": 0.6, "top_p": 0.9, "top_k": 0, "min_p": 0.0, "num_predict": -1},
}

TEMPERATURE_LEVELS = ("low", "medium")

BASE_SEED = 42
DEFAULT_RUNS = 5
SEEDS = tuple(range(BASE_SEED, BASE_SEED + DEFAULT_RUNS))
SEED_RE = re.compile(r"^seed_(\d+)$")

REASONING_LEVELS = ("low", "medium", "high")
REASONING_RE = re.compile(r"@(%s)$" % "|".join(REASONING_LEVELS))
TEMPERATURE_EXPERIMENT_THINK = "low"
REASONING_TEMPERATURE = "medium"

T975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)

GPT_OSS = "gpt-oss:20b"
OTHER_MODELS = ["llama3.1:8b", "gemma3:12b"]
DOVSDOFS_MODELS = ["%s@medium" % GPT_OSS] + OTHER_MODELS
OB_MODELS = DOVSDOFS_MODELS


def check_level(temp: str) -> None:
    if temp not in TEMPERATURE_LEVELS:
        raise ValueError("unknown temperature level %r (expected one of: %s)"
                         % (temp, ", ".join(TEMPERATURE_LEVELS)))


def models_for(temp: str) -> list:
    check_level(temp)
    if temp == REASONING_TEMPERATURE:
        levels = REASONING_LEVELS
    else:
        levels = (TEMPERATURE_EXPERIMENT_THINK,)
    return ["%s@%s" % (GPT_OSS, level) for level in levels] + OTHER_MODELS


def all_models() -> list:
    out = []
    for temp in TEMPERATURE_LEVELS:
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


def model_sort_key(model_dir: str):
    level = reasoning_level(model_dir)
    rank = REASONING_LEVELS.index(level) if level else -1
    return model_dir.partition("@")[0], rank


def reasoning_series(models) -> list:
    levels = sorted((m for m in models if reasoning_level(m)), key=model_sort_key)
    return levels if len(levels) > 1 else []


def temperature_series(models) -> list:
    if not reasoning_series(models):
        return list(models)
    return [m for m in models
            if reasoning_level(m) in (None, TEMPERATURE_EXPERIMENT_THINK)] or list(models)


def sanitize(name: str) -> str:
    return re.sub(r"[:/]", "_", name)


def recommended(model: str) -> dict:
    base = sanitize(model.partition("@")[0])
    for name, sampling in RECOMMENDED.items():
        if sanitize(name) == base:
            return dict(sampling)
    raise ValueError("no recommended sampling for %r -- add it to experiment.RECOMMENDED "
                     "(every option is sent explicitly, so a model cannot run on ollama's "
                     "defaults)" % model)


def temperature_for(model: str, temp: str) -> float:
    check_level(temp)
    rec = recommended(model)["temperature"]
    return {"low": 0.0, "medium": rec}[temp]


def options_for(model: str, temp: str, seed: int, num_ctx: int) -> dict:
    return dict(recommended(model), temperature=temperature_for(model, temp),
                num_ctx=num_ctx, seed=seed)


def temp_dir(root, temp: str) -> Path:
    check_level(temp)
    return Path(root) / ("temp_%s" % temp)


def seeds_for(temp: str, runs: int = None) -> tuple:
    check_level(temp)
    runs = DEFAULT_RUNS if runs is None else runs
    if runs < 1:
        raise ValueError("runs must be >= 1, got %d" % runs)
    return tuple(range(BASE_SEED, BASE_SEED + runs))


def run_idx(seed: int) -> int:
    return seed - BASE_SEED + 1


def seed_dir(root, temp: str, seed: int) -> Path:
    return temp_dir(root, temp) / ("seed_%d" % seed)


def seed_dirs(root, temp: str) -> list:
    base = temp_dir(root, temp)
    if not base.is_dir():
        return []
    found = [(int(m.group(1)), p) for p in base.iterdir()
             if p.is_dir() for m in [SEED_RE.match(p.name)] if m]
    return sorted(found)


def t975(df: int) -> float:
    if df < 1:
        raise ValueError("df must be >= 1, got %d" % df)
    if df <= len(T975):
        return T975[df - 1]
    z = 1.959964
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)


def mean_ci(rates):
    if not rates:
        return 0.0, None, None
    mean = sum(rates) / len(rates)
    if len(rates) < 2:
        return mean, None, None
    std = statistics.stdev(rates)
    return mean, std, t975(len(rates) - 1) * std / math.sqrt(len(rates))


def asr_stats(calls) -> dict:
    """calls: (log, seed, tricked) for every call of one cell. The ASR per seed (tricked /
    calls in that seed), their mean, sample std and t-interval half-width (mean_ci), and
    the interval clipped to [0, 1]. std / ci / low / high are None with one seed."""
    logs = set()
    per_seed = defaultdict(lambda: [0, 0])
    for log, seed, hit in calls:
        logs.add(log)
        per_seed[seed][0] += hit
        per_seed[seed][1] += 1
    mean, std, ci = mean_ci([t / c for _, (t, c) in sorted(per_seed.items())])
    return {"asr": mean, "std": std, "ci": ci,
            "low": None if ci is None else max(0.0, mean - ci),
            "high": None if ci is None else min(1.0, mean + ci),
            "logs": len(logs), "seeds": len(per_seed),
            "calls": sum(c for _, c in per_seed.values()),
            "tricked": sum(t for t, _ in per_seed.values())}
