#!/usr/bin/env python3
"""
experiment.py

What it does:
  The run settings several stages must agree on: each model's recommended sampling, the
  three temperature levels derived from it and the folder each level writes to, which
  models run at which level, the seeds each level is repeated with, and how a model spec
  carries a gpt-oss reasoning level. Imported by run_benchmark.py, check_context.py,
  summarize_results.py and stats_analysis.py.

How to run it:
  Not a CLI. Import it:  import experiment;  experiment.options_for("llama3.1:8b", "high", 42, 100000)

What it outputs:
  Nothing. The settings it holds:
      sampling       RECOMMENDED: one sampling config per model (temperature, top_p,
                     top_k, min_p, num_predict), the creators' published values. Every
                     key is sent in `options` on every call. repeat_penalty is deliberately
                     NOT sent: no creator publishes one, so it is left to the Modelfile or
                     ollama's default (1.1), as in the runs before RECOMMENDED existed.
                     run_config.json records the Modelfile's value. num_predict -1 = no
                     generation cap (no creator publishes one); num_ctx is the only limit.
      temperatures   three levels per model, relative to its recommended temperature:
                         low    = 0.0 (greedy)
                         medium = recommended         llama 0.6, gemma 1.0, gpt-oss 1.0
                         high   = HIGH_FACTOR x rec   llama 0.9, gemma 1.5, gpt-oss 1.5
                     Only the temperature changes between levels. One folder each:
                     <root>/temp_low, <root>/temp_medium, <root>/temp_high
      model specs    "gpt-oss:20b@high" = ollama model gpt-oss:20b sent with think="high".
      two experiments
                     temperature: every level x gpt-oss@low, llama3.1, gemma3.
                     reasoning:   medium (gpt-oss's recommended sampling) x
                                  gpt-oss@low / @medium / @high. medium x gpt-oss@low is the
                                  same call as the temperature experiment's, so it runs once.
                     Both repeat every series DEFAULT_RUNS = 5 times (seeds 42..46), so
                     models_for("medium") lists five series and the other levels three:
                     11 x 5 = 55 model runs per lane.
      seeds          one folder each: <root>/temp_<t>/seed_<n>, so a result is reported as
                     mean +- std across the runs. Level low is repeated too: greedy decoding
                     on a GPU is not guaranteed to be deterministic, and its flip rate is
                     what tests that. --runs N takes seeds 42..42+N-1.
      run_idx        the 1-based position of a seed in that sequence (42 -> 1).
      result dirs    sanitize(spec): ':' and '/' -> '_', so "gpt-oss_20b@high"
"""

import re
from pathlib import Path

# No repeat_penalty: it is left to ollama, as before RECOMMENDED existed. Sending 1.0 made
# gpt-oss at greedy level low loop ("T1059.003. T1059.003. ...") until num_ctx was full.
RECOMMENDED = {
    "gpt-oss:20b": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "num_predict": -1},
    "gemma3:12b":  {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.0, "num_predict": -1},
    "llama3.1:8b": {"temperature": 0.6, "top_p": 0.9, "top_k": 0, "min_p": 0.0, "num_predict": -1},
}

TEMPERATURE_LEVELS = ("low", "medium", "high")
HIGH_FACTOR = 1.5

BASE_SEED = 42
DEFAULT_RUNS = 5
SEEDS = tuple(range(BASE_SEED, BASE_SEED + DEFAULT_RUNS))
SEED_RE = re.compile(r"^seed_(\d+)$")

REASONING_LEVELS = ("low", "medium", "high")
REASONING_RE = re.compile(r"@(%s)$" % "|".join(REASONING_LEVELS))
# the gpt-oss level the temperature experiment runs at
TEMPERATURE_EXPERIMENT_THINK = "low"
# the temperature level the reasoning experiment runs at (gpt-oss's recommended sampling)
REASONING_TEMPERATURE = "medium"

GPT_OSS = "gpt-oss:20b"
OTHER_MODELS = ["llama3.1:8b", "gemma3:12b"]


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


def sanitize(name: str) -> str:
    return re.sub(r"[:/]", "_", name)


def recommended(model: str) -> dict:
    # a spec ("gpt-oss:20b@low") or a results dir name ("gpt-oss_20b@low")
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
    return {"low": 0.0, "medium": rec, "high": round(HIGH_FACTOR * rec, 6)}[temp]


def options_for(model: str, temp: str, seed: int, num_ctx: int) -> dict:
    # the full `options` object sent to ollama: only the temperature varies by level
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
    # every seed_<n>/ already on disk, by seed -- whichever seeds were actually run
    base = temp_dir(root, temp)
    if not base.is_dir():
        return []
    found = [(int(m.group(1)), p) for p in base.iterdir()
             if p.is_dir() for m in [SEED_RE.match(p.name)] if m]
    return sorted(found)
