#!/usr/bin/env python3
"""
experiment.py

What it does:
  The run settings several stages must agree on: the temperature grid and the folder
  each temperature writes to, which models run at which temperature, the seeds each
  temperature is repeated with, and how a model spec carries a gpt-oss reasoning level.
  Imported by run_benchmark.py, check_context.py and summarize_results.py.

How to run it:
  Not a CLI. Import it:  import experiment;  experiment.models_for("0")

What it outputs:
  Nothing. The settings it holds:
      temperatures   0 -> 0.0, low -> 0.3, medium -> 0.7, one folder each:
                     <root>/temp_0, <root>/temp_low, <root>/temp_medium
      model specs    "gpt-oss:20b@high" = ollama model gpt-oss:20b sent with think="high".
                     The three reasoning levels run at temperature 0 only; at the other
                     temperatures gpt-oss runs once, at medium.
      seeds          temperature 0 is greedy, so a repeat gives the same reply: it runs
                     once, at seed 42. low and medium run at seeds 42..46, one folder each:
                     <root>/temp_<t>/seed_<n>, so a result is reported as mean +- std
                     across the 5 runs.
      result dirs    sanitize(spec): ':' and '/' -> '_', so "gpt-oss_20b@high"
"""

import re
from pathlib import Path

TEMPERATURES = {"0": 0.0, "low": 0.3, "medium": 0.7}

SEEDS = (42, 43, 44, 45, 46)
SEED_RE = re.compile(r"^seed_(\d+)$")

REASONING_LEVELS = ("low", "medium", "high")
REASONING_RE = re.compile(r"@(%s)$" % "|".join(REASONING_LEVELS))

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


def seeds_for(temp: str) -> tuple:
    return SEEDS[:1] if temp == "0" else SEEDS


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
