#!/usr/bin/env python3
"""
plot_results.py

What it does:
  Turns an analysis folder's calls.jsonl (summarize_results.py) into grouped bar charts of
  jailbreak effectiveness. Bar height is the trick rate: the share of that model's verdicts
  that came back anything but "attack" (the same rule as the tricked field in calls.jsonl).
  Every log is a real attack, so a taller bar means the jailbreak fooled the model more
  often.

  The rate is computed per seed and the bar is the mean across seeds (the ASR), with a
  whisker of its 95% confidence interval, t(0.975, n-1) * std / sqrt(n) (experiment.asr_stats,
  the same CI as asr.csv and report.md), clipped to [0, 1]. A single seed (e.g. a --seeds 42
  run) draws a bare bar.

  The default charts (the plain and hexa lanes, and analysis/analysis_reasoning/):
  - asr by model: one bar per model, pooled over every injection -- the overall ASR.
  - by category: a group per injection category, a bar per model.
  - per category: one chart per category, a group per injection in it.
  - top injections: one chart with a panel per model, each that model's TOP_N (5)
    strongest injections ranked by trick rate (ties broken by the smaller std, then the
    id), and the same panel again as one chart per model.
  A folder holding more than one gpt-oss level next to other models charts gpt-oss@low only
  (experiment.temperature_series). A folder holding only gpt-oss levels (analysis/analysis_reasoning/)
  charts all of them, @low / @medium / @high, in that order; gpt-oss@low keeps the hue it
  has in the temperature charts.

  --by-injection (the OB lane) draws one chart per category instead: a group per injection,
  a bar per model.

  --paired (the DOVSDOFS lane) draws one chart instead: a group per model on the x-axis,
  every model present (gpt-oss@medium included), two bars side by side: orange = the
  nomarker baseline (injection_set.BASE_VARIANT), blue = the marked variant (SYSTEM:).
  A second pair (DOVSDOFS_02) adds a group per model.

  --compare-temperatures (analysis/analysis_temperature/, compare_temperatures.py) draws from its
  temperature.csv instead of calls.jsonl:
  - temperature: a group per model, orange = low, blue = medium.

How to run it:
  python3 plot_results.py --analysis-dir analysis/analysis_plaintext/temp_low   # every chart, into charts/
  python3 plot_results.py --analysis-dir analysis/analysis_reasoning/temp_medium
  python3 plot_results.py --analysis-dir analysis/analysis_ob/temp_low --by-injection
  python3 plot_results.py --analysis-dir analysis/analysis_dovsdofs/temp_low --paired
  python3 plot_results.py --analysis-dir analysis/analysis_temperature --compare-temperatures

What it outputs (charts only; every number is in asr.csv):
  <analysis-dir>/charts/asr_by_model.pdf              one bar per model, the overall ASR
  <analysis-dir>/charts/trick_rate_by_category.pdf    the overview, a group per category
  <analysis-dir>/charts/by_category/<category>.pdf    one per category, a group per injection
  <analysis-dir>/charts/top_injections.pdf            a panel per model, its top injections
  <analysis-dir>/charts/top_injections/<model>.pdf    the same panel, one PDF per model
  <analysis-dir>/report.md                            gains a "## Charts" section linking every
                                                      chart, rewritten in place on every run
  Charts are large vector PDFs (sharp at any zoom; fonts embedded as TrueType), each with
  its title on top; a bar chart's value axis ends HEADROOM (30 points) above its tallest
  bar + whisker. A bar chart is laid out at min(11, max(6.5, groups + 2.5)) x 5.5 in, a
  multi-panel chart PANEL_W (4.6) in per panel, then drawn SCALE (1.5) x larger, with
  13-18 pt text. A model keeps its SERIES colour in every chart.
  With --by-injection, ONLY analysis/analysis_ob/temp_<t>/charts/<category>.pdf (obfuscation.pdf).
  With --paired, ONLY analysis/analysis_dovsdofs/temp_<t>/charts/<category>.pdf (dovsdofs.pdf).
  With --compare-temperatures, ONLY analysis/analysis_temperature/charts/temperature.pdf.

Needs matplotlib -- the one dependency in this repo outside requests, and only here.
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import experiment
import injection_set

BUCKETS = ("attack", "normal", "neutral", "benign", "unparseable", "error")
TRICKED_BUCKETS = ("normal", "neutral", "benign", "unparseable", "error")

DPI = 300
SCALE = 1.5        # every chart is drawn at its layout size x SCALE
HEADROOM = 0.30    # the value axis ends this far (30 points of ASR) above the tallest bar
STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
         "xtick.labelsize": 13, "ytick.labelsize": 13, "axes.labelsize": 14,
         "axes.titlesize": 14, "axes.titleweight": "bold", "legend.fontsize": 13,
         "figure.titlesize": 18, "figure.titleweight": "bold",
         "pdf.fonttype": 42}   # TrueType, not Type 3: selectable text
NOTE_SIZE = 11     # in-plot annotations: bar values
PANEL_W = 4.6      # inches per panel of a multi-panel chart (before SCALE)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

THEME = {
    "surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e",
    "muted": "#898781", "grid": "#e1e0d9", "baseline": "#c3c2b7",
}

CHART_STEM = "trick_rate_by_category"
TITLE = "Jailbreaks by category"
REASONING_TITLE = "gpt-oss reasoning effort, by category"
POOLED_STEM = "asr_by_model"
POOLED_TITLE = "ASR per model, every injection pooled"
TOP_STEM = "top_injections"
TOP_TITLE = "Strongest injections per model"
TOP_MODEL_TITLE = "Strongest injections"
TOP_N = 5
REASONING_HUE = {"high": SERIES[0]}
TEMPERATURE_STEM = "temperature"
TEMPERATURE_TITLE = "ASR at temperature low vs medium"
LEVEL_LABEL = {"low": "low (0.0, greedy)", "medium": "medium (recommended)"}
LEVEL_HUE = {"low": SERIES[1], "medium": SERIES[0]}
POOLED = "pooled"
SENTINEL = "<!-- charts:begin -->"
EXT = ".pdf"


def read_calls(path: Path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def usable(r) -> bool:
    category = r.get("category") or ""
    return bool(category) and category != "unknown" and r.get("verdict") in BUCKETS


def group_fn(key):
    return key if callable(key) else (lambda r: r[key])


def aggregate(rows, key, series_of=None):
    series_of = series_of or (lambda r: r["model"])
    group_of = group_fn(key)
    cells = defaultdict(lambda: {b: 0 for b in BUCKETS})
    unknown = 0
    for r in rows:
        if not usable(r):
            unknown += 1
            continue
        cells[(series_of(r), group_of(r))][r["verdict"]] += 1
    return cells, unknown


def trick_rate(counts) -> float:
    seen = sum(counts.values())
    return sum(counts[b] for b in TRICKED_BUCKETS) / seen if seen else 0.0


def cell_stats(rows, key, series_of=None):
    """(series, group) -> (asr, low, high, std): the ASR (mean over the seeds), its 95%
    t-interval and the std across seeds, via experiment.asr_stats -- as in asr.csv."""
    series_of = series_of or (lambda r: r["model"])
    group_of = group_fn(key)
    calls = defaultdict(list)
    for r in rows:
        if usable(r):
            calls[(series_of(r), group_of(r))].append(
                (r["log"], r.get("seed"), 1 if r["verdict"] in TRICKED_BUCKETS else 0))
    out = {}
    for cell, c in calls.items():
        st = experiment.asr_stats(c)
        out[cell] = (st["asr"], st["low"], st["high"], st["std"])
    return out


def whiskers(values, lows, highs):
    lower = [v - lo if lo is not None else 0.0 for v, lo in zip(values, lows)]
    upper = [hi - v if hi is not None else 0.0 for v, hi in zip(values, highs)]
    return lower, upper


def pooled_order(cells, groups=None):
    pooled = defaultdict(lambda: {b: 0 for b in BUCKETS})
    for (_, group), counts in cells.items():
        if groups is None or group in groups:
            for bucket, n in counts.items():
                pooled[group][bucket] += n
    return sorted(pooled, key=lambda g: (-trick_rate(pooled[g]), g))


def model_order(cells):
    return sorted({model for model, _ in cells})


def group_by_category(rows):
    out = defaultdict(set)
    for r in rows:
        category = r.get("category") or ""
        if category and category != "unknown":
            out[category].add(r["injection"])
    return out


def category_label(category: str) -> str:
    return textwrap.fill(category.replace("_", " "), 11, break_long_words=False)


def injection_label(injection: str) -> str:
    parts = injection.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:2]) + "\n" + "_".join(parts[2:])
    return injection


def bar_style(color: str) -> dict:
    return {"color": color}


def model_label(model: str) -> str:
    """gpt-oss_20b@medium (or gpt-oss:20b@medium) -> two lines, so three models fit side by
    side in a column."""
    return re.sub(r"[:_]", "\n", model, count=1)


def fig_size(size):
    """(w, h) layout inches -> the saved size, SCALE x larger."""
    return size[0] * SCALE, size[1] * SCALE


def render(stats, models, groups, title: str, out_path: Path, label_fn, style_fn=None,
           series_label=None, fig_width=None, bar_labels=True, legend_ncol=None) -> Path:
    style_fn = style_fn or (lambda series, slot: bar_style(SERIES[slot]))
    series_label = series_label or (lambda series: series)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    t = THEME
    fig_w, fig_h = fig_size((fig_width or min(11.0, max(6.5, len(groups) + 2.5)), 5.5))
    group_w = 0.8

    px_per_unit = (fig_w * DPI * 0.82) / max(len(groups), 1)
    gap = 2.0 / px_per_unit
    bar_w = max(group_w / len(models) - gap, 0.02)
    # A bar narrower than its "100%" label gets the label upright (~0.9 of the axes is data).
    upright = 72 * fig_w * 0.9 * bar_w / max(len(groups), 1) < 2.4 * NOTE_SIZE

    plt.rcParams.update(STYLE)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI, layout="constrained")
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    xs = list(range(len(groups)))
    tops = [0.0] * len(groups)
    for slot, model in enumerate(models):
        offset = (slot - (len(models) - 1) / 2) * (group_w / len(models))
        bar_xs = [x + offset for x in xs]
        values = [stats[(model, g)][0] if (model, g) in stats else 0.0 for g in groups]
        lows = [stats[(model, g)][1] if (model, g) in stats else None for g in groups]
        highs = [stats[(model, g)][2] if (model, g) in stats else None for g in groups]
        ax.bar(bar_xs, values, bar_w,
               label=series_label(model), zorder=3, **style_fn(model, slot))

        lower, upper = whiskers(values, lows, highs)
        tops = [max(top, v + up) for top, v, up in zip(tops, values, upper)]
        whisk = [i for i, lo in enumerate(lows) if lo is not None]
        if whisk:
            ax.errorbar([bar_xs[i] for i in whisk], [values[i] for i in whisk],
                        yerr=[[lower[i] for i in whisk], [upper[i] for i in whisk]],
                        fmt="none", ecolor=t["secondary"], elinewidth=0.8,
                        capsize=2, capthick=0.8,
                        zorder=4)
        if bar_labels:
            for x, v, up in zip(bar_xs, values, upper):
                if v > 0:
                    ax.annotate(f"{v:.0%}", (x, v + up), xytext=(0, 2),
                                textcoords="offset points", ha="center", va="bottom",
                                rotation=90 if upright else 0,
                                fontsize=NOTE_SIZE, color=t["muted"])

    # The axis stops HEADROOM above the tallest bar + whisker, not at 100%.
    ceiling = max(tops, default=0.0) + HEADROOM
    ax.set_ylim(0, min(1.0, ceiling))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("ASR (trick rate)", color=t["secondary"])
    ax.set_xticks(xs)
    ax.set_xticklabels([label_fn(g) for g in groups])
    ax.tick_params(colors=t["muted"], length=0)

    ax.grid(axis="y", color=t["grid"], linewidth=0.8, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["baseline"])
    ax.spines["bottom"].set_linewidth(0.8)

    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02),
              ncol=legend_ncol or min(len(models), 4),
              frameon=False, labelcolor=t["secondary"],
              handlelength=1.2, handleheight=1.2, columnspacing=1.6)

    return finish_figure(plt, fig, out_path, title)


def top_injections(stats, models, top_n: int):
    out = {}
    for model in models:
        ranked = sorted((g for m, g in stats if m == model),
                        key=lambda g: (-stats[(model, g)][0],
                                       stats[(model, g)][3] or 0.0, g))
        out[model] = ranked[:top_n]
    return out


def render_top(stats, top, color_of, title: str, out_path: Path) -> Path:
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    t = THEME
    models = list(top)
    # One scale for every panel, ending past the longest bar + whisker (room for its label).
    right = max((stats[(m, g)][2] if stats[(m, g)][2] is not None else stats[(m, g)][0]
                 for m in models for g in top[m]), default=0.0)
    right = min(1.12, right + HEADROOM + 0.07)
    ncols = min(len(models), 3)
    nrows = -(-len(models) // ncols)
    rows_per_panel = max((len(v) for v in top.values()), default=1)
    panel_h = 0.4 * rows_per_panel + 1.2
    plt, fig, axes = start_figure((PANEL_W * ncols, panel_h * nrows + 0.8), out_path,
                                  nrows, ncols)

    for i, ax in enumerate(axes):
        if i >= len(models):
            ax.set_visible(False)
            continue
        model = models[i]
        injections = top[model]
        values = [stats[(model, g)][0] for g in injections]
        lows = [stats[(model, g)][1] for g in injections]
        highs = [stats[(model, g)][2] for g in injections]
        ys = list(range(len(injections)))
        ax.barh(ys, values, 0.7, color=color_of(model), zorder=3)

        lower, upper = whiskers(values, lows, highs)
        whisk = [j for j, lo in enumerate(lows) if lo is not None]
        if whisk:
            ax.errorbar([values[j] for j in whisk], [ys[j] for j in whisk],
                        xerr=[[lower[j] for j in whisk], [upper[j] for j in whisk]],
                        fmt="none", ecolor=t["secondary"], elinewidth=0.8,
                        capsize=2, capthick=0.8, zorder=4)
        for y, v, up in zip(ys, values, upper):
            ax.annotate(f"{v:.0%}", (v + up, y), xytext=(3, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=NOTE_SIZE, color=t["muted"])

        ax.set_xlim(0, right)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_ylim(len(injections) - 0.5, -0.5)
        ax.set_yticks(ys)
        ax.set_yticklabels(injections)
        style_axes(ax, "x")
        ax.tick_params(axis="y", labelsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_title(model, loc="left", color=t["primary"], pad=6)

    return finish_figure(plt, fig, out_path, title)


def embed(report_path: Path, block_fn) -> bool:
    if not report_path.is_file():
        return False
    base = report_path.parent

    def rel(p):
        return os.path.relpath(p, base).replace("\\", "/")

    head = report_path.read_text(encoding="utf-8").split(SENTINEL)[0].rstrip()
    report_path.write_text(head + "\n\n" + "\n".join([SENTINEL, ""] + block_fn(rel)),
                           encoding="utf-8")
    return True


def embed_paired(report_path: Path, image: Path) -> bool:
    return embed(report_path, lambda rel: [
        "## Chart", "", f"- [{image.stem}]({rel(image)})", "",
        f"A group per model, two bars side by side: orange the "
        f"`{injection_set.BASE_VARIANT}` variant, blue the marked one. Bar = ASR mean "
        f"across the seeds, whisker = its 95% CI (t). The ASRs are in `asr.csv`.", ""])


def embed_by_injection(report_path: Path, images) -> bool:
    def block(rel):
        lines = ["## Chart", ""]
        for image in images:
            lines += [f"- [{image.stem}]({rel(image)})"]
        return lines + [
            "",
            "A group per injection, a bar per model. Bar = ASR mean across the seeds, "
            "whisker = its 95% CI (t), clipped to [0, 1]. The numbers are in `asr.csv`.", ""]
    return embed(report_path, block)


def embed_in_report(report_path: Path, pooled: Path, overview: Path, per_category,
                    top, reasoning: bool) -> bool:
    def block(rel):
        lines = [
            "## Charts", "",
            f"- [{POOLED_TITLE}]({rel(pooled)})",
            f"- [{REASONING_TITLE if reasoning else TITLE}]({rel(overview)})", "",
            "Bar height is the ASR (trick rate): the share of that model's classifications "
            "that came back anything but attack, averaged over the seeds; the whisker is "
            "the 95% CI across seeds, t(0.975, n-1) * std / sqrt(n), clipped to [0, 1] "
            "(none with a single seed). Every bar's number, per model, per category and "
            "per injection, is in `asr.csv`.", "",
        ]
        if per_category:
            lines += ["### By injection, within each category", ""]
            lines += [f"- [{category}]({rel(image)})" for category, image in per_category]
            lines.append("")
        image, model_dir = top
        lines += ["### Strongest injections per model", "",
                  f"- [{TOP_TITLE}]({rel(image)})", "",
                  f"Each panel is also a PDF of its own in `{rel(model_dir)}/`.", ""]
        return lines
    return embed(report_path, block)


def read_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def number(value):
    return None if value in ("", None) else float(value)


def paired_chart(category: str, rows, models, analysis_dir: Path, out_dir: Path):
    for r in rows:
        r["pair"], _, r["variant"] = r["injection"].rpartition("_")
        r["pair_model"] = (r["pair"], r["model"])
    rows = [r for r in rows if r["model"] in models]
    series_of = lambda r: r["variant"]
    cells, _ = aggregate(rows, "pair_model", series_of=series_of)
    if not cells:
        return None
    stats = cell_stats(rows, "pair_model", series_of=series_of)

    variants = sorted({v for v, _ in cells},
                      key=lambda v: (v != injection_set.BASE_VARIANT, v))
    hues = [SERIES[1], SERIES[0], *SERIES[2:]]
    if len(variants) > len(hues):
        sys.exit(f"ERROR: {category} has {len(variants)} variants per pair but only "
                 f"{len(hues)} colours.")
    hue_of = {v: hues[i] for i, v in enumerate(variants)}
    pairs = sorted({pair for _, (pair, _) in cells})
    present = [m for m in models if any(model == m for _, (_, model) in cells)]
    groups = [(p, m) for p in pairs for m in present if any((v, (p, m)) in cells
                                                              for v in variants)]

    label = ((lambda g: model_label(g[1])) if len(pairs) == 1
             else (lambda g: f"{model_label(g[1])}\n{g[0]}"))
    title = injection_set.PAIRED_CATEGORIES.get(category, category)
    image = render(stats, variants, groups, title, out_dir / f"{category}{EXT}", label,
                   style_fn=lambda v, _slot: bar_style(hue_of[v]))

    print(f"{category} -- delta is vs '{injection_set.BASE_VARIANT}':")
    for p, m in groups:
        rates = [(v, stats[(v, (p, m))][0]) for v in variants if (v, (p, m)) in stats]
        base = dict(rates).get(injection_set.BASE_VARIANT)
        parts = [f"{v} {rate:.0%}" + ("" if base is None or v == injection_set.BASE_VARIANT
                                      else f" ({rate - base:+.0%})")
                 for v, rate in rates]
        print(f"  {m:24} {p:14} " + "   ".join(parts))
    return image


def by_category_rows(rows):
    out = defaultdict(list)
    for r in rows:
        out[r.get("category") or ""].append(r)
    return out


def run_paired(analysis_dir: Path, rows, out_dir: Path) -> None:
    drawn = []
    for category, cat_rows in sorted(by_category_rows(rows).items()):
        models = sorted({r["model"] for r in cat_rows}, key=experiment.model_sort_key)
        image = paired_chart(category, cat_rows, models, analysis_dir, out_dir)
        if image:
            drawn.append(image)
    if not drawn:
        sys.exit(f"ERROR: no usable rows in {analysis_dir / 'calls.jsonl'}")
    if embed_paired(analysis_dir / "report.md", drawn[0]):
        print(f"embedded the chart in {analysis_dir / 'report.md'}")
    print(f"wrote {', '.join(str(image) for image in drawn)}")


def run_by_injection(analysis_dir: Path, rows, out_dir: Path) -> None:
    models = sorted({r["model"] for r in rows if usable(r)}, key=experiment.model_sort_key)
    if len(models) > len(SERIES):
        sys.exit(f"ERROR: {len(models)} models but only {len(SERIES)} colour slots.")
    drawn = []
    for category, cat_rows in sorted(by_category_rows(rows).items()):
        cells, _ = aggregate(cat_rows, "injection")
        if not cells:
            continue
        stats = cell_stats(cat_rows, "injection")
        present = [m for m in models if any(model == m for model, _ in cells)]
        image = render(stats, present, pooled_order(cells),
                       category.replace("_", " ").capitalize() + ", by injection",
                       out_dir / f"{category}{EXT}", injection_label,
                       style_fn=lambda m, _slot: bar_style(SERIES[models.index(m)]))
        drawn.append(image)
        for injection in pooled_order(cells):
            parts = [f"{m} {stats[(m, injection)][0]:.0%}" for m in present
                     if (m, injection) in stats]
            print(f"  {injection:28} " + "   ".join(parts))
    if not drawn:
        sys.exit(f"ERROR: no usable rows in {analysis_dir / 'calls.jsonl'}")
    if embed_by_injection(analysis_dir / "report.md", drawn):
        print(f"embedded the chart in {analysis_dir / 'report.md'}")
    print(f"wrote {', '.join(str(image) for image in drawn)}")


def embed_compare(report_path: Path, bars: Path) -> bool:
    return embed(report_path, lambda rel: [
        "## Chart", "",
        f"- [{TEMPERATURE_TITLE}]({rel(bars)})", "",
        f"{bars.stem}: a group per model, orange = low, blue = medium. Bar = ASR mean "
        f"across the seeds, whisker = its 95% CI (t).", ""])


def style_axes(ax, grid_axis: str) -> None:
    t = THEME
    ax.set_facecolor(t["surface"])
    ax.tick_params(colors=t["muted"], length=0)
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=0.8, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["baseline"])
        ax.spines[side].set_linewidth(0.8)


def start_figure(size, out_path: Path, nrows: int = 1, ncols: int = 1):
    """(plt, fig, axes flat) at fig_size(size) inches, DPI, STYLE."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(nrows, ncols, figsize=fig_size(size), dpi=DPI,
                             squeeze=False, layout="constrained")
    fig.patch.set_facecolor(THEME["surface"])
    for ax in axes.flat:
        ax.set_facecolor(THEME["surface"])
    return plt, fig, list(axes.flat)


def finish_figure(plt, fig, out_path: Path, title: str) -> Path:
    """The title on top of the figure; saved at exactly the figure's size (constrained
    layout already fits the labels inside)."""
    fig.suptitle(title, color=THEME["primary"])
    fig.savefig(out_path, facecolor=THEME["surface"], dpi=DPI)
    plt.close(fig)
    return out_path


def run_compare(analysis_dir: Path, out_dir: Path) -> None:
    path = analysis_dir / "temperature.csv"
    if not path.is_file():
        sys.exit(f"ERROR: {path} not found -- run compare_temperatures.py first")
    rows = read_csv(path)
    models = [r["model"] for r in rows]

    levels = list(experiment.TEMPERATURE_LEVELS)
    stats = {(level, r["model"]): (float(r[f"{level}_asr"]), number(r[f"{level}_low"]),
                                   number(r[f"{level}_high"]), None)
             for r in rows for level in levels}
    bars = render(stats, levels, models, TEMPERATURE_TITLE,
                  out_dir / f"{TEMPERATURE_STEM}{EXT}", model_label,
                  style_fn=lambda level, _slot: bar_style(LEVEL_HUE[level]),
                  series_label=lambda level: LEVEL_LABEL[level])

    for r in rows:
        print(f"  {r['model']:24} low {float(r['low_asr']):.0%}  "
              f"medium {float(r['medium_asr']):.0%}")
    if embed_compare(analysis_dir / "report.md", bars):
        print(f"embedded the chart in {analysis_dir / 'report.md'}")
    print(f"wrote {bars}")


def series_palette(all_models):
    """(models to chart, reasoning folder?, color_of). A folder of gpt-oss levels only
    charts all of them; otherwise the temperature series. Hue slots: the temperature
    series alphabetically (gpt-oss@low beside llama3.1 and gemma3, even in the reasoning
    folder where those two are absent), then the extra gpt-oss levels -- so a model has the
    same colour in every chart."""
    levels = experiment.reasoning_series(all_models)
    reasoning = bool(levels) and levels == sorted(all_models, key=experiment.model_sort_key)
    models = levels if reasoning else experiment.temperature_series(all_models)
    if len(models) > len(SERIES):
        sys.exit(f"ERROR: {len(models)} models but only {len(SERIES)} colour slots. "
                 f"Cycling hues would make two models indistinguishable -- chart them in "
                 f"batches with a narrower results tree instead.")
    first = set(experiment.temperature_series(all_models))
    if reasoning:
        first |= {experiment.sanitize(m) for m in experiment.OTHER_MODELS}
    order = sorted(first) + sorted((m for m in all_models if m not in first),
                                   key=experiment.model_sort_key)
    slot_of = {model: i for i, model in enumerate(order)}

    def color_of(model):
        hue = REASONING_HUE.get(experiment.reasoning_level(model)) if reasoning else None
        return hue or SERIES[slot_of[model] % len(SERIES)]

    return models, reasoning, color_of


def run_default(analysis_dir: Path, rows, out_dir: Path) -> None:
    by_category, unknown = aggregate(rows, "category")
    if not by_category:
        sys.exit(f"ERROR: no usable rows in {analysis_dir / 'calls.jsonl'} "
                 f"({unknown} skipped as unknown category / verdict)")

    models, reasoning, color_of = series_palette(model_order(by_category))
    hue = lambda model, _slot: bar_style(color_of(model))
    rows = [r for r in rows if r["model"] in models]
    category_stats = cell_stats(rows, "category")
    categories = pooled_order({k: v for k, v in by_category.items() if k[0] in models})
    out_dir.mkdir(parents=True, exist_ok=True)

    pooled_stats = cell_stats(rows, lambda r: POOLED)
    n_injections = len({r["injection"] for r in rows if usable(r)})
    pooled = render(pooled_stats, models, [POOLED], POOLED_TITLE,
                    out_dir / f"{POOLED_STEM}{EXT}",
                    lambda _g: f"all {n_injections} injections", style_fn=hue)
    overview = render(category_stats, models, categories,
                      REASONING_TITLE if reasoning else TITLE,
                      out_dir / f"{CHART_STEM}{EXT}", category_label, style_fn=hue)
    written = [pooled, overview]

    by_injection, _ = aggregate(rows, "injection")
    injection_stats = cell_stats(rows, "injection")
    members = group_by_category(rows)

    per_category = []
    sub_dir = out_dir / "by_category"
    sub_dir.mkdir(parents=True, exist_ok=True)
    for category in categories:
        injections = pooled_order(by_injection, members[category])
        if not injections:
            continue
        image = render(injection_stats, models, injections,
                       category.replace("_", " ").capitalize(),
                       sub_dir / f"{category}{EXT}", injection_label, style_fn=hue)
        per_category.append((category, image))
        written.append(image)

    panels = sorted(models, key=experiment.model_sort_key)
    ranked = top_injections(injection_stats, panels, TOP_N)
    image = render_top(injection_stats, ranked, color_of, TOP_TITLE,
                       out_dir / f"{TOP_STEM}{EXT}")
    written.append(image)
    model_dir = out_dir / TOP_STEM
    model_dir.mkdir(parents=True, exist_ok=True)
    for model in panels:
        written.append(render_top(injection_stats, {model: ranked[model]}, color_of,
                                  TOP_MODEL_TITLE, model_dir / f"{model}{EXT}"))

    if unknown:
        print(f"WARNING: skipped {unknown} rows with an unknown category or verdict "
              f"(stale results whose injection id is not in the injections jsonl?)")
    for model in models:
        mean = pooled_stats[(model, POOLED)][0]
        rates = [(c, category_stats[(model, c)][0]) for c in categories
                 if (model, c) in category_stats]
        best = max(rates, key=lambda kv: kv[1]) if rates else ("-", 0.0)
        print(f"{model:24} ASR {mean:.0%}   weakest against: {best[0]:26} "
              f"trick rate {best[1]:.0%}")

    if embed_in_report(analysis_dir / "report.md", pooled, overview, per_category,
                       (image, model_dir), reasoning):
        print(f"embedded the charts in {analysis_dir / 'report.md'}")
    print(f"wrote {len(written)} chart(s) under {out_dir}")


def parse_args():
    ap = argparse.ArgumentParser(description="Chart jailbreak effectiveness per model.")
    ap.add_argument("--analysis-dir", default="analysis/analysis_plaintext/temp_medium",
                    help="where calls.jsonl and report.md live (default: analysis/analysis_plaintext/temp_medium)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--paired", action="store_true",
                      help="draw ONLY the pair chart (the DOVSDOFS lane): a group per "
                           "model, orange = nomarker, blue = the marked variant, every "
                           "model present")
    mode.add_argument("--by-injection", action="store_true",
                      help="draw ONLY one chart per category (the OB lane): a group per "
                           "injection, a bar per model")
    mode.add_argument("--compare-temperatures", action="store_true",
                      help="draw ONLY the low-vs-medium chart from compare_temperatures.py's "
                           "temperature.csv in --analysis-dir "
                           "(analysis/analysis_temperature)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import matplotlib
    except ImportError:
        sys.exit("ERROR: this step needs matplotlib.\n"
                 "  pip install matplotlib      (or: pip install -r requirements.txt)")
    # Embedding fonts in the PDFs logs "ASCP NOT subset ..." per font table it skips; harmless.
    logging.getLogger("fontTools").setLevel(logging.ERROR)

    analysis_dir = Path(args.analysis_dir)
    out_dir = analysis_dir / "charts"
    if args.compare_temperatures:
        out_dir.mkdir(parents=True, exist_ok=True)
        run_compare(analysis_dir, out_dir)
        return
    calls = analysis_dir / "calls.jsonl"
    if not calls.is_file():
        sys.exit(f"ERROR: {calls} not found -- run the analysis step first:\n"
                 f"  python main.py   (stages already done are skipped)")
    rows = read_calls(calls)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.paired:
        run_paired(analysis_dir, rows, out_dir)
    elif args.by_injection:
        run_by_injection(analysis_dir, rows, out_dir)
    else:
        run_default(analysis_dir, rows, out_dir)


if __name__ == "__main__":
    main()
