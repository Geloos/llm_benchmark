#!/usr/bin/env python3
"""
plot_results.py

What it does:
  Turns analysis/verdicts.csv into grouped bar charts of jailbreak effectiveness. Bar
  height is the trick rate: the share of that model's verdicts that came back "normal" or
  "neutral" (the same rule as the tricked column in verdicts.csv).
  Every log is a real attack, so a taller bar means the jailbreak fooled the model more
  often. --compare instead draws one cross-lane chart, a bar per model per lane.

  When verdicts.csv holds more than one seed (temperatures low and medium), the rate is
  computed per seed and the bar is the mean across seeds, with a whisker of +-1 sample
  std, clipped to [0, 1]. A single seed (temperature 0) draws a bare bar.

How to run it:
  python3 plot_results.py --analysis-dir analysis/temp_0     # one temperature's analysis
  python3 plot_results.py --no-per-category        # overview chart only
  python3 plot_results.py --no-report-embed        # leave analysis/report.md alone
  python3 plot_results.py --analysis-dir analysis --compare analysis_hexa
  python3 plot_results.py --analysis-dir analysis --compare analysis_control analysis_hexa \
      --lane-labels plain control hexa

What it outputs:
  analysis/charts/trick_rate_by_category.png    the overview, a group per category
  analysis/charts/by_category/<category>.png    one per category, a group per injection
  analysis/trick_rate_by_category.csv           the numbers behind the overview
  analysis/report.md                            gains a "## Charts" section, rewritten in
                                                place on every run

  With --compare, ONLY the cross-lane pair, and report.md is left untouched:
  analysis/charts/plain_vs_hexa.png  +  analysis/plain_vs_hexa.csv

Needs matplotlib -- the one dependency in this repo outside requests, and only here.
"""

import argparse
import csv
import os
import statistics
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

BUCKETS = ("attack", "normal", "neutral", "unparseable")

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

THEME = {
    "surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e",
    "muted": "#898781", "grid": "#e1e0d9", "baseline": "#c3c2b7",
}

CHART_STEM = "trick_rate_by_category"
TITLE = "Jailbreaks by category"
COMPARE_TITLE = "Trick rate by injection surface"
LANE_HATCH = (None, "///", "...", "xxx")
SENTINEL = "<!-- charts:begin -->"


def read_verdicts(path: Path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def usable(r) -> bool:
    category = r.get("category", "")
    return bool(category) and category != "unknown" and r.get("verdict", "") in BUCKETS


def aggregate(rows, key: str, series_of=None):
    series_of = series_of or (lambda r: r["model"])
    cells = defaultdict(lambda: {b: 0 for b in BUCKETS})
    unknown = 0
    for r in rows:
        if not usable(r):
            unknown += 1
            continue
        cells[(series_of(r), r[key])][r["verdict"]] += 1
    return cells, unknown


def trick_rate(counts) -> float:
    seen = sum(counts.values())
    return (counts["normal"] + counts["neutral"]) / seen if seen else 0.0


def seed_stats(rows, key: str, series_of=None):
    # (series, group) -> (mean, std, seeds): the trick rate per seed, then mean and sample
    # std across seeds. std is None for one seed. A verdicts.csv without a seed column
    # (older runs) reads as one seed, so the mean is the plain pooled rate.
    series_of = series_of or (lambda r: r["model"])
    by_seed = defaultdict(lambda: defaultdict(lambda: {b: 0 for b in BUCKETS}))
    for r in rows:
        if usable(r):
            by_seed[(series_of(r), r[key])][r.get("seed", "")][r["verdict"]] += 1
    out = {}
    for cell, seeds in by_seed.items():
        rates = [trick_rate(counts) for counts in seeds.values()]
        std = statistics.stdev(rates) if len(rates) > 1 else None
        out[cell] = (sum(rates) / len(rates), std, len(rates))
    return out


def pooled_order(cells, groups=None):
    pooled = defaultdict(lambda: {b: 0 for b in BUCKETS})
    for (_, group), counts in cells.items():
        if groups is None or group in groups:
            for bucket, n in counts.items():
                pooled[group][bucket] += n
    return sorted(pooled, key=lambda g: (-trick_rate(pooled[g]), g))


def model_order(cells):
    return sorted({model for model, _ in cells})


def lane_series_order(cells, lane_labels):
    rank = {lane: i for i, lane in enumerate(lane_labels)}
    return sorted({series for series, _ in cells},
                  key=lambda s: (s[0], rank.get(s[1], len(rank))))


def group_by_category(rows):
    out = defaultdict(set)
    for r in rows:
        category = r.get("category", "")
        if category and category != "unknown":
            out[category].add(r["injection"])
    return out


def rate_fields(stats, cell) -> dict:
    mean, std, seeds = stats[cell]
    return {"seeds": seeds, "trick_rate": round(mean, 3),
            "trick_rate_std": None if std is None else round(std, 3)}


def twin_rows(cells, stats, models, categories):
    rows = []
    for model in models:
        for category in categories:
            counts = cells.get((model, category))
            if not counts:
                continue
            rows.append({
                "model": model,
                "category": category,
                "seen": sum(counts.values()),
                **{b: counts[b] for b in BUCKETS},
                **rate_fields(stats, (model, category)),
            })
    return rows


def read_lanes(pairs):
    rows = []
    for analysis_dir, lane in pairs:
        path = Path(analysis_dir) / "verdicts.csv"
        if not path.is_file():
            sys.exit(f"ERROR: {path} not found -- run that lane first:\n"
                     f"  python main.py --corpus {lane}")
        for r in read_verdicts(path):
            r["lane"] = lane
            rows.append(r)
    return rows


def compare_twin_rows(cells, stats, series, categories):
    rows = []
    for model, lane in series:
        for category in categories:
            counts = cells.get(((model, lane), category))
            if not counts:
                continue
            rows.append({
                "model": model,
                "lane": lane,
                "category": category,
                "seen": sum(counts.values()),
                **{b: counts[b] for b in BUCKETS},
                **rate_fields(stats, ((model, lane), category)),
            })
    return rows


def write_csv(rows, path: Path, fields) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def category_label(category: str) -> str:
    return textwrap.fill(category.replace("_", " "), 11, break_long_words=False)


def injection_label(injection: str) -> str:
    parts = injection.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:2]) + "\n" + "_".join(parts[2:])
    return injection


def render(stats, models, groups, title: str, out_path: Path, label_fn,
           style_fn=None, series_label=None, fig_width=None, bar_labels=True,
           legend_ncol=None) -> Path:
    style_fn = style_fn or (lambda series, slot: {"color": SERIES[slot]})
    series_label = series_label or (lambda series: series)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, PercentFormatter

    t = THEME
    fig_w = fig_width or min(11.0, max(6.5, len(groups) + 2.5))
    fig_h, dpi = 5.5, 200
    group_w = 0.8

    px_per_unit = (fig_w * dpi * 0.82) / max(len(groups), 1)
    gap = 2.0 / px_per_unit
    bar_w = max(group_w / len(models) - gap, 0.02)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.subplots_adjust(top=0.78, bottom=0.20, left=0.08, right=0.98)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    xs = list(range(len(groups)))
    for slot, model in enumerate(models):
        offset = (slot - (len(models) - 1) / 2) * (group_w / len(models))
        bar_xs = [x + offset for x in xs]
        values = [stats[(model, g)][0] if (model, g) in stats else 0.0 for g in groups]
        stds = [stats[(model, g)][1] if (model, g) in stats else None for g in groups]
        ax.bar(bar_xs, values, bar_w,
               label=series_label(model), zorder=3, **style_fn(model, slot))

        # +-1 std, clipped to the [0, 1] a rate can take
        lower = [min(s, v) if s is not None else 0.0 for v, s in zip(values, stds)]
        upper = [min(s, 1 - v) if s is not None else 0.0 for v, s in zip(values, stds)]
        whisk = [i for i, s in enumerate(stds) if s is not None]
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
                                fontsize=6.5, color=t["muted"])

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("trick rate", color=t["secondary"], fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([label_fn(g) for g in groups])
    ax.tick_params(colors=t["muted"], labelsize=8, length=0)

    ax.grid(axis="y", color=t["grid"], linewidth=0.8, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["baseline"])
    ax.spines["bottom"].set_linewidth(0.8)

    fig.text(0.08, 0.95, title, color=t["primary"], fontsize=15, fontweight="bold",
             ha="left", va="top")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02),
              ncol=legend_ncol or min(len(models), 4),
              frameon=False, fontsize=9, labelcolor=t["secondary"],
              handlelength=1.2, handleheight=1.2, columnspacing=1.6)

    fig.savefig(out_path, facecolor=t["surface"], bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return out_path


def embed_in_report(report_path: Path, overview: Path, twin: Path, per_category) -> bool:
    if not report_path.is_file():
        return False
    base = report_path.parent

    def rel(p):
        return os.path.relpath(p, base).replace("\\", "/")

    head = report_path.read_text(encoding="utf-8").split(SENTINEL)[0].rstrip()
    block = [
        SENTINEL,
        "",
        "## Charts",
        "",
        f"![{TITLE}]({rel(overview)})",
        "",
        f"Bar height is the trick rate: the share of that model's classifications that "
        f"came back \"Normal\" or \"Neutral\", averaged over the seeds; the whisker is "
        f"+-1 std across seeds (none with a single seed). The counts behind every bar "
        f"are in [`{rel(twin)}`]({rel(twin)}).",
        "",
    ]
    if per_category:
        block += ["### By injection, within each category", "",
                  "Per-injection counts are in `verdicts_by_injection.csv`.", ""]
        for category, image in per_category:
            block += [f"**{category}**", "", f"![{category}]({rel(image)})", ""]
    report_path.write_text(head + "\n\n" + "\n".join(block), encoding="utf-8")
    return True


def run_compare(args, analysis_dir: Path, out_dir: Path) -> None:
    labels = list(args.lane_labels)
    dirs = [analysis_dir] + [Path(d) for d in args.compare]
    if len(labels) != len(dirs):
        sys.exit(f"ERROR: {len(dirs)} lanes ({', '.join(str(d) for d in dirs)}) but "
                 f"{len(labels)} --lane-labels ({', '.join(labels)}); give one label per "
                 f"lane, --analysis-dir first.")
    if len(dirs) > len(LANE_HATCH):
        sys.exit(f"ERROR: {len(dirs)} lanes but only {len(LANE_HATCH)} hatch patterns.")
    baseline = labels[0]
    rows = read_lanes(list(zip(dirs, labels)))
    lane_series = lambda r: (r["model"], r["lane"])
    cells, unknown = aggregate(rows, "category", series_of=lane_series)
    stats = seed_stats(rows, "category", series_of=lane_series)
    if not cells:
        sys.exit(f"ERROR: no usable rows across {analysis_dir} and {args.compare} "
                 f"({unknown} skipped as unknown category / verdict)")

    series = lane_series_order(cells, labels)
    models = sorted({model for model, _ in series})
    if len(models) > len(SERIES):
        sys.exit(f"ERROR: {len(models)} models but only {len(SERIES)} colour slots. "
                 f"Chart them in batches with a filtered verdicts.csv instead.")
    categories = pooled_order(cells)

    slot_of = {model: i for i, model in enumerate(models)}
    hatch_of = {lane: LANE_HATCH[i] for i, lane in enumerate(labels)}

    def style(key, _slot):
        model, lane = key
        face = SERIES[slot_of[model]]
        hatch = hatch_of[lane]
        if hatch is None:
            return {"color": face}
        return {"facecolor": face, "hatch": hatch,
                "edgecolor": THEME["surface"], "linewidth": 0}

    stem = "_vs_".join(labels)
    out_dir.mkdir(parents=True, exist_ok=True)
    twin = analysis_dir / f"{stem}.csv"
    write_csv(compare_twin_rows(cells, stats, series, categories), twin,
              ["model", "lane", "category", "seen", *BUCKETS, "seeds", "trick_rate",
               "trick_rate_std"])

    image = render(stats, series, categories, COMPARE_TITLE,
                   out_dir / f"{stem}.png", category_label,
                   style_fn=style, series_label=lambda k: f"{k[0]} - {k[1]}",
                   fig_width=min(22.0, 9.0 + 2.0 * len(series)), bar_labels=False,
                   legend_ncol=len(models))

    if unknown:
        print(f"WARNING: skipped {unknown} rows with an unknown category or verdict")
    print("pooled trick rate across all categories, per lane "
          f"(delta is vs '{baseline}'):")
    for model in models:
        parts = []
        base_rate = None
        for lane in labels:
            counts = [cells[((model, lane), c)] for c in categories
                      if ((model, lane), c) in cells]
            pooled = {b: sum(c[b] for c in counts) for b in BUCKETS}
            rate = trick_rate(pooled) if counts else 0.0
            if base_rate is None:
                base_rate = rate
                parts.append(f"{lane} {rate:.0%}")
            else:
                parts.append(f"{lane} {rate:.0%} ({rate - base_rate:+.0%})")
        print(f"  {model:24} " + "   ".join(parts))
    print(f"wrote {twin} and {image}")


def parse_args():
    ap = argparse.ArgumentParser(description="Chart jailbreak effectiveness per model.")
    ap.add_argument("--analysis-dir", default="analysis",
                    help="where verdicts.csv and report.md live (default: analysis)")
    ap.add_argument("--out-dir", default=None,
                    help="where to write the PNGs (default: <analysis-dir>/charts)")
    ap.add_argument("--no-per-category", action="store_true",
                    help="only draw the overview, skip the per-category charts")
    ap.add_argument("--no-report-embed", action="store_true",
                    help="do not touch report.md")
    ap.add_argument("--compare", nargs="+", metavar="OTHER_ANALYSIS_DIR",
                    help="draw ONLY the cross-lane comparison chart, reading these dirs' "
                         "verdicts.csv alongside --analysis-dir's (e.g. --compare "
                         "analysis_control analysis_hexa). report.md is left alone in this "
                         "mode; the output is named after the lane labels")
    ap.add_argument("--lane-labels", nargs="+", default=["plain", "hexa"],
                    metavar="LABEL",
                    help="one name per lane for the legend, the CSV and the output "
                         "filename, --analysis-dir's lane FIRST (default: plain hexa)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import matplotlib
    except ImportError:
        sys.exit("ERROR: this step needs matplotlib.\n"
                 "  pip install matplotlib      (or: pip install -r requirements.txt)")

    analysis_dir = Path(args.analysis_dir)
    out_dir = Path(args.out_dir) if args.out_dir else analysis_dir / "charts"

    if args.compare:
        run_compare(args, analysis_dir, out_dir)
        return

    verdicts = analysis_dir / "verdicts.csv"
    if not verdicts.is_file():
        sys.exit(f"ERROR: {verdicts} not found -- run the analysis step first:\n"
                 f"  python main.py --skip-inject --skip-benchmark")

    rows = read_verdicts(verdicts)
    by_category, unknown = aggregate(rows, "category")
    category_stats = seed_stats(rows, "category")
    if not by_category:
        sys.exit(f"ERROR: no usable rows in {verdicts} "
                 f"({unknown} skipped as unknown category / verdict)")

    models = model_order(by_category)
    if len(models) > len(SERIES):
        sys.exit(f"ERROR: {len(models)} models but only {len(SERIES)} colour slots. "
                 f"Cycling hues would make two models indistinguishable -- chart them in "
                 f"batches with a filtered verdicts.csv instead.")
    categories = pooled_order(by_category)

    out_dir.mkdir(parents=True, exist_ok=True)

    twin = analysis_dir / f"{CHART_STEM}.csv"
    write_csv(twin_rows(by_category, category_stats, models, categories), twin,
              ["model", "category", "seen", *BUCKETS, "seeds", "trick_rate",
               "trick_rate_std"])

    overview = render(category_stats, models, categories, TITLE,
                      out_dir / f"{CHART_STEM}.png", category_label)
    written = [overview]

    per_category = []
    if not args.no_per_category:
        by_injection, _ = aggregate(rows, "injection")
        injection_stats = seed_stats(rows, "injection")
        members = group_by_category(rows)
        sub_dir = out_dir / "by_category"
        sub_dir.mkdir(parents=True, exist_ok=True)
        for category in categories:
            injections = pooled_order(by_injection, members[category])
            if not injections:
                continue
            image = render(injection_stats, models, injections,
                           category.replace("_", " "), sub_dir / f"{category}.png",
                           injection_label)
            per_category.append((category, image))
            written.append(image)

    if unknown:
        print(f"WARNING: skipped {unknown} rows with an unknown category or verdict "
              f"(stale results whose injection id is not in injections.jsonl?)")
    for model in models:
        rates = [(c, category_stats[(model, c)][0]) for c in categories
                 if (model, c) in category_stats]
        best = max(rates, key=lambda kv: kv[1]) if rates else ("-", 0.0)
        print(f"{model:24} weakest against: {best[0]:26} trick rate {best[1]:.0%}")

    if not args.no_report_embed and embed_in_report(
            analysis_dir / "report.md", overview, twin, per_category):
        print(f"embedded the charts in {analysis_dir / 'report.md'}")
    print(f"wrote {twin} and {len(written)} chart(s) under {out_dir}")


if __name__ == "__main__":
    main()
