#!/usr/bin/env python3
"""
plot_results.py

What it does:
  Turns analysis/verdicts.csv into a grouped bar chart of jailbreak effectiveness: one
  group per injection category, one bar per model, bar height = the share of that model's
  classifications for that category that came back "normal". Every attack_log is a real
  attack, so a taller bar means that family of jailbreak fooled that model more often.
  Rates (not raw counts) because coverage is not guaranteed even across models.

How to run it:
  python3 plot_results.py --analysis-dir analysis --out-dir analysis/charts --theme both
  python3 plot_results.py --no-report-embed        # leave analysis/report.md alone

What it outputs:
  analysis/charts/trick_rate_by_category.png       light render
  analysis/charts/trick_rate_by_category_dark.png  dark render -- a PNG cannot follow the
                                                   viewer's theme, so both get drawn
  analysis/trick_rate_by_category.csv              the numbers behind the chart:
                                                   model,category,seen,attack,normal,
                                                   neutral,unparseable,trick_rate
  analysis/report.md                               gains a "## Charts" section, rewritten
                                                   in place on every run (never appended
                                                   twice)

Needs matplotlib -- the one dependency in this repo outside requests, and only here.
"""

import argparse
import csv
import os
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

# Same buckets summarize_results.py writes into the verdict column.
BUCKETS = ("attack", "normal", "neutral", "unparseable")

# Categorical slots in fixed order: slot N always belongs to the Nth model by name, so a
# model keeps its hue no matter where it ranks or which models are filtered out. Hues and
# ordering are the validated set -- do not re-order or extend past 8 without re-validating
# (worst adjacent CVD dE 9.1 light / 8.4 dark).
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
               "#d55181", "#008300", "#9085e9", "#e66767"]

THEMES = {
    "light": {
        "suffix": "", "surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "baseline": "#c3c2b7", "series": SERIES_LIGHT,
    },
    "dark": {
        "suffix": "_dark", "surface": "#1a1a19", "primary": "#ffffff", "secondary": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "baseline": "#383835", "series": SERIES_DARK,
    },
}

CHART_STEM = "trick_rate_by_category"
TITLE = "Jailbreak effectiveness by category"

SENTINEL = "<!-- charts:begin -->"


def read_verdicts(path: Path):
    """The flat matrix written by summarize_results.py, one dict per model x log x
    injection: model,category,injection,log,verdict,tricked."""
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def aggregate(rows):
    """(model, category) -> per-bucket counts. 'unknown' categories are dropped (stale
    result files whose injection id is no longer in injections.jsonl) and counted so the
    caller can warn about them."""
    cells = defaultdict(lambda: {b: 0 for b in BUCKETS})
    unknown = 0
    for r in rows:
        category = r.get("category", "")
        if category == "unknown" or not category:
            unknown += 1
            continue
        verdict = r.get("verdict", "")
        if verdict not in BUCKETS:
            unknown += 1
            continue
        cells[(r["model"], category)][verdict] += 1
    return cells, unknown


def trick_rate(counts) -> float:
    """Share of classifications the model answered "normal" -- i.e. the injection worked.
    attack / neutral / unparseable all stay in the denominator, matching the `tricked`
    rule in summarize_results.py."""
    seen = sum(counts.values())
    return counts["normal"] / seen if seen else 0.0


def axes_order(cells):
    """Models alphabetically (their colour slot follows that fixed order, not their rank);
    categories by pooled trick rate across all models, best jailbreak first."""
    models = sorted({model for model, _ in cells})
    pooled = defaultdict(lambda: {b: 0 for b in BUCKETS})
    for (_, category), counts in cells.items():
        for bucket, n in counts.items():
            pooled[category][bucket] += n
    categories = sorted(pooled, key=lambda c: (-trick_rate(pooled[c]), c))
    return models, categories


def twin_rows(cells, models, categories):
    """The table twin behind the chart -- every plotted bar as a row of raw counts, so no
    value is reachable only by reading a colour off the picture."""
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
                "trick_rate": round(trick_rate(counts), 3),
            })
    return rows


def write_csv(rows, path: Path, fields) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _tick_label(category: str) -> str:
    """direct_override -> 'direct\\noverride'. Wrapping beats rotating: 10 groups of
    upright text stay readable, angled text does not. Never split a word -- a category
    that reads 'context man / ipulation' is worse than one wide line."""
    return textwrap.fill(category.replace("_", " "), 11, break_long_words=False)


def render(cells, models, categories, theme_name: str, out_path: Path) -> Path:
    """Draw one grouped bar chart for one theme. Returns the file written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, PercentFormatter

    t = THEMES[theme_name]
    fig_w, fig_h, dpi = 11.0, 5.5, 200
    group_w = 0.8

    # A 2px surface gap between adjacent bars instead of a border around each. The plot
    # occupies ~82% of the figure width, so one category unit is that many pixels wide.
    px_per_unit = (fig_w * dpi * 0.82) / max(len(categories), 1)
    gap = 2.0 / px_per_unit
    bar_w = max(group_w / len(models) - gap, 0.02)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.subplots_adjust(top=0.74, bottom=0.20, left=0.07, right=0.98)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    xs = list(range(len(categories)))
    for slot, model in enumerate(models):
        offset = (slot - (len(models) - 1) / 2) * (group_w / len(models))
        values = [trick_rate(cells[(model, c)]) if (model, c) in cells else 0.0
                  for c in categories]
        bars = ax.bar([x + offset for x in xs], values, bar_w,
                      label=model, color=t["series"][slot], zorder=3)
        # Selective labels: a 0% bar has nothing to say and the label would sit on the axis.
        ax.bar_label(bars, labels=[f"{v:.0%}" if v > 0 else "" for v in values],
                     padding=2, fontsize=6.5, color=t["muted"])

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("trick rate", color=t["secondary"], fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([_tick_label(c) for c in categories])
    ax.tick_params(colors=t["muted"], labelsize=8, length=0)

    # Recessive chrome: solid horizontal hairlines only, behind the bars.
    ax.grid(axis="y", color=t["grid"], linewidth=0.8, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["baseline"])
    ax.spines["bottom"].set_linewidth(0.8)

    total = sum(sum(c.values()) for c in cells.values())
    fig.text(0.07, 0.945, TITLE, color=t["primary"], fontsize=15, fontweight="bold",
             ha="left", va="top")
    fig.text(0.07, 0.895,
             f"Share of each model's classifications answered \"Normal\" "
             f"({total} in total across {len(categories)} categories). "
             f"Every log is a real attack, so higher = the jailbreak worked.",
             color=t["secondary"], fontsize=9, ha="left", va="top")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=min(len(models), 4),
              frameon=False, fontsize=9, labelcolor=t["secondary"],
              handlelength=1.2, handleheight=1.2, columnspacing=1.6)

    fig.savefig(out_path, facecolor=t["surface"], bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return out_path


def embed_in_report(report_path: Path, image: Path, twin: Path) -> bool:
    """Add (or refresh) the '## Charts' section of report.md. summarize_results.py rewrites
    that file wholesale, so this re-attaches on every run; the sentinel keeps it to one
    section however many times either script runs."""
    if not report_path.is_file():
        return False
    base = report_path.parent
    img_rel = os.path.relpath(image, base).replace("\\", "/")
    twin_rel = os.path.relpath(twin, base).replace("\\", "/")
    head = report_path.read_text(encoding="utf-8").split(SENTINEL)[0].rstrip()
    block = [
        SENTINEL,
        "",
        "## Charts",
        "",
        f"![{TITLE}]({img_rel})",
        "",
        f"Bar height is the trick rate: the share of that model's classifications for the "
        f"category that came back \"Normal\". The counts behind every bar are in "
        f"[`{twin_rel}`]({twin_rel}).",
        "",
    ]
    report_path.write_text(head + "\n\n" + "\n".join(block), encoding="utf-8")
    return True


def parse_args():
    ap = argparse.ArgumentParser(description="Chart jailbreak effectiveness per model.")
    ap.add_argument("--analysis-dir", default="analysis",
                    help="where verdicts.csv and report.md live (default: analysis)")
    ap.add_argument("--out-dir", default=None,
                    help="where to write the PNGs (default: <analysis-dir>/charts)")
    ap.add_argument("--theme", choices=("light", "dark", "both"), default="both",
                    help="which render(s) to produce (default: both)")
    ap.add_argument("--no-report-embed", action="store_true",
                    help="do not touch report.md")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        sys.exit("ERROR: this step needs matplotlib.\n"
                 "  pip install matplotlib      (or: pip install -r requirements.txt)")

    analysis_dir = Path(args.analysis_dir)
    verdicts = analysis_dir / "verdicts.csv"
    if not verdicts.is_file():
        sys.exit(f"ERROR: {verdicts} not found -- run the analysis step first:\n"
                 f"  python main.py --skip-inject --skip-benchmark")

    cells, unknown = aggregate(read_verdicts(verdicts))
    if not cells:
        sys.exit(f"ERROR: no usable rows in {verdicts} "
                 f"({unknown} skipped as unknown category / verdict)")

    models, categories = axes_order(cells)
    if len(models) > len(SERIES_LIGHT):
        sys.exit(f"ERROR: {len(models)} models but only {len(SERIES_LIGHT)} colour slots. "
                 f"Cycling hues would make two models indistinguishable -- chart them in "
                 f"batches with a filtered verdicts.csv instead.")

    out_dir = Path(args.out_dir) if args.out_dir else analysis_dir / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)

    twin = analysis_dir / f"{CHART_STEM}.csv"
    write_csv(twin_rows(cells, models, categories), twin,
              ["model", "category", "seen", *BUCKETS, "trick_rate"])

    themes = ("light", "dark") if args.theme == "both" else (args.theme,)
    written = [render(cells, models, categories, name,
                      out_dir / f"{CHART_STEM}{THEMES[name]['suffix']}.png")
               for name in themes]

    if unknown:
        print(f"WARNING: skipped {unknown} rows with an unknown category or verdict "
              f"(stale results whose injection id is not in injections.jsonl?)")
    for model in models:
        rates = [(c, trick_rate(cells[(model, c)])) for c in categories if (model, c) in cells]
        best = max(rates, key=lambda kv: kv[1]) if rates else ("-", 0.0)
        print(f"{model:24} worst category: {best[0]:26} trick rate {best[1]:.0%}")

    # The light render is the one markdown embeds; the dark one is for slides.
    if not args.no_report_embed and embed_in_report(
            analysis_dir / "report.md", out_dir / f"{CHART_STEM}.png", twin):
        print(f"embedded the chart in {analysis_dir / 'report.md'}")
    print(f"wrote {twin} and " + ", ".join(str(p) for p in written))


if __name__ == "__main__":
    main()
