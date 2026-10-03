#!/usr/bin/env python3
"""
check_context.py

What it does:
  Preflight for the benchmark: proves every model really reads a whole injected log
  instead of a silently truncated one. Compares each model's trained context length
  against the num_ctx the benchmark asks for, then probes the largest injected file per
  log folder and reads back the token count ollama actually evaluated. While the model is
  still loaded from its first probe, it reads `ollama ps` (/api/ps) and reports how much
  of the model sits on the GPU vs the CPU at that num_ctx -- informational only, since CPU
  offload is allowed; it never fails the preflight.

How to run it:
  python scripts/check_context.py                    # every model, every log folder
  python scripts/check_context.py --models llama3.1:8b --num-ctx 32768
  python scripts/check_context.py --logs drupal ssh
  python scripts/check_context.py --all-files        # every injected file, not just the
                                                     # biggest one per folder
  python scripts/check_context.py --list-only        # trained context only, nothing loaded

  A gpt-oss:20b@low/@medium/@high spec is probed once, as gpt-oss:20b: the reasoning level
  changes the output, not how much of the prompt fits.

What it outputs:
  A per-model table on stdout (log, file, chars, tokens, chars/token, headroom, verdict)
  and GPU/CPU split, plus analysis/context_check.csv (the split in gpu_pct, size_bytes,
  size_vram_bytes). Exit code 1 if anything came back TRUNCATED, OVER-MAX
  or ERROR, so main.py can stop before wasting a full run.
"""

import argparse
import csv
from pathlib import Path

import requests

import experiment
import run_benchmark as bench

OUTPUT_RESERVE = 512

SHOW_URL = bench.api_url("show")
PS_URL = bench.api_url("ps")


def declared_context(model: str):
    try:
        info = requests.post(SHOW_URL, json={"model": model}, timeout=120).json()
    except Exception as e:
        return None, f"unreachable: {e}"
    for key, value in (info.get("model_info") or {}).items():
        if key.endswith(".context_length"):
            return int(value), key.rsplit(".", 1)[0]
    return None, ""


def probe(model: str, log_text: str, num_ctx: int) -> dict:
    resp = requests.post(bench.OLLAMA_URL, json={
        "model": model,
        "messages": [
            {"role": "system", "content": bench.SYSTEM_PROMPT},
            {"role": "user", "content": log_text},
        ],
        "stream": False,
        "keep_alive": "5m",
        "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": 1},
    }, timeout=1800)
    resp.raise_for_status()
    return resp.json()


def gpu_split(model: str) -> dict:
    # what `ollama ps` shows for a loaded model: how much of it sits in VRAM. Read right
    # after a probe, which keeps the model loaded (keep_alive 5m) at the real num_ctx, so
    # the size includes that window's KV cache. Informational: CPU offload is allowed.
    try:
        loaded = requests.get(PS_URL, timeout=30).json().get("models") or []
    except Exception as e:
        return {"gpu_pct": "", "size": "", "size_vram": "", "ps_error": str(e)[:60]}
    entry = next((m for m in loaded if model in (m.get("name"), m.get("model"))), None)
    if not entry or not entry.get("size"):
        return {"gpu_pct": "", "size": "", "size_vram": "", "ps_error": "not loaded"}
    size, vram = int(entry["size"]), int(entry.get("size_vram") or 0)
    return {"gpu_pct": round(100 * vram / size, 1), "size": size, "size_vram": vram,
            "ps_error": ""}


def verdict_for(tokens: int, limit: int) -> str:
    if tokens >= limit - OUTPUT_RESERVE:
        return "TRUNCATED"
    if tokens >= limit * 0.9:
        return "TIGHT"
    return "OK"


def base_models(specs):
    out = []
    for spec in specs:
        base, _ = experiment.parse_model(spec)
        if base not in out:
            out.append(base)
    return out


def parse_args():
    ap = argparse.ArgumentParser(description="Prove the models see the whole log.")
    ap.add_argument("--input-root", default="attack_logs_injected")
    ap.add_argument("--out", default="analysis/context_check.csv")
    ap.add_argument("--models", nargs="*", default=experiment.all_models(),
                    help="model specs; @reasoning suffixes are folded into one probe per "
                         "base model (default: every model any temperature runs)")
    ap.add_argument("--logs", nargs="*", default=[],
                    help="only attack folders whose name contains one of these")
    ap.add_argument("--num-ctx", type=int, default=bench.NUM_CTX,
                    help=f"context window to probe with (default {bench.NUM_CTX}, the "
                         "value run_benchmark.py uses)")
    ap.add_argument("--all-files", action="store_true",
                    help="probe every injected file instead of the largest per folder "
                         "(1020 prefills per model -- slow)")
    ap.add_argument("--list-only", action="store_true",
                    help="just print each model's trained context length and quit -- for "
                         "shopping for a replacement model without loading anything")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.models = base_models(args.models)

    if args.list_only:
        biggest = max((p.stat().st_size for p in Path(args.input_root).rglob("*.txt")),
                      default=0)
        print(f"largest injected file: {biggest} chars (~{biggest // 3}-{biggest // 2} "
              f"tokens for log text)\n")
        print(f"  {'model':32} {'trained context':>15}  {'fits num_ctx=' + str(args.num_ctx):>22}")
        for model in args.models:
            ctx_max, arch = declared_context(model)
            fits = "unknown" if ctx_max is None else ("yes" if ctx_max >= args.num_ctx
                                                     else f"NO (max {ctx_max})")
            shown = ctx_max if ctx_max is not None else (arch.split("(")[0].strip() or "?")
            print(f"  {model:32} {str(shown)[:15]:>15}  {fits:>22}")
        return 0

    input_root = Path(args.input_root)
    if not input_root.is_dir():
        print(f"no {input_root}/ -- run the inject step first.")
        return 1

    folders = sorted(p for p in input_root.iterdir()
                     if p.is_dir() and bench.wanted_folder(p.name, args.logs, []))
    targets = []
    for folder in folders:
        files = sorted(folder.glob("*.txt"), key=lambda p: p.stat().st_size, reverse=True)
        if files:
            targets.extend(files if args.all_files else files[:1])

    print(f"probing {len(targets)} file(s) x {len(args.models)} model(s) "
          f"at num_ctx={args.num_ctx} (largest file per log folder"
          f"{'; --all-files' if args.all_files else ''})\n")

    rows, bad = [], 0
    for model in args.models:
        ctx_max, arch = declared_context(model)
        if ctx_max is None:
            print(f"{model:22} trained context: unknown ({arch})")
        elif ctx_max < args.num_ctx:
            print(f"{model:22} trained context: {ctx_max} ({arch})  "
                  f"*** OVER-MAX: benchmark asks for {args.num_ctx}; anything above "
                  f"{ctx_max} is extrapolated or clamped ***")
        else:
            print(f"{model:22} trained context: {ctx_max} ({arch})  ok for {args.num_ctx}")

        limit = min(args.num_ctx, ctx_max) if ctx_max else args.num_ctx
        print(f"  {'log':30} {'file':26} {'chars':>7} {'tokens':>7} {'c/t':>5} "
              f"{'headroom':>9}  verdict")
        split = None
        for txt in targets:
            text = txt.read_text(encoding="utf-8", errors="replace")
            try:
                data = probe(model, text, args.num_ctx)
                tokens = int(data.get("prompt_eval_count") or 0)
                note = ""
                if split is None:
                    split = gpu_split(model)
            except Exception as e:
                tokens, note = 0, str(e)[:60]

            if note:
                v = "ERROR"
            elif ctx_max and tokens >= ctx_max - OUTPUT_RESERVE:
                v = "OVER-MAX"
            else:
                v = verdict_for(tokens, limit)
            if v not in ("OK", "TIGHT"):
                bad += 1

            ratio = len(text) / tokens if tokens else 0
            print(f"  {txt.parent.name:30} {txt.stem:26} {len(text):>7} {tokens:>7} "
                  f"{ratio:>5.2f} {limit - tokens:>9}  {v} {note}")
            rows.append({
                "model": experiment.sanitize(model), "trained_context": ctx_max or "",
                "log": txt.parent.name, "file": txt.stem, "chars": len(text),
                "prompt_tokens": tokens, "chars_per_token": round(ratio, 2),
                "num_ctx": args.num_ctx, "effective_limit": limit,
                "headroom": limit - tokens, "verdict": v, "error": note,
            })
        split = split or {"gpu_pct": "", "size": "", "size_vram": "",
                          "ps_error": "no successful probe"}
        if split["gpu_pct"] != "":
            print(f"  ollama ps: GPU {split['gpu_pct']:g}% / CPU "
                  f"{100 - split['gpu_pct']:g}%  ({split['size_vram'] / 2**30:.1f} of "
                  f"{split['size'] / 2**30:.1f} GiB in VRAM at num_ctx={args.num_ctx})")
        else:
            print(f"  ollama ps: GPU/CPU split unknown ({split['ps_error']})")
        for row in rows:
            if row["model"] == experiment.sanitize(model):
                row.update(gpu_pct=split["gpu_pct"], size_bytes=split["size"],
                           size_vram_bytes=split["size_vram"])
        bench.unload(model)
        print()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["model"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} rows, {bad} not clean)")

    if bad:
        print("\nFIX BEFORE RUNNING: an ERROR row means the probe never completed (ollama "
              "down, or the KV cache for this num_ctx does not fit on the GPU). A "
              "TRUNCATED/OVER-MAX row means the model never saw the head of that log, so "
              "its verdict is not about the file you think it is -- raise --num-ctx if "
              "the model's trained context allows, otherwise run that model on the logs "
              "that do fit and report the reduced coverage.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
