#!/usr/bin/env python3
"""
check_context.py

Preflight for the benchmark: proves every model can actually read a whole injected log
instead of silently being handed a truncated one. Ollama does not report truncation --
it drops the head of the prompt to make it fit num_ctx and answers as if nothing
happened -- so the only honest evidence is the token count it reports back.

Two checks per model:
  1. declared -- /api/show gives the model's trained context length. If that is smaller
                 than the num_ctx the benchmark asks for, the extra window is fiction
                 (qwen2.5:7b-instruct is trained at 32768, not 60069).
  2. measured -- sends the LARGEST injected file of every log folder with num_predict=1
                 and reads `prompt_eval_count`: the number of tokens really fed to the
                 model. Truncation can only push that count UP TO the limit, so a count
                 comfortably below the limit proves the whole file went in. Probing at
                 the real num_ctx doubles as a KV-cache memory test on the GPU.

How to run it:
  python scripts/check_context.py                      # every model, every log folder
  python scripts/check_context.py --models qwen2.5:7b-instruct --num-ctx 32768
  python scripts/check_context.py --logs drupal ssh
  python scripts/check_context.py --all-files          # probe every injected file, not
                                                       # just the biggest one per folder
  python scripts/check_context.py --list-only --models gemma3:27b mistral-small:24b
                                                       # shop for models: trained context
                                                       # only, nothing loaded

What it outputs:
  a per-model table on stdout (log, file, chars, tokens, chars/token, headroom, verdict)
  plus analysis/context_check.csv. Exit code 1 if anything came back TRUNCATED, OVER-MAX
  or ERROR, so main.py can stop before wasting a full run.
"""

import argparse
import csv
from pathlib import Path

import requests

import run_benchmark as bench

# tokens left free for the model's own JSON answer (10 technique ids + explanation)
OUTPUT_RESERVE = 512

SHOW_URL = bench.OLLAMA_URL.rsplit("/api/", 1)[0] + "/api/show"


# ---------------------------------------------------------------- ollama probes
def declared_context(model: str):
    """(trained context length, arch) straight from the model metadata, or (None, '')."""
    try:
        info = requests.post(SHOW_URL, json={"model": model}, timeout=120).json()
    except Exception as e:
        return None, f"unreachable: {e}"
    for key, value in (info.get("model_info") or {}).items():
        if key.endswith(".context_length"):
            return int(value), key.rsplit(".", 1)[0]
    return None, ""


def probe(model: str, log_text: str, num_ctx: int) -> dict:
    """One real /api/chat call with generation disabled: all we want is how many prompt
    tokens Ollama admits to having evaluated."""
    resp = requests.post(bench.OLLAMA_URL, json={
        "model": model,
        "messages": [
            {"role": "system", "content": bench.SYSTEM_PROMPT},
            {"role": "user", "content": log_text},
        ],
        "stream": False,
        "keep_alive": "5m",          # stay loaded across this model's probes
        "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": 1},
    }, timeout=1800)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------- verdict
def verdict_for(tokens: int, limit: int) -> str:
    """Truncation can only ever land the count ON the limit, never past it."""
    if tokens >= limit - OUTPUT_RESERVE:
        return "TRUNCATED"
    if tokens >= limit * 0.9:
        return "TIGHT"
    return "OK"


# ---------------------------------------------------------------- args
def parse_args():
    ap = argparse.ArgumentParser(description="Prove the models see the whole log.")
    ap.add_argument("--input-root", default="attack_logs_injected")
    ap.add_argument("--out", default="analysis/context_check.csv")
    ap.add_argument("--models", nargs="*", default=bench.MODELS)
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


# ---------------------------------------------------------------- main
def main() -> int:
    args = parse_args()

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
        for txt in targets:
            text = txt.read_text(encoding="utf-8", errors="replace")
            try:
                data = probe(model, text, args.num_ctx)
                tokens = int(data.get("prompt_eval_count") or 0)
                note = ""
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
                "model": bench.sanitize(model), "trained_context": ctx_max or "",
                "log": txt.parent.name, "file": txt.stem, "chars": len(text),
                "prompt_tokens": tokens, "chars_per_token": round(ratio, 2),
                "num_ctx": args.num_ctx, "effective_limit": limit,
                "headroom": limit - tokens, "verdict": v, "error": note,
            })
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
