# LLM log-classifier injection benchmark

Real MITRE ATT&CK attack logs are seeded with jailbreak payloads and given to local LLMs
(through Ollama), which are asked to classify each log as attack or normal. Every log is a
real attack, so any verdict other than "attack" means the injection worked. The share of
calls that were tricked is the **ASR** (attack success rate).

Models: `gpt-oss:20b`, `llama3.1:8b`, `gemma3:12b`.

## How to run it

```bash
pip install -r requirements.txt
python main.py                 # every experiment, every temperature
python main.py --help          # all the flags
```

Ollama must be running with the three models pulled. Finished stages are skipped, so
re-running `python main.py` only does what is missing. To redo a stage, delete its output.

Run only part of it:

```bash
python main.py --plain                 # one experiment (--plain --hexa --dovsdofs --ob)
python main.py --plain --low           # one temperature (--low --medium)
python main.py --plain --skip-reasoning
python main.py --models llama3.1:8b --categories direct_override --logs drupal
python main.py --injections DO_01_canonical PH_ --runs 1
```

## The pipeline

For each experiment, `main.py` runs these steps in order:

| step | script | what it does |
|---|---|---|
| 1 inject | `scripts/inject_prompts.py` | writes the jailbreaks into copies of the attack logs |
| 1b preflight | `scripts/check_context.py` | checks no log is too long for a model's context window |
| 2 benchmark | `scripts/run_benchmark.py` | asks every model to classify every injected log, for every seed |
| 3 summarize | `scripts/summarize_results.py` | parses the replies and computes the ASR (`calls.jsonl`, `asr.csv`, `report.md`) |
| 4 charts | `scripts/plot_results.py` | bar charts of the ASR (PDFs in `charts/`) |
| 5 temperature | `scripts/compare_temperatures.py` | plain experiment only: low vs medium ASR side by side |

Folders:

- `attack_logs/` — the source logs (tracked)
- `jailbreaks/` — the injections
- `attack_logs_injected/`, `results/`, `analysis/` — generated

Results are the mean ASR across seeds, with a 95% interval across the seeds. There are no
significance tests: compare the mean ASRs directly.

## The experiments

Every run is repeated with **5 seeds (42–46)**. There are two temperature levels:
**low** = 0.0 (greedy) and **medium** = each model's recommended temperature (llama 0.6,
gemma and gpt-oss 1.0).

| experiment | flag | temperatures | models | question |
|---|---|---|---|---|
| plain | `--plain` | low, medium | gpt-oss (reasoning low), llama3.1, gemma3 | which jailbreaks work, as readable text? |
| reasoning | part of `--plain` | medium | gpt-oss at reasoning low / medium / high | does more reasoning resist better? |
| hex | `--hexa` | low | gpt-oss (reasoning low), llama3.1, gemma3 | do they still work when the log stores them as hex? |
| DOVSDOFS | `--dovsdofs` | low, medium | gpt-oss (reasoning medium), llama3.1, gemma3 | does a `SYSTEM:` prefix make a direct override stronger? |
| OB | `--ob` | low, medium | gpt-oss (reasoning medium), llama3.1, gemma3 | does hex work if the model is told to decode it? |

The plain experiment uses 47 injections in 10 categories (`jailbreaks/injections.jsonl`).
Its results are also compared across temperatures, low vs medium per model
(`analysis/analysis_temperature/`).
