# LLM log-classifier injection benchmark

Real MITRE ATT&CK attack logs, seeded with jailbreak payloads, are classified by local LLMs
through Ollama. Every log is a real attack, so any verdict other than "attack" means the
injection succeeded. See `CLAUDE.md` for how the pipeline works.

## Running it

```bash
pip install -r requirements.txt
python main.py --corpus both                  # inject, benchmark, summarize, chart: 2 lanes x 2 temps x 5 runs
```

## Reading the results

- **Only the model's randomness is accounted for.** Every series runs 5 times (seeds
  42..46). Each rate is computed per run over the logs, and reported as the mean ± the
  sample std across the runs. The std says how much a result moves from one run to the
  next on these same logs; it says nothing about how the result would carry over to other
  attack logs, and no claim is made about that.
- **Temperature is a level, not a number.** Every model runs at its own recommended
  sampling (`experiment.RECOMMENDED`), and the levels are relative to it: `low` = 0.0
  (greedy), `medium` = recommended. So `medium` is 0.6 for llama3.1 but 1.0 for gemma3 and
  gpt-oss. At `low` the std should be 0; a non-zero one means the GPU did not repeat itself.
- **Two experiments.** The temperature experiment runs every level × gpt-oss `@low`,
  llama3.1 and gemma3. The reasoning experiment runs at `medium` (gpt-oss's recommended
  sampling) with gpt-oss `@low`, `@medium` and `@high`. Both use 5 runs per series, and
  `medium` × `@low` is shared. The temperature comparison for gpt-oss covers `@low` only.
- **No clean baseline.** There is no clean (no-injection) run, so detection on clean logs
  is not measured. Every tricked verdict is attributed to the injection.
