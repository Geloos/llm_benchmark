# third_party

`intervals_clustered.py` is copied **verbatim, unmodified** from the code of

> S. Bowyer, L. Aitchison, D. R. Ivanova, "Position: Don't Use the CLT in LLM Evals With
> Fewer Than a Few Hundred Datapoints," ICML 2025, arXiv:2503.01747.

- source: https://github.com/sambowyer/no_clt_paper/blob/ab812f29f6508f53352a57b397669b2227f06164/eval_lib/src/intervals_clustered.py
- commit: `ab812f29f6508f53352a57b397669b2227f06164`
- sha256: `84b2dc23faf2a2a4a5e182d8e7f4a2ad86a5156c657a16dff22ea2d003f3ae3a`
- license: MIT, see `LICENSE` (copied from the same commit)

`scripts/stats_analysis.py` calls `bayes_subtask_credible_interval_IS` from it, the
paper's clustered Bayesian credible interval. Do not edit the file: the point of copying it
is that the interval is the authors' own implementation. To update it, re-download it from
a newer commit and update the commit and sha256 above.
