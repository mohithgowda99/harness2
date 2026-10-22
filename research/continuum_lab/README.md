# Continuum research lab: first integrated benchmark

**Status:** experimental, deliberately isolated from upstream Harness²; no changes to the core agent executor. No paid model calls.

## Why this exists

Test a falsifiable claim: when a small learner may ask only three questions about an unfamiliar binary world, will selecting queries that distinguish competing hypotheses outperform random exploration on held-out examples?

The environment uses 12 fixed 3-bit Boolean rules. Both strategies see the same 500 seeded hidden worlds and the same number of observations. The agent is never passed the hidden rule name. Only unseen input combinations are graded.

Importantly, all potential rule types **are already in the symbolic learner's hypothesis vocabulary**. This does *not* demonstrate that a model can invent new representations, retain knowledge across tasks, or learn new neural weights.

## Run now: free CPU baseline

Requires Python 3.12+ (also works with 3.13). No pip dependencies.

```bash
python -m unittest discover -s research/continuum_lab -p 'test_*.py' -v
python research/continuum_lab/runner.py --trials 500 --budget 3 --output /tmp/continuum-baseline.json
```

Local development run: random = **0.7964** (1,991/2,500); active = **0.8200** (2,050/2,500); all four unit tests passed. This is a toy-environment lift of 2.36 percentage points, not evidence of AGI/ASI. Each seed and budget is included in the saved JSON results.

## Optional next experiment: real model, entirely on a Mac

Use [Ollama](https://ollama.com/download/mac) locally. On first run download the 523 MB [Qwen3 0.6B](https://ollama.com/library/qwen3:0.6b) model:

```bash
ollama pull qwen3:0.6b
ollama serve  # only if the local service is not already running
python research/continuum_lab/runner.py --backend ollama --model qwen3:0.6b --trials 12 --budget 3 --output /tmp/continuum-ollama.json
```

No model account, cloud subscription, API key, or GPU rental. A local Ollama service must be running on 127.0.0.1:11434. Default settings make **24 model requests** for 12 worlds, because the real-model backend predicts the full held-out set once per strategy and world. Prediction failures are surfaced instead of silently treated as correct.

**Important experimental distinction:** the real model is currently a **predictor**, not the explorer. Query selection is identical to the symbolic benchmark: random versus our algorithmic active-learning policy. A future test will compare model-directed probe selection under the same budgets.

## Method, caveats, next gates

- Keep the evaluation world and hidden labels out of the model prompt; the model sees examples and unobserved query inputs only.
- Compare held-out accuracy and total inference resources, not cherry-picked examples.
- No Internet retrieval, offline dream loop, model weight adaptation, persistent memory, or Harness² recursion is connected yet. This branch establishes a clean low-cost experimental seam, not production integration.
- Next: run the actual model locally; log malformed responses and inference time; test transfer on a **disjoint rule grammar**; only then add persistent episodes and replay, with ablations and independent verification.
- Upstream Harness² itself documents Linux for its benchmark stack. This standalone research lab can run on macOS; that doesn't imply the full upstream benchmark stack is Mac-compatible.
- Do not enable unrestricted agent tooling or benchmark-answer web access. Save generated output under `/tmp` or ignored local paths.

Run upstream's offline demo separately if you have its dependencies installed:
```bash
harness2-demo --output runs/demo --k 2
```
