# Paper coding-domain reproduction

This experiment is the closest public reconstruction of the coding domain in
*The Red Queen Gödel Machine*. It combines:

- executable Aider Polyglot tasks for the coder's fixed utility;
- a frozen learned code-reviewer utility for the coder;
- CRAVE classification accuracy as the reviewer's fixed utility and private
  replacement anchor;
- the core `RQGM` implementation for archive search, checkpoints, replacement,
  and selective erasure.

It is **not** an exact reproduction of the paper's headline result. The paper
does not publish its 10/49/166 Polyglot split, production prompts, model
provider revision, or harness. `configs/paper_full.json` records the reported
settings as a target manifest; the runnable pilot deliberately uses a small
Python-only public subset.

## Data isolation

`prepare_data.py` downloads public metadata into ignored `data/` paths. CRAVE
test examples are supplied only through `AnchorProvider`/`AnchorEvaluator` and
are never placed in a workspace, training feedback, proposer prompt, state
file, or committed artifact. The manifest stores only a SHA-256 fingerprint.

## Run

Prerequisites are a running Docker daemon, the `python:3.12-slim` image, the
official Polyglot repository at `data/polyglot-benchmark`, and an authenticated
Codex CLI.

```powershell
.\.venv\Scripts\python.exe examples\paper_coding\prepare_data.py
.\.venv\Scripts\python.exe examples\paper_coding\run.py \
  --config examples\paper_coding\configs\pilot.json
```

The longer public reconstruction uses `configs/budget128.json`: 128 binary
validation outcomes, checkpoints at 8/16/32/64, three batched training samples
per node, 20 Python Polyglot validation tasks, 100 CRAVE validation examples,
and a 20-example withheld CRAVE anchor. The 20-example anchor is a declared
cost/context approximation, not the paper's 100-item test condition.

Generated code is executed only in Docker with networking disabled, a
read-only root filesystem, a memory cap, a PID cap, and a wall-clock timeout.
The model is instructed not to browse or call tools, and runs in an empty
read-only Codex workspace. These controls cannot remove model pretraining
contamination from a public benchmark.

## Comparability

The paper reports 119/166 held-out Polyglot tasks for both its specialist and
generalist RQGM endpoints. A result from `pilot.json` is a mechanics-and-domain
integration pilot and must not be compared numerically with 119/166. A valid
headline comparison requires all six languages, 12,288 binary validation
outcomes per run, the authors' exact split/harness, and repeated runs.
