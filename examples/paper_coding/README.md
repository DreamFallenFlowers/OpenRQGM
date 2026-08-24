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
settings as a target manifest. The `budget128-v3.json` reconstruction uses a
public, balanced six-language split and must be reported under its own name.

## Data isolation

`prepare_data.py` downloads public metadata into ignored `data/` paths. CRAVE
test examples are supplied only through `AnchorProvider`/`AnchorEvaluator` and
are never placed in a workspace, training feedback, proposer prompt, state
file, or committed artifact. The manifest stores only a SHA-256 fingerprint.

## Run

Prerequisites are a running Docker daemon, the official Polyglot repository at
`data/polyglot-benchmark`, and an authenticated Codex CLI. Build and verify the
six pinned evaluator images before a model-backed run:

```powershell
.\.venv\Scripts\python.exe examples\paper_coding\prepare_data.py
.\examples\paper_coding\build-images.ps1
$env:PYTHONPATH=(Resolve-Path src)
.\.venv\Scripts\python.exe examples\paper_coding\smoke_polyglot.py
.\examples\paper_coding\run-budget128.ps1
```

The current public reconstruction uses `configs/budget128-v3.json`: one shared
budget of 128 binary validation outcomes, checkpoints at 8/16/32/64/96, three
batched training samples per node, one coder training task per node, disjoint
and language-balanced Polyglot train/validation/test splits, 100 CRAVE
validation examples, and a 20-example withheld CRAVE anchor. It is one
cross-language co-evolution run—not six independent 128-outcome runs. Held-out
results report each language and their macro-average. The 20-example anchor is
a declared cost/context approximation, not the paper's 100-item condition.

## Evolved codebase boundary (v3)

An archive node is now a complete sandboxed agent codebase rather than a single
prompt blob. The meta-agent returns the complete next file tree and may add,
replace, or delete any UTF-8 text file inside it, including `agent.py`, helper
modules, prompts, parsers, and coder/repair/reviewer role logic. Omitted files
are deleted. The trusted JSON stdin/stdout protocol, RQGM engine, benchmark,
private anchors, and sandbox policy remain outside the editable tree.

Every evolved tree is protocol-smoked for all three roles and then executed as
UID 65534 in a networkless, capability-free Docker container with a read-only
root and read-only code mount. Evolved code is never imported or executed by
the host Python process. File-count and byte limits bound the mutation surface.

## Sample-aware protocol (v2)

Each workspace node is a reusable agent strategy, not a solution to one fixed
exercise. Every ordinary coder validation consumes a previously unseen
Polyglot sample for that node. The executable-test cell and learned-reviewer
cell share the same sample-level patch; after evaluator replacement, the patch
is reused and reviewed under the new frozen evaluator. It is never counted
twice as fresh fixed evidence. CRAVE validation examples likewise do not repeat
within a node before its pool is exhausted.

The coder receives the public repository tests as task-agent context and up to
two bounded sandbox-feedback repair turns by default, approximating the
paper's bash/editor loop without exposing validation items to the meta-agent.
Lineage-only training contains both CRAVE reviewer feedback and coder test
outcomes, while validation labels remain hidden from the workspace editor.
Final generalist and coder-specialist endpoints are evaluated separately on the
disjoint configured Polyglot test split.

Runs created before this v2 protocol cached one Polyglot solution per node and
must not be used as benchmark estimates. They remain useful only as historical
state-machine integration runs.

The corrected 8-outcome pilot completed with no duplicate ordinary sample ids.
Its disjoint two-task Python endpoint check passed 2/2 after repository-test
context was enabled (versus 0/2 before that context). This is a protocol smoke
test, not a statistically meaningful benchmark result.

Generated code is executed only in Docker with networking disabled, a
read-only root filesystem, a memory cap, a PID cap, and a wall-clock timeout.
The model is instructed not to browse or call tools, and runs in an empty
read-only Codex workspace. These controls cannot remove model pretraining
contamination from a public benchmark.

## Matched 512/1,024-outcome ablation

`configs/ablations` defines six preregistered cells under
`polyglot-matched-v1`. Every cell uses the same GPT-5.6 Sol model, random seed,
six-language train/validation/held-out split, 100-example CRAVE anchor,
20-example anchor inference batches, training settings, repair budget, images,
and sandbox. Only the declared
condition and validation budget/checkpoints may differ:

- `verifier_only`: executable tests are the only utility; reviewer training is
  disabled;
- `fixed_reviewer`: adds a globally fixed reviewer utility and CRAVE reviewer
  role, with no evaluator slot or anchor election;
- `coevolving_reviewer`: uses the same three task cells but makes the reviewer
  an epoch-frozen evaluator slot eligible for anchor-gated replacement.

Each endpoint is tested on 60 disjoint tasks (ten per language). Codex JSONL
usage is recorded as raw tokens and as the paper's
`input_tokens + 5 * output_tokens` blended metric.

```powershell
# No model calls: verify all six cells share data/anchor/config fingerprints.
.\examples\paper_coding\run-matched-ablation.ps1 -Preflight

# Launch one cell in the background. Run cells sequentially to avoid contention.
.\examples\paper_coding\run-matched-ablation.ps1 `
  -Budget 512 -Condition verifier_only

# After all three cells for one budget finish, produce the matched contrast report.
$env:PYTHONPATH=(Resolve-Path src)
.\.venv\Scripts\python.exe examples\paper_coding\ablation.py `
  --report --budget 512
```

`ablation.py --run-all` is guarded by `--confirm-expensive-matrix`; the
PowerShell launcher intentionally starts only one cell at a time. Every
completed cell is automatically checked by the structural RQGM state auditor.

Long cells save an atomic `state.json` every eight validation outcomes and at
every evaluator checkpoint. Re-running the same registered cell restores the
archive, evaluator epochs, checkpoint history, RNG state, initialization state,
and domain training cursor. Model-call usage is appended independently to
`model-calls.jsonl`, so calls spent immediately before a crash remain visible in
the final cost. A resume is rejected if the source commit, configuration, data
split, or private-anchor fingerprint changed. Proposer views omit validation
artifact caches, which reduces archive-copy memory pressure without removing
the lineage training feedback allowed by the algorithm.

## Comparability

The paper reports 119/166 held-out Polyglot tasks for both its specialist and
generalist RQGM endpoints. A result from `pilot.json` is a mechanics-and-domain
integration pilot and must not be compared numerically with 119/166. A valid
headline comparison requires all six languages, 12,288 binary validation
outcomes per run, the authors' exact split/harness, and repeated runs.
