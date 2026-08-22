---
name: rqgm-reproduction
description: Scaffold, run, resume, or audit Red Queen Gödel Machine experiments against the public paper's Algorithm 1. Use for RQGM evaluator co-evolution, CMP/HGM archive search, checkpoint replacement, selective erasure, or paper-alignment checks; do not use for ordinary fixed-evaluator evolutionary search.
---

# RQGM Reproduction

Use the repository's `rqgm` package as the single implementation. The skill
orchestrates experiment work; it must not duplicate or silently weaken the
algorithm.

## Choose the operation

- **Scaffold:** create a domain adapter and experiment manifest. Read
  [domain adapter contract](references/domain-adapter.md), then run
  `scripts/scaffold_experiment.py TARGET`.
- **Run or resume:** inspect the manifest, verify data isolation and sandboxing,
  execute the requested budget, save `state.json`, and audit it before reporting.
- **Audit:** read [paper correspondence](references/paper-correspondence.md) and
  compare code plus run state against every required invariant. Run
  `scripts/audit_run.py STATE`; treat a passing script as necessary, not
  sufficient, because runtime prompt/model stationarity needs human-readable
  evidence.

## Non-negotiable invariants

Keep training feedback out of validation utility. Count the search budget in
binary validation outcomes. Freeze learned evaluators inside an epoch. Allow
replacement only at declared checkpoints and only by evaluator-independent
anchor best-belief. Ties keep the incumbent. After replacement, invalidate only
records owned by changed slots and reuse cached artifacts lazily.

Never expose private anchor examples to the workspace editor, evaluator
proposer, model prompt, run state, or repository. Store only anchor identity,
version and a non-reversible content hash. Never execute evolved code directly
on the host; require an isolated backend with explicit resource limits.

## Reporting

State which parts are paper-exact, which are domain choices, and which are
approximations. Separate epoch-local learned-evaluator scores from globally
comparable fixed/anchor scores. Report validation outcomes, checkpoints,
replacements, erased records, endpoint best-belief, held-out test results,
random seeds, tokens, wall time and cost when available.

Do not call a run a reproduction of the paper's empirical results unless it
uses the same public datasets, splits, model conditions and matched budgets.
A toy run validates mechanics only.
