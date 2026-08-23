# Reproducibility guide

## Before a run

1. Pin the model id, provider revision, prompts and tool image.
2. Define roles and globally unique task ids.
3. Mark every task as fixed or learned; learned tasks name one evaluator slot.
4. Create disjoint train, validation, anchor and test data.
5. Hash private anchor contents and record only provider id/version/hash in the
   experiment manifest—not examples.
6. Choose a validation-outcome budget and checkpoints before observing results.
7. Verify the evolved-code sandbox with a harmless escape test.
8. For multi-item benchmarks, verify that each ordinary validation outcome has
   a distinct sample id within its node/task cell. Reuse is allowed only for a
   declared cross-evaluator re-score of the same cached artifact.

## During a run

- Count budget in validation outcomes, not LLM calls or outer iterations.
- Do not cross checkpoint boundaries with a batch.
- Freeze model, prompt, artifact-generation and scoring protocol per epoch.
- Keep full candidate lineage and model-call metadata, excluding secrets.
- Treat timeout, crash, invalid schema and compile failure as explicit binary
  failures under a predeclared rule.

## After a run

- Report endpoint quality on a held-out test set.
- Report evaluator anchor successes/failures and Beta uncertainty.
- Separate epoch-local learned-evaluator scores from globally comparable fixed
  or anchor scores.
- Report validation outcomes, model tokens, wall time and monetary cost.
- Run `skills/rqgm-reproduction/scripts/audit_run.py state.json`.
- Repeat with several random seeds; the toy's deterministic pass is only a
  mechanics check.

## API credentials

Read credentials from environment variables or the caller's credential store.
Never put them in TOML/YAML, prompts, run state, remote URLs or Git history.
