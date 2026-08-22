# Algorithm correspondence

This table maps the paper's Algorithm 1 to the reference implementation. It is
the acceptance checklist for claims of paper alignment.

| Paper step | Implementation | Required invariant |
|---|---|---|
| Seed archive, epoch vector, frozen slots (17–20) | `RQGM.__post_init__`, `Archive` | one seed; slot incumbents copied and frozen by convention |
| `TrainEval` as lineage evidence (20, 31) | `RQGM._collect_training_feedback` | feedback is visible in node lineage but never becomes utility |
| Next exact checkpoint (23–25) | `RQGM.run`, `_run_epoch_until` | validation batches cannot cross a checkpoint |
| UCB-Air expansion gate (26) | `scheduler.should_expand` | archive growth follows `N**alpha >= abs(T)` after initial evidence |
| CMP parent selection (27) | `Archive.sample_node_by_cmp` | Beta Thompson draw uses valid outcomes pooled over each clade |
| Workspace edit and validation (28–31) | `_expand_once`, `WorkspaceEditor` | invalid/`None` edits do not enter the archive |
| CMP evaluation-node selection (34) | `Archive.sample_node_by_cmp` | expansion and evaluation each select via current CMP evidence |
| Least-measured role/task (35) | `Archive.least_measured_cell` | balance roles first, then tasks within the chosen role |
| Binary validation outcome (36–37) | `TaskEvaluator`, `_evaluate_once` | only `{0,1}` enters search utility; epoch vector is recorded |
| Incumbent plus challengers (41–42) | `_checkpoint` | challenger slot identity must match; candidate ids are unique |
| Anchor best-belief (43–44) | `_score_candidate`, `best_belief` | `Beta.ppf(epsilon, 1+S, 1+F)`; insufficient evidence is ineligible |
| Tie keeps incumbent (44) | `strictly_better`, `_checkpoint` | no replacement or erasure on a tie |
| Freeze and advance slot epoch (45–46) | `_checkpoint` | decisions are planned on the old state, then committed atomically |
| Selective erasure and CMP refresh (47) | `Archive.invalidate_slot` | only records owned by changed slots are tombstoned; CMP is dynamic |
| Endpoint best-belief (52–54) | `Archive.endpoint`, `RunResult` | selection uses only currently valid epoch evidence |

## Additional method invariants

- **Data isolation:** training may create edits; validation selects nodes; test
  data is owned by the external experiment and is never consumed by the core.
- **Anchor isolation:** `AnchorProvider.examples()` is a runtime capability.
  Persistence stores only provider id/version/fingerprint, never examples.
- **Stationarity:** evaluator artifacts, artifact protocol and binary rule must
  remain fixed between checkpoints. Runtime adapters are responsible for model
  version and prompt hashing when external services are used.
- **Lazy recovery:** cached task artifacts remain on nodes after erasure. The
  least-measured scheduler naturally repopulates missing current-epoch cells.
- **Comparable scores:** learned task-agent utilities are epoch-local. Only
  fixed-anchor outcomes can be compared globally across evaluator transitions.

## Explicit boundary

The repository reconstructs Algorithm 1 from the public paper. It cannot
reproduce unspecified model-provider behavior or the authors' private
production harness. A domain adapter that changes prompts/models inside an
epoch, exposes anchors to proposers, mixes training feedback into validation,
or executes evolved code without isolation is not paper-aligned even if it
uses the `RQGM` class.
