# Domain adapter contract

Implement one `rqgm.protocols.Runtime` with these capabilities:

- `WorkspaceEditor`: edits one selected parent into a child workspace. It may
  use lineage training feedback but cannot access validation labels or anchors.
- `TaskEvaluator`: returns a binary `EvaluationOutcome`. For learned tasks it
  receives the slot's frozen evaluator candidate. Reuse `cached_artifact` when
  present instead of calling the task agent again.
- `ChallengerSource`: extracts or proposes evaluator candidates for one slot.
  It receives workspaces, never anchor examples.
- `AnchorProvider`: loads a fixed held-out anchor by slot at runtime.
  It exposes stable `provider_id`, `version`, and non-reversible `fingerprint`
  metadata so restore can reject anchor drift.
- `AnchorEvaluator`: scores a candidate on one anchor example.
- optional `TrainingFeedback`: produces lineage evidence that never enters the
  validation posterior.

Predeclare how exceptions, invalid output, timeout, compilation failure and
sandbox violations map to binary failure. Pin external model and prompt
versions for the duration of each epoch.
