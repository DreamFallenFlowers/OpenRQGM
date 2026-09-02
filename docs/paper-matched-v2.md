# Paper-matched public reconstruction v2

This profile follows arXiv:2606.26294v2 as closely as the public record permits,
without changing the result label into an empirical replication claim.

## Matched settings

- GPT-5.5 with low reasoning effort for search and endpoint calls.
- 12,288 binary validation outcomes per condition.
- epsilon 0.05, UCB-Air alpha 0.6, and a five-outcome eligibility minimum.
- Three training samples per role/node in the public adapter.
- Power-of-two checkpoints with ratio 2.
- Polyglot cardinalities 10 train / 49 validation / 166 held out.
- 100 public CRAVE validation examples and a private 100-example CRAVE anchor.
- Minimal shared task-agent seed derived from Appendix C.5.
- Published timeout and tool/cost-cap values are recorded in the manifest.

The RQGM and HGM-H control manifests differ only in condition, output, claim,
and condition-specific limitations. Run `preflight_paper_matched.py` before any
model calls.

## Remaining non-public or non-equivalent components

The paper does not release the identities of its 10/49/166 tasks, its first
checkpoint, complete production role prompts, model-provider revision, or its
production execution harness. The public replacement split is deterministic,
language-interleaved, and pinned by seed, but is not the authors' split.

OpenRQGM's current coder is a bounded generate-test-repair adapter. It exposes
the public tests and executes them in six isolated containers, but it is not the
paper's persistent bash/editor/delegation loop with a hard 16-tool-call lease.
Likewise, the dollar-denominated expansion/train/validation caps are recorded
but not enforced by the local Codex transport. These differences keep
`paper_comparison_valid` false.
