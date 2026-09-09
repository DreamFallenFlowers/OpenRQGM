# Paper-matched v5 public reconstruction

## Result status

This run is the strongest completed OpenRQGM coding result to date. It follows
the published RQGM Algorithm 1 control flow and uses the paper-reported model,
budget, and split cardinalities. On our deterministic public replacement split,
the saved endpoints achieved:

| Endpoint | Held-out tasks |
|---|---:|
| coder specialist | **163/166 (98.19%)** |
| generalist | **160/166 (96.39%)** |

The paper reports 119/166 (71.69%) for both RQGM endpoints and 116/166
(69.88%) for HGM-H. OpenRQGM therefore has a substantially higher observed
absolute pass rate in this public reconstruction. This is a descriptive result,
not evidence that OpenRQGM is causally better than the paper system: the exact
task identities, complete production prompts, provider revision, and private
agent harness are unpublished, and this run has no matched HGM-H control.

Accordingly, the result remains marked `paper_comparison_valid=false`. The
defensible claim is that OpenRQGM performs strongly under its disclosed public
protocol while reproducing the published algorithmic process.

## Experimental configuration

- paper condition: RQGM coder plus co-evolving code reviewer;
- model: GPT-5.5 with low reasoning effort;
- random seed: `260626294`;
- validation budget: 12,288 binary outcomes;
- public replacement split cardinalities: 10 train / 49 validation / 166 held out;
- training samples per node: 3;
- `epsilon=0.05`, expansion `alpha=0.6`, minimum anchor outcomes 5;
- power-of-two checkpoints: 8 through 8,192;
- six Aider Polyglot languages;
- public CRAVE reconstruction with 100 validation and 100 private-anchor items;
- bounded concurrency: 4;
- source experiment: `paper-matched-v5-context-safe-rqgm-seed-260626294`.

The run took 241,860.56 seconds (67.18 hours) and recorded 447,365,462 blended
tokens across 16,976 successful metered model calls. This is whole-run cost. It
is not directly comparable with the paper's token-efficiency curve, which reads
each agent at the point where it first attains its final best-belief score.

## Co-evolution evidence

The archive contained 285 nodes and 12,288 validation records. Eleven declared
checkpoints produced four evaluator replacements and 2,182 selectively erased
records. The code-reviewer epoch advanced from 1 to 5. The seed reviewer scored
70/100 on the evaluator-independent anchor; the strongest accepted challenger
score observed during replacement was 82/100.

The saved state passes the structural audit:

```text
PASS: structural RQGM state invariants hold
```

This establishes that the run exercised the expected archive growth,
best-belief selection, checkpoint-only evaluator replacement, frozen evaluator
epochs, and slot-local selective erasure. It does not by itself estimate the
causal benefit of co-evolution; matched fixed-reviewer, verifier-only, and HGM-H
controls are still required.

## Endpoint harness audit

The six-language endpoint harness was audited after the run. A test-discovery
issue in one language runner was corrected inside the disposable sandbox, and
both saved endpoints were freshly re-evaluated under complete-suite execution.
The aggregate endpoint scores were unchanged. Because generated held-out patches
were not retained by the original run, this is a fresh evaluation of the same
saved agents rather than a replay of identical patches. The supplement used 96
successful model calls and 6,720,072 blended tokens.

The complete supplemental JSON is preserved with SHA-256
`c8dfd82b6b8bc2ad41c4a9a7bfe2c6b1ebe560ea8a8da5b2712d49d4db05c171`.
The repository commits a compact machine-readable summary rather than the raw
158 KB of sandbox output.

## Paper correspondence and remaining limitations

The implementation matches the published Algorithm 1 invariants documented in
[Algorithm correspondence](../algorithm-correspondence.md): multi-role archive
lineage, UCB-Air/CMP search, role-first task scheduling, validation-only utility,
frozen evaluator epochs, anchor-gated replacement, incumbent-favoring ties,
selective erasure, lazy artifact reuse, and stale-evidence exclusion.

The following remain public reconstruction choices or approximations:

- deterministic replacement task identities because the paper's exact split is unpublished;
- public CRAVE sampling and context-size filtering;
- generate-test-repair in place of the private persistent 16-tool coder harness;
- incomplete production prompts and unknown provider revision;
- one random seed and no same-harness HGM-H control;
- no per-node token checkpoint matching the paper's endpoint-cost convention;
- possible benchmark familiarity or contamination in a public coding suite.

The next causal-quality experiment should run co-evolving reviewer,
fixed-reviewer, verifier-only, and HGM-H conditions with identical fingerprints,
model settings, budgets, and multiple seeds. A harder private held-out suite is
also needed because 96-98% leaves little headroom for discriminating systems.
