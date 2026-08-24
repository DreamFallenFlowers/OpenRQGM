# Paper experiment reproduction status

## Verdict

OpenRQGM reproduces the published Algorithm 1 control flow, but it has **not
reproduced the paper's headline empirical results**. The paper reports 119/166
held-out Polyglot tasks for both its RQGM specialist and generalist endpoints.
No result in this repository should currently be presented as a replication of
that number.

The first 128-outcome local run used a v1 coding adapter that cached one
Polyglot solution per node and could count repeated deterministic test results
as separate evidence. Its evaluator transition remains valid state-machine
evidence, but its endpoint best-belief is not a valid benchmark estimate. The
v2 adapter now uses sample-level artifact keys, fresh validation items,
cross-evaluator patch reuse, coder train feedback, bounded test-and-repair, and
a disjoint endpoint test split.

The v2 corrected pilot was executed at eight validation outcomes. Its records
contained no repeated ordinary sample id within a node/task cell; the fixed and
learned coder cells paired on the same cached `transpose` patch. After public
repository tests were made available to the task agent (matching the paper's
tool-using coder more closely), the disjoint two-task Python endpoint smoke
check passed 2/2 in one attempt per task. The pre-context check was 0/2. Neither
two-task result is comparable with the paper's 166-task endpoint.

The closest public reconstruction is under `examples/paper_coding`. It uses the
official public Aider Polyglot task repository and public CRAVE data, while
keeping CRAVE test examples behind the private anchor interface.

The completed six-language v3 integration run consumed 128 binary outcomes and
224 successful model calls in 7,507 seconds. Its generalist and coder endpoint
were the same archive node and passed a small 12/12 held-out set. All five
checkpoints retained the seed reviewer, so this run validates the challenger
and anchor-election path but does not provide an observed evaluator transition
or an estimate of co-evolution's causal benefit.

The next registered experiment is `polyglot-matched-v1`: verifier-only, fixed
reviewer, and co-evolving reviewer conditions at both 512 and 1,024 outcomes.
It restores the 100-example anchor, expands held-out evaluation to 60 disjoint
tasks, records Codex raw/blended tokens, and enforces identical data and anchor
fingerprints across all six cells. These remain intermediate public
reconstruction experiments, not the paper's unpublished exact split.

Registered long runs use resumable state version 2. The snapshot includes the
search RNG and exact checkpoint prefix, is written by atomic replacement every
eight validation outcomes, and is accepted for continuation only under the
same code/config/data/anchor identity. Model-call accounting uses a separate
append-only ledger. This is an engineering reliability mechanism; it does not
change the RQGM validation budget or evaluator replacement rule.

## What is exact, reconstructed, and approximate

| Component | Status | Notes |
|---|---|---|
| Algorithm 1 search/control flow | Exact to published pseudocode | CMP, UCB-Air, binary budget, checkpoints, frozen evaluators, anchor replacement, and selective erasure are in the core package. |
| Reported paper-scale settings | Recorded | `configs/paper_full.json` records 12,288 outcomes, GPT-5 low, alpha 0.6, epsilon 0.05, three training samples per node, the power-of-two checkpoint ratio, and the 10/49/166 split sizes. The first checkpoint is inherited from HGM and not specified in the RQGM paper. |
| Polyglot task source | Public reconstruction | Official Aider repository pinned locally at commit `7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f`. The paper's exact task split is unpublished. |
| CRAVE reviewer anchor | Public reconstruction | Deterministic withheld sample of the public CRAVE test split; exact paper sample is unpublished. |
| Model and prompts | Approximation | Pilot uses authenticated `gpt-5.6-sol`; the paper used GPT-5 low and does not publish production prompts/provider revision. |
| Execution harness | Public reconstruction | Six isolated language images run the official public Aider commands for C++, Go, Java, JavaScript, Python, and Rust. This is still not the paper's unpublished production harness. |
| Anchor inference | Approximation | Examples are classified in a batch per evaluator candidate to reduce model-call overhead. Binary outcomes are still recorded per example. |

## Executed pilot

Configuration: `examples/paper_coding/configs/pilot.json`.

- two independent executions of the same fixed pilot configuration;
- four validation outcomes per execution;
- one checkpoint after outcome 2;
- two archive nodes;
- nine authenticated model calls in the final execution;
- 141.313 seconds wall time in the final execution;
- one generated `dominoes` solution passed all 13 tests in a network-disabled,
  read-only, non-root Docker container;
- endpoint validation record: 4 successes, 0 failures;
- coder specialist: 2 successes, 0 failures;
- reviewer specialist: 2 successes, 0 failures;
- both reviewer artifacts scored 5/5 on the small CRAVE anchor;
- the reviewer prompts were semantically identical, so the incumbent was kept;
  no evaluator epoch transition or selective erasure occurred;
- the saved state passes `skills/rqgm-reproduction/scripts/audit_run.py`.

The 4/4 result is not a useful estimate of benchmark quality: the budget and
task set are tiny, the same public benchmark may be present in model
pretraining, and no held-out 166-task endpoint evaluation was run. Its value is
integration evidence: real model calls, a real generated program, sandboxed
tests, CRAVE-grounded reviewer validation, and a paper-aligned checkpoint all
completed end to end.

Local ignored artifacts are written to `runs/paper-coding-pilot`, with the
first execution preserved at `runs/paper-coding-pilot-attempt1`.

## What would be required for a defensible empirical replication

1. Obtain the authors' exact 10/49/166 Polyglot split and 100-example CRAVE
   anchor, or publish a preregistered replacement split under a new result name.
2. Implement and pin all six language containers and the exact test commands.
3. Match GPT-5 low, prompts, tool limits, timeout rules, and model-provider
   revision as closely as the released information permits.
4. Run all 12,288 binary validation outcomes for each headline condition and
   multiple seeds, logging model tokens, cost, wall time, and failures.
5. Evaluate frozen specialist/generalist endpoints on all 166 held-out tasks.
6. Report the reconstruction separately from the paper's number and include
   uncertainty and contamination limitations.

Item 2 is now implemented for the public Aider checkout and passes a 6/6
reference-solution smoke test. The remaining items—especially the unpublished
split, 12,288-outcome scale, exact model endpoint, and repeated seeds—still
prevent a headline empirical-replication claim. The 128-outcome v3 run is an
integration and method-reconstruction experiment only.
