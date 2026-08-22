# Paper correspondence

For the authoritative repository-level mapping, read
`../../../docs/algorithm-correspondence.md`. During an audit, verify all of the
following in code and state:

1. archive nodes are multi-role workspaces with parent lineage;
2. UCB-Air controls growth and CMP Thompson sampling selects clades;
3. evaluation balances role first and task second;
4. training feedback never contributes utility records;
5. evaluator slot epochs are frozen between exact validation checkpoints;
6. challengers and incumbents share the same private anchor;
7. replacement uses `Beta.ppf(epsilon, 1+S, 1+F)` and ties retain incumbent;
8. multi-slot decisions are based on the pre-transition state;
9. erasure is slot-local and old records remain auditable as invalid tombstones;
10. cached artifacts support lazy re-evaluation;
11. endpoint selection ignores invalid/stale evidence;
12. private anchor examples are absent from state and proposer inputs.
