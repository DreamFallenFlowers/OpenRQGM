# Architecture

`RQGM` owns the paper's orchestration; a `Runtime` owns domain semantics.

```text
RQGM engine
├── Archive
│   ├── workspace lineage
│   ├── cached artifacts
│   ├── binary utility records
│   └── training feedback (non-utility)
├── Scheduler
│   ├── UCB-Air expansion gate
│   ├── clade CMP Thompson selection
│   └── role/task balancing
├── Evaluator slots
│   ├── frozen incumbent
│   ├── epoch index
│   └── anchor-scored challengers
└── Runtime adapters
    ├── workspace editor
    ├── task evaluator
    ├── challenger source
    ├── private anchor provider
    ├── anchor evaluator
    └── optional training feedback
```

## Trust boundary

The core validates topology, binary outcomes, checkpoints, slot ownership and
state compatibility. It cannot make arbitrary agent code safe. A production
runtime should run evolved code in an isolated process/container with network
off, a read-only base filesystem, a temporary writable directory, hard CPU and
memory limits, and sanitized outputs.

Private anchors must be loaded by `AnchorProvider` at runtime. They must not be
placed in workspace files, prompts, config files, checkpoints or repository
history. Challenger generation receives archive workspaces but never the
anchor examples.
