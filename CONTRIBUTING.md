# Contributing

Changes to algorithmic behavior must include a test and update
`docs/algorithm-correspondence.md` when the paper mapping changes.

Before opening a pull request, run:

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest -q
rqgm toy --output runs/toy
python skills/rqgm-reproduction/scripts/audit_run.py runs/toy/state.json
```

Do not contribute private anchors, benchmark test labels, API credentials,
model transcripts containing secrets, or generated run directories. Evolved
code examples must be harmless and must not assume unrestricted host execution.
