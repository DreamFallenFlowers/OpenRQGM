from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from .persistence import save_state
from .toy import build_toy_engine, result_to_dict

app = typer.Typer(help="Run and inspect the RQGM reference implementation.")


@app.callback()
def main() -> None:
    """Run and inspect the RQGM reference implementation."""


@app.command()
def toy(
    output: Path = typer.Option(Path("runs/toy"), help="Output directory"),  # noqa: B008
    budget: int = typer.Option(24, min=4, help="Validation-outcome budget"),
    seed: int = typer.Option(7, help="Random seed"),
) -> None:
    """Run the deterministic, network-free evaluator co-evolution example."""

    async def execute() -> None:
        engine = build_toy_engine(budget=budget, seed=seed)
        result = await engine.run()
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(
            json.dumps(result_to_dict(result), indent=2, sort_keys=True), encoding="utf-8"
        )
        save_state(engine, output / "state.json")
        typer.echo(f"endpoint={result.endpoint.node_id} bb={result.endpoint.best_belief:.6f}")
        typer.echo(f"epochs={result.epoch_vector} archive={result.archive_size}")

    asyncio.run(execute())


if __name__ == "__main__":
    app()
