import json

import pytest

from rqgm.persistence import restore_state, save_state
from rqgm.toy import build_toy_engine


@pytest.mark.asyncio
async def test_toy_runs_end_to_end_with_real_replacement(tmp_path) -> None:
    engine = build_toy_engine()
    result = await engine.run()
    assert result.validation_outcomes == 24
    assert result.archive_size > 1
    assert any(event.replacements for event in result.checkpoints)
    assert result.epoch_vector["judge"] >= 2

    state_path = save_state(engine, tmp_path / "state.json")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["anchor_provider"] == {
        "provider_id": "toy-ground-truth",
        "version": "v1",
        "fingerprint": "sha256:toy-values-0-through-5-threshold-3",
    }
    assert "examples" not in state


@pytest.mark.asyncio
async def test_state_restores_only_with_same_runtime_contract(tmp_path) -> None:
    engine = build_toy_engine()
    await engine.run()
    path = save_state(engine, tmp_path / "state.json")

    fresh = build_toy_engine()
    restored = restore_state(fresh, path)
    assert restored.validation_outcomes == engine.validation_outcomes
    assert restored.epoch_vector == engine.epoch_vector
    assert len(restored.archive.records) == len(engine.archive.records)

    wrong = build_toy_engine(seed=99)
    with pytest.raises(ValueError, match="configuration changed"):
        restore_state(wrong, path)

    drifted_anchor = build_toy_engine()
    drifted_anchor.runtime.anchor_provider.fingerprint = "sha256:changed"
    with pytest.raises(ValueError, match="anchor provider"):
        restore_state(drifted_anchor, path)
