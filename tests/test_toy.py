import json

import pytest

from rqgm.persistence import restore_state, save_state, snapshot
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


@pytest.mark.parametrize("stop_at", [8, 11])
@pytest.mark.asyncio
async def test_interrupted_run_resumes_with_identical_rng_and_checkpoint_trajectory(
    tmp_path, stop_at
) -> None:
    class PlannedInterruption(RuntimeError):
        pass

    class InterruptingObserver:
        async def update(self, engine, event):  # type: ignore[no-untyped-def]
            if event == "evaluated" and engine.validation_outcomes == stop_at:
                save_state(engine, tmp_path / "state.json")
                raise PlannedInterruption

    interrupted = build_toy_engine(seed=17)
    interrupted.runtime.progress_observer = InterruptingObserver()
    with pytest.raises(PlannedInterruption):
        await interrupted.run()

    resumed = build_toy_engine(seed=17)
    restore_state(resumed, tmp_path / "state.json", require_resumable=True)
    await resumed.run()

    uninterrupted = build_toy_engine(seed=17)
    await uninterrupted.run()
    assert snapshot(resumed) == snapshot(uninterrupted)


def test_atomic_state_write_preserves_previous_snapshot_on_serialization_error(tmp_path) -> None:
    path = tmp_path / "state.json"
    engine = build_toy_engine()
    save_state(engine, path)
    previous = path.read_bytes()
    engine.archive.nodes["seed"].workspace = {"not_json": object()}
    with pytest.raises(TypeError):
        save_state(engine, path)
    assert path.read_bytes() == previous


def test_legacy_state_is_auditable_but_rejected_for_deterministic_resume(tmp_path) -> None:
    path = tmp_path / "state.json"
    engine = build_toy_engine()
    save_state(engine, path)
    state = json.loads(path.read_text(encoding="utf-8"))
    state["state_version"] = 1
    state.pop("rng_state")
    state.pop("initialized")
    path.write_text(json.dumps(state), encoding="utf-8")
    restore_state(build_toy_engine(), path)
    with pytest.raises(ValueError, match="deterministic resume"):
        restore_state(build_toy_engine(), path, require_resumable=True)


def test_workspace_view_excludes_validation_cache_but_keeps_lineage_feedback() -> None:
    engine = build_toy_engine()
    node = engine.archive.nodes["seed"]
    node.cached_artifacts["private-validation"] = {"large": "artifact"}
    node.training_feedback.append({"hint": "lineage only"})
    view = engine._workspace_view(node)
    assert view.cached_artifacts == {}
    assert view.training_feedback == [{"hint": "lineage only"}]
    assert view.training_feedback is not node.training_feedback
