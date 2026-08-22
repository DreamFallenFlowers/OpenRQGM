import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "examples" / "paper_coding" / "run.py"
SPEC = importlib.util.spec_from_file_location("paper_coding_run", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_canonical_hash_is_order_independent() -> None:
    assert MODULE.canonical_hash({"a": 1, "b": 2}) == MODULE.canonical_hash({"b": 2, "a": 1})


def test_review_example_does_not_include_label() -> None:
    row = {
        "pull_request_title": "title",
        "patch": "diff",
        "description": "description",
        "hint": "hint",
        "label": "APPROVE",
    }
    assert "label" not in MODULE.review_example("example", row)


def test_codex_discovery_avoids_windows_store_alias() -> None:
    assert "WindowsApps" not in MODULE.find_codex_executable()
