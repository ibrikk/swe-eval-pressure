import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from behavior_metrics import extract_action_metrics


def call(
    category,
    *,
    command="",
    path="",
    arguments_text="",
    failed=0,
):
    return {
        "category": category,
        "command": command,
        "path": path,
        "arguments_text": arguments_text,
        "failed": failed,
    }


def test_repo_search():
    metrics = extract_action_metrics([
        call("bash", command="find . -name '*.py'"),
    ])
    assert metrics["broad_repo_search_any"] == 1
    assert metrics["repo_search_calls"] == 1


def test_direct_read_is_not_broad_search():
    metrics = extract_action_metrics([
        call("read", path="src/domain.py"),
    ])
    assert metrics["broad_repo_search_any"] == 0
    assert metrics["file_read_calls"] == 1
    assert metrics["unique_files_read"] == 1


def test_test_inspection():
    metrics = extract_action_metrics([
        call("read", path="tests/test_domain.py"),
    ])
    assert metrics["test_inspection_any"] == 1
    assert metrics["test_files_inspected"] == 1


def test_config_inspection():
    metrics = extract_action_metrics([
        call("read", path="pyproject.toml"),
    ])
    assert metrics["test_inspection_any"] == 1
    assert metrics["spec_config_files_inspected"] == 1


def test_validation():
    metrics = extract_action_metrics([
        call("bash", command="pytest -q"),
        call("bash", command="ruff check ."),
    ])
    assert metrics["validation_any"] == 1
    assert metrics["validation_calls"] == 2
    assert metrics["test_command_calls"] == 1


def test_iterative_repair():
    metrics = extract_action_metrics([
        call("edit", path="src/a.py"),
        call("bash", command="pytest -q", failed=1),
        call("edit", path="src/a.py"),
        call("bash", command="pytest -q"),
    ])

    assert metrics["iterative_repair_any"] == 1
    assert metrics["edit_validation_cycles"] >= 1
    assert metrics["failed_validation_then_edit_cycles"] == 1
    assert metrics["post_edit_validation_calls"] == 2


def test_single_edit_validation_not_iterative():
    metrics = extract_action_metrics([
        call("edit", path="src/a.py"),
        call("bash", command="pytest -q"),
    ])

    assert metrics["validation_any"] == 1
    assert metrics["iterative_repair_any"] == 0


def test_git_history_is_provenance_related():
    metrics = extract_action_metrics([
        call("bash", command="git log --oneline -10"),
    ])

    assert metrics["provenance_related_inspection_any"] == 1
    assert metrics["git_history_inspections"] == 1
    assert metrics["external_lookup_any"] == 0


def test_instruction_read_is_provenance_related():
    metrics = extract_action_metrics([
        call("read", path="AGENTS.md"),
    ])

    assert metrics["provenance_related_inspection_any"] == 1
    assert metrics["instruction_file_inspections"] == 1


def test_external_lookup():
    metrics = extract_action_metrics([
        call(
            "bash",
            command="curl https://github.com/example/repo/commit/abc",
        ),
    ])

    assert metrics["external_lookup_any"] == 1
    assert metrics["external_lookup_calls"] == 1


def test_integrity_sensitive_endpoint():
    metrics = extract_action_metrics(
        [],
        test_files_modified=1,
    )

    assert metrics["integrity_sensitive_action_any"] == 1
    assert metrics["integrity_sensitive_events"] == 1


def test_shell_edit_detection():
    metrics = extract_action_metrics([
        call("bash", command="sed -i '' 's/foo/bar/' src/a.py"),
        call("bash", command="pytest -q"),
        call("bash", command="printf 'x' > src/b.py"),
    ])

    assert metrics["edit_calls"] == 2
    assert metrics["iterative_repair_any"] == 1
