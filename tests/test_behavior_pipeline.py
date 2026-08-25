import sys
from importlib import util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PIPELINE_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "12_behavior_pipeline.py"
)

spec = util.spec_from_file_location(
    "behavior_pipeline12",
    PIPELINE_PATH,
)
assert spec is not None
assert spec.loader is not None

pipeline = util.module_from_spec(spec)
sys.modules[spec.name] = pipeline
spec.loader.exec_module(pipeline)


def test_default_profile_set():
    assert pipeline.DEFAULT_PROFILES == (
        "claude",
        "fable",
        "codex",
        "llama",
    )


def test_external_results_path():
    code = Path("/code/repo")
    project = Path("/project/repo")
    results = Path("/external/results")
    output = Path("/analysis/fable")

    command = pipeline.analyzer_command(
        python_executable="python",
        code_root=code,
        project_root=project,
        mode="resource",
        profile="fable",
        results_root=results,
        output_dir=output,
    )

    assert (
        command[
            command.index("--results-dir")
            + 1
        ]
        == str(results)
    )

    assert (
        command[
            command.index("--project-root")
            + 1
        ]
        == str(project)
    )

    assert str(code / "scripts" / "07_analyze.py") in command


def test_semantic_always_disabled():
    command = pipeline.analyzer_command(
        python_executable="python",
        code_root=Path("/code"),
        project_root=Path("/project"),
        mode="full",
        profile="claude",
        results_root=Path("/results"),
        output_dir=Path("/analysis"),
    )

    assert "--no-semantic" in command
    assert "--semantic" not in command


def test_live_forwarded():
    command = pipeline.analyzer_command(
        python_executable="python",
        code_root=Path("/code"),
        project_root=Path("/project"),
        mode="resource",
        profile="fable",
        results_root=Path("/results"),
        output_dir=Path("/analysis"),
        live=True,
    )

    assert "--live" in command


def test_profile_allowlist_forwarded():
    command = pipeline.analyzer_command(
        python_executable="python",
        code_root=Path("/code"),
        project_root=Path("/project"),
        mode="full",
        profile="codex",
        results_root=Path("/results"),
        output_dir=Path("/analysis"),
        strict_reconstruction=True,
        censored_allowlist=Path(
            "/tmp/codex-censored.txt"
        ),
    )

    assert "--strict-reconstruction" in command

    index = command.index(
        "--censored-task-allowlist"
    )

    assert (
        command[index + 1]
        == "/tmp/codex-censored.txt"
    )


def test_profile_path_parser():
    profile, path = (
        pipeline.parse_profile_path(
            "fable=/tmp/refusals.txt"
        )
    )

    assert profile == "fable"
    assert path == Path(
        "/tmp/refusals.txt"
    )

    import argparse

    with pytest.raises(
        argparse.ArgumentTypeError
    ):
        pipeline.parse_profile_path(
            "unknown=/tmp/x"
        )
