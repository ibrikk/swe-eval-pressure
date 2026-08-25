#!/usr/bin/env python3
"""Portable deterministic SWE-EvalPressure behavioral-analysis pipeline.

The code checkout, experiment results, analysis outputs, and report outputs
may live in different directories.

This wrapper performs no semantic LLM judging. It orchestrates:

    07_analyze.py
        ->
    10_behavior_report.py
        ->
    11_behavior_html.py

and records the exact paths/provenance used for the reconstruction.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


CODE_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PROFILES = (
    "claude",
    "fable",
    "codex",
    "llama",
)

ANALYZER_REQUIRED_FILES = (
    "behavior_trials.csv",
    "behavior_prevalence.csv",
    "matched_behavior_pairs.csv",
    "behavior_binary_effects.csv",
    "behavior_secondary_effects.csv",
    "behavior_multiplicity.csv",
)

SYNTHESIS_REQUIRED_FILES = (
    "behavior_report_inventory.csv",
    "behavior_primary_effects_all.csv",
    "behavior_primary_matrix.csv",
    "behavior_secondary_effects_all.csv",
    "behavior_multiplicity_all.csv",
    "behavior_prevalence_all.csv",
    "matched_behavior_pairs_all.csv",
)


def git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(path),
                "rev-parse",
                "HEAD",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def parse_profile_path(
    value: str,
) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "expected PROFILE=/path/to/file"
        )

    profile, raw_path = value.split(
        "=",
        1,
    )

    profile = profile.strip()

    if profile not in DEFAULT_PROFILES:
        raise argparse.ArgumentTypeError(
            f"unknown profile {profile!r}"
        )

    if not raw_path.strip():
        raise argparse.ArgumentTypeError(
            "path must not be empty"
        )

    return (
        profile,
        Path(raw_path).expanduser(),
    )


def analyzer_command(
    *,
    python_executable: str,
    code_root: Path,
    project_root: Path,
    mode: str,
    profile: str,
    results_root: Path,
    output_dir: Path,
    live: bool = False,
    manifest: Path | None = None,
    strict_reconstruction: bool = False,
    censored_allowlist: Path | None = None,
) -> list[str]:
    command = [
        python_executable,
        str(
            code_root
            / "scripts"
            / "07_analyze.py"
        ),
        "--project-root",
        str(project_root),
        "--mode",
        mode,
        "--profile",
        profile,
        "--results-dir",
        str(results_root),
        "--output-dir",
        str(output_dir),
        "--no-semantic",
    ]

    if manifest is not None:
        command.extend([
            "--manifest",
            str(manifest),
        ])

    if strict_reconstruction:
        command.append(
            "--strict-reconstruction"
        )

    if censored_allowlist is not None:
        command.extend([
            "--censored-task-allowlist",
            str(censored_allowlist),
        ])

    if live:
        command.append("--live")

    return command


def synthesis_command(
    *,
    python_executable: str,
    code_root: Path,
    analysis_root: Path,
    synthesis_dir: Path,
) -> list[str]:
    return [
        python_executable,
        str(
            code_root
            / "scripts"
            / "10_behavior_report.py"
        ),
        "--analysis-root",
        str(analysis_root),
        "--output-dir",
        str(synthesis_dir),
    ]


def html_command(
    *,
    python_executable: str,
    code_root: Path,
    synthesis_dir: Path,
    html_path: Path,
    title: str,
) -> list[str]:
    return [
        python_executable,
        str(
            code_root
            / "scripts"
            / "11_behavior_html.py"
        ),
        "--synthesis-dir",
        str(synthesis_dir),
        "--output",
        str(html_path),
        "--title",
        title,
    ]


def print_command(
    command: Sequence[str],
) -> None:
    print(
        "$ "
        + " ".join(
            shlex.quote(part)
            for part in command
        )
    )


def require_files(
    root: Path,
    names: Sequence[str],
) -> None:
    missing = [
        name
        for name in names
        if not (root / name).is_file()
    ]

    if missing:
        raise RuntimeError(
            f"{root}: expected output files missing: "
            + ", ".join(missing)
        )


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    dry_run: bool,
) -> None:
    print_command(command)

    if dry_run:
        return

    subprocess.run(
        list(command),
        cwd=cwd,
        check=True,
    )


def run_pipeline(
    *,
    mode: str,
    profiles: Sequence[str],
    project_root: Path,
    results_root: Path,
    analysis_root: Path,
    output_root: Path,
    live: bool,
    manifest: Path | None,
    strict_reconstruction: bool,
    censored_allowlists: dict[str, Path],
    title: str,
    dry_run: bool,
    code_root: Path = CODE_ROOT,
    python_executable: str = sys.executable,
) -> dict[str, object]:
    code_root = code_root.resolve()
    project_root = project_root.resolve()
    results_root = results_root.resolve()
    analysis_root = analysis_root.resolve()
    output_root = output_root.resolve()

    if manifest is not None:
        manifest = manifest.resolve()

    censored_allowlists = {
        profile: path.resolve()
        for profile, path
        in censored_allowlists.items()
    }

    unknown_profiles = sorted(
        set(profiles)
        - set(DEFAULT_PROFILES)
    )

    if unknown_profiles:
        raise ValueError(
            "unknown profiles: "
            + ", ".join(
                unknown_profiles
            )
        )

    if not profiles:
        raise ValueError(
            "at least one profile is required"
        )

    if not dry_run:
        if not results_root.exists():
            raise FileNotFoundError(
                results_root
            )

        if manifest is not None:
            if not manifest.is_file():
                raise FileNotFoundError(
                    manifest
                )

        for path in (
            censored_allowlists.values()
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    path
                )

    analysis_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile_outputs = {}

    for profile in profiles:
        profile_output = (
            analysis_root / profile
        )

        profile_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        command = analyzer_command(
            python_executable=(
                python_executable
            ),
            code_root=code_root,
            project_root=project_root,
            mode=mode,
            profile=profile,
            results_root=results_root,
            output_dir=profile_output,
            live=live,
            manifest=manifest,
            strict_reconstruction=(
                strict_reconstruction
            ),
            censored_allowlist=(
                censored_allowlists.get(
                    profile
                )
            ),
        )

        run_command(
            command,
            cwd=code_root,
            dry_run=dry_run,
        )

        if not dry_run:
            require_files(
                profile_output,
                ANALYZER_REQUIRED_FILES,
            )

        profile_outputs[profile] = str(
            profile_output
        )

    synthesis_dir = (
        output_root / "synthesis"
    )

    command = synthesis_command(
        python_executable=(
            python_executable
        ),
        code_root=code_root,
        analysis_root=analysis_root,
        synthesis_dir=synthesis_dir,
    )

    run_command(
        command,
        cwd=code_root,
        dry_run=dry_run,
    )

    if not dry_run:
        require_files(
            synthesis_dir,
            SYNTHESIS_REQUIRED_FILES,
        )

    html_path = (
        output_root
        / "SWE_EvalPressure_Behavior_PreRead.html"
    )

    command = html_command(
        python_executable=(
            python_executable
        ),
        code_root=code_root,
        synthesis_dir=synthesis_dir,
        html_path=html_path,
        title=title,
    )

    run_command(
        command,
        cwd=code_root,
        dry_run=dry_run,
    )

    if (
        not dry_run
        and not html_path.is_file()
    ):
        raise RuntimeError(
            f"HTML output missing: {html_path}"
        )

    provenance = {
        "pipeline_schema_version": "1.0",
        "mode": mode,
        "profiles": list(profiles),
        "live": bool(live),
        "strict_reconstruction": bool(
            strict_reconstruction
        ),
        "semantic_analysis": False,
        "code_root": str(code_root),
        "project_root": str(
            project_root
        ),
        "results_root": str(
            results_root
        ),
        "analysis_root": str(
            analysis_root
        ),
        "output_root": str(
            output_root
        ),
        "synthesis_dir": str(
            synthesis_dir
        ),
        "html_path": str(
            html_path
        ),
        "manifest": (
            str(manifest)
            if manifest is not None
            else ""
        ),
        "censored_task_allowlists": {
            profile: str(path)
            for profile, path
            in sorted(
                censored_allowlists.items()
            )
        },
        "code_git_commit": (
            git_commit(code_root)
        ),
        "project_git_commit": (
            git_commit(project_root)
        ),
        "profile_outputs": (
            profile_outputs
        ),
    }

    if not dry_run:
        (
            output_root
            / "pipeline_provenance.json"
        ).write_text(
            json.dumps(
                provenance,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic SWE-EvalPressure "
            "behavior analysis and HTML reporting."
        )
    )

    parser.add_argument(
        "mode",
        choices=[
            "full",
            "resource",
        ],
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=CODE_ROOT,
        help=(
            "Repository/data root used by the "
            "analyzer. Defaults to this checkout."
        ),
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        help=(
            "Raw results directory. Defaults to "
            "<project-root>/results/<mode>."
        ),
    )

    parser.add_argument(
        "--analysis-root",
        type=Path,
        help=(
            "Per-profile canonical analyzer output. "
            "Defaults to "
            "<project-root>/analysis/behavior/<mode>."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Synthesis + HTML output. Defaults to "
            "<project-root>/reports/behavior/<mode>."
        ),
    )

    parser.add_argument(
        "--profile",
        action="append",
        choices=DEFAULT_PROFILES,
        help=(
            "Profile to analyze. Repeat as needed. "
            "Defaults to all four profiles."
        ),
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Explicitly allow a partial reconstruction."
        ),
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Optional common manifest for legacy or "
            "external result reconstruction."
        ),
    )

    parser.add_argument(
        "--strict-reconstruction",
        action="store_true",
    )

    parser.add_argument(
        "--censored-task-allowlist",
        action="append",
        default=[],
        type=parse_profile_path,
        metavar="PROFILE=PATH",
        help=(
            "Per-profile censored-task allowlist. "
            "Repeat as needed."
        ),
    )

    parser.add_argument(
        "--title",
        help="HTML report title.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print all resolved commands without "
            "executing them."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = (
        args.project_root
        .expanduser()
        .resolve()
    )

    results_root = (
        args.results_root.expanduser()
        if args.results_root
        else (
            project_root
            / "results"
            / args.mode
        )
    )

    analysis_root = (
        args.analysis_root.expanduser()
        if args.analysis_root
        else (
            project_root
            / "analysis"
            / "behavior"
            / args.mode
        )
    )

    output_root = (
        args.output_root.expanduser()
        if args.output_root
        else (
            project_root
            / "reports"
            / "behavior"
            / args.mode
        )
    )

    profiles = (
        tuple(args.profile)
        if args.profile
        else DEFAULT_PROFILES
    )

    allowlists: dict[str, Path] = {}

    for profile, path in (
        args.censored_task_allowlist
    ):
        if profile in allowlists:
            raise SystemExit(
                "duplicate censored-task allowlist "
                f"for profile {profile}"
            )

        allowlists[profile] = (
            path.expanduser()
        )

    title = (
        args.title
        or (
            "SWE-EvalPressure Behavioral Analysis "
            f"— {args.mode.title()}"
        )
    )

    provenance = run_pipeline(
        mode=args.mode,
        profiles=profiles,
        project_root=project_root,
        results_root=results_root,
        analysis_root=analysis_root,
        output_root=output_root,
        live=args.live,
        manifest=(
            args.manifest.expanduser()
            if args.manifest
            else None
        ),
        strict_reconstruction=(
            args.strict_reconstruction
        ),
        censored_allowlists=(
            allowlists
        ),
        title=title,
        dry_run=args.dry_run,
    )

    print()
    print("Behavior pipeline complete")
    print(
        "mode:",
        provenance["mode"],
    )
    print(
        "profiles:",
        ", ".join(
            provenance["profiles"]
        ),
    )
    print(
        "results:",
        provenance["results_root"],
    )
    print(
        "analysis:",
        provenance["analysis_root"],
    )
    print(
        "report:",
        provenance["html_path"],
    )


if __name__ == "__main__":
    main()
