#!/usr/bin/env python3
"""Fast/checkpointed execution of scripts/31_current_analysis.py.

Same estimands and output schema as analysis 31, but:
- NumPy-vectorized paired bootstrap
- NumPy-vectorized paired sign-flip tests
- stage-level checkpoint files
- live progress JSON
- immediate CSV writes after every completed stage

No network/API/model/verifier calls.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"

ANALYSIS_PATH = (
    SCRIPT_DIR
    / "31_current_analysis.py"
)

OUT = (
    ROOT
    / "analysis"
    / "current"
    / "results"
)

PROGRESS = (
    ROOT
    / "analysis"
    / "current"
    / "analysis_progress.json"
)

LOG_VERSION = (
    "current-analysis-fast-1.0"
)


# ------------------------------------------------------------
# Load analysis 31 as a module
# ------------------------------------------------------------

spec = (
    importlib.util.spec_from_file_location(
        "current_analysis_31",
        ANALYSIS_PATH,
    )
)

if (
    spec is None
    or spec.loader is None
):
    raise RuntimeError(
        f"Cannot load {ANALYSIS_PATH}"
    )

analysis = (
    importlib.util.module_from_spec(
        spec
    )
)

sys.modules[
    spec.name
] = analysis

spec.loader.exec_module(
    analysis
)

core = analysis.core


# ------------------------------------------------------------
# Fast deterministic resampling
# ------------------------------------------------------------

BOOTSTRAPS = int(
    core.BOOTSTRAPS
)

SIGN_FLIP_DRAWS = int(
    analysis.SIGN_FLIP_DRAWS
)


def seed_from_text(
    prefix: str,
    text: str,
) -> int:
    digest = hashlib.sha256(
        (
            prefix
            + "|"
            + text
        ).encode(
            "utf-8"
        )
    ).digest()

    return int.from_bytes(
        digest[:8],
        "big",
        signed=False,
    )


def fast_paired_bootstrap_ci(
    deltas,
    *,
    seed_text,
):
    values = np.asarray(
        [
            float(x)
            for x in deltas
            if x is not None
        ],
        dtype=np.float64,
    )

    n = values.size

    if n == 0:
        return None, None

    if n == 1:
        value = float(
            values[0]
        )

        return value, value

    seed = seed_from_text(
        str(
            core.BOOTSTRAP_SEED
        ),
        seed_text,
    )

    rng = (
        np.random.default_rng(
            seed
        )
    )

    estimates = np.empty(
        BOOTSTRAPS,
        dtype=np.float64,
    )

    # Keeps memory bounded even on
    # larger paired groups.
    chunk = 5000

    start = 0

    while start < BOOTSTRAPS:
        size = min(
            chunk,
            BOOTSTRAPS
            - start,
        )

        indices = rng.integers(
            0,
            n,
            size=(
                size,
                n,
            ),
            dtype=np.int32,
        )

        estimates[
            start:
            start + size
        ] = (
            values[
                indices
            ].mean(
                axis=1
            )
        )

        start += size

    low, high = np.quantile(
        estimates,
        [
            0.025,
            0.975,
        ],
        method="linear",
    )

    return (
        float(low),
        float(high),
    )


def fast_sign_flip(
    deltas,
    *,
    seed_text,
):
    values = np.asarray(
        [
            float(x)
            for x in deltas
            if x is not None
        ],
        dtype=np.float64,
    )

    n = values.size

    if n == 0:
        return None

    observed = abs(
        float(
            values.mean()
        )
    )

    if np.isclose(
        observed,
        0.0,
        atol=1e-15,
    ):
        return 1.0

    # Exact paired randomization
    # for small samples.
    if n <= 20:
        total = (
            1 << n
        )

        extreme = 0

        bit_positions = (
            np.arange(
                n,
                dtype=np.uint64,
            )
        )

        chunk = 50_000

        for start in range(
            0,
            total,
            chunk,
        ):
            stop = min(
                total,
                start + chunk,
            )

            masks = np.arange(
                start,
                stop,
                dtype=np.uint64,
            )[:, None]

            bits = (
                (
                    masks
                    >> bit_positions
                )
                & 1
            )

            signs = (
                bits.astype(
                    np.float64
                )
                * 2.0
                - 1.0
            )

            estimates = (
                signs @ values
            ) / n

            extreme += int(
                np.count_nonzero(
                    np.abs(
                        estimates
                    )
                    >= (
                        observed
                        - 1e-15
                    )
                )
            )

        return (
            extreme
            / total
        )

    seed = (
        analysis.deterministic_seed(
            seed_text
        )
    )

    rng = (
        np.random.default_rng(
            seed
        )
    )

    extreme = 0
    done = 0
    chunk = 10_000

    while done < (
        SIGN_FLIP_DRAWS
    ):
        size = min(
            chunk,
            SIGN_FLIP_DRAWS
            - done,
        )

        signs = rng.integers(
            0,
            2,
            size=(
                size,
                n,
            ),
            dtype=np.int8,
        )

        signs = (
            signs.astype(
                np.float64
            )
            * 2.0
            - 1.0
        )

        estimates = (
            signs @ values
        ) / n

        extreme += int(
            np.count_nonzero(
                np.abs(
                    estimates
                )
                >= (
                    observed
                    - 1e-15
                )
            )
        )

        done += size

    # Add-one Monte Carlo
    # correction.
    return (
        extreme + 1
    ) / (
        SIGN_FLIP_DRAWS
        + 1
    )


# Patch only computational engines.
core.paired_bootstrap_ci = (
    fast_paired_bootstrap_ci
)

analysis.paired_sign_flip_p = (
    fast_sign_flip
)


# ------------------------------------------------------------
# Progress/checkpoint helpers
# ------------------------------------------------------------

TOTAL_STAGES = 8

completed = []
started_at = time.time()


def write_progress(
    stage,
    description,
    *,
    status="running",
    extra=None,
):
    payload = {
        "runner_version":
            LOG_VERSION,
        "status":
            status,
        "stage":
            stage,
        "total_stages":
            TOTAL_STAGES,
        "description":
            description,
        "completed_stages":
            list(completed),
        "elapsed_seconds":
            round(
                time.time()
                - started_at,
                1,
            ),
    }

    if extra:
        payload.update(
            extra
        )

    PROGRESS.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "=" * 80
    )
    print(
        f"[{stage}/{TOTAL_STAGES}] "
        f"{description}"
    )
    print(
        "=" * 80,
        flush=True,
    )


def finish_stage(
    stage,
    description,
    *,
    extra=None,
):
    completed.append(
        stage
    )

    write_progress(
        stage,
        description,
        status="completed_stage",
        extra=extra,
    )


def save_csv(
    filename,
    rows,
):
    core.write_csv(
        OUT / filename,
        rows,
    )

    print(
        f"  wrote {filename}: "
        f"{len(rows)} rows",
        flush=True,
    )


# ------------------------------------------------------------
# Main checkpointed analysis
# ------------------------------------------------------------

def main():
    if OUT.exists():
        shutil.rmtree(
            OUT
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Sources + matching
    # --------------------------------------------------------

    write_progress(
        1,
        "Load canonical cohorts and build matched pairs",
    )

    studies = {
        study:
            core.load_study(
                study
            )
        for study in (
            "primary",
            "resource",
            "replication",
        )
    }

    pairs = []

    for study, rows in (
        studies.items()
    ):
        pairs.extend(
            core.make_pairs(
                study,
                rows,
            )
        )

    finish_stage(
        1,
        "Load canonical cohorts and build matched pairs",
        extra={
            "matched_pair_records":
                len(pairs),
        },
    )

    # --------------------------------------------------------
    # 2. Cell performance + binary effects
    # --------------------------------------------------------

    write_progress(
        2,
        "Performance and matched binary effects",
    )

    cell_rows = (
        core.cell_summary(
            studies
        )
    )

    binary_rows = (
        analysis.corrected_binary_effects(
            pairs
        )
    )

    save_csv(
        "cell_performance.csv",
        cell_rows,
    )

    save_csv(
        "matched_binary_effects.csv",
        binary_rows,
    )

    finish_stage(
        2,
        "Performance and matched binary effects",
        extra={
            "binary_effect_rows":
                len(binary_rows),
        },
    )

    # --------------------------------------------------------
    # 3. Process effects
    # --------------------------------------------------------

    write_progress(
        3,
        "Canonical paired process effects",
    )

    process_rows = (
        analysis.canonical_process_effects(
            pairs
        )
    )

    focused_resource_rows = (
        analysis.focused_resource_process(
            process_rows
        )
    )

    save_csv(
        "matched_process_effects.csv",
        process_rows,
    )

    save_csv(
        "resource_focused_process.csv",
        focused_resource_rows,
    )

    finish_stage(
        3,
        "Canonical paired process effects",
        extra={
            "process_effect_rows":
                len(process_rows),
            "focused_resource_rows":
                len(
                    focused_resource_rows
                ),
        },
    )

    # --------------------------------------------------------
    # 4. Behavioral taxonomy
    # --------------------------------------------------------

    write_progress(
        4,
        "Behavior prevalence and matched behavior effects",
    )

    prevalence_rows = (
        core.behavior_prevalence(
            studies
        )
    )

    behavior_rows = (
        analysis.corrected_behavior_effects(
            pairs
        )
    )

    save_csv(
        "behavior_prevalence.csv",
        prevalence_rows,
    )

    save_csv(
        "matched_behavior_effects.csv",
        behavior_rows,
    )

    finish_stage(
        4,
        "Behavior prevalence and matched behavior effects",
        extra={
            "prevalence_rows":
                len(prevalence_rows),
            "behavior_effect_rows":
                len(behavior_rows),
        },
    )

    # --------------------------------------------------------
    # 5. Raw semantic jobs + consensus
    # --------------------------------------------------------

    write_progress(
        5,
        "Recompute semantic consensus from raw judge artifacts",
    )

    primary_jobs = (
        core.load_semantic_jobs(
            ROOT
            / "analysis"
            / "semantic-multijudge-v1"
            / "final-repaired-llama-v1",
            analysis.PRIMARY_SEMANTIC_FIELDS,
            "primary",
        )
    )

    resource_jobs = (
        core.load_semantic_jobs(
            ROOT
            / "analysis"
            / "semantic-resource-v1"
            / "full"
            / "production-v1.1",
            analysis.RESOURCE_SEMANTIC_FIELDS,
            "resource",
        )
    )

    (
        primary_consensus,
        primary_pooled_agreement,
        primary_distributions,
    ) = core.semantic_consensus(
        primary_jobs,
        analysis.PRIMARY_SEMANTIC_FIELDS,
    )

    (
        resource_consensus,
        resource_pooled_agreement,
        resource_distributions,
    ) = core.semantic_consensus(
        resource_jobs,
        analysis.RESOURCE_SEMANTIC_FIELDS,
    )

    consensus_rows = (
        primary_consensus
        + resource_consensus
    )

    save_csv(
        "semantic_jobs_primary.csv",
        primary_jobs,
    )

    save_csv(
        "semantic_jobs_resource.csv",
        resource_jobs,
    )

    save_csv(
        "semantic_consensus.csv",
        consensus_rows,
    )

    save_csv(
        "semantic_label_distribution.csv",
        (
            primary_distributions
            + resource_distributions
        ),
    )

    finish_stage(
        5,
        "Recompute semantic consensus from raw judge artifacts",
        extra={
            "primary_semantic_jobs":
                len(primary_jobs),
            "resource_semantic_jobs":
                len(resource_jobs),
            "consensus_trajectories":
                len(consensus_rows),
        },
    )

    # --------------------------------------------------------
    # 6. Reliability
    # --------------------------------------------------------

    write_progress(
        6,
        "Semantic reliability and coverage",
    )

    (
        primary_profile_agreement,
        primary_cell_agreement,
        primary_semantic_coverage,
    ) = (
        analysis.semantic_agreement_breakdowns(
            primary_jobs,
            analysis.PRIMARY_SEMANTIC_FIELDS,
        )
    )

    (
        resource_profile_agreement,
        resource_cell_agreement,
        resource_semantic_coverage,
    ) = (
        analysis.semantic_agreement_breakdowns(
            resource_jobs,
            analysis.RESOURCE_SEMANTIC_FIELDS,
        )
    )

    save_csv(
        "semantic_agreement_pooled.csv",
        (
            primary_pooled_agreement
            + resource_pooled_agreement
        ),
    )

    save_csv(
        "semantic_agreement_by_profile.csv",
        (
            primary_profile_agreement
            + resource_profile_agreement
        ),
    )

    save_csv(
        "semantic_agreement_by_cell.csv",
        (
            primary_cell_agreement
            + resource_cell_agreement
        ),
    )

    save_csv(
        "semantic_coverage.csv",
        (
            primary_semantic_coverage
            + resource_semantic_coverage
        ),
    )

    finish_stage(
        6,
        "Semantic reliability and coverage",
    )

    # --------------------------------------------------------
    # 7. Said X / Did Y
    # --------------------------------------------------------

    write_progress(
        7,
        "Said-X / Did-Y matched semantic-behavior analysis",
    )

    (
        said_did_pair_rows,
        said_did_summary_rows,
    ) = analysis.build_said_did(
        pairs,
        consensus_rows,
    )

    save_csv(
        "said_did_pairs.csv",
        said_did_pair_rows,
    )

    save_csv(
        "said_did_summary.csv",
        said_did_summary_rows,
    )

    finish_stage(
        7,
        "Said-X / Did-Y matched semantic-behavior analysis",
        extra={
            "said_did_pairs":
                len(
                    said_did_pair_rows
                ),
            "said_did_summary_rows":
                len(
                    said_did_summary_rows
                ),
            "figure_eligible":
                sum(
                    row[
                        "figure_eligible"
                    ]
                    == 1
                    for row in (
                        said_did_summary_rows
                    )
                ),
        },
    )

    # --------------------------------------------------------
    # 8. Synthesis + replication + manifest
    # --------------------------------------------------------

    write_progress(
        8,
        "Intervention synthesis, replication comparison, and manifest",
    )

    catalog_rows = (
        analysis.effect_catalog(
            binary_rows,
            behavior_rows,
            process_rows,
        )
    )

    replication_rows = (
        core.replication_direction(
            binary_rows,
            process_rows,
        )
    )

    save_csv(
        "intervention_effect_catalog.csv",
        catalog_rows,
    )

    save_csv(
        "replication_direction.csv",
        replication_rows,
    )

    manifest = {
        "analysis_version":
            analysis.VERSION,
        "fast_runner_version":
            LOG_VERSION,
        "source_root":
            str(
                ROOT
                / "analysis"
                / "current"
                / "source"
            ),
        "bootstrap_replicates":
            BOOTSTRAPS,
        "bootstrap_engine":
            "numpy_vectorized",
        "sign_flip_draws":
            SIGN_FLIP_DRAWS,
        "sign_flip_engine":
            "numpy_vectorized",
        "historical_aggregate_inputs":
            False,
        "historical_inference_inputs":
            False,
        "historical_semantic_summary_inputs":
            False,
        "network_calls":
            0,
        "api_calls":
            0,
        "agent_calls":
            0,
        "verifier_calls":
            0,
        "semantic_judge_calls":
            0,
        "success_endpoint":
            "overall_pass >= 1.0",
        "secondary_test_endpoint":
            "tests_reward >= 1.0",
        "process_test":
            (
                "paired sign-flip randomization "
                "test of zero mean delta"
            ),
        "semantic_consensus":
            (
                "two valid judges with exact "
                "field-label agreement"
            ),
        "said_did_status":
            (
                "descriptive post-treatment; "
                "not causal mediation"
            ),
        "replication_status":
            (
                "partial descriptive cohort; "
                "not pooled with primary"
            ),
        "llama_success_status":
            (
                "descriptive capability floor"
            ),
        "canonical_process_metrics":
            list(
                analysis.CANONICAL_PROCESS_METRICS
            ),
        "row_counts": {
            "cell_performance":
                len(cell_rows),
            "matched_binary_effects":
                len(binary_rows),
            "matched_process_effects":
                len(process_rows),
            "resource_focused_process":
                len(
                    focused_resource_rows
                ),
            "behavior_prevalence":
                len(
                    prevalence_rows
                ),
            "matched_behavior_effects":
                len(
                    behavior_rows
                ),
            "primary_semantic_jobs":
                len(primary_jobs),
            "resource_semantic_jobs":
                len(resource_jobs),
            "semantic_consensus":
                len(consensus_rows),
            "said_did_pairs":
                len(
                    said_did_pair_rows
                ),
            "said_did_summary":
                len(
                    said_did_summary_rows
                ),
            "intervention_effect_catalog":
                len(catalog_rows),
            "replication_direction":
                len(
                    replication_rows
                ),
        },
    }

    core.write_json(
        OUT / "manifest.json",
        manifest,
    )

    finish_stage(
        8,
        "Intervention synthesis, replication comparison, and manifest",
    )

    final_payload = {
        "runner_version":
            LOG_VERSION,
        "status":
            "PASS",
        "stage":
            TOTAL_STAGES,
        "total_stages":
            TOTAL_STAGES,
        "completed_stages":
            list(
                completed
            ),
        "elapsed_seconds":
            round(
                time.time()
                - started_at,
                1,
            ),
        "result_root":
            str(OUT),
    }

    PROGRESS.write_text(
        json.dumps(
            final_payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "=" * 80
    )
    print(
        "CURRENT ANALYSIS FAST: PASS"
    )
    print(
        "=" * 80
    )

    print(
        "elapsed:",
        round(
            time.time()
            - started_at,
            1,
        ),
        "seconds",
    )

    print(
        "outputs:",
        OUT,
    )


if __name__ == "__main__":
    main()
