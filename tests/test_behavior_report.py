import csv
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(PROJECT_ROOT / "scripts"),
)

from behavior_metrics import (  # noqa: E402
    PRIMARY_BINARY_ENDPOINTS,
)
from importlib import util  # noqa: E402

REPORT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "10_behavior_report.py"
)

spec = util.spec_from_file_location(
    "behavior_report10",
    REPORT_PATH,
)
assert spec is not None
assert spec.loader is not None

report = util.module_from_spec(spec)
sys.modules[spec.name] = report
spec.loader.exec_module(report)


def binary_effect(
    endpoint,
    *,
    profile="fable",
    delta=-10.0,
    n=20,
):
    return {
        "analysis_schema_version": "2.6",
        "analysis_mode": "resource",
        "study_signature": "study-r",
        "profile": profile,
        "pair_type": "resource_effect",
        "channel": "scaffold",
        "baseline_condition": "eval_only",
        "baseline_channel": "scaffold",
        "treatment_condition": (
            "eval_resource_deprivation"
        ),
        "treatment_channel": "scaffold",
        "endpoint": endpoint,
        "n_pairs": str(n),
        "baseline_prevalence_pct": "80",
        "treatment_prevalence_pct": "70",
        "risk_difference_pp": str(delta),
        "bootstrap_ci_low_pp": "-20",
        "bootstrap_ci_high_pp": "0",
        "mcnemar_exact_p": "0.04",
        "holm_adjusted_p": "0.2",
        "adjusted_reject": "0",
    }


def test_primary_matrix_has_all_endpoints():
    rows = [
        binary_effect(endpoint)
        for endpoint
        in PRIMARY_BINARY_ENDPOINTS
    ]

    matrix = report.primary_effect_matrix(
        rows
    )

    assert len(matrix) == 1

    result = matrix[0]

    for endpoint in PRIMARY_BINARY_ENDPOINTS:
        assert (
            f"{endpoint}__delta_pp"
            in result
        )
        assert (
            f"{endpoint}__holm_p"
            in result
        )


def test_primary_matrix_uses_adjusted_flag_verbatim():
    rows = [
        binary_effect(endpoint)
        for endpoint
        in PRIMARY_BINARY_ENDPOINTS
    ]

    target = rows[0]
    target["mcnemar_exact_p"] = "0.001"
    target["holm_adjusted_p"] = "0.2"
    target["adjusted_reject"] = "0"

    matrix = report.primary_effect_matrix(
        rows
    )[0]

    endpoint = PRIMARY_BINARY_ENDPOINTS[0]

    assert (
        matrix[
            f"{endpoint}__mcnemar_p"
        ]
        == "0.001"
    )
    assert (
        matrix[
            f"{endpoint}__holm_p"
        ]
        == "0.2"
    )
    assert (
        matrix[
            f"{endpoint}__adjusted_reject"
        ]
        == "0"
    )


def test_primary_matrix_rejects_missing_endpoint():
    rows = [
        binary_effect(endpoint)
        for endpoint
        in PRIMARY_BINARY_ENDPOINTS[:-1]
    ]

    with pytest.raises(
        ValueError,
        match="missing endpoints",
    ):
        report.primary_effect_matrix(rows)


def test_primary_matrix_rejects_mixed_denominators():
    rows = [
        binary_effect(endpoint)
        for endpoint
        in PRIMARY_BINARY_ENDPOINTS
    ]

    rows[-1]["n_pairs"] = "19"

    with pytest.raises(
        ValueError,
        match="inconsistent matched denominators",
    ):
        report.primary_effect_matrix(rows)


def write_table(
    path: Path,
    rows: list[dict[str, str]],
):
    fields = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def test_discovery_accepts_single_profile_dir(
    tmp_path,
):
    profile = tmp_path / "fable"
    profile.mkdir()

    common = {
        "analysis_schema_version": "2.6",
        "analysis_mode": "resource",
        "study_signature": "study-r",
        "profile": "fable",
    }

    for filename in report.REQUIRED_FILES:
        if (
            filename
            == "behavior_binary_effects.csv"
        ):
            rows = [
                binary_effect(endpoint)
                for endpoint
                in PRIMARY_BINARY_ENDPOINTS
            ]
        else:
            rows = [dict(common)]

        write_table(
            profile / filename,
            rows,
        )

    discovered = (
        report.discover_profile_dirs(
            profile
        )
    )

    assert discovered == [profile]
