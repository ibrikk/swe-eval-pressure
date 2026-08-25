import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from behavior_stats import (  # noqa: E402
    benjamini_hochberg_adjust,
    exact_mcnemar_p,
    holm_adjust,
    paired_bootstrap_mean_difference,
    paired_differences,
    paired_log1p_differences,
    paired_sign_flip_p,
    summarize_binary_pairs,
    summarize_continuous_pairs,
)


def test_exact_mcnemar_matches_known_claude_self_root_case():
    # Eight control-only successes and zero treatment-only successes:
    # exact two-sided p = 2 * (1/2)^8 = 0.0078125.
    assert exact_mcnemar_p(0, 8) == pytest.approx(0.0078125)
    assert exact_mcnemar_p(8, 0) == pytest.approx(0.0078125)


def test_mcnemar_zero():
    assert exact_mcnemar_p(0, 0) == 1.0


def test_binary_summary_preserves_pairing():
    control =   [1, 1, 1, 0, 0, 0]
    treatment = [1, 0, 0, 1, 1, 0]

    result = summarize_binary_pairs(
        control,
        treatment,
        bootstrap_replicates=2000,
        bootstrap_seed=123,
    )

    assert result.n_pairs == 6
    assert result.control_positive == 3
    assert result.treatment_positive == 3
    assert result.delta_pp == pytest.approx(0.0)
    assert result.treatment_only_positive == 2
    assert result.control_only_positive == 2
    assert result.discordant_pairs == 4
    assert result.mcnemar_exact_p == pytest.approx(1.0)


def test_paired_differences_are_treatment_minus_control():
    assert paired_differences(
        [1, 2, 3],
        [4, 2, 1],
    ) == [3.0, 0.0, -2.0]


def test_bootstrap_is_reproducible():
    args = dict(
        control=[0, 0, 1, 1, 2],
        treatment=[1, 0, 1, 3, 2],
        replicates=1000,
        seed=42,
    )
    first = paired_bootstrap_mean_difference(**args)
    second = paired_bootstrap_mean_difference(**args)
    assert first == second


def test_direction_counts():
    result = summarize_continuous_pairs(
        control=[1, 2, 3, 4],
        treatment=[2, 2, 1, 7],
        bootstrap_replicates=1000,
        bootstrap_seed=42,
    )

    assert result.increased == 2
    assert result.unchanged == 1
    assert result.decreased == 1
    assert result.increased_fraction == pytest.approx(0.5)
    assert result.unchanged_fraction == pytest.approx(0.25)
    assert result.decreased_fraction == pytest.approx(0.25)


def test_sign_flip_exact_extreme_same_direction():
    # Three equal positive non-zero differences.
    # Only +++ and --- are as extreme as observed:
    # 2 / 2^3 = 0.25.
    assert paired_sign_flip_p(
        [1, 1, 1],
        exact_threshold=20,
    ) == pytest.approx(0.25)


def test_sign_flip_all_zero_is_one():
    assert paired_sign_flip_p([0, 0, 0]) == 1.0


def test_holm_adjustment():
    adjusted = holm_adjust([0.0078125, 0.05, 0.2])
    assert adjusted == pytest.approx([
        0.0234375,
        0.1,
        0.2,
    ])


def test_bh_adjustment():
    adjusted = benjamini_hochberg_adjust(
        [0.01, 0.04, 0.03]
    )
    assert adjusted == pytest.approx([
        0.03,
        0.04,
        0.04,
    ])


def test_log1p_difference():
    result = paired_log1p_differences(
        [0, 3],
        [1, 7],
    )

    assert result[0] == pytest.approx(math.log(2))
    assert result[1] == pytest.approx(math.log(8) - math.log(4))


def test_pair_length_mismatch_fails():
    with pytest.raises(ValueError):
        summarize_binary_pairs(
            [0, 1],
            [1],
        )


def test_binary_validation_fails_for_non_binary_value():
    with pytest.raises(ValueError):
        summarize_binary_pairs(
            [0, 2],
            [1, 1],
        )


def test_invalid_p_values_fail():
    with pytest.raises(ValueError):
        holm_adjust([0.1, -0.1])

    with pytest.raises(ValueError):
        benjamini_hochberg_adjust([0.1, 1.1])
