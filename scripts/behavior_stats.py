#!/usr/bin/env python3
"""Paired statistical primitives for SWE-EvalPressure behavioral analysis.

All inference operates on matched base-task pairs. Tool calls or trajectory
events are never treated as independent statistical units.

This module deliberately has no dependency on semantic LLM judging.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence


DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260824
DEFAULT_SIGNFLIP_REPLICATES = 100_000
DEFAULT_SIGNFLIP_SEED = 20260824


def _as_finite_floats(values: Iterable[float | int], name: str) -> list[float]:
    out = [float(x) for x in values]
    if not out:
        raise ValueError(f"{name} must be non-empty")
    if not all(math.isfinite(x) for x in out):
        raise ValueError(f"{name} contains non-finite values")
    return out


def _validate_pair(
    control: Sequence[float | int],
    treatment: Sequence[float | int],
) -> tuple[list[float], list[float]]:
    c = _as_finite_floats(control, "control")
    t = _as_finite_floats(treatment, "treatment")
    if len(c) != len(t):
        raise ValueError(
            f"paired inputs must have equal length: "
            f"control={len(c)} treatment={len(t)}"
        )
    return c, t


def _validate_binary(values: Sequence[float], name: str) -> list[int]:
    if any(x not in (0.0, 1.0) for x in values):
        raise ValueError(f"{name} must contain only 0/1 values")
    return [int(x) for x in values]


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 1]."""
    if not values:
        raise ValueError("percentile requires non-empty values")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")

    ordered = sorted(float(x) for x in values)
    if len(ordered) == 1:
        return ordered[0]

    position = q * (len(ordered) - 1)
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]

    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def paired_differences(
    control: Sequence[float | int],
    treatment: Sequence[float | int],
) -> list[float]:
    c, t = _validate_pair(control, treatment)
    return [tx - cx for cx, tx in zip(c, t)]


def paired_bootstrap_mean_difference(
    control: Sequence[float | int],
    treatment: Sequence[float | int],
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI for mean paired difference.

    Resamples matched task-pair indices, preserving treatment/control pairing.
    """
    c, t = _validate_pair(control, treatment)

    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    differences = [tx - cx for cx, tx in zip(c, t)]
    n = len(differences)
    rng = random.Random(seed)

    boot = []
    for _ in range(replicates):
        sample_sum = 0.0
        for _ in range(n):
            sample_sum += differences[rng.randrange(n)]
        boot.append(sample_sum / n)

    return (
        percentile(boot, alpha / 2.0),
        percentile(boot, 1.0 - alpha / 2.0),
    )


def exact_mcnemar_p(
    treatment_only: int,
    control_only: int,
) -> float:
    """Exact two-sided McNemar p-value.

    Under the null, conditional on the number of discordant pairs, either
    direction is Binomial(n_discordant, 0.5).
    """
    if treatment_only < 0 or control_only < 0:
        raise ValueError("discordant counts must be non-negative")

    n = treatment_only + control_only
    if n == 0:
        return 1.0

    k = min(treatment_only, control_only)

    lower_tail = sum(
        math.comb(n, j) for j in range(k + 1)
    ) / (2 ** n)

    return min(1.0, 2.0 * lower_tail)


@dataclass(frozen=True)
class BinaryPairedResult:
    n_pairs: int
    control_positive: int
    treatment_positive: int
    control_prevalence: float
    treatment_prevalence: float
    delta: float
    delta_pp: float
    treatment_only_positive: int
    control_only_positive: int
    discordant_pairs: int
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    bootstrap_ci_low_pp: float
    bootstrap_ci_high_pp: float
    mcnemar_exact_p: float


def summarize_binary_pairs(
    control: Sequence[int],
    treatment: Sequence[int],
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> BinaryPairedResult:
    c_raw, t_raw = _validate_pair(control, treatment)
    c = _validate_binary(c_raw, "control")
    t = _validate_binary(t_raw, "treatment")

    n = len(c)
    c_pos = sum(c)
    t_pos = sum(t)

    treatment_only = sum(
        1 for cx, tx in zip(c, t)
        if cx == 0 and tx == 1
    )
    control_only = sum(
        1 for cx, tx in zip(c, t)
        if cx == 1 and tx == 0
    )

    delta = (t_pos - c_pos) / n

    ci_low, ci_high = paired_bootstrap_mean_difference(
        c,
        t,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )

    return BinaryPairedResult(
        n_pairs=n,
        control_positive=c_pos,
        treatment_positive=t_pos,
        control_prevalence=c_pos / n,
        treatment_prevalence=t_pos / n,
        delta=delta,
        delta_pp=100.0 * delta,
        treatment_only_positive=treatment_only,
        control_only_positive=control_only,
        discordant_pairs=treatment_only + control_only,
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        bootstrap_ci_low_pp=100.0 * ci_low,
        bootstrap_ci_high_pp=100.0 * ci_high,
        mcnemar_exact_p=exact_mcnemar_p(
            treatment_only,
            control_only,
        ),
    )


@dataclass(frozen=True)
class ContinuousPairedResult:
    n_pairs: int
    control_mean: float
    treatment_mean: float
    control_median: float
    treatment_median: float
    mean_delta: float
    median_delta: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    increased: int
    unchanged: int
    decreased: int
    increased_fraction: float
    unchanged_fraction: float
    decreased_fraction: float


def summarize_continuous_pairs(
    control: Sequence[float | int],
    treatment: Sequence[float | int],
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    zero_tolerance: float = 0.0,
) -> ContinuousPairedResult:
    c, t = _validate_pair(control, treatment)
    d = [tx - cx for cx, tx in zip(c, t)]
    n = len(d)

    if zero_tolerance < 0:
        raise ValueError("zero_tolerance must be non-negative")

    increased = sum(x > zero_tolerance for x in d)
    decreased = sum(x < -zero_tolerance for x in d)
    unchanged = n - increased - decreased

    ci_low, ci_high = paired_bootstrap_mean_difference(
        c,
        t,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )

    return ContinuousPairedResult(
        n_pairs=n,
        control_mean=statistics.mean(c),
        treatment_mean=statistics.mean(t),
        control_median=statistics.median(c),
        treatment_median=statistics.median(t),
        mean_delta=statistics.mean(d),
        median_delta=statistics.median(d),
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        increased=increased,
        unchanged=unchanged,
        decreased=decreased,
        increased_fraction=increased / n,
        unchanged_fraction=unchanged / n,
        decreased_fraction=decreased / n,
    )


def paired_log1p_differences(
    control: Sequence[float | int],
    treatment: Sequence[float | int],
) -> list[float]:
    """Return log1p(treatment) - log1p(control).

    Intended for non-negative, strongly skewed process metrics.
    """
    c, t = _validate_pair(control, treatment)
    if any(x < 0 for x in c + t):
        raise ValueError("log1p paired ratios require non-negative values")
    return [
        math.log1p(tx) - math.log1p(cx)
        for cx, tx in zip(c, t)
    ]


def paired_sign_flip_p(
    differences: Sequence[float | int],
    *,
    exact_threshold: int = 20,
    replicates: int = DEFAULT_SIGNFLIP_REPLICATES,
    seed: int = DEFAULT_SIGNFLIP_SEED,
) -> float:
    """Two-sided paired sign-flip/randomization p-value for mean difference.

    This is an exploratory test for continuous/count endpoints. Zero
    differences contribute no information and are removed.

    For <= exact_threshold non-zero differences, enumerate all sign
    assignments exactly. Otherwise use a fixed-seed Monte Carlo estimate with
    a +1 correction.

    The test relies on sign-exchangeability/symmetry under the null and is not
    used as the primary test for binary endpoints.
    """
    vals = _as_finite_floats(differences, "differences")
    vals = [x for x in vals if x != 0.0]

    if not vals:
        return 1.0
    if exact_threshold < 0:
        raise ValueError("exact_threshold must be non-negative")
    if replicates <= 0:
        raise ValueError("replicates must be positive")

    observed = abs(statistics.mean(vals))
    eps = 1e-15

    if len(vals) <= exact_threshold:
        extreme = 0
        total = 0
        for signs in product((-1.0, 1.0), repeat=len(vals)):
            stat = abs(
                sum(sign * value for sign, value in zip(signs, vals))
                / len(vals)
            )
            total += 1
            if stat + eps >= observed:
                extreme += 1
        return extreme / total

    rng = random.Random(seed)
    extreme = 0

    for _ in range(replicates):
        stat = abs(
            sum(
                value if rng.random() < 0.5 else -value
                for value in vals
            )
            / len(vals)
        )
        if stat + eps >= observed:
            extreme += 1

    return (extreme + 1) / (replicates + 1)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down family-wise adjusted p-values."""
    p = [float(x) for x in p_values]

    if any(not math.isfinite(x) or x < 0 or x > 1 for x in p):
        raise ValueError("p-values must be finite and in [0, 1]")
    if not p:
        return []

    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    adjusted = [1.0] * m

    running = 0.0
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * p[idx])
        running = max(running, candidate)
        adjusted[idx] = running

    return adjusted


def benjamini_hochberg_adjust(
    p_values: Sequence[float],
) -> list[float]:
    """Benjamini-Hochberg FDR adjusted p-values."""
    p = [float(x) for x in p_values]

    if any(not math.isfinite(x) or x < 0 or x > 1 for x in p):
        raise ValueError("p-values must be finite and in [0, 1]")
    if not p:
        return []

    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    adjusted = [1.0] * m

    running = 1.0
    for reverse_rank in range(m - 1, -1, -1):
        idx = order[reverse_rank]
        rank = reverse_rank + 1
        candidate = min(1.0, p[idx] * m / rank)
        running = min(running, candidate)
        adjusted[idx] = running

    return adjusted
