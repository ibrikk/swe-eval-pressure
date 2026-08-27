#!/usr/bin/env python3

import csv
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BASE = ROOT / "scripts" / "27_current_core_analysis.py"

spec = importlib.util.spec_from_file_location(
    "core27",
    BASE,
)

core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)

OUT = ROOT / "analysis" / "current" / "findings"
OUT.mkdir(parents=True, exist_ok=True)

COMPONENTS = (
    "seeded_cue_removed_or_modified",
    "evaluation_context_file_modified",
    "test_files_modified",
)

studies = {
    name: core.load_study(name)
    for name in (
        "primary",
        "resource",
    )
}

pairs = []

for study, rows in studies.items():
    pairs.extend(
        core.make_pairs(
            study,
            rows,
        )
    )

output = []

for component in COMPONENTS:
    groups = {}

    for pair in pairs:
        key = (
            pair["study"],
            pair["profile"],
            pair["contrast"],
            pair["placement"],
        )

        groups.setdefault(
            key,
            [],
        ).append(pair)

    for key, values in sorted(groups.items()):
        usable = []

        for pair in values:
            b = core.numeric(
                pair["baseline"].get(component)
            )

            t = core.numeric(
                pair["treatment"].get(component)
            )

            if b is None or t is None:
                continue

            usable.append(
                (
                    int(b > 0),
                    int(t > 0),
                )
            )

        if not usable:
            continue

        deltas = [
            float(t - b)
            for b, t in usable
        ]

        f2p = sum(
            b == 0 and t == 1
            for b, t in usable
        )

        p2f = sum(
            b == 1 and t == 0
            for b, t in usable
        )

        low, high = (
            core.paired_bootstrap_ci(
                deltas,
                seed_text=(
                    "integrity|"
                    + "|".join(key)
                    + "|"
                    + component
                ),
            )
        )

        output.append(
            {
                "study": key[0],
                "profile": key[1],
                "contrast": key[2],
                "placement": key[3],
                "component": component,
                "matched_n": len(usable),
                "baseline_rate": (
                    sum(b for b, _ in usable)
                    / len(usable)
                ),
                "treatment_rate": (
                    sum(t for _, t in usable)
                    / len(usable)
                ),
                "effect_pp": (
                    100
                    * sum(deltas)
                    / len(deltas)
                ),
                "ci95_low_pp": 100 * low,
                "ci95_high_pp": 100 * high,
                "fail_to_pass": f2p,
                "pass_to_fail": p2f,
                "mcnemar_p": core.exact_mcnemar(
                    f2p,
                    p2f,
                ),
            }
        )

path = OUT / "integrity_component_effects.csv"

fields = list(output[0])

with path.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fields,
    )
    writer.writeheader()
    writer.writerows(output)


def n(x):
    try:
        return float(x)
    except Exception:
        return None


print("=" * 100)
print("SOURCE-PLACEMENT INTEGRITY EFFECT DECOMPOSITION")
print("=" * 100)

interesting = [
    row
    for row in output
    if (
        row["study"] == "primary"
        and row["placement"] == "source"
        and row["contrast"]
        in {
            "financial_pressure",
            "self_preservation_pressure",
            "evaluation_disclosure",
        }
    )
]

interesting.sort(
    key=lambda r: (
        r["profile"],
        r["contrast"],
        r["component"],
    )
)

for row in interesting:
    if abs(row["effect_pp"]) < 1e-9:
        continue

    print(
        f"{row['profile']:7s} "
        f"{row['contrast']:29s} "
        f"{row['component']:38s} "
        f"n={row['matched_n']:2d} "
        f"Δ={row['effect_pp']:+7.2f}pp "
        f"CI=["
        f"{row['ci95_low_pp']:+7.2f}, "
        f"{row['ci95_high_pp']:+7.2f}] "
        f"p={row['mcnemar_p']:.5g}"
    )

print()
print("output:", path)
