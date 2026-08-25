import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

import semantic_dose as dose


def test_allocation():
    dose.validate_allocation()

    totals = Counter()

    for counts in (
        dose.CELL_PROFILE_COUNTS.values()
    ):
        assert sum(
            counts.values()
        ) == 10

        totals.update(counts)

    assert dict(totals) == {
        "claude": 25,
        "fable": 25,
        "codex": 25,
        "llama": 25,
    }


def test_hash_stable():
    first = dose.dose_hash(
        profile="claude",
        condition="clean",
        placement="none",
        trial_name="abc",
    )

    second = dose.dose_hash(
        profile="claude",
        condition="clean",
        placement="none",
        trial_name="abc",
    )

    assert first == second
    assert len(first) == 64


def test_no_network_code():
    source = (
        ROOT
        / "scripts"
        / "semantic_dose.py"
    ).read_text()

    forbidden = [
        "urlopen(",
        "invoke_judge(",
        "invoke_judge_raw(",
        "requests.",
        "httpx.",
    ]

    for token in forbidden:
        assert token not in source


SPEC_PATH = (
    ROOT
    / "config"
    / "semantic_dose_v1.json"
)


def test_spec_shape():
    spec = json.loads(
        SPEC_PATH.read_text()
    )

    trials = spec["trials"]

    assert (
        spec["selection_status"]
        == "frozen_before_dose_outcomes"
    )

    assert len(trials) == 100
    assert (
        spec["core_judge_jobs"]
        == 200
    )

    assert (
        spec[
            "frozen_pilot_count"
        ]
        == 10
    )

    assert (
        spec[
            "new_trajectory_count"
        ]
        == 90
    )

    assert Counter(
        t["profile"]
        for t in trials
    ) == {
        "claude": 25,
        "fable": 25,
        "codex": 25,
        "llama": 25,
    }

    cells = Counter(
        (
            t["condition"],
            t["placement"],
        )
        for t in trials
    )

    assert set(
        cells
    ) == set(
        dose.CELLS
    )

    assert all(
        n == 10
        for n in cells.values()
    )

    assert sum(
        bool(
            t[
                "from_frozen_pilot"
            ]
        )
        for t in trials
    ) == 10


def test_spec_hashes():
    spec = json.loads(
        SPEC_PATH.read_text()
    )

    for trial in spec["trials"]:
        assert (
            trial["dose_hash"]
            == dose.dose_hash(
                profile=trial[
                    "profile"
                ],
                condition=trial[
                    "condition"
                ],
                placement=trial[
                    "placement"
                ],
                trial_name=trial[
                    "trial_name"
                ],
            )
        )

        assert len(
            trial[
                "trajectory_hash"
            ]
        ) == 64


def test_pilot_subset():
    spec = json.loads(
        SPEC_PATH.read_text()
    )

    pilot_spec = json.loads(
        (
            ROOT
            / "config"
            / "semantic_pilot_v1.json"
        ).read_text()
    )

    dose_ids = {
        (
            t["profile"],
            t["trial_name"],
            t["trajectory_hash"],
        )
        for t in spec["trials"]
        if t[
            "from_frozen_pilot"
        ]
    }

    pilot_ids = {
        (
            t["profile"],
            t["trial_name"],
            t["trajectory_hash"],
        )
        for t
        in pilot_spec["trials"]
    }

    assert dose_ids == pilot_ids
