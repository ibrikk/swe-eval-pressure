from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

spec = importlib.util.spec_from_file_location(
    "swe_eval_pressure_analyzer_replication",
    SCRIPTS / "07_analyze.py",
)

assert spec is not None
assert spec.loader is not None

analyzer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = analyzer
spec.loader.exec_module(analyzer)


def make_run(
    results: Path,
    name: str,
    *,
    replication_id: str | None,
    created_at: str,
) -> Path:
    run = results / name
    run.mkdir(parents=True)

    metadata = {
        "mode": "resource",
        "profile": "llama",
        "model": "model",
        "agent": "agent",
        "created_at": created_at,
        "harbor_repeats": 1,
        "verification_enabled": True,
    }

    if replication_id is not None:
        metadata["replication_id"] = replication_id

    (run / "run_metadata.json").write_text(
        json.dumps(metadata)
    )

    # Identical manifest/configuration -> identical study signature.
    (run / "study_manifest.json").write_text(
        json.dumps({
            "mode": "resource",
            "profile": "llama",
            "variants_per_task": 3,
            "tasks": [
                {
                    "directory": "task-a",
                    "base_task_id": "a",
                    "condition": "clean",
                    "channel": "none",
                }
            ],
        })
    )

    return run


class ReplicationIdentityTests(unittest.TestCase):

    def test_same_study_different_replications_are_not_merged(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            results = root / "results"
            results.mkdir()

            old = make_run(
                results,
                "replication-old",
                replication_id="rep-A",
                created_at="2026-08-25T00:00:00+00:00",
            )

            new = make_run(
                results,
                "replication-new",
                replication_id="rep-B",
                created_at="2026-08-26T00:00:00+00:00",
            )

            selected, excluded, warnings = (
                analyzer.discover_runs(
                    root,
                    results,
                    "resource",
                    "llama",
                    None,
                )
            )

            self.assertEqual(
                [r.root for r in selected],
                [new],
            )

            self.assertEqual(
                [r.root for r in excluded],
                [old],
            )

            self.assertEqual(
                selected[0].replication_id,
                "rep-B",
            )

            self.assertTrue(
                any(
                    "study/replication cohorts"
                    in warning
                    for warning in warnings
                )
            )

    def test_replication_shards_share_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            results = root / "results"
            results.mkdir()

            a = make_run(
                results,
                "shard-1",
                replication_id="rep-A",
                created_at="2026-08-26T00:00:00+00:00",
            )

            b = make_run(
                results,
                "shard-2",
                replication_id="rep-A",
                created_at="2026-08-26T01:00:00+00:00",
            )

            selected, excluded, _ = (
                analyzer.discover_runs(
                    root,
                    results,
                    "resource",
                    "llama",
                    None,
                )
            )

            self.assertEqual(
                {r.root for r in selected},
                {a, b},
            )

            self.assertEqual(excluded, [])

    def test_legacy_runs_remain_backward_compatible(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            results = root / "results"
            results.mkdir()

            a = make_run(
                results,
                "legacy-1",
                replication_id=None,
                created_at="2026-08-25T00:00:00+00:00",
            )

            b = make_run(
                results,
                "legacy-2",
                replication_id=None,
                created_at="2026-08-26T00:00:00+00:00",
            )

            selected, excluded, warnings = (
                analyzer.discover_runs(
                    root,
                    results,
                    "resource",
                    "llama",
                    None,
                )
            )

            self.assertEqual(
                {r.root for r in selected},
                {a, b},
            )

            self.assertEqual(excluded, [])

            self.assertTrue(
                any(
                    "legacy grouping"
                    in warning
                    for warning in warnings
                )
            )


if __name__ == "__main__":
    unittest.main()
