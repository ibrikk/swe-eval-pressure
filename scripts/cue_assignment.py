#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

LANGUAGE_TAGS = ("Go", "TypeScript", "Python", "C", "C++", "JavaScript")
EXTENSION_LANGUAGE = {
    ".go": "Go",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".py": "Python",
    ".c": "C",
    ".cc": "C++",
    ".cpp": "C++",
    ".jsx": "JavaScript",
    ".js": "JavaScript",
    ".cue": "CUE",
    ".yml": "YAML",
    ".yaml": "YAML",
    ".toml": "TOML",
    ".am": "Automake",
}
STRATIFICATION_FIELDS = ("repository", "language", "difficulty", "source_extension")
FORBIDDEN_NEUTRAL_PHRASES = (
    "your deployment depends",
    "may be replaced",
    "funding depends",
    "revenue depends",
    "must outperform",
    "survival depends",
)
FIELD_WEIGHTS = {
    "repository": 4.0,
    "language": 3.0,
    "difficulty": 1.0,
    "source_extension": 2.0,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def short_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:16]


def language_for(record: dict[str, Any]) -> str:
    tags = set(record.get("tags", []))
    for language in LANGUAGE_TAGS:
        if language in tags:
            return language
    return EXTENSION_LANGUAGE.get(str(record.get("source_extension", "")).lower(), "Other")


def normalized_task(record: dict[str, Any]) -> dict[str, str]:
    return {
        "task_id": str(record["task_id"]),
        "repository": str(record.get("repository", "")),
        "language": language_for(record),
        "difficulty": str(record.get("difficulty", "")),
        "source_extension": str(record.get("source_extension", "")),
    }


def load_library(path: Path) -> dict[str, Any]:
    library = json.loads(path.read_text(encoding="utf-8"))
    cues = library.get("cues", [])
    mechanisms = library.get("mechanisms", [])
    errors: list[str] = []
    if len(mechanisms) != 5:
        errors.append(f"expected 5 mechanisms, found {len(mechanisms)}")
    if len(cues) != 10:
        errors.append(f"expected 10 explicit cues, found {len(cues)}")
    ids = [str(c.get("id", "")) for c in cues]
    if len(ids) != len(set(ids)):
        errors.append("cue IDs are not unique")
    source_ids = set(library.get("source_papers", {}))
    for mechanism in mechanisms:
        basis = mechanism.get("literature_basis", {})
        if not basis:
            errors.append(f"{mechanism.get('id')}: missing literature_basis")
        for source in basis.get("sources", []):
            if source.get("source_id") not in source_ids:
                errors.append(
                    f"{mechanism.get('id')}: unknown literature source {source.get('source_id')}"
                )
    for cue in cues:
        text = str(cue.get("text", "")).lower()
        for phrase in FORBIDDEN_NEUTRAL_PHRASES:
            if phrase in text:
                errors.append(f"{cue.get('id')}: neutral cue contains pressure language: {phrase}")
    shape = Counter(c.get("mechanism") for c in cues)
    for mechanism in [m.get("id") for m in mechanisms]:
        if shape[mechanism] != 2:
            errors.append(f"{mechanism}: expected 2 explicit cues, found {shape[mechanism]}")
    if any(c.get("level") != "explicit" for c in cues):
        errors.append("all released evaluation cues must have level=explicit")
    if errors:
        raise ValueError("invalid cue library: " + "; ".join(errors))
    return library


def cues_for_level(library: dict[str, Any], level: str = "explicit") -> list[dict[str, Any]]:
    if level != "explicit":
        raise ValueError("SWE-EvalPressure releases only the explicit evaluation-cue condition")
    mechanism_order = {m["id"]: i for i, m in enumerate(library["mechanisms"])}
    cues = [dict(c) for c in library["cues"]]
    cues.sort(key=lambda c: (mechanism_order[c["mechanism"]], c["wording"]))
    return cues


def registry_fingerprint(records: Iterable[dict[str, Any]]) -> str:
    payload = [normalized_task(record) for record in sorted(records, key=lambda x: x["task_id"])]
    return short_hash(payload)


def _objective(
    tasks: list[dict[str, str]],
    cue_ids: list[str],
    cue_to_mechanism: dict[str, str],
    assignment: dict[str, str],
) -> float:
    n_cues = len(cue_ids)
    mechanisms = sorted(set(cue_to_mechanism.values()))
    n_mechanisms = len(mechanisms)
    totals = {
        field: Counter(task[field] for task in tasks)
        for field in STRATIFICATION_FIELDS
    }
    by_id = {task["task_id"]: task for task in tasks}
    cue_counts: dict[str, dict[str, Counter[str]]] = {
        cue_id: {field: Counter() for field in STRATIFICATION_FIELDS}
        for cue_id in cue_ids
    }
    mechanism_counts: dict[str, dict[str, Counter[str]]] = {
        mechanism: {field: Counter() for field in STRATIFICATION_FIELDS}
        for mechanism in mechanisms
    }
    for task_id, cue_id in assignment.items():
        task = by_id[task_id]
        mechanism = cue_to_mechanism[cue_id]
        for field in STRATIFICATION_FIELDS:
            cue_counts[cue_id][field][task[field]] += 1
            mechanism_counts[mechanism][field][task[field]] += 1

    score = 0.0
    for field in STRATIFICATION_FIELDS:
        weight = FIELD_WEIGHTS[field]
        for value, total in totals[field].items():
            cue_target = total / n_cues
            mechanism_target = total / n_mechanisms
            cue_denominator = max(cue_target, 0.5)
            mechanism_denominator = max(mechanism_target, 0.5)
            for cue_id in cue_ids:
                delta = cue_counts[cue_id][field][value] - cue_target
                score += weight * (delta * delta) / cue_denominator
            for mechanism in mechanisms:
                delta = mechanism_counts[mechanism][field][value] - mechanism_target
                score += 0.75 * weight * (delta * delta) / mechanism_denominator
    return score


def build_assignment(
    records: list[dict[str, Any]],
    cues: list[dict[str, Any]],
    seed: int,
) -> dict[str, str]:
    if not records:
        raise ValueError("cannot assign cues to an empty task registry")
    if len(records) % len(cues) != 0:
        raise ValueError(f"{len(records)} tasks cannot be divided evenly across {len(cues)} cues")

    rng = random.Random(seed)
    tasks = [normalized_task(record) for record in records]
    cue_ids = [str(cue["id"]) for cue in cues]
    cue_to_mechanism = {str(cue["id"]): str(cue["mechanism"]) for cue in cues}
    capacity = len(tasks) // len(cue_ids)

    totals = {
        field: Counter(task[field] for task in tasks)
        for field in STRATIFICATION_FIELDS
    }
    tie_break = {task["task_id"]: rng.random() for task in tasks}
    tasks.sort(
        key=lambda task: (
            totals["repository"][task["repository"]],
            totals["language"][task["language"]],
            totals["source_extension"][task["source_extension"]],
            totals["difficulty"][task["difficulty"]],
            tie_break[task["task_id"]],
            task["task_id"],
        )
    )

    assignment: dict[str, str] = {}
    cue_load = Counter()
    cue_strata = {
        cue_id: {field: Counter() for field in STRATIFICATION_FIELDS}
        for cue_id in cue_ids
    }
    mechanism_strata: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {field: Counter() for field in STRATIFICATION_FIELDS}
    )

    for task in tasks:
        candidates = [cue_id for cue_id in cue_ids if cue_load[cue_id] < capacity]
        rng.shuffle(candidates)
        best_cue = ""
        best_cost = float("inf")
        for cue_id in candidates:
            mechanism = cue_to_mechanism[cue_id]
            cost = 0.0
            for field in STRATIFICATION_FIELDS:
                value = task[field]
                total = totals[field][value]
                cue_target = total / len(cue_ids)
                mechanism_target = total / len(set(cue_to_mechanism.values()))
                old_cue = cue_strata[cue_id][field][value]
                old_mechanism = mechanism_strata[mechanism][field][value]
                weight = FIELD_WEIGHTS[field]
                cost += weight * (((old_cue + 1 - cue_target) ** 2) - ((old_cue - cue_target) ** 2)) / max(cue_target, 0.5)
                cost += 0.75 * weight * (((old_mechanism + 1 - mechanism_target) ** 2) - ((old_mechanism - mechanism_target) ** 2)) / max(mechanism_target, 0.5)
            cost += 0.05 * cue_load[cue_id]
            if cost < best_cost:
                best_cost = cost
                best_cue = cue_id
        assignment[task["task_id"]] = best_cue
        cue_load[best_cue] += 1
        mechanism = cue_to_mechanism[best_cue]
        for field in STRATIFICATION_FIELDS:
            cue_strata[best_cue][field][task[field]] += 1
            mechanism_strata[mechanism][field][task[field]] += 1

    # Deterministic local improvement. Swaps retain exactly equal cue sizes.
    task_ids = sorted(assignment)
    score = _objective(tasks, cue_ids, cue_to_mechanism, assignment)
    for _ in range(30000):
        left, right = rng.sample(task_ids, 2)
        if assignment[left] == assignment[right]:
            continue
        assignment[left], assignment[right] = assignment[right], assignment[left]
        candidate_score = _objective(tasks, cue_ids, cue_to_mechanism, assignment)
        if candidate_score + 1e-12 < score:
            score = candidate_score
        else:
            assignment[left], assignment[right] = assignment[right], assignment[left]

    counts = Counter(assignment.values())
    if set(counts.values()) != {capacity}:
        raise AssertionError(f"unequal cue assignment: {counts}")
    return assignment


def assignment_payload(
    records: list[dict[str, Any]],
    library: dict[str, Any],
    level: str,
    seed: int,
) -> dict[str, Any]:
    cues = cues_for_level(library, level)
    assignment = build_assignment(records, cues, seed)
    cue_by_id = {cue["id"]: cue for cue in cues}
    mechanism_by_id = {m["id"]: m for m in library["mechanisms"]}
    record_by_id = {record["task_id"]: record for record in records}

    tasks = []
    for task_id in sorted(assignment):
        record = record_by_id[task_id]
        cue = cue_by_id[assignment[task_id]]
        mechanism = mechanism_by_id[cue["mechanism"]]
        tasks.append(
            {
                "base_task_id": task_id,
                "repository": record.get("repository", ""),
                "language": language_for(record),
                "difficulty": record.get("difficulty", ""),
                "source_extension": record.get("source_extension", ""),
                "cue_id": cue["id"],
                "cue_mechanism": cue["mechanism"],
                "cue_mechanism_label": mechanism["label"],
                "cue_level": cue["level"],
                "cue_wording": cue["wording"],
                "cue_text": cue["text"],
            }
        )

    cue_counts = Counter(item["cue_id"] for item in tasks)
    mechanism_counts = Counter(item["cue_mechanism"] for item in tasks)
    balance_by_cue = {
        cue_id: {
            field: dict(sorted(Counter(item[field] for item in tasks if item["cue_id"] == cue_id).items()))
            for field in STRATIFICATION_FIELDS
        }
        for cue_id in [cue["id"] for cue in cues]
    }
    balance_by_mechanism = {
        mechanism["id"]: {
            field: dict(sorted(Counter(item[field] for item in tasks if item["cue_mechanism"] == mechanism["id"]).items()))
            for field in STRATIFICATION_FIELDS
        }
        for mechanism in library["mechanisms"]
    }
    return {
        "schema_version": "1.0",
        "assignment_seed": seed,
        "cue_level": level,
        "stratification_fields": list(STRATIFICATION_FIELDS),
        "registry_fingerprint": registry_fingerprint(records),
        "cue_library_fingerprint": short_hash(library),
        "base_task_count": len(records),
        "cue_count": len(cues),
        "tasks_per_cue": len(records) // len(cues),
        "cue_ids": [cue["id"] for cue in cues],
        "summary": {
            "base_tasks_by_cue": dict(sorted(cue_counts.items())),
            "base_tasks_by_mechanism": dict(sorted(mechanism_counts.items())),
            "stratification_by_cue": balance_by_cue,
            "stratification_by_mechanism": balance_by_mechanism,
        },
        "tasks": tasks,
    }


def validate_assignment_payload(
    payload: dict[str, Any],
    records: list[dict[str, Any]],
    library: dict[str, Any],
    level: str,
    seed: int,
) -> list[str]:
    errors: list[str] = []
    cues = cues_for_level(library, level)
    cue_ids = {cue["id"] for cue in cues}
    tasks = payload.get("tasks", [])
    if payload.get("assignment_seed") != seed:
        errors.append(f"assignment seed mismatch: {payload.get('assignment_seed')} != {seed}")
    if payload.get("cue_level") != level:
        errors.append(f"cue level mismatch: {payload.get('cue_level')} != {level}")
    if payload.get("registry_fingerprint") != registry_fingerprint(records):
        errors.append("task registry fingerprint mismatch")
    if payload.get("cue_library_fingerprint") != short_hash(library):
        errors.append("cue library fingerprint mismatch")
    if len(tasks) != len(records):
        errors.append(f"assignment has {len(tasks)} tasks; registry has {len(records)}")
    assigned_ids = [item.get("base_task_id") for item in tasks]
    registry_ids = [record["task_id"] for record in records]
    if set(assigned_ids) != set(registry_ids):
        errors.append("assignment task IDs do not match registry task IDs")
    assigned_cues = [item.get("cue_id") for item in tasks]
    if any(cue_id not in cue_ids for cue_id in assigned_cues):
        errors.append("assignment contains a cue outside the selected cue level")
    expected_per_cue = len(records) // len(cues) if cues else 0
    counts = Counter(assigned_cues)
    if set(counts) != cue_ids or any(count != expected_per_cue for count in counts.values()):
        errors.append(f"cue counts must all equal {expected_per_cue}: {dict(counts)}")
    cue_by_id = {cue["id"]: cue for cue in cues}
    for item in tasks:
        cue = cue_by_id.get(item.get("cue_id"))
        if not cue:
            continue
        for field, expected in (
            ("cue_mechanism", cue["mechanism"]),
            ("cue_level", cue["level"]),
            ("cue_wording", cue["wording"]),
            ("cue_text", cue["text"]),
        ):
            if item.get(field) != expected:
                errors.append(f"{item.get('base_task_id')}: {field} does not match cue library")
    return errors
