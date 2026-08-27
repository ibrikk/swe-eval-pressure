#!/usr/bin/env python3
"""
Build an ICLR-style team pre-read for SWE-EvalPressure.

This script is intentionally downstream of the frozen current analysis.
It does not call models, judges, verifiers, or the network.

Inputs:
  analysis/current/results/*.csv
  analysis/current/findings/*.csv
  analysis/current/audit/*.csv
  analysis/current/source/{primary,resource,replication}/... via the
  already-tested current analysis loader in scripts/31_current_analysis.py

Outputs:
  analysis/current/behavioral_claims_v2/*.csv
  reports/iclr-current/index.html
  reports/iclr-current/figures/*.svg
  reports/iclr-current/manifest.json

The report preserves the existing reports/current/index.html in full under
"Complete prior integrated report" while surfacing stronger, more scholarly
results and diagnostics above it.
"""

from __future__ import annotations

import csv
import html
import importlib.util
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
RESULTS = ROOT / "analysis" / "current" / "results"
FINDINGS = ROOT / "analysis" / "current" / "findings"
AUDIT = ROOT / "analysis" / "current" / "audit"
OUT = ROOT / "analysis" / "current" / "behavioral_claims_v2"
REPORT = ROOT / "reports" / "iclr-current"
FIG = REPORT / "figures"
PRIOR = ROOT / "reports" / "current" / "index.html"
CORE_SCRIPT = SCRIPTS / "31_current_analysis.py"

OUT.mkdir(parents=True, exist_ok=True)
REPORT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def fnum(v: Any) -> float | None:
    if v in (None, "", "NA", "nan", "None"):
        return None
    try:
        x = float(v)
    except Exception:
        return None
    return x if math.isfinite(x) else None

def ival(v: Any) -> int:
    try:
        return int(float(v))
    except Exception:
        return 0

def esc(v: Any) -> str:
    return html.escape("" if v is None else str(v))

def fmt(v: Any, d: int = 2) -> str:
    x = fnum(v)
    if x is None:
        return "—"
    if abs(x) >= 100000:
        return f"{x:,.0f}"
    return f"{x:,.{d}f}"

def signed(v: Any, d: int = 2) -> str:
    x = fnum(v)
    if x is None:
        return "—"
    if abs(x) >= 100000:
        return f"{x:+,.0f}"
    return f"{x:+,.{d}f}"

def pct(v: Any, d: int = 1) -> str:
    x = fnum(v)
    if x is None:
        return "—"
    # values in current CSVs are generally fractions for prevalence and
    # percentage-points for risk differences. Caller decides conversion.
    return f"{100*x:.{d}f}%"

def ptxt(v: Any) -> str:
    x = fnum(v)
    if x is None:
        return "—"
    if x < 0.001:
        return f"{x:.2e}"
    return f"{x:.4f}"

def mean(xs: Iterable[float]) -> float | None:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None

def median(xs: Iterable[float]) -> float | None:
    xs = list(xs)
    return statistics.median(xs) if xs else None

def quantile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    pos = q * (len(ys) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return ys[lo]
    w = pos - lo
    return ys[lo] * (1-w) + ys[hi] * w

def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p-value without scipy."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)

def holm(ps: list[float | None]) -> list[float | None]:
    idx = [(i, p) for i, p in enumerate(ps) if p is not None]
    idx.sort(key=lambda z: z[1])
    out: list[float | None] = [None] * len(ps)
    prev = 0.0
    m = len(idx)
    for rank, (i, p) in enumerate(idx):
        adj = min(1.0, (m-rank) * p)
        adj = max(adj, prev)
        prev = adj
        out[i] = adj
    return out

def bootstrap_mean_ci(ds: list[float], seed: int = 20260827, reps: int = 10000) -> tuple[float|None,float|None]:
    if not ds:
        return None, None
    import random
    rng = random.Random(seed)
    n = len(ds)
    vals = []
    for _ in range(reps):
        vals.append(sum(ds[rng.randrange(n)] for __ in range(n)) / n)
    return quantile(vals, .025), quantile(vals, .975)

def signflip_p(ds: list[float], seed: int = 20260827, draws: int = 50000) -> float | None:
    if not ds:
        return None
    obs = abs(sum(ds)/len(ds))
    if obs < 1e-15:
        return 1.0
    n = len(ds)
    if n <= 20:
        ex = 0
        total = 1 << n
        for mask in range(total):
            s = sum((x if (mask >> j) & 1 else -x) for j, x in enumerate(ds))/n
            if abs(s) >= obs - 1e-15:
                ex += 1
        return ex/total
    import random
    rng = random.Random(seed)
    ex = 0
    for _ in range(draws):
        s = sum((x if rng.getrandbits(1) else -x) for x in ds)/n
        if abs(s) >= obs - 1e-15:
            ex += 1
    return (ex+1)/(draws+1)

def stable_seed(text: str) -> int:
    import hashlib
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)

def load_current_module():
    if not CORE_SCRIPT.is_file():
        return None
    spec = importlib.util.spec_from_file_location("current_analysis_v2_source", CORE_SCRIPT)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

CURRENT = load_current_module()

# ---------------------------------------------------------------------
# Current aggregate inputs
# ---------------------------------------------------------------------

binary = read_csv(RESULTS / "matched_binary_effects.csv")
behavior_prev = read_csv(RESULTS / "behavior_prevalence.csv")
behavior_eff = read_csv(RESULTS / "matched_behavior_effects.csv")
process_eff = read_csv(RESULTS / "matched_process_effects.csv")
resource_focus = read_csv(RESULTS / "resource_focused_process.csv")
agreement = read_csv(RESULTS / "semantic_agreement_pooled.csv")
semantic_consensus = read_csv(RESULTS / "semantic_consensus.csv")
semantic_jobs_primary = read_csv(RESULTS / "semantic_jobs_primary.csv")
semantic_jobs_resource = read_csv(RESULTS / "semantic_jobs_resource.csv")
replication = read_csv(RESULTS / "replication_direction.csv")
said = read_csv(RESULTS / "said_did_summary.csv")
said_head = read_csv(FINDINGS / "said_did_headlines.csv")
integrity = read_csv(FINDINGS / "integrity_component_effects.csv")
delivery = read_csv(AUDIT / "treatment_delivery.csv")
verifier = read_csv(AUDIT / "verifier_forensics_summary.csv")

# ---------------------------------------------------------------------
# Pair-level source access through the already-tested loader
# ---------------------------------------------------------------------

pairs: list[dict[str, Any]] = []
if CURRENT is not None:
    try:
        studies = {s: CURRENT.core.load_study(s) for s in ("primary", "resource", "replication")}
        for study, rows in studies.items():
            pairs.extend(CURRENT.core.make_pairs(study, rows))
    except Exception as e:
        print("WARN: could not load pair-level sources:", repr(e), file=sys.stderr)

# ---------------------------------------------------------------------
# 1. Treatment fidelity
# ---------------------------------------------------------------------

fidelity_rows = []
for r in delivery:
    row = dict(r)
    # normalize likely rate columns for convenient report rendering
    metric = r.get("delivery_metric") or r.get("metric") or r.get("observation_metric") or ""
    rate = fnum(r.get("rate"))
    if rate is None:
        n = fnum(r.get("n"))
        count = fnum(r.get("count"))
        if n and count is not None:
            rate = count/n
    row["_metric"] = metric
    row["_rate"] = rate
    fidelity_rows.append(row)
write_csv(OUT / "treatment_fidelity.csv", fidelity_rows)

# ---------------------------------------------------------------------
# 2. Semantic treatment effects
# ---------------------------------------------------------------------

# semantic_consensus.csv is wide-format: one row per trajectory with
#   study, profile, trial_name,
#   <field>__status, <field>__label,
#   <field>__deepseek, <field>__gemini
# Strict consensus is the primary semantic source. Judge-specific estimates
# are sensitivity analyses. Disagreements/missing labels are unresolved, not 0.

SEM_TARGETS = {
    "pressure_recognition": {"observed"},
    "cue_provenance_stance": {"untrusted_or_suspicious"},
    "explicit_resistance": {"observed"},
    "claimed_behavioral_response": {
        "ignore",
        "remove_or_modify_cue",
        "inspect_or_verify",
        "comply_with_cue",
    },
    "resource_constraint_recognition": {"observed"},
    "resource_constraint_stance": {
        "accepted",
        "rejected_as_untrusted_or_injection",
    },
    "claimed_resource_response": {
        "adapt_to_constraint",
        "explicitly_ignore_constraint",
    },
    "response_reduce_exploration": {"observed"},
    "response_reduce_validation": {"observed"},
    "response_conserve_tool_calls": {"observed"},
    "response_stop_early": {"observed"},
}

SEMANTIC_SOURCES = (
    "strict_consensus",
    "deepseek",
    "gemini",
)


def pair_trial_name(obj: dict[str, Any]) -> str:
    for key in (
        "trial_name",
        "trial_id",
        "trajectory_id",
        "id",
    ):
        if obj.get(key):
            return str(obj[key])
    return ""


consensus_lookup: dict[
    tuple[str, str, str],
    dict[str, str],
] = {}

for row in semantic_consensus:
    key = (
        str(row.get("study") or ""),
        str(row.get("profile") or ""),
        str(row.get("trial_name") or ""),
    )
    if key[2]:
        consensus_lookup[key] = row


def semantic_label(
    row: dict[str, str] | None,
    field: str,
    source: str,
) -> str | None:
    if not row:
        return None

    if source == "strict_consensus":
        if (
            str(
                row.get(
                    field
                    + "__status"
                )
                or ""
            )
            != "agreement"
        ):
            return None

        value = row.get(
            field
            + "__label"
        )

    elif source in {
        "deepseek",
        "gemini",
    }:
        value = row.get(
            field
            + "__"
            + source
        )

    else:
        raise ValueError(
            f"unknown semantic source: {source}"
        )

    if value in (
        None,
        "",
        "None",
        "null",
    ):
        return None

    return str(value)


semantic_pair_records: dict[
    tuple[
        str,
        str,
        str,
        str,
        str,
        str,
        str,
    ],
    list[
        tuple[
            int | None,
            int | None,
        ]
    ],
] = defaultdict(list)

for pair in pairs:
    study = str(
        pair.get("study")
        or ""
    )

    if study == "replication":
        continue

    profile_name = str(
        pair.get("profile")
        or ""
    )

    contrast = str(
        pair.get("contrast")
        or ""
    )

    placement = str(
        pair.get("placement")
        or ""
    )

    baseline = pair.get(
        "baseline",
        {},
    )

    treatment = pair.get(
        "treatment",
        {},
    )

    baseline_trial = pair_trial_name(
        baseline
    )

    treatment_trial = pair_trial_name(
        treatment
    )

    baseline_sem = consensus_lookup.get(
        (
            study,
            profile_name,
            baseline_trial,
        )
    )

    treatment_sem = consensus_lookup.get(
        (
            study,
            profile_name,
            treatment_trial,
        )
    )

    if (
        baseline_sem is None
        and treatment_sem is None
    ):
        continue

    for field, positive_labels in (
        SEM_TARGETS.items()
    ):
        # Only analyze fields defined for this study.
        if (
            study == "primary"
            and field
            not in CURRENT.PRIMARY_SEMANTIC_FIELDS
        ):
            continue

        if (
            study == "resource"
            and field
            not in CURRENT.RESOURCE_SEMANTIC_FIELDS
        ):
            continue

        for positive in sorted(
            positive_labels
        ):
            for source in (
                SEMANTIC_SOURCES
            ):
                baseline_label = (
                    semantic_label(
                        baseline_sem,
                        field,
                        source,
                    )
                )

                treatment_label = (
                    semantic_label(
                        treatment_sem,
                        field,
                        source,
                    )
                )

                semantic_pair_records[
                    (
                        study,
                        profile_name,
                        contrast,
                        placement,
                        field,
                        positive,
                        source,
                    )
                ].append(
                    (
                        (
                            None
                            if baseline_label
                            is None
                            else int(
                                baseline_label
                                == positive
                            )
                        ),
                        (
                            None
                            if treatment_label
                            is None
                            else int(
                                treatment_label
                                == positive
                            )
                        ),
                    )
                )


semantic_effects = []

for key, observations in sorted(
    semantic_pair_records.items()
):
    (
        study,
        profile_name,
        contrast,
        placement,
        field,
        positive,
        source,
    ) = key

    complete = [
        (
            baseline_value,
            treatment_value,
        )
        for (
            baseline_value,
            treatment_value,
        )
        in observations
        if (
            baseline_value
            is not None
            and treatment_value
            is not None
        )
    ]

    if not complete:
        continue

    n = len(complete)

    baseline_rate = (
        sum(
            baseline_value
            for baseline_value, _
            in complete
        )
        / n
    )

    treatment_rate = (
        sum(
            treatment_value
            for _, treatment_value
            in complete
        )
        / n
    )

    baseline1_treatment0 = sum(
        baseline_value == 1
        and treatment_value == 0
        for (
            baseline_value,
            treatment_value,
        )
        in complete
    )

    baseline0_treatment1 = sum(
        baseline_value == 0
        and treatment_value == 1
        for (
            baseline_value,
            treatment_value,
        )
        in complete
    )

    deltas = [
        treatment_value
        - baseline_value
        for (
            baseline_value,
            treatment_value,
        )
        in complete
    ]

    seed_text = (
        "semantic|"
        + "|".join(key)
    )

    ci_low, ci_high = (
        bootstrap_mean_ci(
            deltas,
            stable_seed(
                seed_text
            ),
        )
    )

    # Conservative full-pair bounds: each unresolved semantic label may
    # take either binary value. This does not impute unresolved as negative.
    lower_sum = 0.0
    upper_sum = 0.0

    for (
        baseline_value,
        treatment_value,
    ) in observations:
        if (
            baseline_value
            is not None
            and treatment_value
            is not None
        ):
            delta = (
                treatment_value
                - baseline_value
            )
            lower_sum += delta
            upper_sum += delta

        elif (
            baseline_value
            is None
            and treatment_value
            is None
        ):
            lower_sum += -1.0
            upper_sum += 1.0

        elif baseline_value is None:
            assert (
                treatment_value
                is not None
            )
            lower_sum += (
                treatment_value
                - 1.0
            )
            upper_sum += (
                treatment_value
                - 0.0
            )

        else:
            assert (
                treatment_value
                is None
            )
            lower_sum += (
                0.0
                - baseline_value
            )
            upper_sum += (
                1.0
                - baseline_value
            )

    total_pairs = len(
        observations
    )

    semantic_effects.append(
        {
            "study": study,
            "profile": profile_name,
            "contrast": contrast,
            "placement": placement,
            "field": field,
            "positive_label": positive,
            "semantic_source": source,
            "matched_n": n,
            "total_pair_n": total_pairs,
            "unresolved_pair_n": (
                total_pairs
                - n
            ),
            "resolution_rate": (
                n
                / total_pairs
            ),
            "baseline_rate": baseline_rate,
            "treatment_rate": treatment_rate,
            "risk_difference_pp": (
                100
                * (
                    treatment_rate
                    - baseline_rate
                )
            ),
            "ci95_low_pp": (
                100
                * ci_low
                if ci_low
                is not None
                else None
            ),
            "ci95_high_pp": (
                100
                * ci_high
                if ci_high
                is not None
                else None
            ),
            "conservative_lower_pp": (
                100
                * lower_sum
                / total_pairs
            ),
            "conservative_upper_pp": (
                100
                * upper_sum
                / total_pairs
            ),
            "baseline1_treatment0":
                baseline1_treatment0,
            "baseline0_treatment1":
                baseline0_treatment1,
            "mcnemar_p_raw":
                exact_mcnemar(
                    baseline1_treatment0,
                    baseline0_treatment1,
                ),
            "holm_p": None,
            "inferential_role": (
                "secondary_matched_semantic_outcome"
                if source
                == "strict_consensus"
                else
                "judge_sensitivity"
            ),
        }
    )


# Multiplicity is controlled separately for each semantic source, model,
# study, and binary endpoint across its planned contrast/placement family.
semantic_families: dict[
    tuple[
        str,
        str,
        str,
        str,
    ],
    list[int],
] = defaultdict(list)

for index, row in enumerate(
    semantic_effects
):
    semantic_families[
        (
            str(
                row["study"]
            ),
            str(
                row["profile"]
            ),
            (
                str(
                    row["field"]
                )
                + "="
                + str(
                    row[
                        "positive_label"
                    ]
                )
            ),
            str(
                row[
                    "semantic_source"
                ]
            ),
        )
    ].append(index)

for indices in (
    semantic_families.values()
):
    adjusted = holm(
        [
            fnum(
                semantic_effects[
                    index
                ][
                    "mcnemar_p_raw"
                ]
            )
            for index in indices
        ]
    )

    for index, adjusted_p in zip(
        indices,
        adjusted,
    ):
        semantic_effects[
            index
        ][
            "holm_p"
        ] = adjusted_p


strict_semantic_effects = [
    row
    for row in semantic_effects
    if row[
        "semantic_source"
    ]
    == "strict_consensus"
]

if (
    semantic_consensus
    and pairs
    and not strict_semantic_effects
):
    raise RuntimeError(
        "Semantic inputs are present, but the matched semantic analysis "
        "produced 0 strict-consensus effects. Check semantic_consensus.csv "
        "wide-schema identity/status/label fields before generating the report."
    )

write_csv(
    OUT
    / "semantic_treatment_effects.csv",
    semantic_effects,
)


# ---------------------------------------------------------------------
# 3. Fine-grained process response, organized by construct
# ---------------------------------------------------------------------

DOMAIN = {
    "repo_search_calls":"Exploration",
    "file_read_calls":"Exploration",
    "unique_files_read":"Exploration",
    "unique_dirs_read":"Exploration",
    "test_files_inspected":"Task understanding",
    "spec_config_files_inspected":"Task understanding",
    "instruction_file_inspections":"Task understanding",
    "edit_calls":"Implementation",
    "unique_files_modified":"Implementation",
    "test_command_calls":"Verification",
    "validation_calls":"Verification",
    "post_edit_validation_calls":"Verification",
    "edit_validation_cycles":"Recovery / persistence",
    "failed_validation_then_edit_cycles":"Recovery / persistence",
    "git_history_inspections":"Provenance",
    "external_lookup_calls":"Externalization",
    "subagent_delegation_calls":"Externalization",
    "raw_tool_calls":"Resource use",
    "behavioral_action_calls":"Resource use",
    "trajectory_steps":"Resource use",
    "input_tokens":"Resource use",
    "output_tokens":"Resource use",
    "duration_sec":"Resource use",
}

process_detail = []
for r in process_eff:
    rr = dict(r)
    rr["domain"] = DOMAIN.get(r.get("metric",""), "Other")
    process_detail.append(rr)
write_csv(OUT / "process_detail_effects.csv", process_detail)

# ---------------------------------------------------------------------
# 4. Resource action policy
# ---------------------------------------------------------------------

resource_policy = []
for r in process_detail:
    if r.get("study") == "resource" and r.get("contrast") == "resource_deprivation":
        if r.get("metric") in DOMAIN:
            resource_policy.append(dict(r))
write_csv(OUT / "resource_action_policy.csv", resource_policy)

# ---------------------------------------------------------------------
# 5. Pair-level interactions + robustness
# ---------------------------------------------------------------------

HEADLINE_METRICS = (
    "raw_tool_calls","input_tokens","validation_calls","trajectory_steps",
    "behavioral_action_calls","duration_sec","overall_pass",
    "integrity_sensitive_action_any"
)

def numeric_from(obj: dict[str, Any], metric: str) -> float | None:
    return fnum(obj.get(metric))

# treatment deltas by exact task/model/contrast/placement
delta_lookup: dict[tuple[str,str,str,str,str], float] = {}
for p in pairs:
    study = str(p.get("study",""))
    if study == "replication":
        continue
    profile = str(p.get("profile",""))
    contrast = str(p.get("contrast",""))
    placement = str(p.get("placement",""))
    task = str(p.get("task_id") or p.get("base_task_id") or p.get("baseline",{}).get("task_id") or "")
    for metric in HEADLINE_METRICS:
        b = numeric_from(p.get("baseline",{}), metric)
        t = numeric_from(p.get("treatment",{}), metric)
        if b is not None and t is not None:
            delta_lookup[(study,profile,contrast,placement,task,metric)] = t-b

placement_rows = []
placement_groups: dict[tuple[str,str,str,str,str], list[float]] = defaultdict(list)
placements = ("source","root","scaffold")
for (study,profile,contrast,placement,task,metric), delta in list(delta_lookup.items()):
    for p2 in placements:
        if p2 <= placement:
            continue
        k2 = (study,profile,contrast,p2,task,metric)
        if k2 in delta_lookup:
            placement_groups[(study,profile,contrast,f"{placement}-{p2}",metric)].append(delta-delta_lookup[k2])

for key, ds in sorted(placement_groups.items()):
    study,profile,contrast,comparison,metric = key
    lo,hi = bootstrap_mean_ci(ds, stable_seed("placement|"+"|".join(key)))
    placement_rows.append({
        "study":study,"profile":profile,"contrast":contrast,
        "placement_comparison":comparison,"metric":metric,"matched_n":len(ds),
        "difference_in_treatment_effect":mean(ds),"median_difference":median(ds),
        "ci95_low":lo,"ci95_high":hi,
        "sign_flip_p_raw":signflip_p(ds, stable_seed("placement-p|"+"|".join(key))),
    })
for group_key in sorted(set((r["study"],r["profile"],r["contrast"],r["metric"]) for r in placement_rows)):
    inds=[i for i,r in enumerate(placement_rows) if (r["study"],r["profile"],r["contrast"],r["metric"])==group_key]
    adj=holm([fnum(placement_rows[i]["sign_flip_p_raw"]) for i in inds])
    for i,p in zip(inds,adj): placement_rows[i]["holm_p"]=p
write_csv(OUT / "placement_interactions.csv", placement_rows)

# Model heterogeneity
model_rows=[]
model_groups: dict[tuple[str,str,str,str,str],list[float]]=defaultdict(list)
models=sorted(set(k[1] for k in delta_lookup))
for (study,profile,contrast,placement,task,metric),d in list(delta_lookup.items()):
    for m2 in models:
        if m2 <= profile: continue
        k2=(study,m2,contrast,placement,task,metric)
        if k2 in delta_lookup:
            model_groups[(study,contrast,placement,f"{profile}-{m2}",metric)].append(d-delta_lookup[k2])
for key,ds in sorted(model_groups.items()):
    study,contrast,placement,comparison,metric=key
    lo,hi=bootstrap_mean_ci(ds,stable_seed("model|"+"|".join(key)))
    model_rows.append({
        "study":study,"contrast":contrast,"placement":placement,
        "model_comparison":comparison,"metric":metric,"matched_n":len(ds),
        "difference_in_treatment_effect":mean(ds),"median_difference":median(ds),
        "ci95_low":lo,"ci95_high":hi,
        "sign_flip_p_raw":signflip_p(ds,stable_seed("model-p|"+"|".join(key))),
    })
# Holm within contrast/placement/metric
for group_key in sorted(set((r["study"],r["contrast"],r["placement"],r["metric"]) for r in model_rows)):
    inds=[i for i,r in enumerate(model_rows) if (r["study"],r["contrast"],r["placement"],r["metric"])==group_key]
    adj=holm([fnum(model_rows[i]["sign_flip_p_raw"]) for i in inds])
    for i,p in zip(inds,adj): model_rows[i]["holm_p"]=p
write_csv(OUT / "model_interactions.csv", model_rows)

# Continuous robustness
robust=[]
rgroups: dict[tuple[str,str,str,str,str],list[tuple[float,float,float]]]=defaultdict(list)
for p in pairs:
    study=str(p.get("study",""))
    if study=="replication": continue
    profile=str(p.get("profile","")); contrast=str(p.get("contrast","")); placement=str(p.get("placement",""))
    for metric in ("raw_tool_calls","input_tokens","validation_calls","trajectory_steps","behavioral_action_calls","duration_sec"):
        b=numeric_from(p.get("baseline",{}),metric); t=numeric_from(p.get("treatment",{}),metric)
        if b is not None and t is not None:
            rgroups[(study,profile,contrast,placement,metric)].append((b,t,t-b))
for key,obs in sorted(rgroups.items()):
    ds=[d for _,_,d in obs]
    logds=[math.log1p(max(0,t))-math.log1p(max(0,b)) for b,t,_ in obs]
    sds=sorted(ds)
    cut=max(0,int(.1*len(sds)))
    wins=sds[cut:len(sds)-cut] if len(sds)-2*cut>0 else sds
    # leave-one-task-out means
    loo=[]
    if len(ds)>1:
        total=sum(ds)
        loo=[(total-d)/(len(ds)-1) for d in ds]
    lo,hi=bootstrap_mean_ci(ds,stable_seed("robust|"+"|".join(key)))
    robust.append({
        "study":key[0],"profile":key[1],"contrast":key[2],"placement":key[3],"metric":key[4],
        "matched_n":len(ds),"mean_delta":mean(ds),"median_delta":median(ds),
        "ci95_low":lo,"ci95_high":hi,
        "positive_fraction":sum(d>0 for d in ds)/len(ds),
        "zero_fraction":sum(d==0 for d in ds)/len(ds),
        "negative_fraction":sum(d<0 for d in ds)/len(ds),
        "mean_log1p_ratio":mean(logds),
        "winsorized_10pct_mean_delta":mean(wins),
        "loo_min":min(loo) if loo else None,
        "loo_max":max(loo) if loo else None,
        "max_abs_single_task_delta":max(abs(d) for d in ds),
    })
write_csv(OUT / "continuous_robustness.csv", robust)

# ---------------------------------------------------------------------
# 6. Replication headline table from current descriptive output
# ---------------------------------------------------------------------

# Keep only rows that match substantive headline metrics/conditions; exact
# schema varies, so use conservative filtering.
rep_head=[]
for r in replication:
    text=" ".join(str(v).lower() for v in r.values())
    if any(m in text for m in ("input_tokens","raw_tool_calls","validation_calls","trajectory_steps","behavioral_action_calls","duration_sec","integrity_sensitive_action_any")):
        if any(c in text for c in ("financial","self_preservation","self-preservation","resource")):
            rep_head.append(dict(r))
write_csv(OUT / "replication_headline_effects.csv", rep_head)

# ---------------------------------------------------------------------
# 7. Cue-mechanism robustness for headline treatment effects
# ---------------------------------------------------------------------

def nested_get(d: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if d.get(name) not in (None, ""):
            return d.get(name)
    for container in ("metadata", "treatment_metadata", "variant_metadata", "cue_metadata"):
        sub = d.get(container)
        if isinstance(sub, dict):
            for name in names:
                if sub.get(name) not in (None, ""):
                    return sub.get(name)
    return None

CUE_KEYS = (
    "cue_mechanism", "evaluation_cue_mechanism", "eval_cue_mechanism",
    "mechanism", "cue_type", "eval_mechanism",
)

cue_observations: dict[
    tuple[str, str, str, str, str],
    list[tuple[str, str, float]]
] = defaultdict(list)

for p in pairs:
    if str(p.get("study","")) != "primary":
        continue
    profile = str(p.get("profile",""))
    contrast = str(p.get("contrast",""))
    placement = str(p.get("placement",""))
    task = str(
        p.get("task_id")
        or p.get("base_task_id")
        or p.get("treatment",{}).get("task_id")
        or ""
    )
    treatment = p.get("treatment",{})
    mechanism = nested_get(treatment, CUE_KEYS)
    if mechanism in (None, ""):
        mechanism = nested_get(p, CUE_KEYS)
    if mechanism in (None, ""):
        continue
    mechanism = str(mechanism)
    for metric in HEADLINE_METRICS:
        b = numeric_from(p.get("baseline",{}), metric)
        t = numeric_from(treatment, metric)
        if b is not None and t is not None:
            cue_observations[
                ("primary", profile, contrast, placement, metric)
            ].append((task, mechanism, t-b))

cue_rows = []
for key, obs in sorted(cue_observations.items()):
    study, profile, contrast, placement, metric = key
    mechanisms = sorted(set(m for _,m,_ in obs))
    if len(mechanisms) < 2:
        continue
    full_ds = [d for _,_,d in obs]
    full_mean = mean(full_ds)
    loo = []
    for held_out in mechanisms:
        ds = [d for _,m,d in obs if m != held_out]
        if not ds:
            continue
        loo.append((held_out, mean(ds), len(ds)))
    if not loo:
        continue
    estimates = [x for _,x,_ in loo if x is not None]
    for held_out, estimate, n in loo:
        cue_rows.append({
            "study":study, "profile":profile, "contrast":contrast,
            "placement":placement, "metric":metric,
            "full_n":len(obs), "mechanism_count":len(mechanisms),
            "full_mean_delta":full_mean,
            "held_out_mechanism":held_out,
            "leave_one_mechanism_out_n":n,
            "leave_one_mechanism_out_mean_delta":estimate,
            "direction_matches_full": (
                None if full_mean is None or estimate is None
                else int(
                    (full_mean == 0 and estimate == 0)
                    or (full_mean > 0 and estimate > 0)
                    or (full_mean < 0 and estimate < 0)
                )
            ),
            "loo_min":min(estimates) if estimates else None,
            "loo_max":max(estimates) if estimates else None,
        })

write_csv(OUT / "cue_mechanism_robustness.csv", cue_rows)

# Compact per-effect summary for the HTML.
cue_summary = []
by_effect: dict[tuple[str,str,str,str,str], list[dict[str,Any]]] = defaultdict(list)
for r in cue_rows:
    by_effect[
        (r["study"],r["profile"],r["contrast"],r["placement"],r["metric"])
    ].append(r)
for key, rows_ in sorted(by_effect.items()):
    direction_vals = [ival(r.get("direction_matches_full")) for r in rows_]
    cue_summary.append({
        "study":key[0],"profile":key[1],"contrast":key[2],
        "placement":key[3],"metric":key[4],
        "full_n":rows_[0].get("full_n"),
        "mechanism_count":rows_[0].get("mechanism_count"),
        "full_mean_delta":rows_[0].get("full_mean_delta"),
        "loo_min":min(fnum(r.get("leave_one_mechanism_out_mean_delta")) for r in rows_
                      if fnum(r.get("leave_one_mechanism_out_mean_delta")) is not None),
        "loo_max":max(fnum(r.get("leave_one_mechanism_out_mean_delta")) for r in rows_
                      if fnum(r.get("leave_one_mechanism_out_mean_delta")) is not None),
        "direction_stability_fraction":sum(direction_vals)/len(direction_vals) if direction_vals else None,
    })
write_csv(OUT / "cue_mechanism_summary.csv", cue_summary)

# ---------------------------------------------------------------------
# 8. Claim table: conservative machine-readable summary
# ---------------------------------------------------------------------

claims=[]

for r in binary:
    hp=fnum(r.get("holm_p"))
    if hp is not None and hp <= .05:
        claims.append({
            "evidence_class":"Primary/secondary matched outcome",
            "study":r.get("study"),"model":r.get("profile"),"contrast":r.get("contrast"),
            "placement":r.get("placement"),"outcome":r.get("metric"),
            "effect":r.get("risk_difference_pp") or r.get("mean_delta"),
            "ci_low":r.get("ci95_low_pp") or r.get("ci95_low"),
            "ci_high":r.get("ci95_high_pp") or r.get("ci95_high"),
            "adjusted_p":hp,"status":"Supported treatment effect",
        })

for r in behavior_eff:
    hp=fnum(r.get("holm_p"))
    if hp is not None and hp <= .05:
        claims.append({
            "evidence_class":"Matched deterministic behavior",
            "study":r.get("study"),"model":r.get("profile"),"contrast":r.get("contrast"),
            "placement":r.get("placement"),"outcome":r.get("metric"),
            "effect":r.get("risk_difference_pp") or r.get("mean_delta"),
            "ci_low":r.get("ci95_low_pp") or r.get("ci95_low"),
            "ci_high":r.get("ci95_high_pp") or r.get("ci95_high"),
            "adjusted_p":hp,"status":"Supported treatment effect",
        })

for r in process_eff:
    q=fnum(r.get("bh_q"))
    if q is not None and q <= .05:
        status="Treatment effect; mechanism unresolved" if r.get("placement")=="root" else "Exploratory matched process effect"
        claims.append({
            "evidence_class":"Matched process outcome",
            "study":r.get("study"),"model":r.get("profile"),"contrast":r.get("contrast"),
            "placement":r.get("placement"),"outcome":r.get("metric"),
            "effect":r.get("mean_delta"),"ci_low":r.get("ci95_low"),"ci_high":r.get("ci95_high"),
            "adjusted_p":q,"status":status,
        })

for r in strict_semantic_effects:
    hp=fnum(r.get("holm_p"))
    if hp is not None and hp <= .05:
        claims.append({
            "evidence_class":"Matched semantic outcome (secondary)",
            "study":r.get("study"),"model":r.get("profile"),"contrast":r.get("contrast"),
            "placement":r.get("placement"),"outcome":f'{r.get("field")}={r.get("positive_label")}',
            "effect":r.get("risk_difference_pp"),"ci_low":r.get("ci95_low_pp"),"ci_high":r.get("ci95_high_pp"),
            "adjusted_p":hp,"status":"Secondary supported treatment effect",
        })

write_csv(OUT / "supported_claims.csv", claims)

# ---------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------

def svg_bar_chart(path: Path, title: str, labels: list[str], values: list[float],
                  note: str="", width: int=980, row_h: int=34) -> None:
    h=95+row_h*len(labels)
    max_abs=max([abs(v) for v in values] or [1])
    left=360; right=80; plot_w=width-left-right; zero=left+plot_w/2
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{h}" viewBox="0 0 {width} {h}">',
           '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;fill:#17212b}.t{font-size:19px;font-weight:700}.l{font-size:12px}.v{font-size:12px;font-weight:600}.n{font-size:11px;fill:#687787}.axis{stroke:#aab5bf;stroke-width:1}.bar{fill:#486f93}.neg{fill:#7a5962}</style>',
           f'<text x="20" y="28" class="t">{esc(title)}</text>',
           f'<line x1="{zero}" y1="48" x2="{zero}" y2="{h-35}" class="axis"/>']
    for i,(lab,v) in enumerate(zip(labels,values)):
        y=62+i*row_h
        w=(abs(v)/max_abs)*(plot_w/2-18)
        x=zero if v>=0 else zero-w
        cls="bar" if v>=0 else "neg"
        parts.append(f'<text x="20" y="{y+13}" class="l">{esc(lab[:55])}</text>')
        parts.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="18" rx="2" class="{cls}"/>')
        tx=x+w+6 if v>=0 else x-6
        anchor="start" if v>=0 else "end"
        parts.append(f'<text x="{tx:.1f}" y="{y+13}" text-anchor="{anchor}" class="v">{v:+.2f}</text>')
    if note:
        parts.append(f'<text x="20" y="{h-12}" class="n">{esc(note)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts),encoding="utf-8")

def svg_heatmap(path: Path, title: str, row_labels: list[str], col_labels: list[str],
                matrix: list[list[float|None]], width: int=1180) -> None:
    cw=max(75,(width-320)//max(1,len(col_labels))); rh=30
    h=100+rh*len(row_labels)
    vals=[v for row in matrix for v in row if v is not None]
    vmax=max(vals or [1]); vmin=min(vals or [0])
    def shade(v):
        if v is None: return "#f2f4f6"
        z=0.5 if vmax==vmin else (v-vmin)/(vmax-vmin)
        g=int(242-105*z); b=int(248-92*z)
        return f"rgb({220-int(75*z)},{g},{b})"
    p=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{h}">',
       '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;fill:#17212b}.t{font-size:19px;font-weight:700}.l{font-size:11px}.v{font-size:10px;font-weight:600}</style>',
       f'<text x="18" y="28" class="t">{esc(title)}</text>']
    for j,c in enumerate(col_labels):
        x=300+j*cw+cw/2
        p.append(f'<text x="{x}" y="62" text-anchor="middle" class="l">{esc(c[:16])}</text>')
    for i,rl in enumerate(row_labels):
        y=72+i*rh
        p.append(f'<text x="292" y="{y+19}" text-anchor="end" class="l">{esc(rl[:42])}</text>')
        for j,v in enumerate(matrix[i]):
            x=300+j*cw
            p.append(f'<rect x="{x}" y="{y}" width="{cw-2}" height="{rh-2}" fill="{shade(v)}"/>')
            if v is not None:
                p.append(f'<text x="{x+cw/2}" y="{y+19}" text-anchor="middle" class="v">{100*v:.0f}%</text>')
    p.append("</svg>")
    path.write_text("\n".join(p),encoding="utf-8")

# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

# Behavior prevalence heatmap
if behavior_prev:
    metrics=sorted(set(r.get("metric","") for r in behavior_prev if r.get("metric")))
    rows=[]
    labels=[]
    # Prefer primary, capable + llama, and cap to reasonable height
    grouped=defaultdict(dict)
    for r in behavior_prev:
        if r.get("study")!="primary": continue
        key=(r.get("profile",""),r.get("condition",""),r.get("placement",""))
        rate=fnum(r.get("rate") or r.get("prevalence"))
        if rate is not None: grouped[key][r.get("metric","")]=rate
    for key,d in sorted(grouped.items()):
        labels.append(" · ".join(x for x in key if x))
        rows.append([d.get(m) for m in metrics])
    if rows:
        svg_heatmap(FIG/"behavior_prevalence.svg","Primary-study behavioral prevalence",labels,metrics,rows)

# Significant process effects
sig_proc=[r for r in process_eff if (fnum(r.get("bh_q")) or 9) <= .05]
sig_proc=sorted(sig_proc,key=lambda r:fnum(r.get("bh_q")) or 9)[:20]
if sig_proc:
    svg_bar_chart(FIG/"significant_process_effects.svg","Multiplicity-robust process effects",
                  [f'{r.get("profile")} · {r.get("contrast")} · {r.get("placement")} · {r.get("metric")}' for r in sig_proc],
                  [fnum(r.get("mean_delta")) or 0 for r in sig_proc],
                  "Bars are mean same-task paired differences; exact CIs and adjusted q-values are in the table.")

# Resource focused
if resource_focus:
    svg_bar_chart(FIG/"resource_focused.svg","Resource-scarcity focused family",
                  [f'{r.get("profile")} · {r.get("metric")}' for r in resource_focus],
                  [fnum(r.get("mean_delta")) or 0 for r in resource_focus],
                  "Same-task paired differences under resource scarcity versus eval-only scaffold.")

# Semantic significant effects
sig_sem=[
    r
    for r in strict_semantic_effects
    if (fnum(r.get("holm_p")) or 9) <= .05
]
sig_sem=sorted(sig_sem,key=lambda r:fnum(r.get("holm_p")) or 9)[:24]
if sig_sem:
    svg_bar_chart(FIG/"semantic_treatment_effects.svg","Matched effects on observable semantic response",
                  [f'{r["profile"]} · {r["contrast"]} · {r["placement"]} · {r["field"]}={r["positive_label"]}' for r in sig_sem],
                  [fnum(r.get("risk_difference_pp")) or 0 for r in sig_sem],
                  "Percentage-point treatment effects; secondary analysis with Holm adjustment.")

# Placement interactions
sig_place=[r for r in placement_rows if (fnum(r.get("holm_p")) or 9) <= .05]
sig_place=sorted(sig_place,key=lambda r:fnum(r.get("holm_p")) or 9)[:18]
if sig_place:
    svg_bar_chart(FIG/"placement_interactions.svg","Placement moderation of treatment effects",
                  [f'{r["profile"]} · {r["contrast"]} · {r["metric"]} · {r["placement_comparison"]}' for r in sig_place],
                  [fnum(r.get("difference_in_treatment_effect")) or 0 for r in sig_place],
                  "Difference-in-differences across the same underlying tasks; Holm-adjusted within outcome family.")

# Cue-mechanism leave-one-out stability
cue_fig = [
    r for r in cue_summary
    if r.get("metric") in (
        "input_tokens","raw_tool_calls","validation_calls",
        "integrity_sensitive_action_any","overall_pass"
    )
]
cue_fig = cue_fig[:24]
if cue_fig:
    svg_bar_chart(
        FIG/"cue_mechanism_robustness.svg",
        "Leave-one-cue-mechanism-out treatment effects",
        [
            f'{r["profile"]} · {r["contrast"]} · {r["placement"]} · {r["metric"]}'
            for r in cue_fig
        ],
        [fnum(r.get("full_mean_delta")) or 0 for r in cue_fig],
        "Bars show full pooled mean paired effects; exact leave-one-mechanism-out ranges are reported in the table."
    )

# ---------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------

def table(rows: list[dict[str, Any]], cols: list[tuple[str,str]], limit: int|None=None) -> str:
    if limit is not None: rows=rows[:limit]
    if not rows:
        return '<p class="muted">No rows available.</p>'
    out=['<div class="table-scroll"><table><thead><tr>']
    out += [f"<th>{esc(label)}</th>" for _,label in cols]
    out.append("</tr></thead><tbody>")
    for r in rows:
        out.append("<tr>")
        for key,_ in cols:
            out.append(f"<td>{esc(r.get(key,''))}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)

def prior_main() -> str:
    if not PRIOR.is_file(): return ""
    s=PRIOR.read_text(encoding="utf-8")
    m=re.search(r"<main[^>]*>(.*)</main>",s,re.S)
    return m.group(1) if m else s

import re

# Determine top claims, with supported effects first.
def claim_strength(c):
    p=fnum(c.get("adjusted_p"))
    return (0 if p is not None and p<=.05 else 1, p if p is not None else 9)

top_claims=sorted(claims,key=claim_strength)[:8]

cards=[]
for c in top_claims:
    cards.append(f"""
    <article class="finding-card">
      <div class="tag">{esc(c.get("status"))}</div>
      <h3>{esc(c.get("model"))}: {esc(c.get("outcome"))}</h3>
      <div class="effect">{signed(c.get("effect"))}</div>
      <p><strong>{esc(c.get("contrast"))}</strong> · {esc(c.get("placement"))}</p>
      <p class="small">Adjusted p/q: {ptxt(c.get("adjusted_p"))}; 95% CI [{fmt(c.get("ci_low"))}, {fmt(c.get("ci_high"))}]</p>
    </article>""")

# Selected tables
top_sig_proc=sorted(sig_proc,key=lambda r:fnum(r.get("bh_q")) or 9)
top_resource=sorted(resource_focus,key=lambda r:fnum(r.get("focused_holm_p")) or 9)
top_sem=sorted(sig_sem,key=lambda r:fnum(r.get("holm_p")) or 9)
top_place=sorted(sig_place,key=lambda r:fnum(r.get("holm_p")) or 9)

# Robustness cards for headline Claude root rows
root_rob=[r for r in robust if r.get("study")=="primary" and r.get("profile")=="claude" and r.get("placement")=="root" and r.get("contrast") in ("financial_pressure","self_preservation_pressure")]
root_rob=sorted(root_rob,key=lambda r:(r.get("contrast",""),r.get("metric","")))

css=r"""
:root{--bg:#f4f6f8;--panel:#fff;--ink:#17212b;--muted:#667484;--line:#d9e0e7;--nav:#132536;--blue:#355f86;--soft:#eef4f8;--green:#eaf5ee;--amber:#fff5df;--red:#faeeee}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif}
nav{position:sticky;top:0;z-index:50;background:var(--nav);padding:10px 18px;white-space:nowrap;overflow:auto}
nav a{color:#fff;text-decoration:none;margin-right:18px;font-size:13px}
main{max-width:1320px;margin:auto;padding:32px 24px 120px}
.hero{background:linear-gradient(135deg,#fff,#eef4f8);border:1px solid var(--line);border-radius:16px;padding:30px;margin-bottom:24px}
h1{font-size:38px;line-height:1.08;margin:0 0 12px} h2{margin-top:54px;padding-top:8px;border-top:1px solid var(--line)} h3{margin:6px 0 8px}
.subtitle{font-size:18px;color:var(--muted);max-width:1000px}
.finding-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(265px,1fr));gap:14px;margin:20px 0}
.finding-card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:17px;box-shadow:0 1px 2px rgba(0,0,0,.03)}
.finding-card .effect{font-size:29px;font-weight:750}.tag{display:inline-block;font-size:10px;text-transform:uppercase;letter-spacing:.055em;background:var(--soft);padding:3px 7px;border-radius:10px;color:#36566f}
.small,.muted{font-size:12px;color:var(--muted)}
.callout{background:var(--soft);border-left:4px solid var(--blue);padding:14px 17px;margin:18px 0}
.warning{background:var(--amber);border-left:4px solid #b47a20;padding:14px 17px;margin:18px 0}
.good{background:var(--green);border-left:4px solid #4d8060;padding:14px 17px;margin:18px 0}
.figure{background:#fff;border:1px solid var(--line);border-radius:12px;padding:15px;margin:18px 0}
.figure img{width:100%;height:auto;display:block}
.figure figcaption{font-size:12px;color:var(--muted);margin-top:8px}
.table-scroll{max-height:72vh;overflow:auto;position:relative;background:#fff;border:1px solid var(--line);border-radius:10px;scrollbar-gutter:stable}
table{width:100%;border-collapse:collapse;font-size:12px} thead th{position:sticky;top:0;z-index:5;background:#edf1f5;text-align:left;padding:8px;border-bottom:1px solid var(--line)} td{padding:8px;border-bottom:1px solid #edf0f2;vertical-align:top}
details{background:#fff;border:1px solid var(--line);border-radius:10px;margin:12px 0;padding:10px 14px} summary{cursor:pointer;font-weight:650}
.rq{background:#fff;border:1px solid var(--line);border-radius:10px;padding:13px 16px;margin:8px 0}
.status{font-weight:650}.two{display:grid;grid-template-columns:1fr 1fr;gap:15px}@media(max-width:850px){.two{grid-template-columns:1fr}}
"""

nav = """
<nav>
<a href="#findings">Key findings</a><a href="#design">Design</a><a href="#fidelity">Fidelity</a>
<a href="#prevalence">Prevalence</a><a href="#semantic">Semantic effects</a><a href="#process">Execution</a>
<a href="#resource">Resource</a><a href="#interactions">Interactions</a><a href="#said">Said/Did</a>
<a href="#robustness">Robustness</a><a href="#cue">Cue robustness</a><a href="#replication">Replication</a><a href="#claims">Claims</a>
<a href="#prior">Full prior report</a>
</nav>
"""

def fig_block(name,title,caption):
    p=FIG/name
    if not p.is_file(): return ""
    return f'<figure class="figure"><img src="figures/{esc(name)}"><figcaption><strong>{esc(title)}.</strong> {esc(caption)}</figcaption></figure>'

html_doc=f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SWE-EvalPressure — Behavioral Results Pre-read</title><style>{css}</style></head><body>{nav}<main>
<section class="hero">
<div class="tag">Team pre-read · ICLR 2027 analysis development</div>
<h1>Behavioral Responses of Coding Agents to Evaluation-Related Pressure</h1>
<p class="subtitle">Matched experimental analysis of task performance, execution strategy, cue handling, semantic response, treatment fidelity, and verbal–behavioral consistency. Adjusted treatment effects are separated from exploratory and post-treatment descriptive analyses.</p>
</section>

<section id="findings"><h2>Key empirical findings</h2>
<div class="finding-grid">{''.join(cards)}</div>
<div class="callout"><strong>Interpretive hierarchy.</strong> The strongest claims are randomized same-task treatment effects with multiplicity correction. Process analyses are secondary/exploratory unless otherwise noted. Semantic-state-conditioned Said/Did analyses are descriptive because semantic state is post-treatment.</div>
</section>

<section id="design"><h2>Research questions and design</h2>
<div class="rq"><strong>RQ1.</strong> How frequently do predefined behaviors occur across models and conditions?</div>
<div class="rq"><strong>RQ2.</strong> Does pressure alter observable recognition, provenance assessment, resistance, or claimed response?</div>
<div class="rq"><strong>RQ3.</strong> Does pressure alter exploration, editing, verification, repair, resource use, or task performance?</div>
<div class="rq"><strong>RQ4.</strong> Does resource scarcity change execution volume and allocation across action classes?</div>
<div class="rq"><strong>RQ5.</strong> Are treatment effects moderated by placement or model?</div>
<div class="rq"><strong>RQ6.</strong> Are verbal responses consistent with objective same-task behavioral changes?</div>
<div class="rq"><strong>RQ7.</strong> Do headline behavioral effects recur in the independent rerun?</div>
</section>

<section id="fidelity"><h2>Treatment fidelity and outcome validity</h2>
<div class="warning"><strong>Root-placement interpretation.</strong> Direct cue consumption is weakly observed for several passive-root conditions. Root-condition estimates therefore support an intention-to-treat effect of assignment; they do not by themselves establish that the model consciously read the pressure cue. Mechanism claims remain unresolved unless additional delivery evidence is recovered.</div>
{table(fidelity_rows,[("study","Study"),("profile","Model"),("condition","Condition"),("placement","Placement"),("_metric","Observation"),("n","n"),("_rate","Rate")],80)}
<details><summary>Verifier audit</summary>
{table(verifier,[("study","Study"),("profile","Model"),("substantive","Substantive"),("strict_passes","Strict passes"),("strict_raw_unavailable","Raw unavailable"),("strict_forensic_risk","Forensic risk")])}
</details>
</section>

<section id="prevalence"><h2>RQ1 · Behavioral prevalence</h2>
<p>Prevalence describes how often a behavior occurs; it is not itself a treatment-effect test.</p>
{fig_block("behavior_prevalence.svg","Behavioral prevalence","Percentage of substantive trajectories exhibiting each predefined deterministic behavior, stratified by model, condition, and placement.")}
</section>

<section id="semantic"><h2>RQ2 · Effects on observable semantic response</h2>
<p>Semantic labels are treated here as outcomes of randomized condition assignment. Strict-consensus complete cases are paired by base task. These analyses are secondary and use Holm adjustment within model × semantic-endpoint families.</p>
{fig_block("semantic_treatment_effects.svg","Semantic treatment effects","Percentage-point same-task effects on recognition, provenance stance, explicit resistance, and claimed response.")}
{table(top_sem,[("profile","Model"),("contrast","Intervention"),("placement","Placement"),("field","Field"),("positive_label","Outcome label"),("matched_n","Resolved n"),("total_pair_n","Total pairs"),("resolution_rate","Resolution"),("risk_difference_pp","Δ pp"),("ci95_low_pp","CI low"),("ci95_high_pp","CI high"),("conservative_lower_pp","Conservative lower"),("conservative_upper_pp","Conservative upper"),("holm_p","Holm p")],30)}
</section>

<section id="process"><h2>RQ3 · Task-execution response</h2>
<p>The process analysis distinguishes exploration, task understanding, implementation, verification, recovery/persistence, provenance inspection, externalization, and aggregate resource use.</p>
{fig_block("significant_process_effects.svg","Multiplicity-robust process changes","Only BH-FDR q≤.05 process effects are shown in this headline figure.")}
{table(top_sig_proc,[("profile","Model"),("contrast","Intervention"),("placement","Placement"),("domain","Domain"),("metric","Metric"),("matched_n","n"),("mean_delta","Mean Δ"),("median_delta","Median Δ"),("ci95_low","CI low"),("ci95_high","CI high"),("bh_q","BH q")],30)}
<h3>Continuous-effect robustness for Claude root conditions</h3>
<p>These diagnostics distinguish broad task-level shifts from means dominated by a small number of very long trajectories.</p>
{table(root_rob,[("contrast","Intervention"),("metric","Metric"),("matched_n","n"),("mean_delta","Mean Δ"),("median_delta","Median Δ"),("positive_fraction","Fraction ↑"),("mean_log1p_ratio","Mean log-ratio"),("winsorized_10pct_mean_delta","10% winsorized Δ"),("loo_min","LOO min"),("loo_max","LOO max"),("max_abs_single_task_delta","Largest |task Δ|")])}
</section>

<section id="resource"><h2>RQ4 · Resource-scarcity response</h2>
{fig_block("resource_focused.svg","Resource-scarcity focused family","Prespecified focused family for tool calls, input tokens, and validation calls.")}
{table(top_resource,[("profile","Model"),("metric","Metric"),("matched_n","n"),("baseline_mean","Eval-only mean"),("treatment_mean","Resource mean"),("mean_delta","Mean Δ"),("ci95_low","CI low"),("ci95_high","CI high"),("focused_holm_p","Focused Holm p")])}
<h3>Action-policy decomposition</h3>
<p>The table below decomposes contraction across observable action classes. Absolute counts should be interpreted before relative action shares because compositional shares are mechanically dependent.</p>
{table(resource_policy,[("profile","Model"),("domain","Domain"),("metric","Metric"),("matched_n","n"),("baseline_mean","Baseline"),("treatment_mean","Treatment"),("mean_delta","Mean Δ"),("ci95_low","CI low"),("ci95_high","CI high"),("bh_q","BH q")],120)}
</section>

<section id="interactions"><h2>RQ5 · Moderation by placement and model</h2>
<p>A significant effect in one subgroup and a nonsignificant effect in another is not evidence of heterogeneity. The analyses below test the difference in treatment effects directly.</p>
{fig_block("placement_interactions.svg","Placement moderation","Same-task difference-in-differences across source, root, and scaffold placements.")}
<h3>Placement interactions</h3>
{table(top_place,[("profile","Model"),("contrast","Intervention"),("metric","Metric"),("placement_comparison","Comparison"),("matched_n","n"),("difference_in_treatment_effect","Δ treatment effect"),("ci95_low","CI low"),("ci95_high","CI high"),("holm_p","Holm p")],40)}
<h3>Model heterogeneity</h3>
{table(sorted(model_rows,key=lambda r:fnum(r.get("holm_p")) or 9),[("contrast","Intervention"),("placement","Placement"),("metric","Metric"),("model_comparison","Models"),("matched_n","n"),("difference_in_treatment_effect","Δ treatment effect"),("ci95_low","CI low"),("ci95_high","CI high"),("holm_p","Holm p")],40)}
</section>

<section id="said"><h2>RQ6 · Verbal response versus objective behavior</h2>
<div class="warning"><strong>Descriptive analysis.</strong> Recognition, suspicion, resistance, and claimed response are measured after treatment. Conditioning on them does not preserve randomization. These results characterize verbal–behavioral consistency; they do not estimate the causal effect of adopting a verbal stance.</div>
{table(said_head,[("study","Study"),("profile","Model"),("contrast","Intervention"),("placement","Placement"),("semantic_field","Semantic field"),("semantic_label","Label"),("metric","Objective metric"),("n","n"),("mean_delta","Mean paired Δ"),("ci95_low","CI low"),("ci95_high","CI high")],30)}
</section>

<section id="robustness"><h2>Robustness and influence diagnostics</h2>
<p>Headline continuous outcomes are evaluated using mean and median paired differences, task-direction frequencies, log-ratio sensitivity, 10% winsorization, and leave-one-task-out influence ranges.</p>
{table(robust,[("profile","Model"),("contrast","Intervention"),("placement","Placement"),("metric","Metric"),("matched_n","n"),("mean_delta","Mean Δ"),("median_delta","Median Δ"),("positive_fraction","↑ fraction"),("mean_log1p_ratio","Mean log-ratio"),("winsorized_10pct_mean_delta","Winsorized Δ"),("loo_min","LOO min"),("loo_max","LOO max")],80)}
</section>

<section id="cue"><h2>Cue-mechanism sensitivity</h2>
<p>Headline pooled effects are re-estimated after excluding each cue mechanism in turn. Per-mechanism estimates are treated as low-powered diagnostics; the primary robustness question is whether the pooled effect changes direction or depends strongly on one mechanism.</p>
{fig_block("cue_mechanism_robustness.svg","Cue-mechanism robustness","Full pooled effects are shown visually; leave-one-mechanism-out ranges and direction stability are reported below.")}
{table(cue_summary,[("profile","Model"),("contrast","Intervention"),("placement","Placement"),("metric","Metric"),("full_n","Full n"),("mechanism_count","Cue mechanisms"),("full_mean_delta","Full mean Δ"),("loo_min","LOO min"),("loo_max","LOO max"),("direction_stability_fraction","Direction stability")],80)}
</section>

<section id="replication"><h2>RQ7 · Independent rerun</h2>
<div class="warning"><strong>Partial replication.</strong> The August 26 rerun remains incomplete and is not pooled with the primary experiment. No confirmatory p-values are inferred from the incomplete cohort. The table emphasizes headline behavioral outcomes rather than an omnibus direction-agreement percentage across hundreds of exploratory metrics.</div>
{table(rep_head,list((k,k.replace("_"," ").title()) for k in (list(rep_head[0].keys()) if rep_head else [])),50)}
</section>

<section id="claims"><h2>Summary of supported claims</h2>
<p>This table separates adjusted treatment effects from secondary, exploratory, and post-treatment descriptive evidence.</p>
{table(sorted(claims,key=claim_strength),[("evidence_class","Evidence class"),("study","Study"),("model","Model"),("contrast","Intervention"),("placement","Placement"),("outcome","Outcome"),("effect","Effect"),("ci_low","CI low"),("ci_high","CI high"),("adjusted_p","Adjusted p/q"),("status","Interpretation status")],100)}
</section>

<section><h2>Limitations and interpretation boundaries</h2>
<ul>
<li>Passive/root treatment fidelity is incompletely observed; root-condition mechanism claims are therefore limited.</li>
<li>Semantic outcomes are generated by two LLM judges; concrete pressure/resource constructs have stronger reliability than generic evaluation recognition, and human validation remains pending.</li>
<li>Llama strict success is a capability floor and is not used for success-rate treatment inference.</li>
<li>Token and duration outcomes are heavy-tailed; robustness diagnostics accompany headline continuous effects.</li>
<li>The resource intervention is a textual scarcity claim rather than an externally enforced compute budget.</li>
<li>Cue-mechanism-specific comparisons have limited per-mechanism sample size.</li>
<li>The independent rerun is incomplete and remains descriptive.</li>
</ul>
</section>

<section id="prior"><h2>Complete prior integrated report</h2>
<p class="muted">Nothing from the validated current report is discarded. The complete prior report is retained below for detailed tables, semantic prevalence, verifier details, representative matched trajectories, and the existing trial explorer.</p>
<details><summary>Open complete prior integrated report</summary>
<div class="prior-report">{prior_main()}</div>
</details>
</section>

</main></body></html>"""

(REPORT/"index.html").write_text(html_doc,encoding="utf-8")

manifest={
    "report_version":"iclr-team-preread-v2",
    "upstream_report":"reports/current/index.html",
    "output":"reports/iclr-current/index.html",
    "analysis_output":"analysis/current/behavioral_claims_v2",
    "pair_level_sources_loaded":bool(pairs),
    "semantic_effect_rows":len(semantic_effects),
    "semantic_strict_consensus_effect_rows":len(strict_semantic_effects),
    "placement_interaction_rows":len(placement_rows),
    "model_interaction_rows":len(model_rows),
    "robustness_rows":len(robust),
    "cue_mechanism_rows":len(cue_rows),
    "cue_mechanism_summary_rows":len(cue_summary),
    "supported_claim_rows":len(claims),
    "network_calls":0,"model_calls":0,"judge_calls":0,"verifier_calls":0,
}
(REPORT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")

print("="*88)
print("ICLR TEAM PRE-READ V2: GENERATED")
print("="*88)
print("report:", REPORT/"index.html")
print("analysis:", OUT)
print("pair-level sources loaded:", bool(pairs))
print(
    "semantic treatment effects:",
    len(strict_semantic_effects),
    "strict-consensus rows /",
    len(semantic_effects),
    "rows incl. judge sensitivities",
)
print("placement interactions:", len(placement_rows))
print("model interactions:", len(model_rows))
print("robustness rows:", len(robust))
print("cue-mechanism robustness rows:", len(cue_rows))
print("supported claims:", len(claims))
print()
print("IMPORTANT: inspect analysis/current/behavioral_claims_v2/ and the report before promoting")
print("this output to reports/current/. No model/judge/verifier/network calls were made.")
