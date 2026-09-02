# Campaign V2 Execution Plan — `replication-20260902-v1`

Prepared 2026-09-02. **Nothing in this campaign has been executed.** No Harbor
job was launched, no benchmark trajectory was run, and no historical result
directory was read, moved or modified by the preparation work described here.

---

## 1. What this campaign is

A second, completely fresh experimental campaign for SWE-EvalPressure, covering
all four model stacks and both modes.

| mode | base tasks | arms | per profile | profiles | fresh trajectories |
|---|---:|---:|---:|---:|---:|
| FULL | 70 | 10 | 700 | 4 | **2,800** |
| RESOURCE | 70 | 3 | 210 | 4 | **840** |
| | | | | | **3,640** |

**FULL arms (10):** `clean-n`, `eval-src`, `eval-fin-src`, `eval-self-src`,
`eval-root`, `eval-fin-root`, `eval-self-root`, `eval-scaf`, `eval-fin-scaf`,
`eval-self-scaf`.

**RESOURCE arms (3):** `clean-n`, `eval-scaf`, `eval-resource-scaf`.

### 1.1 RESOURCE is deliberately self-contained

`clean-n` and `eval-scaf` in RESOURCE are byte-identical, as task definitions,
to the FULL cells of the same name — verified by sha256 of
`environment/benchmark_seed/seed.json`: **140 identical, 0 differing, per
profile**. They are executed independently anyway.

This is a deliberate design decision, not an oversight. RESOURCE gets its own
freshly executed control condition, its own evaluation-only comparator and its
own resource-deprivation treatment, so that RESOURCE analysis never depends on a
FULL trajectory. The duplication costs 560 of the 3,640 trajectories
(140 × 4 profiles) and buys the guarantee that FULL and RESOURCE are each
internally complete within one campaign.

The lineage is recorded, per trial, as `resource_derivation_parent` in the cell
manifests, purely for transparency. It is never used to substitute data.
`campaign/validate.py` check **R11** fails the campaign if that lineage field is
missing, and `campaign/analyze.py` runs an explicit `cross_mode_leak_check` that
fails if any run id or trial directory is shared between modes.

---

## 2. Campaign root and isolation

```
results/campaigns/replication-20260902-v1/
  CAMPAIGN_MANIFEST.json     frozen design, pins, cost model, policy
  CAMPAIGN_README.md         operator-facing summary
  datasets/                  hardlink snapshot of generated/ + 24 pre-built shards
  full/                      Harbor job outputs, mode=full
  resource/                  Harbor job outputs, mode=resource
  manifests/cells/           24 frozen per-(mode,profile,shard) manifests
  provenance/                attempts.jsonl, accepted_runs.json, build_report.json, corpus.jsonl
  validation/                validation_report.json
  analysis/                  campaign_summary.json
  logs/                      preflight.jsonl, runner.jsonl
```

The namespace mechanism is the repo's existing `RESULTS_ROOT` override, which is
already honoured end-to-end by `scripts/00_common.sh`, `scripts/05_run_profile.sh`
and `scripts/06_run_matrix_adaptive.sh`. `campaign.sh` exports it, so **every V2
output lands under the campaign root from the moment it is generated**. Nothing
is written to a historical location and moved later.

The isolation is enforced structurally, not by convention:
`campaign/lib.py:assert_campaign_path()` raises on any path outside
`results/campaigns/replication-20260902-v1/`. Provenance recording, corpus
building and validation all route through it, so a historical Aug 2026 run
directory cannot enter this campaign even if someone passes one deliberately.
Test `test_08_historical_run_dir_refused` proves it.

### 2.1 Dataset integrity

Strategy: **campaign-local immutable snapshot, built with hardlinks**, plus
frozen content hashes.

Chosen over simply pointing at `generated/` read-only because `./lab.sh prepare`
rebuilds `generated/<mode>/<profile>` with `rmtree` + rewrite. A hardlink
snapshot survives that untouched (the rebuild creates new inodes); a read-only
reference would silently start pointing at different content mid-campaign.
Hardlinks share blocks, so the 809 MB corpus was snapshotted at effectively zero
incremental disk cost (verified: `nlink=3` on snapshot files).

Hardlinks still share an inode, so an *in-place* edit would propagate. That
residual hole is closed by content hashes: every task directory's sha256 is
frozen in its cell manifest and re-verified by `preflight` before every shard
launch.

---

## 3. Sharding

Fixed-size over the 70 ordered base task ids, 30 per shard — the same scheme the
existing `scripts/05_shard_dataset.py` uses, reimplemented campaign-locally with
hard assertions on the resulting counts.

| shard | base tasks | FULL /profile | FULL ×4 | RESOURCE /profile | RESOURCE ×4 |
|---:|---|---:|---:|---:|---:|
| 1 | 1–30 | 300 | 1,200 | 90 | 360 |
| 2 | 31–60 | 300 | 1,200 | 90 | 360 |
| 3 | 61–70 | 100 | 400 | 30 | 120 |
| | | 700 | **2,800** | 210 | **840** |

All 24 shard datasets are already materialised and hashed.

---

## 4. Version pinning

Every stack is explicitly pinned. Preflight **fails closed** if any pin env var
is unset — an unpinned stack is treated as a campaign-stopping error, not a
default.

| profile | agent | version | pin env | model |
|---|---|---|---|---|
| claude | claude-code | 2.1.247 | `CLAUDE_CODE_VERSION` | `anthropic/claude-opus-4-8` |
| fable | claude-code | 2.1.247 | `CLAUDE_CODE_VERSION` | `anthropic/claude-fable-5` |
| codex | codex | 0.147.0 | `CODEX_VERSION` | `openai/gpt-5.6` |
| llama | mini-swe-agent | 2.4.5 | `MINI_SWE_VERSION` | `openai/llmengine/llama-3-3-70b-instruct` |

Runtime also pinned: Harbor `0.20.0`, Modal VM runtime, `harbor_repeats=1`,
mini-swe-agent's `litellm==1.83.0` (`MINI_SWE_LITELLM_VERSION`), uv bootstrap
`0.7.13`. All pinned versions were confirmed to exist upstream (npm / PyPI).

### 4.1 Gap that had to be closed

Claude Code was **not pinnable** before this work. Harbor's `ClaudeCode` agent
accepts a `version` kwarg exactly like the codex and mini-swe agents
(`BaseInstalledAgent.__init__(..., version: str | None = None, ...)`), but
`scripts/05_run_profile.sh` only forwarded it for codex and llama. So Harbor
installed *latest* on every claude and fable job, and the Aug 2026 corpus drifted
across claude-code 2.1.241 / 2.1.243 / 2.1.245 / 2.1.246 / 2.1.247 **mid-study**.

The fix is a five-line, backwards-compatible addition to
`scripts/05_run_profile.sh`; with `CLAUDE_CODE_VERSION` unset the behaviour is
byte-for-byte what it was before. Validator check `:7` fails the campaign if more
than one agent version appears in a profile, or if it differs from the pin.

---

## 5. Attempt isolation

The Aug 2026 provenance failure is the thing this design exists to prevent.
`audits/claude_opus5/scripts/01_build_inventory.py` hard-codes a `RUN_DIRS` list
mixing a good shard-1 run, a partial shard-2 run, a budget-failed shard-2 run, an
archived abort and a repair run — then **dedupes by `task_name`**. Because a
failed attempt and a good attempt produce directories with the same task name,
whichever the loop saw last silently won. A failed trial could be promoted into
the primary corpus with no trace.

Five structural fixes:

1. **No globbing.** Runs enter the corpus only via an explicit, append-only
   ledger, `provenance/attempts.jsonl`.
2. **No paths outside.** Every run dir passes `assert_campaign_path`.
3. **One winner per cell.** Exactly one attempt per cell may be `complete`.
   A second is a hard error, never a dedupe.
4. **No dedupe by name.** A repeated `(cell, base_task, arm)` raises.
5. **Superseding is explicit.** A failed attempt is marked `superseded_by` its
   retry, stays in the ledger as evidence, and stays out of the corpus.

Every attempt records: `campaign_id`, `mode`, `shard`, `shard_index`, `profile`,
`attempt_id` (`<mode>-<profile>-s<N>-a<NN>`), `run_id`, `run_dir`, `status`,
`model_id`, `agent`, `agent_version_requested`, `agent_version_observed`,
`started_at`, `finished_at`, `expected_trials`, `observed_trials`,
`status_counts`, `superseded_by`, `recorded_at`, `note`.

**If an attempt fails it is preserved as FAILED and the campaign stops.** A
replacement gets a new `attempt_id`. There is no backfilling and no
partial-shard acceptance.

### 5.1 A fail-open gap found and closed during testing

The duplicate-cell test initially *passed the validator*. The corpus builder
correctly detected the duplicate and returned a non-zero exit code — but it had
already written a corpus with the duplicate skipped, which happened to contain
exactly 3,640 rows and therefore validated clean.

That is the Aug 2026 failure mode one layer up: a builder that knows the data is
wrong, handing the validator something that looks right. `cmd_build` now fails
closed — it writes no corpus at all when it has errors, deletes any stale one,
and records `provenance/build_report.json`, which validator check **X4** reads.

---

## 6. Validation

`campaign/validate.py` certifies the campaign only if all of the following hold.

**Cross-cutting**

| id | check |
|---|---|
| X0 | corpus is non-empty |
| X1 | totals equal 2,800 (FULL) + 840 (RESOURCE) = 3,640 |
| X2 | no contributing run dir lives outside the campaign namespace |
| X3 | every accepted attempt's dataset hash matches its frozen cell manifest |
| X4 | the provenance corpus builder finished with zero errors |

**Per mode × profile** (`F:` for FULL, `R:` for RESOURCE — 8 groups)

| id | check |
|---|---|
| `:1` | all 3 shards present, each with exactly one accepted attempt |
| `:2` | 70 distinct base tasks |
| `:3` | every base task carries the complete canonical arm set |
| `:4` | exact trial count, zero duplicates, zero missing cells |
| `:5` | zero synthetic trials |
| `:6` | zero budget-censored trials |
| `:7` | one agent version across the profile, equal to the pin |
| `:8` | one model id across the profile, equal to the pin |
| `:9` | every run id traces to an accepted campaign attempt |
| `:10` | every row carries this campaign id |
| R11 | RESOURCE control arms carry explicit FULL lineage (recorded, never pooled) |

FULL expects exactly 2,800 cells; RESOURCE exactly 840. **RESOURCE is validated
as its own 3-arm execution set — its controls are never sourced from FULL.**

The validator rejects: missing cells, duplicate cells, synthetic trials,
budget-censored trials, cross-attempt backfills, historical results, a wrong
campaign id, and wrong mode/shard/profile provenance.

### 6.1 Local test evidence

`campaign/tests/test_tooling.py` — 12 tests, all passing, fully offline. Each
builds a synthetic campaign namespace, injects exactly one defect, and asserts
the tooling fails with the expected check id. A control test asserts a clean
3,640-trial corpus passes, so the failures are meaningful.

| test | defect injected | detected by |
|---|---|---|
| 00 | none (control) | validates clean, 3,640 trials |
| 00b | — | expected totals 2,800 / 840 / 3,640 / 24 cells |
| 01 | incomplete shard (17 trials dropped) | X1 + `F:full/claude:1,:2,:4` |
| 02 | duplicate cell | build fails closed → X4 + X0 |
| 03 | second attempt for a cell | earlier attempt preserved as `superseded`; forcing both accepted is a hard error |
| 04 | 12 synthetic trials | `F:full/fable:5` |
| 05 | 9 budget-censored trials | `R:resource/claude:6` |
| 06 | claude-code 2.1.241 drift | `F:full/claude:7` |
| 07 | missing arm | `R:resource/llama:3` + `:4` |
| 08 | historical run dir supplied | `assert_campaign_path` raises |
| 09 | unvalidated campaign | analysis refuses to run |
| 10 | — | no cross-mode substitution; RESOURCE owns 280 `clean-n` + 280 `eval-scaf` |

Tests 04, 05 and 07 deliberately **force-accept** the defective attempt, so they
exercise the validator rather than stopping at the earlier attempt-status gate.
Both layers are therefore proven independently.

---

## 7. Budget audit

### 7.1 Key identity — the historical key was not replaced

The matrix runner would use, right now, the key in `.env` line 2
(`export LITE_LLM_KEY=...`). `scripts/00_common.sh:set_litellm_aliases()` maps
that single key onto `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`ANTHROPIC_AUTH_TOKEN`, `GEMINI_API_KEY` and `MSWEA_API_KEY` — so **all four
profiles draw on one budget pool**.

Fingerprint: `sha256:dc0b76aef04c918e… / suffix …S_Vw`.

That fingerprint is **byte-identical** to this session's `ANTHROPIC_API_KEY` and
`OPENAI_API_KEY`, and its suffix matches the historical failure text:

```
Budget has been exceeded! Key=pub-sprl-eval-awareness (sk-...S_Vw)
Current cost: 10000.701821100023, Max budget: 10000.0
```

**So: this is the same key that was exhausted in Aug 2026, alias
`pub-sprl-eval-awareness`. It has not been rotated or replaced. Its budget was
reset between 2026-08-27 and 2026-09-02** (spend fell from $10,000.70 to under
$100). The reset looks like a monthly cycle, but that is inferred from two data
points, not confirmed — treat the reset date as unknown.

There is exactly one active benchmark key. There is no second key to report.

### 7.2 Does it work now

Yes. `GET /v1/models` returns HTTP 200 with the key and HTTP 401 without it, and
all four pinned models are exposed:

- `anthropic/claude-opus-4-8` — present
- `anthropic/claude-fable-5` — present
- `openai/gpt-5.6` — present
- `openai/llmengine/llama-3-3-70b-instruct` — present as
  `llmengine/llama-3-3-70b-instruct`; litellm strips the leading `openai/`
  provider prefix client-side before dispatch. Preflight accepts either form.

### 7.3 Budget readings

Admin routes `/key/info`, `/user/info` and `/key/health` are HTTP 403 on this
gateway. Budget is therefore read from response headers on a **non-billable**
request: a pinned, key-allowed model with an empty message list, which the
upstream provider rejects before any inference. Every reading below returned
`x-litellm-response-cost: 0`.

| reading (UTC) | spend | max | remaining |
|---|---:|---:|---:|
| 03:03:48 | $98.22 | $10,000.00 | $9,901.78 |
| 03:37 | $107.34 | $10,000.00 | $9,892.66 |
| 03:40:06 | $107.76 | $10,000.00 | $9,892.24 |

TPM 5,000,000 / RPM 5,000,000.

**Spend rose ~$9.54 in 37 minutes while nothing of mine was running.** This key
is shared and has concurrent consumers. Two consequences:

1. The remaining budget at launch time will be **lower** than the figure above.
   Re-read it immediately before starting.
2. A concurrent consumer can exhaust this pool mid-campaign — precisely how the
   Aug 2026 study died. `preflight` therefore re-checks the budget before every
   single shard and stops the campaign if the remaining plan no longer fits.

An earlier probe technique — an unroutable model name — must not be used: this
key carries a model ACL, and LiteLLM rejects an out-of-ACL model at HTTP 401
*before* stamping budget headers, which reads as an auth failure and hides the
budget entirely. `lib.probe_budget()` uses the allowed-model / empty-messages
form and additionally refuses any reading whose response cost is non-zero.

### 7.4 Cost model

Derived from the Aug 2026 corpus **for cost provenance only** — no historical
trajectory is reused as data. Synthetic and budget-censored trials are excluded,
so the means reflect trials that actually called a model.

| cell | historical n | mean $ | p90 $ | V2 trials | expected $ | p90 $ |
|---|---:|---:|---:|---:|---:|---:|
| full/claude | 356 | 2.5164 | 4.8542 | 700 | 1,761.46 | 3,397.91 |
| full/codex | 537 | 1.2946 | 2.2853 | 700 | 906.25 | 1,599.74 |
| full/fable | 331 | 3.5893 | 6.2605 | 700 | 2,512.54 | 4,382.35 |
| full/llama | 0 | — | — | 700 | 0.00 | 0.00 |
| resource/claude | 209 | 2.8388 | 6.3119 | 210 | 596.14 | 1,325.49 |
| resource/codex | 210 | 1.0245 | 2.0607 | 210 | 215.16 | 432.75 |
| resource/fable | 198 | 4.1513 | 8.8210 | 210 | 871.78 | 1,852.42 |
| resource/llama | 0 | — | — | 210 | 0.00 | 0.00 |

- **Expected FULL: $5,180.25**
- **Expected RESOURCE: $1,683.08**
- **Expected TOTAL: $6,863.33**
- Planning total (expected + 20% contingency): **$8,236.00**
- p90 total: **$12,990.66**

**Two caveats that must not be glossed over:**

1. **Llama is priced at $0 because the gateway returned zero cost for every one
   of its historical trials** — the `llmengine/` route appears not to be billed
   through this key's cost accounting. If that changes, llama's 910 trials become
   an unmodelled cost. The 20% contingency is the only cover for this.
2. The p90 total **exceeds** the remaining budget. The campaign fits on
   *expectation*, not on a heavy-tail outcome. Preflight passes but emits this
   warning explicitly, and re-evaluates between every shard.

### 7.5 Fail-closed gate

`preflight` requires `remaining ≥ remaining_planning_cost × 1.10`.

- Required at launch: **$9,059.52**
- Remaining (03:40): **$9,892.24**
- **Surplus: ~$832.72 (~9%)**

The gate recomputes the *remaining* plan as shards complete, so cost already
spent stops being demanded of the budget and the margin naturally widens as the
campaign proceeds. If a concurrent consumer erodes the pool, the campaign stops
between shards rather than mid-shard.

This is a genuinely tight margin. It is a pass, not a comfortable one.

---

## 8. Execution contract

`run-full` and `run-resource` each execute, for shard i in 1, 2, 3:

```
preflight  →  claude  →  fable  →  codex  →  llama  →  validate  →  next shard
```

- Any failed **preflight** stops the campaign before anything is launched.
- Any failed **validation** stops the campaign.
- A failed attempt is recorded as FAILED in the ledger and stops the campaign.
- Every launch and every outcome is appended to `logs/runner.jsonl`.

The runner refuses a campaign id that does not match the id this tree is pinned
to, rather than writing into the wrong namespace.

---

## 9. What changed in the repository

**Modified (both backwards-compatible; no-ops when the new env vars are unset):**

- `scripts/00_common.sh` — added `CLAUDE_CODE_VERSION` (empty default preserves
  historical behaviour); made `GENERATED_ROOT` overridable; added
  `MINI_SWE_LITELLM_VERSION`, `GENERATED_ROOT`, `CLAUDE_CODE_VERSION` to the
  `.env`-sourcing preserve list.
- `scripts/05_run_profile.sh` — added `--shard-label` so a pre-sharding caller
  keeps a meaningful job name; forward `CLAUDE_CODE_VERSION` to Harbor as
  `--ak version=<v>` for the claude-code agent.

**New:**

- `campaign.sh` — the runner.
- `campaign/lib.py` — shared constants, namespace guard, trial scanning, budget probe.
- `campaign/prepare.py` — snapshot, shard, cell manifests, campaign manifest + README.
- `campaign/preflight.py` — fail-closed namespace / integrity / version / model / budget gate.
- `campaign/provenance.py` — attempt ledger and corpus builder.
- `campaign/validate.py` — the validation gate.
- `campaign/analyze.py` — analysis entry point; refuses an unvalidated campaign.
- `campaign/cost_model.py` — cost estimates from the historical corpus.
- `campaign/tests/test_tooling.py` — 12 offline detection tests.

Nothing under `results/full`, `results/resource`, `results/archive`,
`results/quarantine`, `results/failed_repair_attempts`, `generated/` or
`analysis/` was modified. The canonical benchmark content was not regenerated.

---

## 10. Local status

- `./campaign.sh prepare` — **done**. 24 cells, 3,640 expected trials, 24 shard
  datasets materialised and hashed.
- `./campaign.sh preflight --quick` — **passes**, with the documented p90 warning.
- Full-hash preflight (all 3,640 task dirs) — **passes** (integrity section).
- `campaign/tests/test_tooling.py` — **12/12 pass**.
- `./campaign.sh validate` on the empty campaign — **correctly refuses**.
- `./campaign.sh analyze` on the empty campaign — **correctly refuses**.

**Zero trajectories executed. Zero Harbor jobs launched. Zero model inference
calls.** The only network requests made were gateway metadata reads and
non-billable budget probes, each confirmed at `x-litellm-response-cost: 0`.
