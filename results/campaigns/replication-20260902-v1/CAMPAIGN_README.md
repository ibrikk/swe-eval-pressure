# Campaign `replication-20260902-v1`

Second, completely fresh experimental campaign for SWE-EvalPressure.

## Hard rule

**Every trajectory under this directory is newly executed by this campaign.**
Nothing from the Aug 2026 corpus (`results/full`, `results/resource`,
`results/archive`, `results/quarantine`, `results/failed_repair_attempts`) is
reused, pooled, salvaged or backfilled. The tooling enforces this structurally:
`campaign/lib.py:assert_campaign_path` refuses any run directory outside
`results/campaigns/replication-20260902-v1/`.

## Design

| mode | base tasks | arms | per profile | profiles | total |
|---|---|---|---|---|---|
| full | 70 | 10 | 700 | 4 | **2800** |
| resource | 70 | 3 | 210 | 4 | **840** |
| | | | | | **3640** |

FULL arms: `clean-n`, `eval-src`, `eval-fin-src`, `eval-self-src`, `eval-root`, `eval-fin-root`, `eval-self-root`, `eval-scaf`, `eval-fin-scaf`, `eval-self-scaf`

RESOURCE arms: `clean-n`, `eval-scaf`, `eval-resource-scaf`

### RESOURCE is deliberately self-contained

`clean-n` and `eval-scaf` task definitions in RESOURCE are byte-identical to the
FULL cells of the same name (verified: 140/140 matching `seed.json` hashes per
profile). They are executed **independently anyway**. RESOURCE gets its own
freshly executed control, eval-only comparator and resource-deprivation
treatment, so RESOURCE analysis never depends on a FULL trajectory. The
duplication is recorded per trial as `resource_derivation_parent` for
transparency; it is never used to substitute data.

## Sharding

Fixed-size over the 70 ordered base task ids, 30 per shard.

| shard | base tasks | FULL /profile | FULL all | RESOURCE /profile | RESOURCE all |
|---|---|---|---|---|---|
| 1 | 1-30 | 300 | 1200 | 90 | 360 |
| 2 | 31-60 | 300 | 1200 | 90 | 360 |
| 3 | 61-70 | 100 | 400 | 30 | 120 |

## Version pins

| profile | agent | version | pin env | model |
|---|---|---|---|---|
| claude | `claude-code` | `2.1.247` | `CLAUDE_CODE_VERSION` | `anthropic/claude-opus-4-8` |
| fable | `claude-code` | `2.1.247` | `CLAUDE_CODE_VERSION` | `anthropic/claude-fable-5` |
| codex | `codex` | `0.147.0` | `CODEX_VERSION` | `openai/gpt-5.6` |
| llama | `mini-swe-agent` | `2.4.5` | `MINI_SWE_VERSION` | `openai/llmengine/llama-3-3-70b-instruct` |

Runtime: Harbor `0.20.0`, Modal VM runtime, `mini-swe-agent` extra `litellm==1.83.0`,
uv bootstrap `0.7.13`, `harbor_repeats=1`.

## Layout

```
replication-20260902-v1/
  CAMPAIGN_MANIFEST.json   frozen design, pins, cost model, policy
  CAMPAIGN_README.md       this file
  datasets/                hardlink snapshot of generated/ + pre-built shards
  full/                    Harbor job outputs, mode=full
  resource/                Harbor job outputs, mode=resource
  manifests/cells/         one frozen manifest per (mode, profile, shard)
  provenance/              attempts.jsonl, accepted_runs.json, corpus.jsonl
  validation/              validation_report.json
  analysis/                analysis outputs
  logs/                    preflight + runner logs
```

## Operating it

```
./campaign.sh prepare      replication-20260902-v1
./campaign.sh preflight    replication-20260902-v1
./campaign.sh run-full     replication-20260902-v1
./campaign.sh run-resource replication-20260902-v1
./campaign.sh validate     replication-20260902-v1
./campaign.sh analyze      replication-20260902-v1
```

`run-full` and `run-resource` preflight before every shard, run all four
profiles for that shard, validate, and **stop on the first failure**. A failed
attempt is preserved as FAILED; the replacement gets a new `attempt_id`. There
is no backfilling and no partial-shard acceptance.
