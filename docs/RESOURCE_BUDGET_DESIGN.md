# RESOURCE mode: 840 fresh trajectories, or 280 plus reused controls?

**Status: analysis and recommendation only. Nothing in RESOURCE has been changed.**
No dataset, manifest, or trial in `resource/` was touched to produce this note; every
number below comes from the shard dataset manifests and from the trajectories the
interrupted FULL shard-1 run already wrote.

## The two options

RESOURCE has three arms per base task — `clean-n`, `eval-scaf`, `eval-resource-scaf` —
across 70 base tasks (30 + 30 + 10 over three shards) and 4 profiles.

| | Trials | Description |
|---|---|---|
| **Option A** | 70 × 3 × 4 = **840** | Run all three arms fresh inside RESOURCE. |
| **Option B** | 70 × 1 × 4 = **280** | Run only `eval-resource-scaf`; use the same campaign's FULL `clean-n` and `eval-scaf` as controls. |

## Are the reused controls actually the same treatment?

This is the question that decides it, and it is answerable from disk rather than by
argument. For all 70 base tasks × 2 shared arms × 4 profiles = **560 cells**:

```
shared (clean-n + eval-scaf) task cells across all shards/profiles: 560
identical manifest ENTRIES : 560
identical ON-DISK TREES    : 560
```

The manifest entries match field-for-field (including `content_hash`), and
`sha256_tree` over the materialised task directories matches for every one. The
mode-level manifests differ only in fields that govern *which additional arms exist*
(`delivery_channels`, `resource_deprivation_message_index`, `variants_per_task`), not
in the content of `clean-n` or `eval-scaf`. `scripts/05_run_profile.sh` uses `MODE`
only to select the dataset path and the results subdirectory: same agent pin, same
model pin, same `allow_internet`, same Modal runtime, same verifier.

So a RESOURCE `clean-n` trial is not a *different* treatment from a FULL `clean-n`
trial. It is a **temporal replicate of a byte-identical task under an identical
runtime configuration**. Option B is therefore not a silent substitution of one
condition for another; it is a decision not to purchase a second replicate.

## What Option B actually gives up

Reusing FULL's controls gives up exactly one thing: **contemporaneity**. FULL and
RESOURCE run in different wall-clock windows, so any control reuse assumes nothing
drifted in between — provider-side model updates behind a pinned alias, gateway
routing changes, Modal base-image changes, verifier changes.

That assumption is not free, and this campaign has already produced direct evidence
of time-varying infrastructure: three Modal image IDs
(`im-x6EXQtv6VljLdOL1LimybV`, `im-Y3lpPwyqkg1CMY3rSAAy3O`, `im-gerqAtzpAdNqKhzhs9yFdy`)
failed identically across all four profiles hours apart within a single shard. Infra
state is a live variable here, not a constant.

Two things bound the risk:

1. **Same-campaign, not historical.** Option B reuses FULL trajectories from
   `replication-20260902-v1` under the same pinned models and agent versions — days
   apart, not the August corpus. Reusing the Aug-2026 runs would be a different and
   unacceptable proposal, and `campaign.cells` structurally refuses it.
2. **The comparison of interest is within-arm.** The RESOURCE hypothesis concerns
   `eval-resource-scaf` versus `eval-scaf` — resource-deprivation pressure added to
   an otherwise identical scaffold cue. If drift between the FULL and RESOURCE windows
   is a real concern, it applies to `eval-scaf` and `eval-resource-scaf` alike only
   under Option A; under Option B the treatment is measured in one window and the
   control in another, so drift becomes a confound *of the effect being measured*.

That last point is the honest cost of Option B and should not be glossed: it converts
a fully within-window contrast into a partly across-window one.

## Cost

Per-trial costs are measured from the 976 valid FULL shard-1 trajectories, not from
the historical corpus:

| profile | n | mean $/trial | p90 $/trial |
|---|---|---|---|
| claude | 197 | 2.250 | 4.242 |
| fable  | 287 | 3.681 | 6.520 |
| codex  | 295 | 1.196 | 2.080 |
| llama  | 197 | 0.000 | 0.000 |

(`llama` bills $0 on this gateway route; its true cost is unknown-but-small and is
covered by the 20% contingency.)

| work | trials | mean | +20% contingency | p90 |
|---|---|---|---|---|
| shard-1 repair | 224 | $285.58 | $342.70 | $532.09 |
| remaining FULL (shards 2+3) | 1,600 | $2,850.80 | $3,420.96 | $5,136.80 |
| **RESOURCE Option A** | 840 | **$1,496.67** | $1,796.00 | $2,696.82 |
| **RESOURCE Option B** | 280 | **$498.89** | $598.67 | $898.94 |

**Difference: $997.78 at mean cost, $1,197.34 with contingency, $1,797.88 at p90.**

Against a gateway key at $3,299.60 spent of $10,000 ($6,700.40 remaining):

| | remaining work (+20%) | projected final spend | headroom | p90 remaining work | fits at p90? |
|---|---|---|---|---|---|
| Option A | $5,559.66 | $8,859.26 | $1,140.74 | $8,365.71 | **no — over by $1,665.31** |
| Option B | $4,362.32 | $7,661.92 | $2,338.08 | $6,567.83 | yes, by $132.57 |

## Recommendation

**Option B, with two conditions.**

The deciding factor is not the $998. It is that Option A does not survive its own
pessimistic cost path: at p90 per-trial costs, Option A needs $8,365.71 against
$6,700.40 remaining. A design that only completes if costs land at or below the mean
is a design that ends in a budget-censored RESOURCE mode — and this project already
has a 1,065-marker example of what that produces. Option B fits at mean comfortably
and at p90 barely, which is a meaningfully different risk position.

Separate contemporaneous controls are **not materially necessary for validity** —
the reused controls are byte-identical tasks from the same campaign under identical
pins — but they are not worthless either, because the drift they would absorb lands
directly on the effect being estimated. That is a real, bounded methodological cost,
accepted deliberately here rather than hidden.

Conditions:

1. **Re-decide after FULL completes**, using measured FULL shard-2/3 costs rather than
   these projections. If actual spend lands near the mean path, Option A's p90 gap may
   close, and the extra 560 trajectories are then worth buying. The controller now
   polls the gateway on a timer and records spend in its result file, so this will be a
   measurement rather than an estimate.
2. **Label the reuse explicitly in the corpus.** Every control row used in a RESOURCE
   comparison must carry the FULL run directory and trial it came from, so a reader can
   see the arm was measured in a different window. Reuse that is not visible in the
   data is the failure mode to avoid, not reuse itself.

If the drift concern is judged dominant, there is a middle path: run
`eval-resource-scaf` **and** `eval-scaf` fresh (70 × 2 × 4 = 560 trials) and reuse only
`clean-n`. That keeps the treatment/control contrast the mode exists to measure inside
a single window, at $997.78 mean / $1,197.34 with contingency / $1,797.88 at p90 —
exactly two-thirds of Option A. It fits at mean cost ($4,960.99 remaining work with
contingency, $1,739.41 headroom) but, like Option A, does **not** fit at p90
($7,466.77 against $6,700.40 remaining). It is the right choice only if the budget
ceiling is raised or FULL comes in under the mean path.
