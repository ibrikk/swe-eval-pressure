# Scheduler audit — `scale-5m-adaptive` (Section 1)

Read-only audit. Nothing modified. Evidence: `scripts/06_run_matrix_adaptive.sh` (379 lines),
`scripts/05_run_profile_pool.sh` (160), `scripts/05_run_profile.sh`, 5 historical
`adaptive-controller-*` dirs, 3,374 historical trials with complete timing + token telemetry.

## 1. How each profile's concurrency is chosen

| Profile | Mechanism | Adaptive? |
|---|---|---|
| Claude | `ADAPTIVE_CLAUDE_CONCURRENCY:-20`, exported once as `HARBOR_CONCURRENCY`, `exec`'d into one Harbor job | **No — static** |
| Fable  | `ADAPTIVE_FABLE_CONCURRENCY:-16`, same | **No — static** |
| Codex  | `ADAPTIVE_CODEX_CONCURRENCY:-16`, same | **No — static** |
| Llama  | while-loop, ×2 growth / ÷2 backoff between sequential batches, floored by `phase_floor()` | Partially |

`phase_floor()` (lines 224–232) keys off how many *fixed* profiles are still alive:
`0 → LLAMA_MAX(40)`, `1 → 24`, `2 → 12`, `else → LLAMA_MIN(4)`.

## 2. Redistribution, donation, starvation

- **Capacity flows in one direction only.** Fixed profiles finishing raises Llama's floor. Claude
  finishing never lets Fable or Codex expand — their `-n` was fixed at launch.
- **Nothing is work-conserving.** No queued-work signal exists between profiles.
- **Llama starvation is structural, not incidental.** While all three fixed profiles are alive,
  `phase_floor()` pins Llama at 4. This is the direct cause of the observed "relatively little Llama".

## 3. What the feedback signal actually is

Rate-limit **text-grep only**: `grep -Eqi 'rate.?limit|too many requests|429'` over the batch log,
plus `scripts/11_rate_limit_failures.py`. **Aggregate TPM is never measured anywhere in the codebase.**
The script's own help text concedes it: *"the gateway does not expose an authoritative live
remaining-TPM counter to this runner."*

Consequence: the controller reasons about **process counts**, never token consumption.

## 4. Blocking architectural constraint

`harbor run ... -n "$HARBOR_CONCURRENCY"` is a **launch-time CLI flag**. Concurrency cannot be
retuned on an in-flight Harbor job. Any adaptive controller must re-chunk work into fresh Harbor
invocations (as the Llama path does) or run multiple concurrent jobs per profile. This constrains
the entire design.

Also: with a single key, `05_run_profile_pool.sh` line 43 `exec`s straight through — its multi-key
rate-limit failover is dead code.

## 5. Why the 5M allowance was "underutilized" — the premise is wrong

Measured over 3,374 historical trials, minute-bucketed, tokens spread across each trial's
`started_at`→`finished_at`:

| Basis | median | p90 | max | max vs 5M |
|---|---|---|---|---|
| Metered (fresh input + cache-creation + output) | 19,718 | 340,378 | **467,693** | **9.4%** |
| Cache-inclusive (prompt+completion as logged)   | 37,301 | 6,509,957 | 11,348,569 | 227% |

`total_cached_tokens == total_cache_read_input_tokens` exactly (verified field-by-field), so the
metered row is the real cost basis.

**Zero rate-limit events in the entire historical record**: `rate_event` sums to 0 across all five
controller logs, and zero 429/rate-limit lines in any batch log. The ÷2 backoff never once fired.
The single `ApiRateLimitError` on Aug-27 is Harbor's `_classify_exec_error` heuristically
relabelling a codex `exit 1` — not a gateway 429.

The cache-inclusive row peaked at **227% of the cap for 12+ consecutive minutes with no throttling**.
That is only possible if either (a) the gateway does not meter cache reads, or (b) the TPM/RPM
limits are not enforced at all — `TPM == RPM == 5,000,000` exactly is the signature of unset
placeholder values. The evidence cannot fully separate (a) from (b), **but both readings imply the
same thing: the 5M TPM cap was never the binding constraint, and never came close to binding.**

### What the actual bottleneck was

- **47.0% of active minutes had only ONE profile running.** Only 39.3% had all four.
- Mean concurrent workers while active: claude 16.7, fable 13.7, codex 14.1, **llama 8.3 (median 4)**.
- Llama batches run **sequentially**, 2 base tasks at a time, each paying fresh Harbor startup.

The long wall clock came from **serialization and static ceilings**, not from failure to
redistribute a token budget that was 90% idle the whole time.

## 6. Empirical per-profile worker cost (basis for new defaults)

| Profile | n | median tok/trial (metered) | median sec/trial | metered TPM per worker |
|---|---|---|---|---|
| claude | 565 | — | 838 | ~6,893 |
| fable  | 529 | — | 810 | ~6,481 |
| codex  | 747 | — | 703 | ~7,291 |
| llama  | 554 | 13,095 | 453 | ~1,881 |

Cache-inclusive per-worker rates are ~20× higher (claude 149k, fable 106k, codex 127k, llama 1.9k)
and are what a naive `prompt+completion` meter would report.

## 7. Cost is the real scarce resource

Historical mean cost/trial: claude $2.64, fable $3.65, codex $1.22 (llama: no cost data —
self-hosted, $0). At 910 trials/profile → **~$6,831 estimated for the 3,640-trial campaign**.

Live budget probe (non-billable, `x-litellm-response-cost: 0`):
`max_budget $10,000 / spend $118.81` → **~$9,881 remaining**. The key was reset after Aug-27.

Headroom is ~31%, and per-trial cost is heavy-tailed (max observed: fable $24.90, claude $18.43).
**Budget, not TPM, is the constraint that can actually stop this campaign.**

## 8. Design implication

Targeting 4,600,000 metered TPM would require roughly **10× historical concurrency (~550 concurrent
workers)**. A pure AIMD loop that ramps while utilisation < 85% of target would observe ~9% and
increase without bound until something untested breaks — Modal VM capacity, local process/memory
limits, or upstream per-model provider limits that are separate from the gateway key.

Therefore the TPM target is implemented as a **safety ceiling that throttles down**, never as a
goal that drives unbounded ramp-up. Real throughput gain comes from work-conservation, overlap,
and removing the Llama floor — which the measurements say is where the loss actually is.
