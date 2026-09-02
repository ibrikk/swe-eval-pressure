#!/usr/bin/env python3
"""Deterministic offline tests for the bounded adaptive scheduler.

NO Harbor job, NO model call, NO benchmark trajectory is launched anywhere in
this file. Everything is synthetic state or a temporary directory.

Coverage maps 1:1 onto the requested behaviours:
   1 capacity redistribution when claude finishes
   2 capacity redistribution when fable finishes
   3 llama never starves while queued
   4 a single remaining profile expands to its safe cap
   5 real metered TPM at the ceiling causes scale-down (live signal only)
   6 low TPM alone does NOT cause unbounded scale-up
   7 a corroborated 429 causes backoff and is retryable
   8 a false Harbor rate-limit verdict does NOT trigger a retry
   9 transient pre-model setup failure retries
  10 budget exhaustion stops immediately and is never retried
  11 partial model trajectory preserved with explicit retry lineage
  12 retry count is bounded
  13 failed attempts are never silently backfilled
  14 validator accepts explicit retry provenance
  15 validator rejects silent replacement
  16 controller never schedules beyond per-profile caps or the global ceiling
"""
from __future__ import annotations

import importlib
import json
import unittest
from pathlib import Path

from campaign import failures
from campaign.scheduler import (Scheduler, TpmMeter, PROFILE_LIMITS,
                                GLOBAL_WORKER_CEILING, SOFT_TPM_CEILING)


def loaded(queued: dict[str, int], **kw) -> Scheduler:
    s = Scheduler(**kw)
    for p, n in queued.items():
        s.states[p].queued = n
    return s


class TestRedistribution(unittest.TestCase):

    def test_01_claude_finishing_redistributes_capacity(self):
        s = loaded({"claude": 300, "fable": 300, "codex": 300, "llama": 300})
        before = s.allocate().allocation
        s.states["claude"].queued = 0                      # claude drains
        after = s.allocate().allocation
        self.assertEqual(after["claude"], 0, "a profile with no queue must hold no capacity")
        freed = before["claude"]
        gained = sum(after[p] - before[p] for p in ("fable", "codex", "llama"))
        self.assertGreater(gained, 0, "freed capacity must be redistributed, not left idle")
        self.assertGreaterEqual(gained, freed * 0.5,
                                f"only {gained} of {freed} freed workers were reused")

    def test_02_fable_finishing_redistributes_capacity(self):
        s = loaded({"claude": 300, "fable": 300, "codex": 300, "llama": 300})
        before = s.allocate().allocation
        s.states["fable"].queued = 0
        after = s.allocate().allocation
        self.assertEqual(after["fable"], 0)
        self.assertGreater(sum(after[p] - before[p] for p in ("claude", "codex", "llama")), 0)

    def test_02b_stalled_profile_donates_capacity(self):
        s = loaded({"claude": 300, "fable": 300, "codex": 300, "llama": 300})
        before = s.allocate().allocation
        s.states["codex"].stalled = True                   # blocked, still queued
        after = s.allocate().allocation
        self.assertEqual(after["codex"], 0, "a stalled profile must not hold capacity")
        self.assertGreater(sum(after[p] for p in ("claude", "fable", "llama")),
                           sum(before[p] for p in ("claude", "fable", "llama")))

    def test_03_llama_never_starves(self):
        """The historical bug: llama pinned to 4 whenever the other three ran."""
        s = loaded({"claude": 999, "fable": 999, "codex": 999, "llama": 300})
        alloc = s.allocate().allocation
        self.assertGreaterEqual(alloc["llama"], PROFILE_LIMITS["llama"]["floor"],
                                "llama fell below its floor with all profiles hot")
        self.assertGreater(alloc["llama"], 4, "llama is back at the historical starvation value")
        # and it still holds under sustained pressure on everyone else
        for _ in range(5):
            s.note_rate_limit("claude")
        self.assertGreaterEqual(s.allocate().allocation["llama"],
                                PROFILE_LIMITS["llama"]["floor"])

    def test_03b_every_queued_profile_gets_its_floor(self):
        s = loaded({p: 500 for p in PROFILE_LIMITS})
        alloc = s.allocate().allocation
        for p, lim in PROFILE_LIMITS.items():
            self.assertGreaterEqual(alloc[p], lim["floor"], f"{p} below floor")

    def test_04_single_remaining_profile_expands_to_cap(self):
        s = loaded({"claude": 0, "fable": 0, "codex": 0, "llama": 400})
        d = s.allocate()
        self.assertEqual(d.allocation["llama"], PROFILE_LIMITS["llama"]["cap"])
        self.assertIn("may expand to its cap", d.reason)
        s2 = loaded({"claude": 400, "fable": 0, "codex": 0, "llama": 0})
        self.assertEqual(s2.allocate().allocation["claude"], PROFILE_LIMITS["claude"]["cap"])

    def test_04b_queue_shorter_than_floor_is_not_over_allocated(self):
        s = loaded({"claude": 2, "fable": 0, "codex": 0, "llama": 0})
        self.assertEqual(s.allocate().allocation["claude"], 2)


class TestTpmCeiling(unittest.TestCase):

    def test_05_observed_tpm_over_ceiling_scales_down(self):
        """Only a LIVE signal may throttle -- see test_05b for why."""
        s = loaded({p: 500 for p in PROFILE_LIMITS}, live_tpm_signal=True)
        baseline = s.allocate()
        # metered tokens far above the soft ceiling within the rolling window
        s.meter.record("claude", int(SOFT_TPM_CEILING * 1.5 * (s.meter.window_sec / 60.0)), 1000.0)
        d = s.allocate(now=1000.0)
        self.assertTrue(d.throttled)
        self.assertLess(d.total_workers, baseline.total_workers,
                        "TPM over the soft ceiling must reduce workers")
        self.assertIn("over soft ceiling", d.reason)

    def test_05b_retrospective_tpm_alone_never_throttles(self):
        """The 2026-09-02 regression, pinned.

        The controller read 12,298,656 TPM -- 2.46x the hard ceiling -- and cut
        llama from 21 workers to 15. Real traffic that minute was 31,521. The
        reading was an artefact of charging whole trials at batch-end, so the
        default scheduler must not act on the retrospective series at all.
        """
        s = loaded({p: 500 for p in PROFILE_LIMITS})
        self.assertFalse(s.live_tpm_signal,
                         "no live per-request meter exists; default must be off")
        baseline = s.allocate()
        s.meter.record("llama", int(12_298_656 * (s.meter.window_sec / 60.0)), 1000.0)
        d = s.allocate(now=1000.0)
        self.assertGreater(s.meter.tpm(1000.0), s.hard_tpm,
                           "fixture must reproduce the phantom magnitude")
        self.assertFalse(d.throttled,
                         "a retrospective series must never throttle live work")
        self.assertEqual(d.total_workers, baseline.total_workers)
        self.assertTrue(d.observed_is_retrospective)

    def test_06_low_tpm_does_not_cause_unbounded_growth(self):
        """The whole point of the ceiling design: 9% utilisation must not ramp."""
        s = loaded({p: 100_000 for p in PROFILE_LIMITS})
        first = s.allocate()
        for i in range(50):                       # many windows of very low TPM
            s.meter.record("llama", 10, 1000.0 + i)
            s.note_healthy_window(now=1000.0 + i * 120)
            d = s.allocate(now=1000.0 + i * 120)
            self.assertLessEqual(d.total_workers, GLOBAL_WORKER_CEILING)
            for p, n in d.allocation.items():
                self.assertLessEqual(n, PROFILE_LIMITS[p]["cap"])
        self.assertEqual(d.total_workers, first.total_workers,
                         "sustained low TPM changed the allocation - growth is not bounded")

    def test_16_never_exceeds_caps_or_global_ceiling(self):
        import itertools
        for queues in itertools.product((0, 1, 7, 50, 100_000), repeat=4):
            s = loaded(dict(zip(("claude", "fable", "codex", "llama"), queues)))
            d = s.allocate()
            self.assertLessEqual(d.total_workers, GLOBAL_WORKER_CEILING, f"queues={queues}")
            for p, n in d.allocation.items():
                self.assertLessEqual(n, PROFILE_LIMITS[p]["cap"], f"{p} over cap, queues={queues}")
                self.assertLessEqual(n, s.states[p].queued, f"{p} over its queue")
                self.assertGreaterEqual(n, 0)
            self.assertLessEqual(d.projected_tpm, SOFT_TPM_CEILING)

    def test_16b_projected_tpm_never_targets_the_ceiling(self):
        """Honest expectation: at full caps we reach ~10% of 5M, not 92%."""
        s = loaded({p: 100_000 for p in PROFILE_LIMITS})
        d = s.allocate()
        self.assertLess(d.projected_tpm, 1_000_000)

    def test_meter_excludes_cache_reads(self):
        fm = {"total_prompt_tokens": 1_233_785, "total_cached_tokens": 1_191_612,
              "total_completion_tokens": 13_637}
        self.assertEqual(TpmMeter.metered_tokens(fm), (1_233_785 - 1_191_612) + 13_637)


class TestBackoff(unittest.TestCase):

    def test_07_real_429_causes_backoff(self):
        s = loaded({p: 500 for p in PROFILE_LIMITS})
        before = s.allocate().allocation["claude"]
        s.note_rate_limit("claude", now=1000.0)
        after = s.allocate(now=1000.0)
        self.assertLess(after.allocation["claude"], before)
        self.assertTrue(after.throttled)
        self.assertGreaterEqual(after.allocation["claude"], PROFILE_LIMITS["claude"]["floor"],
                                "backoff must never drop a queued profile below its floor")

    def test_07b_recovery_is_additive_and_cooldown_gated(self):
        s = loaded({p: 500 for p in PROFILE_LIMITS})
        s.note_rate_limit("claude", now=1000.0)
        hurt = s.states["claude"].headroom
        s.note_healthy_window(now=1001.0)          # inside cooldown -> no change
        self.assertEqual(s.states["claude"].headroom, hurt)
        s.note_healthy_window(now=1000.0 + 10_000)
        self.assertEqual(s.states["claude"].headroom, hurt + 1, "recovery must be additive")


class TestFailureClassification(unittest.TestCase):

    def test_07c_corroborated_429_is_transient(self):
        for text in ("HTTP/1.1 429 Too Many Requests",
                     'server said {"type": "rate_limit_error"}',
                     "status_code: 429", "Rate limit reached for model"):
            c = failures.classify(text)
            self.assertEqual(c.failure_class, failures.TRANSIENT_PROVIDER, text)
            self.assertTrue(c.retryable and c.corroborated)

    def test_08_false_harbor_rate_limit_verdict_is_not_retried(self):
        """The real Aug-27 shape: Harbor's classifier said rate limit, nothing else did."""
        text = ("Traceback (most recent call last):\n"
                "  File \"harbor/agents/installed/base.py\", line 559, in _exec\n"
                "    raise self._classify_exec_error(command, result)\n"
                "harbor.agents.installed.base.ApiRateLimitError: Command failed (exit 1): "
                "codex exec --model openai/gpt-5.6 -- '# Task Description ...'")
        c = failures.classify(text)
        self.assertEqual(c.failure_class, failures.PERMANENT)
        self.assertFalse(c.retryable)
        self.assertFalse(c.corroborated)
        self.assertIn("no HTTP status", c.reason)

    def test_08b_the_actual_aug27_artifact_classifies_as_budget(self):
        """Ground truth from the corpus: that trial was budget exhaustion."""
        p = Path("results/full/swe-eval-pressure-full-codex-chunk-2-size-30-20260827-041001/"
                 "swe-eval-pressure-full-codex-chunk-2-size-30-20260827-041001/"
                 "ea-ed98812f-eval-fin-root__5eYXG9x/exception.txt")
        if not p.is_file():
            self.skipTest("historical artifact not present")
        c = failures.classify(p.read_text(errors="replace"))
        self.assertEqual(c.failure_class, failures.BUDGET)
        self.assertFalse(c.retryable)

    def test_09_pre_model_setup_failure_retries(self):
        for text in ("Error: failed to build image for sandbox",
                     "modal.exception.TimeoutError: container startup failed",
                     "environment setup failed: no space left on device"):
            c = failures.classify(text, model_started=False)
            self.assertEqual(c.failure_class, failures.PRE_MODEL_INFRA, text)
            self.assertTrue(c.retryable)

    def test_10_budget_never_retried_and_outranks_everything(self):
        text = ("HTTP/1.1 429 Too Many Requests\n"
                "Budget has been exceeded! Key=x Current cost: 10001.9, Max budget: 10000.0")
        c = failures.classify(text, model_started=True)
        self.assertEqual(c.failure_class, failures.BUDGET,
                         "budget must outrank a co-occurring 429")
        self.assertFalse(c.retryable)

    def test_11_partial_model_trajectory_is_its_own_class(self):
        c = failures.classify("HTTP/1.1 503 Service Unavailable", model_started=True)
        self.assertEqual(c.failure_class, failures.PARTIAL_MODEL)
        self.assertTrue(c.model_started and c.retryable)
        c2 = failures.classify("agent produced malformed patch", model_started=True)
        self.assertEqual(c2.failure_class, failures.PARTIAL_MODEL)
        self.assertFalse(c2.retryable, "no transient evidence -> not retried")




# --------------------------------------------------------------------------- #
# ledger + validator behaviours (need a throwaway campaign namespace)
# --------------------------------------------------------------------------- #
import shutil                                                    # noqa: E402
import tempfile                                                  # noqa: E402
from campaign import lib                                         # noqa: E402
from campaign.tests.test_tooling import Fixture, _validate       # noqa: E402


class TestRetryLedgerAndProvenance(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="campv2-sched-")
        self.fx = Fixture(Path(self.tmpdir))
        self.ledger = failures.RetryLedger(lib.campaign_paths()["provenance"] / "retries.jsonl")

    def tearDown(self):
        self.fx.restore()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _cls(self, kind=failures.TRANSIENT_PROVIDER):
        return failures.Classification(kind, "corroborated HTTP 429", False, True, True, "429")

    def test_12_retry_count_is_bounded(self):
        cls = self._cls()
        for i in range(failures.MAX_RETRIES):
            ok, why = self.ledger.may_retry("ea-trial-1", cls)
            self.assertTrue(ok, why)
            self.ledger.open_retry("ea-trial-1", f"ea-trial-1--retry{i+1}", cls)
        ok, why = self.ledger.may_retry("ea-trial-1", cls)
        self.assertFalse(ok, "retries must be bounded")
        self.assertIn("retry budget exhausted", why)
        self.assertEqual(self.ledger.retry_count("ea-trial-1"), failures.MAX_RETRIES)

    def test_12b_backoff_is_exponential_and_capped(self):
        b = [self.ledger.backoff_seconds(n) for n in (1, 2, 3, 10)]
        self.assertEqual(b[0], 30.0)
        self.assertEqual(b[1], 60.0)
        self.assertLessEqual(b[3], 600.0)
        self.assertTrue(all(b[i] <= b[i + 1] for i in range(len(b) - 1)))

    def test_budget_never_retried(self):
        cls = failures.Classification(failures.BUDGET, "budget exceeded", False, False, True)
        ok, why = self.ledger.may_retry("ea-trial-x", cls)
        self.assertFalse(ok)
        self.assertIn("never retried", why)

    def test_11b_partial_model_retry_records_full_lineage(self):
        cls = failures.classify("HTTP/1.1 503 Service Unavailable", model_started=True)
        rec = self.ledger.open_retry("ea-orig", "ea-orig--retry1", cls, cell="full/claude/chunk-1-size-30")
        d = rec.as_dict()
        for k in ("original_trial_id", "retry_trial_id", "retry_number", "failure_class",
                  "failure_reason", "model_started", "started_at", "accepted_status"):
            self.assertIn(k, d)
            self.assertNotIn(d[k], (None, ""), f"{k} must be populated")
        self.assertTrue(d["model_started"])
        self.assertEqual(d["failure_class"], failures.PARTIAL_MODEL)
        self.assertEqual(d["accepted_status"], "pending")
        # the original is referenced, never overwritten
        self.assertEqual(d["original_trial_id"], "ea-orig")
        self.assertNotEqual(d["retry_trial_id"], d["original_trial_id"])
        self.ledger.close_retry("ea-orig--retry1", "accepted")
        self.assertEqual(self.ledger.records()[0].accepted_status, "accepted")
        self.assertTrue(self.ledger.records()[0].finished_at)

    def test_13_failed_attempt_is_preserved_not_backfilled(self):
        from campaign import provenance
        importlib.reload(provenance)
        self.fx.write_cell_manifest("full", "claude", 3)
        bad = self.fx.make_run("full", "claude", 3, drop=5, suffix="-a1")
        self.fx.record(provenance, "full", "claude", 3, bad)          # -> failed
        good = self.fx.make_run("full", "claude", 3, suffix="-a2")
        self.fx.record(provenance, "full", "claude", 3, good)         # -> complete

        attempts = [json.loads(l) for l in
                    (lib.campaign_paths()["provenance"] / "attempts.jsonl").read_text().splitlines() if l.strip()]
        self.assertEqual(len(attempts), 2, "the failed attempt must still be in the ledger")
        self.assertEqual(attempts[0]["status"], "failed")
        self.assertNotEqual(attempts[0]["attempt_id"], attempts[1]["attempt_id"])
        self.assertTrue(Path(bad).is_dir(), "the failed run directory must be preserved on disk")
        accepted = json.loads((lib.campaign_paths()["provenance"] / "accepted_runs.json").read_text())
        ids = [a["attempt_id"] for a in accepted["accepted"]]
        self.assertIn(attempts[1]["attempt_id"], ids)
        self.assertNotIn(attempts[0]["attempt_id"], ids)

    def _write_ledger(self, records):
        p = lib.campaign_paths()["provenance"] / "retries.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in records))

    @staticmethod
    def _base_record(**kw):
        rec = {"original_trial_id": "", "retry_trial_id": "", "retry_number": 1,
               "failure_class": failures.TRANSIENT_PROVIDER, "failure_reason": "corroborated 429",
               "model_started": True, "started_at": lib.now_iso(), "finished_at": lib.now_iso(),
               "accepted_status": "accepted", "cell": "full/claude/chunk-1-size-30",
               "evidence": "429"}
        rec.update(kw)
        return rec

    def test_14_validator_accepts_explicit_retry_lineage(self):
        self.fx.build()
        corpus = [json.loads(l) for l in
                  (lib.campaign_paths()["provenance"] / "corpus.jsonl").read_text().splitlines() if l.strip()]
        rows = [r for r in corpus if r["cell"] == "full/claude/chunk-1-size-30"]
        # both the original and the retry are present in the corpus: nothing hidden
        self._write_ledger([self._base_record(original_trial_id=rows[0]["trial_dir"],
                                              retry_trial_id=rows[1]["trial_dir"])])
        rep = _validate()
        self.assertNotIn("RT1", rep["failed"], "well-formed lineage must pass RT1")
        self.assertNotIn("RT2", rep["failed"], "explicit lineage must be accepted")
        self.assertTrue(rep["ok"], f"failed={rep['failed']}")

    def test_15_validator_rejects_silent_replacement(self):
        self.fx.build()
        corpus = [json.loads(l) for l in
                  (lib.campaign_paths()["provenance"] / "corpus.jsonl").read_text().splitlines() if l.strip()]
        rows = [r for r in corpus if r["cell"] == "full/claude/chunk-1-size-30"]
        # the retry is accepted into the corpus but its original is nowhere:
        # neither in the corpus nor preserved on disk. That is the Aug 2026 shape.
        self._write_ledger([self._base_record(original_trial_id="ea-vanished-original",
                                              retry_trial_id=rows[0]["trial_dir"])])
        rep = _validate()
        self.assertIn("RT2", rep["failed"], "a retry whose original vanished must be rejected")
        self.assertFalse(rep["ok"])

    def test_15b_validator_rejects_malformed_and_budget_retry_records(self):
        self.fx.build()
        self._write_ledger([self._base_record(original_trial_id="ea-a", retry_trial_id="ea-b",
                                              failure_class=failures.BUDGET)])
        rep = _validate()
        self.assertIn("RT1", rep["failed"], "a budget failure recorded as a retry must be rejected")

        self._write_ledger([self._base_record(original_trial_id="ea-a", retry_trial_id="ea-b",
                                              retry_number=99)])
        rep = _validate()
        self.assertIn("RT1", rep["failed"], "retry_number beyond the bound must be rejected")

        self._write_ledger([{"original_trial_id": "ea-a"}])
        rep = _validate()
        self.assertIn("RT1", rep["failed"], "missing lineage fields must be rejected")

    def test_14b_no_ledger_and_no_retries_is_clean(self):
        self.fx.build()
        rep = _validate()
        self.assertTrue(rep["ok"], f"failed={rep['failed']}")




class TestInflightCeiling(unittest.TestCase):
    """`harbor run -n` is launch-time only, so in-flight workers are pinned and
    must still be charged against the global ceiling."""

    def test_16c_inflight_workers_count_against_global_ceiling(self):
        s = loaded({p: 500 for p in PROFILE_LIMITS})
        s.states["claude"].active = 24          # a big batch already running
        s.states["fable"].active = 20
        d = s.allocate()
        self.assertEqual(d.allocation["claude"], 24, "in-flight concurrency must be pinned")
        self.assertEqual(d.allocation["fable"], 20)
        self.assertLessEqual(d.total_workers, GLOBAL_WORKER_CEILING,
                             "in-flight + newly launched breached the global ceiling")

    def test_16d_fully_inflight_launches_nothing_new(self):
        s = loaded({p: 500 for p in PROFILE_LIMITS})
        for p in PROFILE_LIMITS:
            s.states[p].active = 10
        d = s.allocate()
        self.assertEqual(d.total_workers, 40)
        self.assertIn("in flight", d.reason)

    def test_16e_ceiling_holds_across_reachable_inflight_mixes(self):
        # Only mixes the scheduler could itself have produced (pinned sum <= ceiling).
        import itertools
        for actives in itertools.product((0, 4, 20), repeat=4):
            if sum(actives) > GLOBAL_WORKER_CEILING:
                continue
            s = loaded({p: 500 for p in PROFILE_LIMITS})
            for p, a in zip(("claude", "fable", "codex", "llama"), actives):
                s.states[p].active = a
            d = s.allocate()
            self.assertLessEqual(d.total_workers, GLOBAL_WORKER_CEILING,
                                 f"breach with in-flight {actives}: {d.allocation}")

    def test_16f_over_ceiling_inflight_adds_no_new_work(self):
        # Defensive: in-flight workers cannot be recalled (launch-time -n), so the
        # scheduler cannot retroactively enforce the ceiling -- but it must never
        # make an over-subscribed state worse.
        s = loaded({p: 500 for p in PROFILE_LIMITS})
        for p in PROFILE_LIMITS:
            s.states[p].active = 20          # 80 pinned, above the ceiling
        d = s.allocate()
        self.assertEqual(d.total_workers, 80, "pinned workers must be reported, not invented away")
        self.assertEqual(sum(d.allocation[p] - s.states[p].active for p in PROFILE_LIMITS), 0,
                         "must not add new workers while over the ceiling")


class TestReapMetering(unittest.TestCase):
    """The controller must charge the meter with METERED tokens only."""

    def test_16g_scanned_trials_carry_cache_reads(self):
        import json as _j, tempfile
        from campaign import lib
        with tempfile.TemporaryDirectory() as td:
            trial = Path(td) / "job" / "ea-fake__clean-n" / "agent"
            trial.mkdir(parents=True)
            (trial / "trajectory.json").write_text(_j.dumps({
                "agent": {"name": "a", "version": "1", "model_name": "m"},
                "final_metrics": {"total_prompt_tokens": 1000,
                                  "total_completion_tokens": 100,
                                  "total_cached_tokens": 900},
            }))
            rows = lib.scan_run_dir(Path(td))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].cached_tokens, 900,
                             "cache reads must survive scan_run_dir")
            metered = TpmMeter.metered_tokens({
                "total_prompt_tokens": rows[0].prompt_tokens,
                "total_completion_tokens": rows[0].completion_tokens,
                "total_cached_tokens": rows[0].cached_tokens,
            })
            self.assertEqual(metered, 200, "(1000-900)+100")

    def test_16h_controller_reap_does_not_count_cache_reads(self):
        """Metering excludes cache reads, and happens per step, not per batch."""
        from campaign import tokens
        metered, prompt, completion, cached = tokens.metered_of({
            "prompt_tokens": 1000, "completion_tokens": 100, "cached_tokens": 900})
        self.assertEqual(metered, 200, "(1000-900)+100")
        src = (Path(__file__).resolve().parents[1] / "controller.py").read_text()
        seg = src[src.index("def reap"):src.index("def _close_retry")]
        self.assertNotIn("b.finished_at", seg,
                         "reap must not charge a whole trial at batch-end")
        self.assertNotIn("u.finished_at)", seg,
                         "reap must not charge a whole trial at unit-end")
        self.assertIn("_record_tokens", seg,
                      "reap must route tokens through the per-step recorder")


if __name__ == "__main__":
    unittest.main(verbosity=2)
