"""Recovery tests use temporary fixtures and mocked trial processes only."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest import mock

from experiments.sailboat_BO import recover_iteration1 as recovery
from experiments.sailboat_BO import bo_campaign_evaluator as ev
from experiments.sailboat_BO.tests import test_bo_campaign_state as sf
from experiments.sailboat_BO.tests import test_bo_campaign_evaluator as tf


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        fixture = sf.BoCampaignStateTests(); fixture.setUp()
        self.c, self.provenance = fixture._freeze(self.root)
        self.baseline = self.root/"iter_008.json"
        self.r = self.root/"recovery"
        self.params = dict(fixture.params); self.params["ATC_SAIL_P"] += .01
        self.fixture = tf.BoCampaignEvaluatorTests(); self.fixture.setUp(); self.fixture.params = self.params
        it = self.c/"iter_001"; it.mkdir()
        (it/"params.json").write_bytes(recovery.encoded(self.params))
        (it/"proposal.json").write_bytes(recovery.encoded(dict(
            iteration=1, source="local_warmup", seed=42, params=self.params)))
        self.crosswind = []
        for repeat in (0, 1):
            t = self.fixture._write_trial(it/"trials", repeat_idx=repeat)
            self.crosswind.append(t)
        p = self.crosswind[1]/"logs/launch.stdout.log"
        lines = p.read_text().splitlines(keepends=True)
        lines[-1] = lines[-1].replace("boomAngleRad=0.500000", "boomAngleRad=0.970000").replace(
            "effectivePenetrationRad=0.000000", "effectivePenetrationRad=0.270000").replace(
            "requestedTorqueNm=0.000000", "requestedTorqueNm=-1.000000").replace(
            "appliedTorqueNm=0.000000", "appliedTorqueNm=-1.000000").replace("sheetTaut=false", "sheetTaut=true")
        p.write_text("".join(lines))
        incomplete = it/"trials/incomplete__train_upwind_tack__r00"; incomplete.mkdir()
        (incomplete/"trial_manifest.json").write_bytes(recovery.encoded(dict(
            scenario_id="train_upwind_tack", repeat_idx=0, params=self.params)))
        for target in (recovery.state, ev):
            patch = mock.patch.object(target, "validate_provenance", return_value=self.provenance)
            patch.start(); self.addCleanup(patch.stop)
        patch = mock.patch.object(recovery, "MANIFEST", ev.sha256_file(self.c/"campaign_state.json"))
        patch.start(); self.addCleanup(patch.stop)
        self.initial = recovery.hashes(self.c)

    def prepare(self):
        return recovery.prepare(self.c, self.r, self.baseline)

    def fake_trial(self, command, **kwargs):
        scenario = command[command.index("--scenario-id")+1]
        repeat = int(command[command.index("--repeat-idx")+1])
        out = Path(command[command.index("--results-dir")+1])
        self.fixture._write_trial(out, scenario_id=scenario, repeat_idx=repeat)
        return types.SimpleNamespace(returncode=0)

    def collect(self):
        with mock.patch.object(recovery, "runtime_preflight"), mock.patch.object(
                recovery.subprocess, "run", side_effect=self.fake_trial) as run:
            for slot in recovery.PENDING:
                recovery.run_slot(self.c, self.r, self.baseline, slot)
            self.assertEqual(run.call_count, 4)

    def evaluated(self):
        self.prepare(); self.collect()
        return recovery.evaluate(self.c, self.r, self.baseline)

    def test_dry_run_is_read_only_and_never_proposes(self):
        with mock.patch("experiments.sailboat_BO.run_bo.propose_iteration_params",
                        side_effect=AssertionError("must not propose")), mock.patch.object(
                        recovery.subprocess, "run", side_effect=AssertionError("must not execute")):
            d = recovery.prepare(self.c, self.r, self.baseline, dry_run=True)
        self.assertTrue(d["dry_run"]); self.assertFalse(self.r.exists())
        self.assertEqual(d["reviews"]["crosswind-r01"]["layer2"], "OPEN")
        self.assertEqual(self.initial, recovery.hashes(self.c))

    def test_fresh_plan_and_exact_proposal_preserved(self):
        self.prepare(); plan = recovery.check(self.c, self.r, self.baseline)
        self.assertEqual(plan["proposal"]["params"], self.params)
        self.assertEqual(set(plan["pending"]), set(recovery.PENDING))
        self.assertEqual(self.initial, recovery.hashes(self.c))
        with self.assertRaisesRegex(ev.ContractError, "fresh recovery"):
            self.prepare()

    def test_original_tamper_blocks_before_any_process(self):
        self.prepare(); (self.c/"iter_001/params.json").write_text("{}")
        with mock.patch.object(recovery.subprocess, "run") as run, self.assertRaises(ev.ContractError):
            recovery.run_slot(self.c, self.r, self.baseline, "upwind-r00")
        run.assert_not_called()

    def test_corrupt_plan_or_baseline_copy_rejected(self):
        self.prepare(); p = self.r/"baseline/history.csv"; p.write_text("corrupt")
        with self.assertRaisesRegex(ev.ContractError, "baseline copy"):
            recovery.check(self.c, self.r, self.baseline)
        p.write_bytes((self.c/"history.csv").read_bytes())
        (self.r/"plan.json").write_text("{}")
        with self.assertRaisesRegex(ev.ContractError, "plan SHA"):
            recovery.check(self.c, self.r, self.baseline)

    def test_crosswind_cannot_execute(self):
        self.prepare()
        with self.assertRaisesRegex(ev.ContractError, "only four"):
            recovery.run_slot(self.c, self.r, self.baseline, "crosswind-r00")

    def test_slot_dry_run_and_repeat_are_idempotent(self):
        self.prepare()
        with mock.patch.object(recovery.subprocess, "run") as run:
            d = recovery.run_slot(self.c, self.r, self.baseline, "upwind-r00", dry_run=True)
            run.assert_not_called(); self.assertFalse((self.r/"slots").exists())
            self.assertIn("--repeat-idx", d["command"])
        with mock.patch.object(recovery, "runtime_preflight"), mock.patch.object(
                recovery.subprocess, "run", side_effect=self.fake_trial) as run:
            recovery.run_slot(self.c, self.r, self.baseline, "upwind-r00")
            d = recovery.run_slot(self.c, self.r, self.baseline, "upwind-r00")
            self.assertEqual(run.call_count, 1); self.assertEqual(d["status"], "ALREADY_COMPLETED")
        self.assertEqual(self.initial, recovery.hashes(self.c))

    def test_incomplete_attempt_requires_explicit_new_attempt(self):
        self.prepare(); p = self.r/"slots/upwind-r00/attempt_0001"; p.mkdir(parents=True)
        (p/"console.log").write_text("preserved incomplete")
        with self.assertRaisesRegex(ev.ContractError, "retry-incomplete"):
            recovery.run_slot(self.c, self.r, self.baseline, "upwind-r00", dry_run=True)
        with mock.patch.object(recovery, "runtime_preflight"), mock.patch.object(
                recovery.subprocess, "run", side_effect=self.fake_trial):
            recovery.run_slot(self.c, self.r, self.baseline, "upwind-r00", retry=True)
        self.assertEqual((p/"console.log").read_text(), "preserved incomplete")
        self.assertTrue((p.parent/"attempt_0002/completion.json").is_file())

    def test_missing_slots_cannot_evaluate(self):
        self.prepare()
        with self.assertRaisesRegex(ev.ContractError, "pending slot"):
            recovery.evaluate(self.c, self.r, self.baseline)
        self.assertFalse((self.r/"evaluations").exists())

    def test_frozen_evaluator_view_preserves_originals_and_open_penalty(self):
        result = self.evaluated()
        self.assertEqual(result["failure_count"], 1)
        self.assertFalse(result["candidate_feasible"])
        self.assertEqual(self.initial, recovery.hashes(self.c))
        c = recovery.read(Path(result["candidate"]))
        self.assertEqual(c["trial_count"], 6)
        cw = next(t for t in c["trials"] if t["scenario_id"] == "train_crosswind_straight" and t["repeat_idx"] == 1)
        self.assertEqual(cw["effective_cost"], 5.)
        self.assertEqual(cw["layer2"]["classification"], "OPEN")
        for t in self.crosswind:
            self.assertFalse((t/"physics_diagnostics").exists())
        with mock.patch.object(ev, "evaluate_candidate", side_effect=AssertionError("must not re-evaluate")):
            self.assertEqual(recovery.evaluate(self.c, self.r, self.baseline)["status"], "ALREADY_EVALUATED")

    def test_commit_exactly_once_and_incumbent_unchanged(self):
        self.evaluated(); original_best = (self.c/"best_params.json").read_bytes()
        before = recovery.hashes(self.c)
        self.assertTrue(recovery.commit(self.c, self.r, self.baseline, dry_run=True)["dry_run"])
        self.assertEqual(before, recovery.hashes(self.c))
        for _ in range(2):
            result = recovery.commit(self.c, self.r, self.baseline)
            self.assertEqual(result["observation_count"], 2)
        self.assertEqual(len((self.c/"observations.jsonl").read_text().splitlines()), 2)
        self.assertEqual((self.c/"best_params.json").read_bytes(), original_best)
        for n, h in self.initial.items():
            if n not in ("history.csv", "observations.jsonl"):
                self.assertEqual(ev.sha256_file(self.c/n), h, n)

    def test_interruption_between_history_and_observation_recovers_once(self):
        self.evaluated(); append = recovery.append_checked
        def interrupted(path, base, suffix):
            if path.name == "observations.jsonl":
                raise OSError("injected interruption")
            append(path, base, suffix)
        with mock.patch.object(recovery, "append_checked", side_effect=interrupted), self.assertRaises(OSError):
            recovery.commit(self.c, self.r, self.baseline)
        self.assertEqual(len((self.c/"observations.jsonl").read_text().splitlines()), 1)
        self.assertEqual(recovery.commit(self.c, self.r, self.baseline)["next_iteration"], 2)
        self.assertEqual(len((self.c/"observations.jsonl").read_text().splitlines()), 2)

    def test_torn_own_append_recovered_without_truncation(self):
        self.evaluated(); append = recovery.append_checked
        def torn(path, base, suffix):
            if path.name == "history.csv":
                with path.open("ab") as f: f.write(suffix[:7])
                raise OSError("injected torn append")
            append(path, base, suffix)
        with mock.patch.object(recovery, "append_checked", side_effect=torn), self.assertRaises(OSError):
            recovery.commit(self.c, self.r, self.baseline)
        prefix = (self.c/"history.csv").read_bytes()
        recovery.commit(self.c, self.r, self.baseline)
        self.assertTrue((self.c/"history.csv").read_bytes().startswith(prefix))
        self.assertEqual(recovery.validate_state(self.c, self.baseline)["observation_count"], 2)

    def test_foreign_append_rejected_not_deleted(self):
        self.evaluated()
        with (self.c/"observations.jsonl").open("ab") as f: f.write(b'foreign\n')
        before = (self.c/"observations.jsonl").read_bytes()
        with self.assertRaises(ev.ContractError):
            recovery.commit(self.c, self.r, self.baseline)
        self.assertEqual((self.c/"observations.jsonl").read_bytes(), before)

    def test_changed_collected_trial_rejected(self):
        self.prepare(); self.collect()
        plan = recovery.read(self.r/"plan.json")
        t = recovery.completed_slot(self.r, "upwind-r00", plan)
        (t/"raw/samples.csv").write_text("corrupt")
        with self.assertRaisesRegex(ev.ContractError, "completed trial changed"):
            recovery.evaluate(self.c, self.r, self.baseline)

    def test_directory_lock_excludes_other_recovery(self):
        with recovery.locked(self.c), self.assertRaisesRegex(ev.ContractError, "another recovery"):
            with recovery.locked(self.c):
                pass

    def test_docker_timeout_and_wrong_runtime_fail(self):
        with mock.patch.object(recovery.subprocess, "run", side_effect=subprocess.TimeoutExpired("docker", 12)):
            with self.assertRaisesRegex(ev.ContractError, "Docker daemon"):
                recovery.docker_preflight()
        with mock.patch.dict(recovery.os.environ, {"ACCEPTED_IMAGE_ID": "shadow-image"}):
            with self.assertRaisesRegex(ev.ContractError, "pinned"):
                recovery.runtime_preflight()

    def test_shell_mounts_only_tooling_readonly_and_never_starts_trial(self):
        self.prepare()
        with mock.patch.object(recovery, "docker_preflight"), mock.patch.object(
                recovery.subprocess, "call", return_value=0) as call:
            recovery.shell(self.c, self.r, self.baseline)
        cmd = call.call_args.args[0]
        self.assertEqual(cmd[-2:], [recovery.IMAGE, "bash"])
        self.assertIn(str(self.c)+":/frozen_campaign:ro", cmd)
        self.assertNotIn("run_trial.py", cmd[-1])
        self.assertTrue(any('generated_worlds' in x for x in cmd))
        self.assertFalse(any('mesh-dry' in x or 'shadow_kernel' in x for x in cmd))

    def test_shell_entry_does_not_hold_host_lock(self):
        self.prepare()
        def container_shell(_):
            with recovery.locked(self.c):
                return 0
        with mock.patch.object(recovery, "docker_preflight"), mock.patch.object(
                recovery.subprocess, "call", side_effect=container_shell):
            self.assertEqual(recovery.main(["shell", "--campaign", str(self.c),
                "--recovery", str(self.r), "--baseline", str(self.baseline)]), 0)

    def test_container_cannot_evaluate_or_commit_host_paths(self):
        self.prepare(); self.collect()
        plan = recovery.read(self.r/"plan.json")
        plan["cwd"] = "/different/container/path"
        with mock.patch.object(recovery, "check", return_value=plan):
            with self.assertRaisesRegex(ev.ContractError, "WSL host"):
                recovery.evaluate(self.c, self.r, self.baseline)
            with self.assertRaisesRegex(ev.ContractError, "WSL host"):
                recovery.commit(self.c, self.r, self.baseline)

    def test_docker_failure_precedes_any_prepare_write(self):
        with mock.patch.object(recovery, "docker_preflight", side_effect=ev.ContractError("engine offline")), \
                contextlib.redirect_stderr(io.StringIO()):
            rc = recovery.main(["prepare", "--campaign", str(self.c), "--recovery", str(self.r),
                                "--baseline", str(self.baseline), "--docker"])
        self.assertEqual(rc, 1); self.assertFalse(self.r.exists())
        self.assertEqual(self.initial, recovery.hashes(self.c))


if __name__ == "__main__":
    unittest.main()
