"""Offline fixtures only. Never executes run_trial, Gazebo or Docker."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml

from experiments.sailboat_validation import validate_ab as v
from experiments.sailboat_BO.tests import test_bo_campaign_evaluator as frozen_tests


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name) / 'validation'
        self.output.mkdir()
        self.m = dict(slots=v.schedule(), params={a: dict(path=f'/original/{a}.json', values={}, sha256=v.PARAM_SHA[a])
                                                  for a in ('A', 'B')})

    def records(self, phase='core3', a=0.5, b=0.4):
        return [dict(slot=s, returncode=0, evaluation=dict(runtime=dict(valid=True), bo_feasible=True,
                     failure_count=0, effective_cost=a if s['arm'] == 'A' else b))
                for s in v.schedule() if s['phase'] == phase]

    def test_exact_twenty_slots_and_balanced_order(self):
        slots = v.schedule()
        self.assertEqual(len(slots), 20)
        self.assertEqual(len({s['slot'] for s in slots}), 20)
        self.assertEqual(sum(s['phase'] == 'core3' for s in slots), 12)
        self.assertEqual(sum(s['phase'] == 'heldout' for s in slots), 8)
        for i in range(0, 20, 4):
            self.assertEqual([(s['repeat'], s['arm']) for s in slots[i:i+4]],
                             [(0, 'A'), (0, 'B'), (1, 'B'), (1, 'A')])

    def test_yaml_reuses_accepted_globals_and_only_appends_shifted(self):
        accepted = yaml.safe_load((v.ROOT / 'experiments/sailboat_BO/scenario.yaml').read_text())
        legacy = yaml.safe_load((v.ROOT / 'experiments/sailboat_BO/scenario_generalization.yaml').read_text())
        original = copy.deepcopy(accepted)
        result = v.validation_scenario(accepted, legacy)
        self.assertEqual(accepted, original)
        self.assertEqual(result['scenarios'][:-1], accepted['scenarios'])
        for key in accepted.keys() - {'scenarios'}:
            self.assertEqual(result[key], accepted[key])
        self.assertEqual(result['scenarios'][-1], next(s for s in legacy['scenarios'] if s['id'] == v.HELD[1]))
        self.assertEqual(result['scenarios'][-1]['world']['wind_mag_sin_amp_percent'], 0.08)
        for params in (v.BASELINE, v.CAMPAIGN / 'iter_014/params.json'):
            actual = v.read(params)
            for bound in result['search_space']:
                self.assertLessEqual(bound['low'], actual[bound['name']])
                self.assertLessEqual(actual[bound['name']], bound['high'])

    def test_yaml_missing_or_duplicate_heldout_definition_rejected(self):
        accepted = dict(scenarios=[dict(id=s) for s in (*v.CORE, v.HELD[0])])
        for definitions in ([], [dict(id=v.HELD[1]), dict(id=v.HELD[1])]):
            with self.assertRaises(v.ev.ContractError):
                v.validation_scenario(accepted, dict(scenarios=definitions))

    def test_no_clobber_publication(self):
        path = self.output / 'artifact.json'
        v.save(path, dict(status='original'))
        v.save(path, dict(status='original'))
        with self.assertRaises(v.ev.ContractError):
            v.save(path, dict(status='replacement'))
        self.assertEqual(v.read(path), dict(status='original'))

    def test_dry_run_has_no_writes_or_subprocess_or_evaluator_calls(self):
        with mock.patch.object(v, 'check', return_value=self.m), \
             mock.patch.object(v.subprocess, 'run', side_effect=AssertionError('simulation forbidden')), \
             mock.patch.object(v.ev, 'evaluate_trial', side_effect=AssertionError('evaluation forbidden')):
            before = list(self.output.rglob('*'))
            result = v.run_phase(self.output, 'core3', dry_run=True)
            self.assertEqual(result['pending_trial_count'], 12)
            self.assertFalse(result['simulation_started'])
            self.assertEqual(before, list(self.output.rglob('*')))
            self.assertIn('/original/A.json', result['commands'][0]['command'])

    def test_partial_slot_prevents_any_new_simulation(self):
        (self.output / 'slots' / v.schedule()[2]['slot']).mkdir(parents=True)
        with mock.patch.object(v, 'check', return_value=self.m), mock.patch.object(v.subprocess, 'run') as run:
            with self.assertRaisesRegex(v.ev.ContractError, 'PARTIAL'):
                v.run_phase(self.output, 'core3', dry_run=True)
            run.assert_not_called()

    def test_stop_artifact_blocks_restart(self):
        v.save(self.output / 'STOP.json', dict(status='STOP_PRESERVED'))
        with mock.patch.object(v, 'check', return_value=self.m), mock.patch.object(v.subprocess, 'run') as run:
            with self.assertRaises(v.ev.ContractError):
                v.run_phase(self.output, 'core3')
            run.assert_not_called()

    def test_heldout_requires_core_feasibility_and_improvement(self):
        bad = v.aggregate('core3', self.records(a=.4, b=.5))
        self.assertEqual(bad['status'], 'FAIL')
        with mock.patch.object(v, 'check', return_value=self.m), \
             mock.patch.object(v, 'phase_results', return_value=bad), \
             mock.patch.object(v.subprocess, 'run') as run:
            with self.assertRaisesRegex(v.ev.ContractError, 'Core-3'):
                v.run_phase(self.output, 'heldout', dry_run=True)
            run.assert_not_called()

    def test_core_and_heldout_aggregation_are_separate_and_paired(self):
        core = v.aggregate('core3', self.records(a=.5, b=.4))
        held = v.aggregate('heldout', self.records('heldout', a=.3, b=.3))
        self.assertEqual(core['status'], 'PASS')
        self.assertEqual(core['feasible_counts']['B'], 6)
        self.assertEqual(held['status'], 'PASS')
        self.assertEqual(held['feasible_counts']['B'], 4)
        self.assertEqual(len(core['paired_results']), 6)
        self.assertEqual(len(held['paired_results']), 4)
        self.assertAlmostEqual(core['B_minus_A'], -.1)
        self.assertEqual(v.aggregate('core3', self.records(a=.3, b=.3))['status'], 'FAIL')

    def test_unbalanced_missing_duplicate_reordered_slots_rejected(self):
        records = self.records()
        for bad in (records[:-1], records + [records[0]], list(reversed(records)), [records[0]] * 12):
            with self.assertRaises(v.ev.ContractError):
                v.aggregate('core3', bad)

    def test_frozen_penalty_and_open_not_reclassified(self):
        records = self.records()
        b = next(r for r in records if r['slot']['arm'] == 'B')
        b['evaluation'].update(bo_feasible=False, failure_count=1, effective_cost=5.,
                               layer2=dict(classification='OPEN'))
        result = v.aggregate('core3', records)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['feasible_counts']['B'], 5)
        self.assertEqual(b['evaluation']['layer2']['classification'], 'OPEN')
        b['evaluation'].update(runtime=dict(valid=False), effective_cost=10.)
        self.assertEqual(v.aggregate('core3', records)['status'], 'OPEN')

    def test_a_framework_failure_keeps_cost_5(self):
        records = self.records()
        records[0]['evaluation'].update(bo_feasible=False, failure_count=1, effective_cost=5.)
        result = v.aggregate('core3', records)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['feasible_counts']['A'], 5)
        self.assertEqual(result['trials'][0]['evaluation']['effective_cost'], 5.)

    def test_receipt_hash_identity_and_raw_integrity(self):
        slot = v.schedule()[0]
        directory = self.output / 'slots' / slot['slot']
        directory.mkdir(parents=True)
        v.save(self.output / 'validation_manifest.json', {})
        raw = directory / 'raw.csv'
        raw.write_text('original')
        receipt = dict(slot=slot, manifest_sha256=v.ev.sha256_file(self.output / 'validation_manifest.json'),
                       files={'raw.csv': v.ev.sha256_file(raw)})
        v.save(directory / 'completion.json', receipt)
        self.assertEqual(v.completion(self.output, slot), receipt)
        raw.write_text('tampered')
        with self.assertRaisesRegex(v.ev.ContractError, 'changed'):
            v.completion(self.output, slot)

    def test_docker_mounts_campaign_readonly_and_no_repository_overlay(self):
        output = v.RESULTS / 'bo_validation/example'
        cmd = v.docker_command(output, self.m)
        mounts = [cmd[i+1] for i, s in enumerate(cmd) if s == '--mount']
        self.assertIn(f'type=bind,src={v.RESULTS},dst={v.RESULTS},readonly', mounts)
        self.assertIn(f'type=bind,src={output}/inputs,dst={output}/inputs,readonly', mounts)
        self.assertFalse(any(f'src={v.ROOT},' in m for m in mounts))
        self.assertFalse(any('Shadow' in m or 'ablation' in m for m in mounts))
        self.assertEqual(cmd[-2:], [v.IMAGE, 'bash'])
        self.assertTrue(any('generated_worlds' in m and not m.endswith(',readonly') for m in mounts))

    def test_docker_timeout_fails_without_trials(self):
        with mock.patch.object(v.subprocess, 'run', side_effect=v.subprocess.TimeoutExpired('docker', 15)):
            with self.assertRaisesRegex(v.ev.ContractError, 'Docker unavailable'):
                v.docker_preflight()

    def test_runtime_wrong_image_fails(self):
        with mock.patch.dict(v.os.environ, {'ACCEPTED_IMAGE_ID': 'wrong'}):
            with self.assertRaises(v.ev.ContractError):
                v.runtime_preflight()

    def test_process_error_stops_with_no_automatic_retry(self):
        with mock.patch.object(v, 'check', return_value=self.m), \
             mock.patch.object(v, 'runtime_preflight'), \
             mock.patch.object(v.subprocess, 'run', side_effect=OSError('fixture process error')) as run:
            with self.assertRaises(OSError):
                v.run_phase(self.output, 'core3')
            self.assertEqual(run.call_count, 1)
            self.assertTrue((self.output / 'STOP.json').exists())
            with self.assertRaises(v.ev.ContractError):
                v.run_phase(self.output, 'core3')
            self.assertEqual(run.call_count, 1)

    def test_complete_twenty_fixture_slots_final_compare_and_idempotence(self):
        v.save(self.output / 'validation_manifest.json', {})
        manifest_sha = v.ev.sha256_file(self.output / 'validation_manifest.json')
        for record in self.records() + self.records('heldout', a=.3, b=.3):
            record.update(manifest_sha256=manifest_sha, files={})
            v.save(self.output / 'slots' / record['slot']['slot'] / 'completion.json', record)
        with mock.patch.object(v, 'check', return_value=self.m), mock.patch.object(v, 'runtime_preflight'), \
             mock.patch.object(v.subprocess, 'run', side_effect=AssertionError('no reruns')):
            self.assertEqual(v.run_phase(self.output, 'core3')['status'], 'PASS')
            self.assertEqual(v.run_phase(self.output, 'heldout')['status'], 'PASS')
            result = v.compare(self.output)
            self.assertEqual(result['validation'], 'PASS')
            self.assertEqual(result['completed_trial_count'], 20)
            self.assertFalse(result['production_promotion'])
            self.assertFalse(result['statistical_significance_claim'])
            before = (self.output / 'final_comparison.json').read_bytes()
            self.assertEqual(v.compare(self.output), result)
            self.assertEqual((self.output / 'final_comparison.json').read_bytes(), before)

    def test_descriptive_metrics_are_not_mixed_across_scenarios_or_phases(self):
        records = self.records()
        for record in records:
            record['navigation'] = dict(metadata=dict(status='success'),
                constraints=dict(constraint_mission_complete=True),
                metrics=dict(xte_rms_m=1. if record['slot']['arm'] == 'A' else 2.))
        result = v.aggregate('core3', records)
        self.assertEqual(result['descriptive_metrics']['A']['xte_rms_m']['mean'], 1.)
        self.assertEqual(result['descriptive_metrics']['B']['mission_success_count'], 6)
        self.assertEqual(set(result['per_scenario']), set(v.CORE))
        for scenario in v.CORE:
            self.assertEqual(result['per_scenario'][scenario]['B']['metrics']['xte_rms_m']['count'], 2)

    def test_runtime_invalid_keeps_evaluation_and_stops(self):
        def receipt(*args):
            slot = args[2]
            return dict(slot=slot, returncode=0, evaluation=dict(runtime=dict(valid=False),
                        layer1={}, layer2={}, layer3={}, bo_feasible=False, effective_cost=10.))
        with mock.patch.object(v, 'check', return_value=self.m), \
             mock.patch.object(v, 'runtime_preflight'), \
             mock.patch.object(v.subprocess, 'run', return_value=mock.Mock(returncode=0)) as run, \
             mock.patch.object(v, 'evaluate_new_trial', side_effect=receipt):
            with self.assertRaisesRegex(v.ev.ContractError, 'runtime-invalid'):
                v.run_phase(self.output, 'core3')
            self.assertEqual(run.call_count, 1)
            self.assertEqual(v.read(self.output / 'STOP.json')['status'], 'STOP_PRESERVED')

    def test_actual_patch1_evaluator_fixture_and_no_re_evaluation(self):
        fixture = frozen_tests.BoCampaignEvaluatorTests()
        fixture.setUp()
        slot = v.schedule()[0]
        directory = self.output / 'slots' / slot['slot']
        trial = fixture._write_trial(directory / 'trials')
        scenario = dict(scenarios=[dict(id=slot['scenario'])], termination=dict(timeout_s=240., min_speed_mps=.25),
                        study=dict(seed=42), logging={})
        manifest = v.read(trial / 'trial_manifest.json')
        manifest.update(scenario=scenario['scenarios'][0], study=scenario['study'], logging={})
        (trial / 'trial_manifest.json').write_text(json.dumps(manifest))
        (self.output / 'inputs').mkdir()
        (self.output / 'inputs/scenario_validation.yaml').write_text(yaml.safe_dump(scenario))
        (self.output / 'inputs' / v.CONTRACT).write_bytes((v.CAMPAIGN / v.CONTRACT).read_bytes())
        v.save(self.output / 'validation_manifest.json', {})
        self.m['params']['A']['values'] = fixture.params
        receipt = v.evaluate_new_trial(self.output, self.m, slot, directory, 0)
        self.assertTrue(receipt['evaluation']['runtime']['valid'])
        self.assertTrue(receipt['evaluation']['bo_feasible'])
        self.assertEqual(receipt['evaluation']['objective']['evaluation_no_go_angle_deg'], 60.)
        self.assertEqual(v.completion(self.output, slot)['evaluation']['effective_cost'],
                         receipt['evaluation']['effective_cost'])
        with self.assertRaisesRegex(v.ev.ContractError, 'existing evaluation'):
            v.evaluate_new_trial(self.output, self.m, slot, directory, 0)

    def test_original_param_hashes_and_frozen_sources(self):
        for arm, path in v.original_params(v.CAMPAIGN, v.BASELINE).items():
            self.assertEqual(v.ev.sha256_file(path), v.PARAM_SHA[arm])
        for name, digest in v.FROZEN.items():
            self.assertEqual(v.ev.sha256_file(v.ROOT / name), digest, name)

    def test_prepare_fresh_output_and_deterministic_dry_run(self):
        output = self.output / 'fresh'
        real_git = v.subprocess.check_output
        with mock.patch.object(v, 'inspect_originals'), mock.patch.object(v.subprocess, 'check_output', wraps=real_git):
            v.prepare(output)
        before = {str(p): v.ev.sha256_file(p) for p in output.rglob('*') if p.is_file()}
        first = v.run_phase(output, 'core3', dry_run=True)
        self.assertEqual(first, v.run_phase(output, 'core3', dry_run=True))
        self.assertEqual(before, {str(p): v.ev.sha256_file(p) for p in output.rglob('*') if p.is_file()})
        self.assertEqual(v.check(output)['maximum_trials'], 20)
        with self.assertRaisesRegex(v.ev.ContractError, 'fresh'):
            v.prepare(output)
        (output / 'inputs/A_params.json').write_text('{}')
        with self.assertRaisesRegex(v.ev.ContractError, 'snapshot mismatch'):
            v.check(output)


if __name__ == '__main__':
    unittest.main()
