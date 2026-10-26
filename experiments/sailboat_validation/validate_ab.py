"""Frozen iter_008 vs iter_014 validation. Only `run` starts Gazebo trials.

No resume/retry of partial trials, no optimizer writes, no production overlay.
The patch1 per-trial evaluator is reused unchanged; its BO-only six-trial
aggregator is deliberately NOT used for the four-trial held-out comparison.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile

import yaml

from experiments.sailboat_BO import bo_campaign_evaluator as ev
from experiments.sailboat_BO import bo_campaign_state as state

ROOT = Path(__file__).resolve().parents[2]
CROOT = Path('/home/docker/sailboat_ws/src/dave')
RESULTS = Path('/home/jeong/sailboat_rl_results')
CAMPAIGN = RESULTS / 'bo/bo_campaign_contract_v1_patch1_20261020T042842Z'
BASELINE = RESULTS / 'bo/final_optimized_baseline/iter_008_original_params.json'
IMAGE = 'sha256:c7dd2ca9507f72a6937b694f7318560a1f6c923035e561a9884d2b68458c16bc'
SOURCE = '4ec49538912fe88ca3be569c1a787dba4c07994b'
STATE_SHA = '82a7e66fdc1c9506c4ecf6e25a17c9c0b1a9c65e09952e50c22c66a99824eb96'
CONTRACT = 'bo_campaign_contract_v1_patch1.yaml'
EVAL_ARTIFACT = 'physics_diagnostics/bo_campaign_evaluation_v1_patch1.json'
PARAM_SHA = dict(A='ec830d81b7be876685e1bd93f5e096bbd9971926cca3e2c01880939866393b12',
                 B='1e52aed28be6c660e55fe831cb718411dd38edee3b66c670db0125d18a4fe663')
CORE = ('train_crosswind_straight', 'train_upwind_tack', 'train_dogleg')
HELD = ('eval_long_oblique', 'eval_shifted_wind_loop')
SETTINGS = dict(initial_random=12, local_warmup=6, local_radius=.15, candidate_pool=1024,
                repeats=2, seed=42, objective_column='scenario_cost', worst_case_weight=0.,
                no_baseline=False, trial_dry_run=False, stop_on_failure=False)
FROZEN = {
    'experiments/sailboat_BO/scenario.yaml': 'ad2d8d0028571b16471722fc8edbae7f9b9111b10606e59e5b23a89707189011',
    'experiments/sailboat_BO/scenario_generalization.yaml': '2d1a52357ad606cad73da1418ac9ef3f7e44371d797a59c4819a66dc0f4dfa4c',
    'experiments/sailboat_BO/bo_campaign_evaluator.py': 'cdaca4770944a6af4f5551378815def97b912f18527a15612ec4c7290e4ed7ff',
    'experiments/sailboat_BO/score.py': 'de38730311821b15867f7403307b43e150fb88e3de98e1ff1ab147661c229a92',
    'experiments/sailboat_physics/virtual_sheet_validation.py': '6138a5822a23a3e5b40142be339bea4adddf252c0e89bae5cad472ad51523035',
    'experiments/sailboat_BO/run_trial.py': 'c39fe13500131e0ba28b1d1933445f3530f086624a38df42e0d0b910d11fdb72',
}
CONTRACT_SHA = '5454dfc50e4b5f7528bc7c27e531d1b3488b632f54c92b2bc1c4f99449c3f05b'
SDF_SHA = '5fb8d0288e39eabac47f14c92c7c62785ca4fc8021274e4744280bdf6e398ecc'
PLUGIN_SHA = '66877a1da116237123ef2931846e08702b6a6a8803f9bc7dbf2427166982b849'
PRODUCTION = (
    'gazebo/dave_gz_model_plugins/CMakeLists.txt',
    'gazebo/dave_gz_model_plugins/include/dave_gz_model_plugins/VirtualSheetActuator.hh',
    'gazebo/dave_gz_model_plugins/src/VirtualSheetActuatorSystem.cc',
    'gazebo/dave_gz_model_plugins/test/VirtualSheetActuator_TEST.cc',
    'models/dave_robot_models/description/sailboat/model.sdf',
    'experiments/sailboat_physics/tests/test_sheet_boom_contract.py',
)


def require(condition, message):
    if not condition:
        raise ev.ContractError(message)


def read(path):
    return json.loads(Path(path).read_text())


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def publish(path, data):
    """Atomic no-clobber artifact; an identical artifact can be read again."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, 'refusing overwrite: ' + str(path))
        return
    fd, temporary = tempfile.mkstemp(prefix='.validation-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            require(path.read_bytes() == data, 'concurrent artifact conflict: ' + str(path))
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save(path, value):
    publish(path, encoded(value))


@contextmanager
def locked(output):
    fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ev.ContractError('another validation command is active')
        yield
    finally:
        os.close(fd)


def file_hashes(paths, root):
    return {str(p.relative_to(root)): ev.sha256_file(p) for p in sorted(paths)}


def tooling_hashes():
    paths = list((ROOT / 'experiments/sailboat_BO').glob('*.py'))
    paths += [ROOT / p for p in FROZEN]
    paths += list((ROOT / 'experiments/sailboat_validation').rglob('*.py'))
    paths += list((ROOT / 'experiments/sailboat_validation').glob('*.md'))
    return file_hashes(set(paths), ROOT)


def campaign_hashes(campaign):
    # Preserve all config/history/observations and per-iteration decision/eval
    # artifacts. Large original raw logs stay under an entirely read-only mount.
    paths = [p for p in campaign.iterdir() if p.is_file()]
    for iteration in sorted(campaign.glob('iter_[0-9][0-9][0-9]')):
        paths += list(iteration.glob('*.json'))
        paths += list(iteration.rglob('*evaluation*.json'))
    return file_hashes(set(paths), campaign)


def schedule():
    slots = []
    for phase, scenarios in (('core3', CORE), ('heldout', HELD)):
        for scenario in scenarios:
            for repeat, order in ((0, ('A', 'B')), (1, ('B', 'A'))):
                for arm in order:
                    slots.append(dict(slot=f'{len(slots)+1:02d}_{scenario}_r{repeat:02d}_{arm}',
                                      phase=phase, scenario=scenario, repeat=repeat, arm=arm))
    return slots


def validation_scenario(accepted, generalization):
    """No legacy narrow bounds: only append the pre-existing shifted task."""
    result = copy.deepcopy(accepted)
    matches = [s for s in generalization['scenarios'] if s['id'] == HELD[1]]
    require(len(matches) == 1, 'missing/duplicate existing shifted-wind definition')
    require([s['id'] for s in result['scenarios']] == list(CORE) + [HELD[0]],
            'unexpected accepted scenario set')
    result['scenarios'].append(copy.deepcopy(matches[0]))
    return result


def original_params(campaign, baseline):
    return dict(A=baseline, B=campaign / 'iter_014/params.json')


def inspect_originals(campaign, baseline):
    require(ev.sha256_file(campaign / CONTRACT) == CONTRACT_SHA, 'patch1 contract SHA mismatch')
    for name, digest in FROZEN.items():
        require(ev.sha256_file(ROOT / name) == digest, 'frozen source mismatch: ' + name)
    for arm, path in original_params(campaign, baseline).items():
        require(ev.sha256_file(path) == PARAM_SHA[arm], arm + ' original JSON SHA mismatch')
    validated = state.validate_frozen_campaign_state(
        contract_path=campaign / CONTRACT, scenario_path=ROOT / 'experiments/sailboat_BO/scenario.yaml',
        baseline_params_path=baseline, runtime_image_id=IMAGE, source_commit=SOURCE,
        bo_dir=campaign, expected_optimizer_settings=SETTINGS)
    require(validated['manifest_sha256'] == STATE_SHA, 'campaign identity mismatch')
    require(validated['observation_count'] == validated['next_iteration'] == 40,
            'campaign must be complete at exactly 40 observations')
    best = read(campaign / 'best_params.json')
    require(best['iteration'] == 14 and best['failure_count'] == 0 and
            best['objective'] == 0.32607422077940834 and
            best['params'] == read(campaign / 'iter_014/params.json'), 'incumbent mismatch')
    require(read(campaign / 'iter_000/params.json') == read(baseline),
            'reference is not this campaign iteration 0')
    return validated


def prepare(output, campaign=CAMPAIGN, baseline=BASELINE):
    require(not output.exists(), 'fresh validation output required; never overwrite')
    require(not output.is_relative_to(campaign) and not output.is_relative_to(ROOT) and
            not output.is_relative_to(baseline.parent), 'validation sidecar must not be inside source/original BO data')
    # All checks before creating the new sidecar. No trial APIs called here.
    inspect_originals(campaign, baseline)
    require(subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
            == SOURCE, 'host accepted source HEAD mismatch')
    production_hashes = file_hashes([ROOT / p for p in PRODUCTION], ROOT)
    for name, digest in production_hashes.items():
        committed = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{SOURCE}:{name}'])
        require(hashlib.sha256(committed).hexdigest() == digest, 'host production source drift: ' + name)
    accepted = yaml.safe_load((ROOT / 'experiments/sailboat_BO/scenario.yaml').read_text())
    general = yaml.safe_load((ROOT / 'experiments/sailboat_BO/scenario_generalization.yaml').read_text())
    scenario_bytes = yaml.safe_dump(validation_scenario(accepted, general), sort_keys=False).encode()
    manifest = dict(format='bo_independent_validation_v1', status='FROZEN_BEFORE_TRIALS',
        created_at_utc=datetime.now(timezone.utc).isoformat(), source=SOURCE, image=IMAGE,
        campaign=str(campaign), baseline=str(baseline), maximum_trials=20, seed=42, repeats=2,
        params={arm: dict(path=str(path), sha256=PARAM_SHA[arm], values=read(path))
                for arm, path in original_params(campaign, baseline).items()},
        contract_sha256=CONTRACT_SHA, frozen_sha256=FROZEN, tooling_sha256=tooling_hashes(),
        production_source_sha256=production_hashes,
        campaign_sha256=campaign_hashes(campaign), slots=schedule(),
        scenario_sha256=hashlib.sha256(scenario_bytes).hexdigest(),
        scenario_delta=['append existing eval_shifted_wind_loop from scenario_generalization.yaml',
                        'accepted globals, search_space and all four original scenarios unchanged'],
        parameter_policy='exact A/B original JSON SHA only; no clamp, sampling or tuning',
        feasibility='runtime_valid AND L1 PASS AND L2 BOUNDED TRANSIENT AND L3 PASS',
        evaluation_no_go_angle_deg=60.0, framework_failure_cost=5.0, runtime_invalid_cost=10.0,
        core3_gate='B 6/6 feasible AND paired-condition mean cost B < A; all runtime valid',
        heldout_gate='B 4/4 feasible AND paired-condition mean cost B <= A; all runtime valid',
        aggregation='separate core3/heldout; arithmetic mean of effective_cost over same scenario/repeat pairs',
        abort_policy='runtime invalid, process error or partial trial: STOP, preserve, no automatic retry',
        limitations=['two repeats; no statistical significance or broad robustness claim',
                     'seed 42 retained as study metadata; run_trial has no simulator RNG seed injection',
                     'known pitch/daggerboard limitations deferred; no new physical gate',
                     'no BO history or optimizer observation is created'])
    output.mkdir(parents=True, exist_ok=False)
    publish(output / 'inputs/scenario_validation.yaml', scenario_bytes)
    publish(output / 'inputs' / CONTRACT, (campaign / CONTRACT).read_bytes())
    for arm, path in original_params(campaign, baseline).items():
        publish(output / f'inputs/{arm}_params.json', path.read_bytes())
    save(output / 'validation_manifest.json', manifest)
    publish(output / 'validation_manifest.sha256',
            (ev.sha256_file(output / 'validation_manifest.json') + '  validation_manifest.json\n').encode())
    (output / 'runtime/generated_worlds').mkdir(parents=True)
    return dict(prepared=str(output), manifest_sha256=ev.sha256_file(output / 'validation_manifest.json'),
                trial_slots=20, simulation_started=False)


def check(output):
    m = read(output / 'validation_manifest.json')
    require((output / 'validation_manifest.sha256').read_text().split()[0] ==
            ev.sha256_file(output / 'validation_manifest.json'), 'manifest SHA mismatch')
    require(m['format'] == 'bo_independent_validation_v1' and m['image'] == IMAGE and
            m['source'] == SOURCE and m['slots'] == schedule() and m['maximum_trials'] == 20,
            'manifest identity/schedule mismatch')
    require(m['tooling_sha256'] == tooling_hashes(), 'validation/runtime tooling changed after freeze')
    require(m['production_source_sha256'] == file_hashes([ROOT / p for p in PRODUCTION], ROOT),
            'production host/image source snapshot drift')
    require(m['campaign_sha256'] == campaign_hashes(Path(m['campaign'])), 'original BO campaign changed')
    for arm, record in m['params'].items():
        require(record['sha256'] == PARAM_SHA[arm] and ev.sha256_file(Path(record['path'])) == PARAM_SHA[arm],
                'original parameter SHA mismatch: ' + arm)
        require(ev.sha256_file(output / f'inputs/{arm}_params.json') == PARAM_SHA[arm], 'parameter snapshot mismatch')
    require(ev.sha256_file(output / 'inputs' / CONTRACT) == CONTRACT_SHA, 'contract snapshot mismatch')
    require(ev.sha256_file(output / 'inputs/scenario_validation.yaml') == m['scenario_sha256'],
            'validation YAML changed')
    expected = validation_scenario(yaml.safe_load((ROOT / 'experiments/sailboat_BO/scenario.yaml').read_text()),
        yaml.safe_load((ROOT / 'experiments/sailboat_BO/scenario_generalization.yaml').read_text()))
    require(yaml.safe_load((output / 'inputs/scenario_validation.yaml').read_text()) == expected,
            'validation YAML semantic mismatch')
    return m


def docker_preflight():
    try:
        p = subprocess.run(['docker', 'image', 'inspect', IMAGE, '--format', '{{.Id}}'],
                           capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ev.ContractError('Docker unavailable: ' + str(exc))
    require(p.returncode == 0 and p.stdout.strip() == IMAGE, 'accepted image unavailable: ' + p.stderr)
    return dict(docker='PASS', image=IMAGE)


def runtime_preflight():
    require(os.environ.get('ACCEPTED_IMAGE_ID') == IMAGE and ROOT == CROOT,
            'use validation shell in pinned accepted image')
    require(os.environ.get('ACCEPTED_SOURCE_COMMIT') == SOURCE, 'runtime source provenance missing')
    installed = Path('/home/docker/sailboat_ws/install')
    paths = {ROOT / 'models/dave_robot_models/description/sailboat/model.sdf': SDF_SHA,
        installed / 'dave_robot_models/share/dave_robot_models/description/sailboat/model.sdf': SDF_SHA,
        installed / 'dave_gz_model_plugins/lib/dave_gz_model_plugins/libVirtualSheetActuatorSystem.so': PLUGIN_SHA}
    for path, digest in paths.items():
        require(ev.sha256_file(path) == digest, 'accepted runtime mismatch: ' + str(path))
    return dict(runtime='PASS', shadow_gate_overlay=False)


def docker_command(output, m):
    cmd = ['docker', 'run', '--rm', '-it', '--net=host', '--privileged', '--user', 'docker',
           '--name', 'sailboat-bo-independent-validation',
           '-e', 'DISPLAY=' + os.environ.get('DISPLAY', ':0'), '-e', 'QT_QPA_PLATFORM=xcb',
           '-e', 'WAYLAND_DISPLAY=', '-e', 'XDG_RUNTIME_DIR=/tmp/runtime-docker', '-e', 'START_QGC=0',
           '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'ACCEPTED_IMAGE_ID=' + IMAGE,
           '-e', 'ACCEPTED_SOURCE_COMMIT=' + SOURCE, '-e', 'VALIDATION_DIR=' + str(output)]
    # Frozen BO results and A/B originals stay read-only at their original paths.
    mounts = [(RESULTS, RESULTS, True), (output, output, False),
              (output / 'inputs', output / 'inputs', True),
              (output / 'validation_manifest.json', output / 'validation_manifest.json', True),
              (output / 'validation_manifest.sha256', output / 'validation_manifest.sha256', True),
              (output / 'runtime/generated_worlds', CROOT / 'experiments/sailboat_BO/results/generated_worlds', False),
              (ROOT / 'experiments/sailboat_validation', CROOT / 'experiments/sailboat_validation', True),
              (ROOT / 'experiments/sailboat_physics/virtual_sheet_validation.py',
               CROOT / 'experiments/sailboat_physics/virtual_sheet_validation.py', True)]
    # No entire-repository overlay: production physics/libs/SDF remain image-owned.
    for path in sorted((ROOT / 'experiments/sailboat_BO').glob('*.py')):
        mounts.append((path, CROOT / 'experiments/sailboat_BO' / path.name, True))
    for name in ('scenario.yaml', 'scenario_generalization.yaml'):
        mounts.append((ROOT / 'experiments/sailboat_BO' / name, CROOT / 'experiments/sailboat_BO' / name, True))
    for path in ('/tmp/.X11-unix', '/mnt/wslg'):
        if Path(path).exists():
            mounts.append((Path(path), Path(path), False))
    for source, target, readonly in mounts:
        cmd += ['--mount', f'type=bind,src={source},dst={target}' + (',readonly' if readonly else '')]
    cmd += ['-w', str(CROOT), IMAGE, 'bash']
    return cmd


def trial_command(output, m, slot):
    return [sys.executable, '-B', 'experiments/sailboat_BO/run_trial.py', '--scenario',
        str(output / 'inputs/scenario_validation.yaml'), '--scenario-id', slot['scenario'],
        '--params-file', m['params'][slot['arm']]['path'], '--repeat-idx', str(slot['repeat']),
        '--results-dir', str(output / 'slots' / slot['slot'] / 'trials'), '--execution-backend', 'local']


def completion(output, slot):
    directory = output / 'slots' / slot['slot']
    if not directory.exists():
        return None
    require((directory / 'completion.json').is_file(), 'PARTIAL/INTERRUPTED slot preserved; no retry: ' + slot['slot'])
    receipt = read(directory / 'completion.json')
    require(receipt['slot'] == slot and receipt['manifest_sha256'] == ev.sha256_file(output / 'validation_manifest.json'),
            'slot receipt identity mismatch')
    actual_files = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    require(actual_files == set(receipt['files']) | {'completion.json'}, 'unexpected completed-slot files/trials')
    for name, digest in receipt['files'].items():
        require(ev.sha256_file(directory / name) == digest, 'completed trial/artifact changed: ' + name)
    return receipt


def measured_metrics(records):
    """Descriptive means, separate from frozen effective objective/feasibility."""
    names = ('mission_time_s', 'xte_rms_m', 'xte_integral_m_s', 'final_distance_to_wp_m',
             'max_abs_roll_deg', 'max_abs_roll_deg_after_grace', 'progress_ratio')
    result = {}
    for name in names:
        values = [r.get('navigation', {}).get('metrics', {}).get(name) for r in records]
        finite = [x for x in values if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)]
        result[name] = dict(count=len(finite), mean=statistics.mean(finite) if finite else None,
                           minimum=min(finite) if finite else None, maximum=max(finite) if finite else None)
    result['mission_success_count'] = sum(r.get('navigation', {}).get('metadata', {}).get('status') == 'success'
                                          for r in records)
    result['mission_complete_count'] = sum(r.get('navigation', {}).get('constraints', {}).get(
                                            'constraint_mission_complete', False) for r in records)
    return result


def aggregate(phase, records):
    expected = [s for s in schedule() if s['phase'] == phase]
    require(len(records) == len(expected) and [r['slot'] for r in records] == expected,
            'phase slots missing, duplicate, reordered or unpaired')
    by_arm = {arm: [r for r in records if r['slot']['arm'] == arm] for arm in ('A', 'B')}
    means = {a: statistics.mean(r['evaluation']['effective_cost'] for r in rows) for a, rows in by_arm.items()}
    counts = {a: sum(r['evaluation']['bo_feasible'] for r in rows) for a, rows in by_arm.items()}
    valid = all(r['evaluation']['runtime']['valid'] and r['returncode'] == 0 for r in records)
    improves = means['B'] < means['A'] if phase == 'core3' else means['B'] <= means['A']
    passed = valid and counts['B'] == len(by_arm['B']) and improves
    pairs = []
    for scenario in (CORE if phase == 'core3' else HELD):
        for repeat in (0, 1):
            pair = {r['slot']['arm']: r for r in records if
                    r['slot']['scenario'] == scenario and r['slot']['repeat'] == repeat}
            pairs.append(dict(scenario=scenario, repeat=repeat,
                A_cost=pair['A']['evaluation']['effective_cost'], B_cost=pair['B']['evaluation']['effective_cost'],
                B_minus_A=pair['B']['evaluation']['effective_cost'] - pair['A']['evaluation']['effective_cost']))
    per_scenario = {}
    for scenario in (CORE if phase == 'core3' else HELD):
        per_scenario[scenario] = {
            arm: dict(mean_effective_cost=statistics.mean(r['evaluation']['effective_cost'] for r in rows
                                                         if r['slot']['scenario'] == scenario),
                      metrics=measured_metrics([r for r in rows if r['slot']['scenario'] == scenario]))
            for arm, rows in by_arm.items()}
    return dict(phase=phase, status='PASS' if passed else ('FAIL' if valid else 'OPEN'),
        trial_count=len(records), feasible_counts=counts, mean_effective_cost=means,
        B_minus_A=means['B'] - means['A'], runtime_valid=valid, B_feasibility_pass=counts['B'] == len(by_arm['B']),
        objective_rule='B < A' if phase == 'core3' else 'B <= A', objective_pass=improves,
        paired_results=pairs, per_scenario=per_scenario,
        descriptive_metrics={arm: measured_metrics(rows) for arm, rows in by_arm.items()},
        trials=records, confidence='descriptive only; two repeats; no significance claim')


def phase_results(output, phase):
    records = [completion(output, s) for s in schedule() if s['phase'] == phase]
    require(all(r is not None for r in records), 'phase not complete: ' + phase)
    return aggregate(phase, records)


def preflight(output, runtime=False, docker=False):
    m = check(output)
    require(not (output / 'STOP.json').exists(), 'STOP preserved; no automatic restart')
    if (output / 'slots').exists():
        require({p.name for p in (output / 'slots').iterdir()} <= {s['slot'] for s in m['slots']},
                'unexpected slot/attempt; fixed 20-trial maximum')
    slots = {s['slot']: 'COMPLETED' if completion(output, s) else 'PENDING' for s in m['slots']}
    result = dict(preflight='PASS', slots=slots, simulation_started=False,
                  disk_free_bytes=shutil.disk_usage(output).free)
    if docker:
        result.update(docker_preflight())
    if runtime:
        result.update(runtime_preflight())
        require(os.access(output / 'runtime/generated_worlds', os.W_OK), 'private world output not writable')
    return result


def evaluate_new_trial(output, m, slot, directory, returncode):
    trials = ev._discover_trials(directory / 'trials')
    require(set(trials) == {(slot['scenario'], slot['repeat'])}, 'missing/unexpected trial identity')
    trial = trials[(slot['scenario'], slot['repeat'])]
    require(not (trial / EVAL_ARTIFACT).exists(), 'existing evaluation preserved; no reevaluation')
    manifest = read(trial / 'trial_manifest.json')
    require(manifest['params'] == m['params'][slot['arm']]['values'], 'trial parameter values mismatch')
    scenario = yaml.safe_load((output / 'inputs/scenario_validation.yaml').read_text())
    definition = next(s for s in scenario['scenarios'] if s['id'] == slot['scenario'])
    termination = dict(scenario['termination'], **definition.get('termination', {}))
    require(manifest['termination'] == termination, 'trial termination differs from validation scenario')
    require(manifest['scenario'] == definition and manifest['study'] == scenario['study'] and
            manifest['logging'] == scenario['logging'], 'trial world/mission/study/logging mismatch')
    payload = ev.evaluate_trial(contract=ev.load_contract(output / 'inputs' / CONTRACT), trial_dir=trial,
        expected_scenario_id=slot['scenario'], expected_repeat_idx=slot['repeat'],
        expected_params=m['params'][slot['arm']]['values'])
    summary = read(trial / 'summary/summary.json') if (trial / 'summary/summary.json').exists() else {}
    metric_names = ('mission_time_s', 'xte_rms_m', 'xte_integral_m_s', 'final_distance_to_wp_m',
                    'max_abs_roll_deg', 'max_abs_roll_deg_after_grace', 'progress_ratio', 'waypoint_switch_count')
    files = [p for p in directory.rglob('*') if p.is_file()]
    receipt = dict(slot=slot, returncode=returncode, trial=str(trial), evaluation=payload,
        navigation=dict(metadata=summary.get('metadata', {}), constraints=summary.get('constraints', {}),
                        metrics={k: summary.get('metrics', {}).get(k) for k in metric_names}),
        manifest_sha256=ev.sha256_file(output / 'validation_manifest.json'),
        files=file_hashes(files, directory))
    save(directory / 'completion.json', receipt)
    return receipt


def run_phase(output, phase, dry_run=False):
    m = check(output)
    require(not (output / 'STOP.json').exists(), 'STOP present: preserve and inspect; no automatic retry')
    if phase == 'heldout':
        require(phase_results(output, 'core3')['status'] == 'PASS', 'Core-3 prerequisite not PASS')
    selected = [s for s in m['slots'] if s['phase'] == phase]
    # Inspect ALL slots before starting anything, not just the current slot.
    if (output / 'slots').exists():
        require({p.name for p in (output / 'slots').iterdir()} <= {s['slot'] for s in m['slots']},
                'unexpected slot/attempt; fixed 20-trial maximum')
    for s in m['slots']:
        completion(output, s)
    commands = [dict(slot=s['slot'], command=trial_command(output, m, s)) for s in selected
                if completion(output, s) is None]
    if dry_run:
        return dict(dry_run=True, phase=phase, pending_trial_count=len(commands), commands=commands,
                    simulation_started=False, writes=False)
    runtime_preflight()
    for slot in selected:
        prior = completion(output, slot)
        if prior is not None:
            require(prior['returncode'] == 0 and prior['evaluation']['runtime']['valid'], 'prior runtime-invalid slot')
            continue
        directory = output / 'slots' / slot['slot']
        try:
            check(output)  # Revalidate original params, frozen code/campaign before every trial.
            command = trial_command(output, m, slot)
            directory.mkdir(parents=True, exist_ok=False)
            save(directory / 'command.json', dict(argv=command, slot=slot, image=IMAGE,
                parameter_sha256=m['params'][slot['arm']]['sha256']))
            print('SIMULATION STARTING: ' + slot['slot'] + '; console=' + str(directory / 'console.log'), flush=True)
            with (directory / 'console.log').open('x') as console:
                process = subprocess.run(command, cwd=ROOT, stdout=console, stderr=subprocess.STDOUT,
                    env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1'))
            record = evaluate_new_trial(output, m, slot, directory, process.returncode)
            check(output)
            evaluation = record['evaluation']
            print(slot['slot'] + ': ' + json.dumps(dict(runtime=evaluation['runtime'],
                layer1=evaluation['layer1'].get('status'), layer2=evaluation['layer2'].get('classification'),
                layer3=evaluation['layer3'].get('status'), bo_feasible=evaluation['bo_feasible'],
                effective_cost=evaluation['effective_cost'])), flush=True)
            require(process.returncode == 0 and record['evaluation']['runtime']['valid'],
                    'runtime-invalid/process error; frozen cost retained; stopping')
            if slot['arm'] == 'B':
                require(record['evaluation']['bo_feasible'], 'B feasibility criterion failed; stopping')
        except BaseException as exc:
            save(output / 'STOP.json', dict(status='STOP_PRESERVED', phase=phase, slot=slot,
                error=type(exc).__name__, message=str(exc), automatic_retry=False,
                timestamp_utc=datetime.now(timezone.utc).isoformat()))
            raise
    result = phase_results(output, phase)
    save(output / f'{phase}_comparison.json', result)
    require(result['status'] == 'PASS', phase + ' comparison criterion not met; preserved; stop')
    return result


def compare(output):
    check(output)
    require(not (output / 'STOP.json').exists(), 'STOP present; no successful final comparison')
    core = phase_results(output, 'core3')
    held = phase_results(output, 'heldout')
    save(output / 'core3_comparison.json', core)
    save(output / 'heldout_comparison.json', held)
    passed = core['status'] == held['status'] == 'PASS'
    result = dict(validation='PASS' if passed else 'FAIL', completed_trial_count=20,
        core3=dict(status=core['status'], feasible_counts=core['feasible_counts'], means=core['mean_effective_cost']),
        heldout=dict(status=held['status'], feasible_counts=held['feasible_counts'], means=held['mean_effective_cost']),
        residual_rl_baseline='SUPPORTED_FOR_NEXT_STAGE_REVIEW' if passed else 'NOT_SUPPORTED',
        production_promotion=False, statistical_significance_claim=False,
        original_campaign_unchanged=True, manifest_sha256=ev.sha256_file(output / 'validation_manifest.json'))
    save(output / 'final_comparison.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'preflight', 'shell', 'run', 'compare', 'status'))
    parser.add_argument('--output', type=Path, default=os.environ.get('VALIDATION_DIR'))
    parser.add_argument('--campaign', type=Path, default=CAMPAIGN)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--phase', choices=('core3', 'heldout'))
    parser.add_argument('--docker', action='store_true')
    parser.add_argument('--runtime', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        require(args.output is not None, '--output or VALIDATION_DIR required')
        output = args.output.resolve()
        if args.action == 'prepare':
            require(not args.dry_run, 'prepare freezes files only; never starts simulations')
            result = prepare(output, args.campaign.resolve(), args.baseline.resolve())
        elif args.action == 'shell':
            m = check(output)
            preflight(output, docker=True)
            print('PINNED CONTAINER SHELL ONLY; no simulation started', flush=True)
            return subprocess.call(docker_command(output, m))
        else:
            with locked(output):
                if args.action == 'preflight':
                    result = preflight(output, args.runtime, args.docker)
                elif args.action == 'run':
                    require(args.phase is not None, '--phase required')
                    result = run_phase(output, args.phase, args.dry_run)
                elif args.action == 'compare':
                    result = compare(output)
                else:
                    m = check(output)
                    result = dict(status='STOP_PRESERVED' if (output / 'STOP.json').exists() else 'VALIDATION_ONLY',
                        slots={s['slot']: ('COMPLETED' if (output / 'slots' / s['slot'] / 'completion.json').is_file()
                               else 'PARTIAL' if (output / 'slots' / s['slot']).exists() else 'PENDING') for s in m['slots']},
                        comparisons={p: read(output / p) for p in ('core3_comparison.json', 'heldout_comparison.json',
                                      'final_comparison.json', 'STOP.json') if (output / p).exists()})
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        if args.action == 'compare' and result['validation'] != 'PASS':
            return 1
        return 0
    except KeyboardInterrupt:
        print('INTERRUPTED: original/partial artifacts preserved; never retry automatically', file=sys.stderr)
        return 130
    except (ev.ContractError, OSError, ValueError, KeyError, StopIteration) as exc:
        print(json.dumps(dict(status='FAIL_OR_OPEN', error=type(exc).__name__, message=str(exc),
                              simulation_automatically_retried=False)), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
