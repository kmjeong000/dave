"""BO evidence collection, read-only verification and fresh local raw copy.

Never launches simulations/BO, edits original evidence, or modifies Git refs.
GitHub stores compact evidence; the raw mirror is deliberately a separate step.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ORIGINAL = Path('/home/jeong/sailboat_ws/virtual-sheet-nominal30-production-candidate')
RESULTS = Path('/home/jeong/sailboat_rl_results')
CAMPAIGN = Path('bo/bo_campaign_contract_v1_patch1_20261020T042842Z')
VALIDATION = Path('bo_validation/iter014_vs_iter008_v1_20261026T055333Z')
SOURCE = '4ec49538912fe88ca3be569c1a787dba4c07994b'
IMAGE = 'sha256:c7dd2ca9507f72a6937b694f7318560a1f6c923035e561a9884d2b68458c16bc'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def publish(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(data)


def save(path, data):
    publish(path, (json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+'\n').encode())


def table(path, rows):
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
    publish(path, out.getvalue().encode())


def roots():
    # Include external iteration-0 and iteration-1 recovery dependencies, and
    # the accepted plant's original repeat0/1/2 validation provenance.
    return [CAMPAIGN, VALIDATION,
        Path('bo/bo_campaign_contract_v1_iter0_20261020T014743Z'),
        Path('bo/bo_patch1_iter1_recovery_20261024T090735Z'),
        Path('bo/final_optimized_baseline/iter_008_original_params.json'),
        Path('incremental_rebuild/virtual_sheet_nominal30_production_core3_r00_20261019T083721Z'),
        Path('incremental_rebuild/virtual_sheet_nominal30_production_core3_r01_r02_20261019T113319Z'),
        Path('incremental_rebuild/accepted_parents')]


def collect():
    sys.path.insert(0, str(ORIGINAL))
    from experiments.sailboat_validation import validate_ab as v
    v.inspect_originals(RESULTS/CAMPAIGN, v.BASELINE)
    v.check(RESULTS/VALIDATION)
    original_status = subprocess.check_output(['git','-C',str(ORIGINAL),'status','--porcelain=v1','-uall'], text=True)
    dirty = [x[3:] for x in original_status.splitlines()]
    code = {n: sha(REPO/n) for n in v.tooling_hashes()}
    assert all(sha(ORIGINAL/n) == digest for n, digest in code.items())
    c = RESULTS/CAMPAIGN; val = RESULTS/VALIDATION
    observations = [json.loads(x) for x in (c/'observations.jsonl').read_text().splitlines()]
    assert len(observations)==40 and [x['iteration'] for x in observations]==list(range(40))
    trials=[]; iterations=[]
    for ob in observations:
        it=ob['iteration']; idir=c/f'iter_{it:03d}'
        e=idir/'candidate_evaluation_v1_patch1.json' if it==0 else idir/'trials/candidate_evaluation_v1_patch1.json'
        ev=load(e)
        assert ev['trial_count']==ev['expected_trial_count']==6
        iterations.append(dict(iteration=it, objective=ev['objective'], failure_count=ev['failure_count'],
            feasible=ev['candidate_feasible'],params_sha256=ev['params_sha256'],evaluation_path=str(e.relative_to(RESULTS)),evaluation_sha256=sha(e)))
        for t in ev['trials']:
            trials.append(dict(iteration=it,scenario=t['scenario_id'],repeat=t['repeat_idx'],
                runtime_valid=t['runtime']['valid'],layer1=t['layer1']['status'],
                layer2=t['layer2']['classification'],layer3=t['layer3']['status'],
                feasible=t['bo_feasible'],effective_cost=t['effective_cost'],trial_dir=t['trial_dir']))
    assert sum(x['feasible'] for x in iterations)==30
    assert sum(x['feasible'] for x in trials)==223 and sum(x['runtime_valid'] for x in trials)==239
    core=v.phase_results(val,'core3'); assert core==load(val/'core3_comparison.json')
    validation=[]
    for s in v.schedule():
        d=val/'slots'/s['slot']
        if not d.exists(): continue
        r=v.completion(val,s); t=r['evaluation']; n=r['navigation']
        validation.append(dict(slot=s['slot'],phase=s['phase'],arm=s['arm'],scenario=s['scenario'],repeat=s['repeat'],
            runtime_valid=t['runtime']['valid'],layer1=t['layer1']['status'],layer2=t['layer2']['classification'],
            layer3=t['layer3']['status'],feasible=t['bo_feasible'],effective_cost=t['effective_cost'],
            failure_reason=n['metadata']['failure_reason'],max_abs_roll_deg=n['metrics']['max_abs_roll_deg']))
    assert len(validation)==14 and load(val/'STOP.json')['slot']['slot']=='14_eval_long_oblique_r00_B'
    inventory=[]
    for relative in roots():
        p=RESULTS/relative
        assert p.exists(), p
        paths=sorted(q for q in p.rglob('*') if q.is_file()) if p.is_dir() else [p]
        for q in paths:
            inventory.append(dict(path=str(q.relative_to(RESULTS)),bytes=q.stat().st_size,sha256=sha(q)))
            if len(inventory)%100==0: print('RAW HASHED',len(inventory),flush=True)
    publish(HERE/'raw_inventory.jsonl', ''.join(json.dumps(x,sort_keys=True)+'\n' for x in inventory).encode())
    for source,dest in [(c/'iter_014/params.json','params_iter014.json'),
                        (v.BASELINE,'params_reference_iter008.json'),
                        (c/'bo_config.json','bo_config.json'),(c/'campaign_state.json','campaign_state.json'),
                        (c/'observations.jsonl','observations.jsonl'),(c/'history.csv','history.csv'),
                        (val/'validation_manifest.json','validation_manifest.json'),
                        (val/'inputs/scenario_validation.yaml','scenario_validation.yaml'),
                        (val/'STOP.json','STOP.json')]:
        publish(HERE/dest,source.read_bytes())
    table(HERE/'iterations.csv',iterations);table(HERE/'trials.csv',trials);table(HERE/'validation_trials.csv',validation)
    manifest=dict(format='completed_bo_iter014_research_archive_v1',created_at_utc=datetime.now(timezone.utc).isoformat(),
        accepted_source=SOURCE,accepted_runtime_image=IMAGE,original_workspace=str(ORIGINAL),
        original_branch='sailboat-virtual-sheet-nominal30-production',
        original_status=original_status,original_dirty_sha256={n:sha(ORIGINAL/n) for n in dirty},
        host_build=dict(build_exists=(ORIGINAL/'build').exists(),install_exists=(ORIGINAL/'install').exists(),
                        validated_runtime='image-owned /home/docker/sailboat_ws/install; do not use shared WSL install'),
        code_sha256=code,production_source_sha256=v.file_hashes([ORIGINAL/p for p in v.PRODUCTION],ORIGINAL),
        campaign=dict(path=str(c),iterations=40,trials=240,feasible_candidates=30,feasible_trials=223,runtime_valid_trials=239,
                      best_iteration=14,best_objective=.32607422077940834,params_sha256=v.PARAM_SHA['B'],
                      immutable_state_sha256=v.STATE_SHA),
        validation=dict(path=str(val),planned_maximum=20,completed=14,unexecuted=6,
            core3=dict(status='PASS',feasible_counts=core['feasible_counts'],mean_effective_cost=core['mean_effective_cost'],
                improvement_percent=100*(core['mean_effective_cost']['A']-core['mean_effective_cost']['B'])/core['mean_effective_cost']['A']),
            heldout='FAIL: Long-Oblique r00 excessive_roll for A and B; Layer1 PASS / Layer2 BOUNDED TRANSIENT / Layer3 FAIL',
            stop_sha256=sha(val/'STOP.json'),final_comparison_exists=False),
        raw_roots=[str(x) for x in roots()],raw_file_count=len(inventory),raw_logical_bytes=sum(x['bytes'] for x in inventory),
        raw_inventory_sha256=sha(HERE/'raw_inventory.jsonl'),
        tests_before_commit=dict(bo=80,validation=23,framework_v2=8,result='111 PASS; offline fixtures only'),
        limitations=['GitHub contains code, config, params and compact evidence, not raw logs or Docker image layers',
                     'Iteration14 is provisional Core-3 validated, not held-out/generalization approved',
                     'Known pitch/daggerboard limitations and common roll failure remain unresolved',
                     'Seed metadata does not guarantee identical Gazebo stochastic trajectories',
                     'Local raw copy on the same WSL disk is not independent disaster-recovery backup',
                     'Iteration1 recovered Crosswind r01 OPEN retained with penalty5; iteration34 runtime-invalid cost10 retained'])
    save(HERE/'manifest.json',manifest)
    print('COLLECT PASS',len(inventory),manifest['raw_logical_bytes'],flush=True)


def verify(root):
    m=load(HERE/'manifest.json')
    assert sha(HERE/'raw_inventory.jsonl')==m['raw_inventory_sha256']
    rows=[json.loads(x) for x in (HERE/'raw_inventory.jsonl').read_text().splitlines()]
    for i,r in enumerate(rows,1):
        p=root/r['path'];assert p.is_file() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],p
        if i%100==0:print('VERIFIED',i,flush=True)
    print('RAW SHA PASS',len(rows),flush=True)
    return m


def backup(destination):
    m=load(HERE/'manifest.json')
    assert not destination.exists(),'fresh backup required'
    assert not destination.resolve().is_relative_to(RESULTS.resolve()),'separate backup root required'
    destination.parent.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(destination.parent).free>m['raw_logical_bytes'],'insufficient disk space'
    destination.mkdir()
    subprocess.run(['cp','-aL','--parents','--',*m['raw_roots'],str(destination)],cwd=RESULTS,check=True)
    verify(destination)
    save(destination/'BACKUP_VERIFIED.json',dict(status='PASS',manifest_sha256=sha(HERE/'manifest.json'),
        files=m['raw_file_count'],logical_bytes=m['raw_logical_bytes'],independent_storage=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('collect','verify','backup'))
    p.add_argument('--results-root',type=Path,default=RESULTS);p.add_argument('--destination',type=Path)
    a=p.parse_args()
    if a.action=='collect':collect()
    elif a.action=='verify':verify(a.results_root)
    else:
        assert a.destination is not None,'--destination required'
        backup(a.destination)

if __name__=='__main__':main()
