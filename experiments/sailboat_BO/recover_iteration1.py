"""Recover only the interrupted patch1 iteration 1; never proposes parameters.

inspect/dry-run are read-only. prepare freezes a fresh sidecar, run-slot starts
ONE explicitly requested trial, evaluate writes ONLY a fresh evaluation view,
and commit appends ONE observation under a replayable append-only journal.
No subcommand starts another iteration or modifies optimizer/evaluator math.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import fcntl
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from . import bo_campaign_evaluator as ev
from . import bo_campaign_state as state

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "sha256:c7dd2ca9507f72a6937b694f7318560a1f6c923035e561a9884d2b68458c16bc"
SOURCE = "4ec49538912fe88ca3be569c1a787dba4c07994b"
MANIFEST = "82a7e66fdc1c9506c4ecf6e25a17c9c0b1a9c65e09952e50c22c66a99824eb96"
CONTRACT = "bo_campaign_contract_v1_patch1.yaml"
ARTIFACT = "candidate_evaluation_v1_patch1.json"
PENDING = {"upwind-r00": ("train_upwind_tack", 0),
           "upwind-r01": ("train_upwind_tack", 1),
           "dogleg-r00": ("train_dogleg", 0), "dogleg-r01": ("train_dogleg", 1)}
SETTINGS = dict(initial_random=12, local_warmup=6, local_radius=.15,
                candidate_pool=1024, repeats=2, seed=42, objective_column="scenario_cost",
                worst_case_weight=0., no_baseline=False, trial_dry_run=False, stop_on_failure=False)
SDF_SHA = "5fb8d0288e39eabac47f14c92c7c62785ca4fc8021274e4744280bdf6e398ecc"
PLUGIN_SHA = "66877a1da116237123ef2931846e08702b6a6a8803f9bc7dbf2427166982b849"


def require(ok, message):
    if not ok:
        raise ev.ContractError(message)


def read(path):
    return json.loads(Path(path).read_text())


def encoded(data):
    return (json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()


def publish(path, content):
    """Atomic no-clobber publication; an identical existing receipt is reusable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == content, "existing artifact differs: "+str(path))
        return
    fd, name = tempfile.mkstemp(prefix=".recovery-publish-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content); f.flush(); os.fsync(f.fileno())
        try:
            os.link(name, path)  # Fails rather than overwriting an existing name.
        except FileExistsError:
            require(path.read_bytes() == content, "concurrent artifact conflict: "+str(path))
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        Path(name).unlink(missing_ok=True)


def save(path, data):
    publish(path, encoded(data))


def hashes(root):
    return {str(p.relative_to(root)): ev.sha256_file(p) for p in sorted(root.rglob("*"))
            if p.is_file()}


def code_hashes():
    names = ["common.py", "run_trial.py", "run_batch.py", "run_bo.py", "score.py",
             "bo_campaign_state.py", "bo_campaign_evaluator.py", "recover_iteration1.py", "scenario.yaml"]
    paths = [ROOT/"experiments/sailboat_BO"/n for n in names]
    paths += [ROOT/"experiments/sailboat_physics/virtual_sheet_validation.py"]
    return {str(p.relative_to(ROOT)): ev.sha256_file(p) for p in paths}


@contextmanager
def locked(campaign):
    # Lock the existing directory inode: no lockfile or original file is edited.
    fd = os.open(campaign, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ev.ContractError("another recovery command holds this campaign lock")
        yield
    finally:
        os.close(fd)


def validate_state(campaign, baseline):
    return state.validate_frozen_campaign_state(
        contract_path=campaign/CONTRACT, scenario_path=ROOT/"experiments/sailboat_BO/scenario.yaml",
        baseline_params_path=baseline, runtime_image_id=IMAGE, source_commit=SOURCE,
        bo_dir=campaign, expected_optimizer_settings=SETTINGS)


def inspect(campaign, baseline):
    v = validate_state(campaign, baseline)
    require(v["manifest_sha256"] == MANIFEST, "not the approved frozen campaign root")
    require(v["observation_count"] == v["next_iteration"] == 1, "iteration 1 already recorded or state changed")
    proposal = read(campaign/"iter_001/proposal.json")
    params = read(campaign/"iter_001/params.json")
    require(proposal["iteration"] == 1 and proposal["seed"] == 42
            and proposal["source"] == "local_warmup" and proposal["params"] == params,
            "saved iteration-1 proposal/params/seed mismatch")
    slots = ev._discover_trials(campaign/"iter_001/trials")
    reused = {}
    for repeat in (0, 1):
        trial = slots.get(("train_crosswind_straight", repeat))
        require(trial is not None and (trial/"summary/summary.json").is_file()
                and (trial/"raw/samples.csv").is_file(), "completed Crosswind input missing")
        manifest = read(trial/"trial_manifest.json")
        ev._assert_trial_identity(manifest=manifest, expected_scenario_id="train_crosswind_straight",
                                  expected_repeat_idx=repeat, expected_params=params)
        reused["crosswind-r0"+str(repeat)] = str(trial.relative_to(campaign))
    for scenario, repeat in PENDING.values():
        t = slots.get((scenario, repeat))
        require(t is None or not (t/"summary/summary.json").exists(),
                "a pending slot has completed data; do not re-run it blindly")
    return dict(observation_count=1, next_iteration=1, manifest_sha256=v["manifest_sha256"],
                reused=reused, pending=PENDING, proposal=proposal,
                original_upwind_incomplete_preserved=True, code_sha256=code_hashes())


def review_trial(trial, contract, params, scenario, repeat):
    """Read-only composition of the frozen evaluator's existing pure functions."""
    manifest = read(trial/"trial_manifest.json")
    ev._assert_trial_identity(manifest=manifest, expected_scenario_id=scenario,
                             expected_repeat_idx=repeat, expected_params=params)
    summary = read(trial/"summary/summary.json")
    raw = ev._read_csv(trial/"raw/samples.csv")
    logs = sorted((trial/"logs").glob("launch.*.log"))
    samples = ev.parse_virtual_sheet_logs(logs)
    runtime = ev._runtime_validity(trial, summary=summary, raw_samples=raw,
                                  framework_samples=samples, log_paths=logs)
    f = contract["framework"]
    l1 = ev.evaluate_layer1(samples, maximum_allowance_rad=math.radians(f["maximum_allowance_deg"]),
        mechanical_limit_rad=math.radians(f["mechanical_limit_deg"]),
        reference_rad=math.radians(f["penetration_reference_deg"]),
        sheet_taut_states=ev.parse_sheet_taut_states(logs, samples))
    l2 = ev.evaluate_compliance(samples, framework_version=f["version"],
        reference_rad=math.radians(f["penetration_reference_deg"]),
        maximum_allowance_rad=math.radians(f["maximum_allowance_deg"]))
    l3 = ev.evaluate_layer3(summary)
    return dict(runtime_valid=runtime["valid"], layer1=l1["status"],
                layer2=l2.classification, layer3=l3["status"])


def prepare(campaign, recovery, baseline, dry_run=False):
    require(recovery != campaign and campaign not in recovery.parents
            and recovery not in campaign.parents, "use a separate recovery directory, not inside campaign")
    info = inspect(campaign, baseline)
    before = hashes(campaign)
    contract = ev.load_contract(campaign/CONTRACT)
    params = info["proposal"]["params"]
    reviews = {slot: review_trial(campaign/p, contract, params, "train_crosswind_straight", repeat)
               for repeat, (slot, p) in enumerate(info["reused"].items())}
    require(reviews["crosswind-r00"] == dict(runtime_valid=True, layer1="PASS",
            layer2="BOUNDED TRANSIENT", layer3="PASS"), "Crosswind r00 changed")
    require(reviews["crosswind-r01"] == dict(runtime_valid=True, layer1="PASS",
            layer2="OPEN", layer3="PASS"), "Crosswind r01 OPEN must not be changed")
    require(before == hashes(campaign), "campaign changed during preflight")
    plan = dict(format="bo_iteration1_recovery_v1", source=SOURCE, image=IMAGE,
                manifest_sha256=MANIFEST, original_files=before,
                code_sha256=info["code_sha256"], reused=info["reused"], reviews=reviews,
                pending=PENDING, proposal=info["proposal"], optimizer_settings=SETTINGS,
                cwd=str(ROOT), original_campaign=str(campaign))
    if dry_run:
        return dict(dry_run=True, simulation_started=False, campaign_written=False,
                    reviews=reviews, pending_slots=list(PENDING), proposal_source="saved local_warmup",
                    observation_count=1, next_iteration=1, manifest_sha256=MANIFEST)
    require(not recovery.exists(), "fresh recovery directory required")
    recovery.mkdir(parents=True)
    parent_fd = os.open(recovery.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    (recovery/"runtime/generated_worlds").mkdir(parents=True)
    for name in ("history.csv", "observations.jsonl", "best_params.json"):
        publish(recovery/"baseline"/name, (campaign/name).read_bytes())
    save(recovery/"plan.json", plan)
    publish(recovery/"plan.sha256", (ev.sha256_file(recovery/"plan.json")+"\n").encode())
    return dict(prepared=True, recovery=str(recovery), pending_slots=list(PENDING), reviews=reviews)


def check(campaign, recovery, baseline, committing=False):
    plan = read(recovery/"plan.json")
    require(ev.sha256_file(recovery/"plan.json") == (recovery/"plan.sha256").read_text().strip(),
            "recovery plan SHA mismatch")
    require(plan["format"] == "bo_iteration1_recovery_v1" and plan["image"] == IMAGE
            and plan["source"] == SOURCE and plan["manifest_sha256"] == MANIFEST,
            "recovery identity mismatch")
    require(code_hashes() == plan["code_sha256"], "recovery/runtime tooling changed")
    for name in ("history.csv", "observations.jsonl", "best_params.json"):
        require(ev.sha256_file(recovery/"baseline"/name) == plan["original_files"][name],
                "recovery baseline copy changed: "+name)
    for name, h in plan["original_files"].items():
        if committing and name in ("history.csv", "observations.jsonl"):
            continue  # Only a verified journal may append to these two files.
        require((campaign/name).is_file() and ev.sha256_file(campaign/name) == h,
                "original campaign file changed: "+name)
    if not committing:
        v = validate_state(campaign, baseline)
        require(v["observation_count"] == 1 and v["next_iteration"] == 1, "state no longer at iteration 1")
    else:
        # Full state validation can fail transiently between our two journaled
        # appends. Validate frozen source/contract independently, then validate
        # the complete optimizer state after both appends finish.
        ev.validate_provenance(contract_path=campaign/CONTRACT,
            scenario_path=ROOT/"experiments/sailboat_BO/scenario.yaml", baseline_params_path=baseline,
            runtime_image_id=IMAGE, source_commit=SOURCE)
    return plan


def docker_preflight():
    try:
        p = subprocess.run(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"],
                           capture_output=True, text=True, timeout=12)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ev.ContractError("Docker daemon/image unavailable: "+str(exc))
    require(p.returncode == 0 and p.stdout.strip() == IMAGE, "accepted Docker image inaccessible: "+p.stderr)
    return dict(docker_daemon="PASS", accepted_image=IMAGE)


def runtime_preflight():
    require(os.environ.get("ACCEPTED_IMAGE_ID") == IMAGE, "use the pinned accepted-image container shell")
    home = Path("/home/docker/sailboat_ws")
    sdf = "dave_robot_models/share/dave_robot_models/description/sailboat/model.sdf"
    for p, h in ((ROOT/"models/dave_robot_models/description/sailboat/model.sdf", SDF_SHA),
                 (home/"install"/sdf, SDF_SHA),
                 (home/"install/dave_gz_model_plugins/lib/dave_gz_model_plugins/libVirtualSheetActuatorSystem.so", PLUGIN_SHA)):
        require(p.is_file() and ev.sha256_file(p) == h, "accepted runtime hash mismatch: "+str(p))
    if os.environ.get("BO_RECOVERY_DIR"):
        output = Path(os.environ["BO_RECOVERY_DIR"])
        require(os.access(output, os.W_OK)
                and os.access(output/"runtime/generated_worlds", os.W_OK),
                "fresh recovery results/generated-world mounts are not writable")
    return dict(runtime="PASS", image=IMAGE, diagnostic_overlay=False)


def shell(campaign, recovery, baseline):
    check(campaign, recovery, baseline)
    docker_preflight()
    cmd = ["docker", "run", "--rm", "-it", "--net=host", "--privileged", "--user", "docker",
           "-e", "DISPLAY="+os.environ.get("DISPLAY", ":0"), "-e", "QT_QPA_PLATFORM=xcb",
           "-e", "WAYLAND_DISPLAY=", "-e", "XDG_RUNTIME_DIR=/tmp/runtime-docker", "-e", "START_QGC=0",
           "-e", "ACCEPTED_IMAGE_ID="+IMAGE, "-e", "BO_RECOVERY_CAMPAIGN=/frozen_campaign",
           "-e", "BO_RECOVERY_DIR=/recovery", "-e", "BO_RECOVERY_BASELINE=/baseline.json",
           "-v", str(campaign)+":/frozen_campaign:ro", "-v", str(recovery)+":/recovery",
           "-v", str(baseline)+":/baseline.json:ro",
           "-v", str(recovery/"runtime/generated_worlds")+
                 ":/home/docker/sailboat_ws/src/dave/experiments/sailboat_BO/results/generated_worlds",
           "-v", str(ROOT/"experiments/sailboat_physics/virtual_sheet_validation.py")+
                 ":/home/docker/sailboat_ws/src/dave/experiments/sailboat_physics/virtual_sheet_validation.py:ro"]
    # Mount tooling files, not the whole BO directory: run_trial's generated
    # world output must remain writable without modifying host source/config.
    for name in [p.name for p in (ROOT/"experiments/sailboat_BO").glob("*.py")] + ["scenario.yaml"]:
        cmd += ["-v", str(ROOT/"experiments/sailboat_BO"/name)+
                ":/home/docker/sailboat_ws/src/dave/experiments/sailboat_BO/"+name+":ro"]
    for p in ("/tmp/.X11-unix", "/mnt/wslg"):
        if Path(p).exists():
            cmd += ["-v", p+":"+p]
    cmd += ["-w", "/home/docker/sailboat_ws/src/dave", IMAGE, "bash"]
    print("CONTAINER SHELL ONLY; no simulation started by shell entry", flush=True)
    return subprocess.call(cmd)


def completed_slot(recovery, slot, plan):
    receipts = sorted((recovery/"slots"/slot).glob("attempt_*/completion.json"))
    require(len(receipts) <= 1, "multiple completed attempts for "+slot)
    if not receipts:
        return None
    receipt = read(receipts[0])
    attempt = receipts[0].parent
    require(receipt["slot"] == slot and receipt["plan_sha256"] == ev.sha256_file(recovery/"plan.json"),
            "slot receipt identity mismatch")
    require(receipt["files"] == hashes(attempt/"trials"), "completed trial changed: "+slot)
    require(receipt["image"] == IMAGE, "wrong runtime image in receipt")
    t = attempt/"trials"/receipt["trial_name"]
    scenario, repeat = PENDING[slot]
    ev._assert_trial_identity(manifest=read(t/"trial_manifest.json"), expected_scenario_id=scenario,
                             expected_repeat_idx=repeat, expected_params=plan["proposal"]["params"])
    return t


def run_slot(campaign, recovery, baseline, slot, dry_run=False, retry=False):
    require(slot in PENDING, "only four pending slots are executable; Crosswind never re-runs")
    plan = check(campaign, recovery, baseline)
    prior = completed_slot(recovery, slot, plan)
    if prior:
        return dict(slot=slot, status="ALREADY_COMPLETED", simulation_started=False, trial=str(prior))
    attempts = sorted((recovery/"slots"/slot).glob("attempt_*"))
    require(not attempts or retry, "incomplete attempt preserved; inspect it, then explicitly use --retry-incomplete")
    attempt = recovery/"slots"/slot/("attempt_%04d" % (len(attempts)+1))
    scenario, repeat = PENDING[slot]
    command = [sys.executable, "-B", "experiments/sailboat_BO/run_trial.py", "--scenario",
               "experiments/sailboat_BO/scenario.yaml", "--scenario-id", scenario,
               "--params-file", str(campaign/"iter_001/params.json"), "--repeat-idx", str(repeat),
               "--results-dir", str(attempt/"trials"), "--execution-backend", "local"]
    if dry_run:
        return dict(slot=slot, command=command, simulation_started=False, dry_run=True)
    runtime_preflight()
    require(os.access(recovery, os.W_OK), "recovery bind mount is not writable")
    attempt.mkdir(parents=True, exist_ok=False)
    save(attempt/"command.json", dict(argv=command, image=IMAGE, plan_sha256=ev.sha256_file(recovery/"plan.json")))
    print("SIMULATION STARTING: "+slot+"; console="+str(attempt/"console.log"), flush=True)
    with (attempt/"console.log").open("x") as console:
        result = subprocess.run(command, cwd=ROOT, stdout=console, stderr=subprocess.STDOUT,
                                env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1"))
    trials = ev._discover_trials(attempt/"trials")
    require(set(trials) == {(scenario, repeat)}, "missing/unexpected trial slot in attempt")
    t = trials[(scenario, repeat)]
    require((t/"summary/summary.json").exists() and (t/"raw/samples.csv").exists(),
            "attempt incomplete: preserved, no completion receipt; retry requires explicit flag")
    ev._assert_trial_identity(manifest=read(t/"trial_manifest.json"), expected_scenario_id=scenario,
                             expected_repeat_idx=repeat, expected_params=plan["proposal"]["params"])
    check(campaign, recovery, baseline)
    save(attempt/"completion.json", dict(slot=slot, trial_name=t.name, returncode=result.returncode,
        image=IMAGE, plan_sha256=ev.sha256_file(recovery/"plan.json"), files=hashes(attempt/"trials")))
    return dict(slot=slot, status="COLLECTED_NOT_YET_FEASIBILITY_EVALUATED", returncode=result.returncode,
                trial=str(t), simulation_started=True, console=str(attempt/"console.log"))


def selected_trials(campaign, recovery, plan):
    selected = {k: campaign/v for k, v in plan["reused"].items()}
    for slot in PENDING:
        t = completed_slot(recovery, slot, plan)
        require(t is not None, "pending slot not complete: "+slot)
        selected[slot] = t
    return selected


def host_side(plan, campaign):
    require(str(ROOT) == plan["cwd"] and str(campaign) == plan["original_campaign"],
            "evaluate/commit must run on the original WSL host, not in the container")


def evaluate(campaign, recovery, baseline, dry_run=False):
    plan = check(campaign, recovery, baseline)
    host_side(plan, campaign)
    selected = selected_trials(campaign, recovery, plan)
    complete = recovery/"evaluation_complete.json"
    if complete.exists():
        receipt = read(complete)
        require(receipt["selected_sha256"] == {k: hashes(v) for k, v in selected.items()},
                "selected trial data changed after evaluation")
        view = recovery/receipt["view"]
        require(ev.sha256_file(view/ARTIFACT) == receipt["candidate_sha256"], "candidate artifact changed")
        return dict(status="ALREADY_EVALUATED", candidate=str(view/ARTIFACT), simulation_started=False)
    if dry_run:
        return dict(dry_run=True, selected={k: str(v) for k,v in selected.items()}, writes=False)
    attempts = sorted((recovery/"evaluations").glob("attempt_*"))
    view = recovery/"evaluations"/("attempt_%04d" % (len(attempts)+1))/"trials"
    view.mkdir(parents=True, exist_ok=False)
    # Real view directories, symlinks ONLY for immutable inputs. The evaluator
    # writes physics_diagnostics into the new view, never the original trials.
    for slot, source in selected.items():
        target = view/slot; target.mkdir()
        for name in ("trial_manifest.json", "trial.parm", "logs", "raw", "summary"):
            if (source/name).exists():
                (target/name).symlink_to(source/name, target_is_directory=(source/name).is_dir())
    before = {k: hashes(v) for k,v in selected.items()}
    # Match run_batch --keep-going: return 1 if any executed trial returned nonzero.
    returncodes = {slot: read(t.parent.parent/"completion.json")["returncode"]
                   for slot, t in selected.items() if slot in PENDING}
    batch_returncode = int(any(v != 0 for v in returncodes.values()))
    candidate = ev.evaluate_candidate(contract_path=campaign/CONTRACT,
        scenario_path=ROOT/"experiments/sailboat_BO/scenario.yaml", baseline_params_path=baseline,
        params_path=campaign/"iter_001/params.json", candidate_dir=view,
        runtime_image_id=IMAGE, source_commit=SOURCE, iteration=1, batch_returncode=batch_returncode)
    require(before == {k: hashes(v) for k,v in selected.items()}, "selected originals were modified")
    require(candidate["trial_count"] == candidate["expected_trial_count"] == 6, "not six trials")
    cw1 = next(t for t in candidate["trials"] if t["scenario_id"] == "train_crosswind_straight" and t["repeat_idx"] == 1)
    require(cw1["layer2"]["classification"] == "OPEN" and cw1["effective_cost"] == 5.
            and cw1["failure_count"] == 1 and not cw1["bo_feasible"], "frozen OPEN penalty changed")
    fields = ["scenario_id", "repeat_idx", "bo_feasible", "failure_count", "effective_cost", "layer1", "layer2", "layer3"]
    output = io.StringIO(); writer = csv.DictWriter(output, fieldnames=fields); writer.writeheader()
    for t in candidate["trials"]:
        writer.writerow(dict(scenario_id=t["scenario_id"], repeat_idx=t["repeat_idx"],
            bo_feasible=t["bo_feasible"], failure_count=t["failure_count"], effective_cost=t["effective_cost"],
            layer1=t["layer1"]["status"], layer2=t["layer2"]["classification"], layer3=t["layer3"]["status"]))
    publish(view/"summary.csv", output.getvalue().encode())
    check(campaign, recovery, baseline)
    save(complete, dict(view=str(view.relative_to(recovery)), candidate_sha256=ev.sha256_file(view/ARTIFACT),
                        summary_sha256=ev.sha256_file(view/"summary.csv"), selected_sha256=before,
                        executed_trial_returncodes=returncodes))
    return dict(status="EVALUATED_NOT_COMMITTED", candidate=str(view/ARTIFACT),
                failure_count=candidate["failure_count"], objective=candidate["objective"],
                candidate_feasible=candidate["candidate_feasible"])


def append_checked(path, base, suffix):
    """Recover ONLY our exact journaled prefix, including a torn own append."""
    with path.open("r+b") as f:
        current = f.read()
        require(current.startswith(base), "optimizer history prefix changed: "+str(path))
        tail = current[len(base):]
        require(suffix.startswith(tail), "foreign/duplicate optimizer append: "+str(path))
        f.seek(0, os.SEEK_END)
        f.write(suffix[len(tail):]); f.flush(); os.fsync(f.fileno())


def commit(campaign, recovery, baseline, dry_run=False):
    plan = check(campaign, recovery, baseline, committing=True)
    host_side(plan, campaign)
    receipt = read(recovery/"evaluation_complete.json")
    view = recovery/receipt["view"]
    require(ev.sha256_file(view/ARTIFACT) == receipt["candidate_sha256"]
            and ev.sha256_file(view/"summary.csv") == receipt["summary_sha256"], "evaluation artifact changed")
    selected = selected_trials(campaign, recovery, read(recovery/"plan.json"))
    require(receipt["selected_sha256"] == {k: hashes(v) for k,v in selected.items()}, "trial inputs changed")
    candidate = read(view/ARTIFACT)
    require(candidate["iteration"] == 1 and candidate["trial_count"] == 6
            and candidate["failure_count"] >= 1, "invalid iteration-1 aggregate")
    result = {k: candidate[k] for k in ("iteration", "params", "objective", "mean_scenario_cost",
              "worst_scenario_cost", "trial_count", "expected_trial_count", "failure_count")}
    result.update(returncode=candidate["batch_returncode"], summary_csv=str(view/"summary.csv"))
    # Retained CW r01 is infeasible, so the feasible iteration-0 incumbent stays
    # unchanged under the frozen failure-count-first ordering.
    best = read(campaign/"best_params.json")
    require(best["iteration"] == 0 and best["failure_count"] == 0, "unexpected incumbent")
    base_history = (recovery/"baseline/history.csv").read_bytes()
    base_obs = (recovery/"baseline/observations.jsonl").read_bytes()
    require(base_history.endswith(b"\n") and base_obs.endswith(b"\n"), "unterminated baseline state")
    fields = next(csv.reader(io.StringIO(base_history.decode())))
    row = {k: result.get(k) for k in fields if not k.startswith("param__")}
    row.update({k: result["params"][k.removeprefix("param__")] for k in fields if k.startswith("param__")})
    buf = io.StringIO(); csv.DictWriter(buf, fieldnames=fields).writerow(row)
    suffixes = {"history.csv": buf.getvalue().encode(),
                "observations.jsonl": (json.dumps(result, sort_keys=True, allow_nan=False)+"\n").encode()}
    if not (recovery/"commit_journal.json").exists():
        require((campaign/"history.csv").read_bytes() == base_history
                and (campaign/"observations.jsonl").read_bytes() == base_obs,
                "history changed without this recovery's commit journal")
    for name, base in (("history.csv", base_history), ("observations.jsonl", base_obs)):
        data = (campaign/name).read_bytes()
        require(data.startswith(base) and suffixes[name].startswith(data[len(base):]), "history append conflict: "+name)
    if dry_run:
        return dict(dry_run=True, would_append_iteration=1, observation_count_after=2, writes=False)
    journal = dict(candidate_sha256=receipt["candidate_sha256"], result=result,
                   suffixes={k: v.decode() for k,v in suffixes.items()})
    save(recovery/"commit_journal.json", journal)
    # No-clobber artifacts first, then two append-only, journal-recoverable logs.
    publish(campaign/"iter_001/trials"/ARTIFACT, (view/ARTIFACT).read_bytes())
    save(campaign/"iter_001/iteration_result.json", result)
    append_checked(campaign/"history.csv", base_history, suffixes["history.csv"])
    append_checked(campaign/"observations.jsonl", base_obs, suffixes["observations.jsonl"])
    validated = validate_state(campaign, baseline)
    require(validated["observation_count"] == validated["next_iteration"] == 2, "post-commit state is not exactly two observations")
    save(recovery/"committed.json", dict(iteration=1, observation_count=2, next_iteration=2,
        candidate_sha256=receipt["candidate_sha256"], incumbent_unchanged=True))
    return dict(status="COMMITTED_OR_ALREADY_COMMITTED", observation_count=2, next_iteration=2,
                failure_count=candidate["failure_count"], incumbent_iteration=0, simulation_started=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("inspect", "prepare", "preflight", "shell", "runtime", "run-slot", "evaluate", "commit"))
    p.add_argument("--campaign", type=Path, default=os.environ.get("BO_RECOVERY_CAMPAIGN"))
    p.add_argument("--recovery", type=Path, default=os.environ.get("BO_RECOVERY_DIR"))
    p.add_argument("--baseline", type=Path, default=os.environ.get("BO_RECOVERY_BASELINE",
        str(Path.home()/"sailboat_rl_results/bo/final_optimized_baseline/iter_008_original_params.json")))
    p.add_argument("--slot", choices=list(PENDING))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--docker", action="store_true", help="read-only daemon + pinned image inspection")
    p.add_argument("--retry-incomplete", action="store_true")
    a = p.parse_args(argv)
    try:
        docker_result = docker_preflight() if a.docker else None
        if a.action == "runtime":
            result = runtime_preflight()
        else:
            require(a.campaign is not None, "--campaign required")
            campaign = a.campaign.resolve(); baseline = a.baseline.resolve()
            if a.action == "inspect":
                result = inspect(campaign, baseline)
            else:
                require(a.recovery is not None, "--recovery required")
                recovery = a.recovery.resolve()
                # Do not retain a host directory lock across the interactive
                # shell: the container's run-slot must acquire that same inode.
                if a.action == "shell":
                    require(not a.dry_run, "shell has no dry-run action; use preflight --docker")
                    return shell(campaign, recovery, baseline)
                with locked(campaign):
                    if a.action == "prepare":
                        result = prepare(campaign, recovery, baseline, a.dry_run)
                    elif a.action == "preflight":
                        plan = check(campaign, recovery, baseline)
                        result = dict(preflight="PASS", pending={s: "COLLECTED" if completed_slot(recovery,s,plan)
                                      else "PENDING" for s in PENDING}, simulation_started=False)
                    elif a.action == "run-slot":
                        result = run_slot(campaign, recovery, baseline, a.slot, a.dry_run, a.retry_incomplete)
                    elif a.action == "evaluate":
                        result = evaluate(campaign, recovery, baseline, a.dry_run)
                    else:
                        result = commit(campaign, recovery, baseline, a.dry_run)
            if docker_result is not None:
                result["docker"] = docker_result
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ev.ContractError, OSError, ValueError, KeyError) as exc:
        print(json.dumps(dict(status="FAIL", error=type(exc).__name__, message=str(exc))), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('{"status":"INTERRUPTED","existing_attempt_preserved":true}', file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
