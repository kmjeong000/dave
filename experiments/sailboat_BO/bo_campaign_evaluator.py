"""Frozen post-trial feasibility and aggregation for BO campaign v1.

The runtime trial summary remains a useful raw artifact, but it is not the BO
observation contract.  This module is the single post-trial interface that:

* validates immutable campaign provenance;
* evaluates framework-v2 Layers 1, 2 and 3;
* recomputes sailing metrics with a fixed 60 degree no-go reference;
* applies exactly one failure and one effective cost per expected trial; and
* aggregates the six expected trials into one BO observation.

Raw telemetry is never deleted or rewritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from .common import load_yaml
    from .score import compute_objective, compute_sailing_behavior_metrics
except ImportError:
    from common import load_yaml
    from score import compute_objective, compute_sailing_behavior_metrics

from experiments.sailboat_physics.virtual_sheet_validation import (
    BOUNDED_TRANSIENT,
    FRAMEWORK_V2,
    VirtualSheetSample,
    evaluate_compliance,
    parse_virtual_sheet_logs,
)


CONTRACT_ID = "bo_campaign_contract_v1"
TRIAL_EVALUATION_FILENAME = "bo_campaign_evaluation_v1.json"
CANDIDATE_EVALUATION_FILENAME = "candidate_evaluation.json"
PATCH1_REVISION = 1
PATCH1_TRIAL_EVALUATION_FILENAME = (
    "bo_campaign_evaluation_v1_patch1.json"
)
PATCH1_CANDIDATE_EVALUATION_FILENAME = (
    "candidate_evaluation_v1_patch1.json"
)
_RUNTIME_INVALID_PATTERN = re.compile(
    r"(?:paused|pause)=true|action=(?:PAUSE|OnPause)|"
    r"BrokenPipeError|runtime_exception"
)
_JSON_SENSOR_PATTERN = re.compile(r"No JSON sensor message received")
_INVALID_STATE_PATTERN = re.compile(
    r"\[VirtualSheetActuator\]\[physics\].*"
    r"(?:=[+-]?(?:nan|inf)(?:\s|$))",
    re.IGNORECASE,
)
_VIRTUAL_SHEET_MARKER = "[VirtualSheetActuator][physics]"
_ANSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_SIM_TIME_PATTERN = re.compile(
    r"(?:^|\s)simTimeS=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(?=\s|$)"
)
_SHEET_TAUT_PATTERN = re.compile(
    r"(?:^|\s)sheetTaut=(true|false)(?=\s|$)"
)


class ContractError(RuntimeError):
    """The requested run does not match the frozen campaign contract."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_contract(path: str | Path) -> dict[str, Any]:
    contract = load_yaml(path)
    identity = contract.get("contract", {})
    if identity.get("id") != CONTRACT_ID or int(identity.get("version", 0)) != 1:
        raise ContractError(f"unsupported campaign contract: {identity!r}")
    return contract


def _patch_revision(contract: Mapping[str, Any]) -> int:
    return int(contract.get("contract", {}).get("patch_revision", 0))


def _trial_evaluation_filename(contract: Mapping[str, Any]) -> str:
    if _patch_revision(contract) == PATCH1_REVISION:
        return PATCH1_TRIAL_EVALUATION_FILENAME
    return TRIAL_EVALUATION_FILENAME


def _candidate_evaluation_filename(contract: Mapping[str, Any]) -> str:
    if _patch_revision(contract) == PATCH1_REVISION:
        return PATCH1_CANDIDATE_EVALUATION_FILENAME
    return CANDIDATE_EVALUATION_FILENAME


def _normalized_search_space(items: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": str(item["name"]),
            "type": str(item.get("type", "float")).lower(),
            "low": float(item["low"]),
            "high": float(item["high"]),
        }
        for item in items
    ]


def _git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_provenance(
    *,
    contract_path: str | Path,
    scenario_path: str | Path,
    baseline_params_path: str | Path,
    runtime_image_id: str,
    source_commit: str | None = None,
) -> dict[str, Any]:
    """Fail fast when source, runtime, scenario, evaluator, or seed drifted."""

    contract_path = Path(contract_path).resolve()
    scenario_path = Path(scenario_path).resolve()
    baseline_params_path = Path(baseline_params_path).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    contract = load_contract(contract_path)

    accepted = contract["accepted_parent"]
    scenario_contract = contract["scenario"]
    framework = contract["framework"]
    optimization = contract["optimization"]
    evaluation = contract["evaluation"]
    post_trial_evaluator = contract.get("post_trial_evaluator")

    actual_source_commit = source_commit or _git_head(repo_root)
    if actual_source_commit != accepted["source_commit"]:
        raise ContractError(
            "source commit mismatch: "
            f"{actual_source_commit} != {accepted['source_commit']}"
        )
    if runtime_image_id != accepted["runtime_image_id"]:
        raise ContractError(
            "runtime image mismatch: "
            f"{runtime_image_id} != {accepted['runtime_image_id']}"
        )

    actual_scenario_sha = sha256_file(scenario_path)
    if actual_scenario_sha != scenario_contract["sha256"]:
        raise ContractError(
            "scenario SHA mismatch: "
            f"{actual_scenario_sha} != {scenario_contract['sha256']}"
        )

    evaluator_path = repo_root / framework["evaluator_path"]
    actual_evaluator_sha = sha256_file(evaluator_path)
    if actual_evaluator_sha != framework["evaluator_sha256"]:
        raise ContractError(
            "framework evaluator SHA mismatch: "
            f"{actual_evaluator_sha} != {framework['evaluator_sha256']}"
        )

    actual_post_trial_evaluator_sha = None
    if post_trial_evaluator is not None:
        post_trial_evaluator_path = (
            repo_root / post_trial_evaluator["path"]
        )
        actual_post_trial_evaluator_sha = sha256_file(
            post_trial_evaluator_path
        )
        if actual_post_trial_evaluator_sha != post_trial_evaluator["sha256"]:
            raise ContractError(
                "post-trial evaluator SHA mismatch: "
                f"{actual_post_trial_evaluator_sha} != "
                f"{post_trial_evaluator['sha256']}"
            )

    score_path = repo_root / evaluation["score_path"]
    actual_score_sha = sha256_file(score_path)
    if actual_score_sha != evaluation["score_sha256"]:
        raise ContractError(
            "score implementation SHA mismatch: "
            f"{actual_score_sha} != {evaluation['score_sha256']}"
        )

    actual_baseline_sha = sha256_file(baseline_params_path)
    if actual_baseline_sha != optimization["baseline_params_sha256"]:
        raise ContractError(
            "baseline parameter SHA mismatch: "
            f"{actual_baseline_sha} != "
            f"{optimization['baseline_params_sha256']}"
        )

    scenario_config = load_yaml(scenario_path)
    if _normalized_search_space(scenario_config.get("search_space", [])) != (
        _normalized_search_space(contract.get("search_space", []))
    ):
        raise ContractError("scenario search space differs from the contract")

    selected_ids = [
        str(item["id"])
        for item in scenario_config.get("scenarios", [])
        if str(item.get("id")) in set(scenario_contract["scenario_ids"])
    ]
    if selected_ids != list(scenario_contract["scenario_ids"]):
        raise ContractError(
            f"scenario order/set mismatch: {selected_ids!r} != "
            f"{scenario_contract['scenario_ids']!r}"
        )
    if int(scenario_config.get("study", {}).get("seed", -1)) != int(
        optimization["seed"]
    ):
        raise ContractError("scenario seed differs from the contract")
    if int(
        scenario_config.get("study", {}).get("repeats_per_scenario", -1)
    ) != int(scenario_contract["repeats_per_scenario"]):
        raise ContractError("scenario repeat policy differs from the contract")

    return {
        "contract_id": CONTRACT_ID,
        "contract_patch_revision": _patch_revision(contract),
        "contract_sha256": sha256_file(contract_path),
        "source_commit": actual_source_commit,
        "runtime_image_id": runtime_image_id,
        "scenario_sha256": actual_scenario_sha,
        "framework_evaluator_sha256": actual_evaluator_sha,
        "post_trial_evaluator_sha256": actual_post_trial_evaluator_sha,
        "score_sha256": actual_score_sha,
        "baseline_params_sha256": actual_baseline_sha,
        "search_space": _normalized_search_space(
            scenario_config["search_space"]
        ),
    }


def validate_fresh_history(bo_dir: str | Path) -> None:
    path = Path(bo_dir)
    if path.exists() and any(path.iterdir()):
        raise ContractError(
            f"fresh BO history required; directory is not empty: {path}"
        )


def validate_campaign_preflight(
    *,
    contract_path: str | Path,
    scenario_path: str | Path,
    baseline_params_path: str | Path,
    runtime_image_id: str,
    bo_dir: str | Path,
    source_commit: str | None = None,
) -> dict[str, Any]:
    provenance = validate_provenance(
        contract_path=contract_path,
        scenario_path=scenario_path,
        baseline_params_path=baseline_params_path,
        runtime_image_id=runtime_image_id,
        source_commit=source_commit,
    )
    validate_fresh_history(bo_dir)
    return provenance


def validate_run_settings(
    *,
    contract: Mapping[str, Any],
    scenario_ids: Sequence[str],
    repeats: int,
    seed: int,
    objective_column: str,
    worst_case_weight: float,
    resume: bool,
    no_baseline: bool,
    trial_dry_run: bool,
    stop_on_failure: bool,
) -> None:
    scenario_contract = contract["scenario"]
    optimization = contract["optimization"]
    errors: list[str] = []
    if list(scenario_ids) != list(scenario_contract["scenario_ids"]):
        errors.append("scenario IDs/order")
    if repeats != int(scenario_contract["repeats_per_scenario"]):
        errors.append("repeats")
    if seed != int(optimization["seed"]):
        errors.append("seed")
    if objective_column != optimization["objective_column"]:
        errors.append("objective column")
    if not math.isclose(
        worst_case_weight,
        float(optimization["worst_case_weight"]),
        abs_tol=1.0e-12,
    ):
        errors.append("worst-case weight")
    if resume:
        errors.append("resume is forbidden by fresh-history policy")
    if no_baseline:
        errors.append("iteration-0 iter_008 reference is required")
    if trial_dry_run:
        errors.append("dry-run trials cannot become BO observations")
    if stop_on_failure:
        errors.append("all six trials must be collected for aggregation")
    if errors:
        raise ContractError(
            "run settings violate campaign contract: " + ", ".join(errors)
        )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _framework_serializable(evaluation: Any) -> dict[str, Any]:
    payload = asdict(evaluation)
    payload["reference_deg"] = math.degrees(evaluation.reference_rad)
    payload["max_penetration_deg"] = math.degrees(
        evaluation.max_penetration_rad
    )
    payload["excess_integral_deg_s"] = math.degrees(
        evaluation.excess_integral_rad_s
    )
    for episode in payload["episodes"]:
        episode["peak_penetration_deg"] = math.degrees(
            episode["peak_penetration_rad"]
        )
    return payload


def parse_sheet_taut_states(
    paths: Iterable[Path],
    samples: Sequence[VirtualSheetSample],
) -> tuple[bool | None, ...]:
    """Return telemetry sheet-taut states aligned with parsed physics samples.

    Framework-v2 parsing is intentionally unchanged.  This BO contract patch
    reads the plant's explicit ``sheetTaut`` field in parallel and falls back
    to exact penetration semantics when that field is absent or cannot be
    aligned safely.
    """

    parsed: list[tuple[float, int, bool | None]] = []
    sequence = 0
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8", errors="replace") as stream:
            for raw_line in stream:
                line = _ANSI_PATTERN.sub("", raw_line)
                if _VIRTUAL_SHEET_MARKER not in line:
                    continue
                time_match = _SIM_TIME_PATTERN.search(line)
                if time_match is None:
                    continue
                taut_match = _SHEET_TAUT_PATTERN.search(line)
                sheet_taut = (
                    None
                    if taut_match is None
                    else taut_match.group(1) == "true"
                )
                parsed.append(
                    (float(time_match.group(1)), sequence, sheet_taut)
                )
                sequence += 1

    parsed.sort(key=lambda item: (item[0], item[1]))
    if len(parsed) != len(samples):
        return tuple(None for _ in samples)

    states: list[bool | None] = []
    for sample, (sim_time_s, _, sheet_taut) in zip(samples, parsed):
        if not math.isclose(
            sample.sim_time_s,
            sim_time_s,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            states.append(None)
        else:
            states.append(sheet_taut)
    return tuple(states)


def evaluate_layer1(
    samples: Sequence[VirtualSheetSample],
    *,
    maximum_allowance_rad: float,
    mechanical_limit_rad: float,
    reference_rad: float,
    tolerance_rad: float = 2.0e-6,
    sheet_taut_states: Sequence[bool | None] | None = None,
) -> dict[str, Any]:
    """Evaluate structural virtual-sheet validity without quality thresholds."""

    issues: list[str] = []
    if not samples:
        return {"status": "FAIL", "issues": ["missing virtual-sheet telemetry"]}

    if sheet_taut_states is None or len(sheet_taut_states) != len(samples):
        sheet_taut_states = tuple(None for _ in samples)

    authoritative_taut_count = sum(
        state is not None for state in sheet_taut_states
    )
    for index, sample in enumerate(samples):
        numeric_values = (
            sample.sim_time_s,
            sample.raw_allowance_rad,
            sample.effective_allowance_rad,
            sample.allowance_lag_rad,
            sample.boom_angle_rad,
            sample.boom_velocity_rad_s,
            sample.penetration_rad,
            sample.requested_torque_nm,
            sample.applied_torque_nm,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            issues.append(f"non-finite telemetry at sample {index}")
            continue
        if sample.raw_allowance_rad < -tolerance_rad or (
            sample.raw_allowance_rad
            > maximum_allowance_rad + tolerance_rad
        ):
            issues.append(f"raw allowance outside contract at sample {index}")
        if sample.effective_allowance_rad + tolerance_rad < (
            sample.raw_allowance_rad
        ):
            issues.append(f"q_eff < q_raw at sample {index}")
        if sample.effective_allowance_rad > (
            maximum_allowance_rad + tolerance_rad
        ):
            issues.append(f"q_eff exceeds maximum allowance at sample {index}")
        expected_lag = (
            sample.effective_allowance_rad - sample.raw_allowance_rad
        )
        if abs(expected_lag - sample.allowance_lag_rad) > tolerance_rad:
            issues.append(f"allowance lag mismatch at sample {index}")
        expected_penetration = max(
            abs(sample.boom_angle_rad) - sample.effective_allowance_rad,
            0.0,
        )
        if abs(expected_penetration - sample.penetration_rad) > tolerance_rad:
            issues.append(f"penetration mismatch at sample {index}")
        if abs(sample.boom_angle_rad) >= (
            mechanical_limit_rad - tolerance_rad
        ):
            issues.append(f"mechanical hard-stop contact at sample {index}")
        if (
            sample.applied_torque_nm * sample.boom_angle_rad
            > tolerance_rad
        ):
            issues.append(f"outward sheet torque at sample {index}")
        sheet_taut = sheet_taut_states[index]
        is_slack = (
            not sheet_taut
            if sheet_taut is not None
            else sample.penetration_rad <= 0.0
        )
        if is_slack and abs(sample.applied_torque_nm) > 1.0e-4:
            issues.append(f"nonzero sheet torque while slack at sample {index}")
        if sample.sheet_out_release and (
            sample.tightening
            or abs(
                sample.effective_allowance_rad
                - sample.raw_allowance_rad
            )
            > tolerance_rad
        ):
            issues.append(f"invalid sheet-out release at sample {index}")

        if index == 0:
            continue
        previous = samples[index - 1]
        if sample.sim_time_s <= previous.sim_time_s:
            issues.append(f"non-monotonic simulation time at sample {index}")
        if (
            sample.tightening
            and sample.effective_allowance_rad
            > previous.effective_allowance_rad + tolerance_rad
        ):
            issues.append(
                f"q_eff increased during tightening at sample {index}"
            )
        if sample.sheet_out_release and (
            sample.effective_allowance_rad
            <= previous.effective_allowance_rad + tolerance_rad
        ):
            issues.append(
                f"sheet-out release did not expand allowance at sample {index}"
            )

    fixed_raw_divergence = False
    run_start = 0
    for end in range(1, len(samples) + 1):
        boundary = end == len(samples) or (
            abs(
                samples[end].raw_allowance_rad
                - samples[end - 1].raw_allowance_rad
            )
            > tolerance_rad
        )
        if not boundary:
            continue
        run = samples[run_start:end]
        if len(run) >= 3 and run[0].allowance_lag_rad > tolerance_rad:
            lag_deltas = [
                later.allowance_lag_rad - earlier.allowance_lag_rad
                for earlier, later in zip(run, run[1:])
            ]
            if lag_deltas and all(delta > tolerance_rad for delta in lag_deltas):
                fixed_raw_divergence = True
        run_start = end
    if fixed_raw_divergence:
        issues.append("fixed-raw allowance lag diverged")

    if samples[-1].torque_capped and (
        samples[-1].penetration_rad > reference_rad
    ):
        issues.append("torque saturation remained unrecovered at trial end")

    unique_issues = list(dict.fromkeys(issues))
    return {
        "status": "PASS" if not unique_issues else "FAIL",
        "issues": unique_issues,
        "sample_count": len(samples),
        "sheet_taut_authoritative_sample_count": authoritative_taut_count,
        "sheet_taut_fallback_sample_count": (
            len(samples) - authoritative_taut_count
        ),
        "minimum_mechanical_limit_margin_rad": min(
            mechanical_limit_rad - abs(sample.boom_angle_rad)
            for sample in samples
        ),
    }


def evaluate_layer3(summary: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(summary.get("metadata", {}))
    constraints = dict(summary.get("constraints", {}))
    issues: list[str] = []
    if metadata.get("status") != "success":
        issues.append(
            f"trial status is {metadata.get('status')!r}, not 'success'"
        )
    if not constraints.get("constraint_mission_complete", False):
        issues.append("mission did not complete")
    for key in (
        "constraint_timeout",
        "constraint_stuck",
        "constraint_no_progress",
        "constraint_excessive_roll",
    ):
        if constraints.get(key, False):
            issues.append(key)
    return {
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
    }


def _runtime_validity(
    trial_dir: Path,
    *,
    summary: Mapping[str, Any] | None,
    raw_samples: Sequence[Mapping[str, Any]],
    framework_samples: Sequence[VirtualSheetSample],
    log_paths: Sequence[Path],
) -> dict[str, Any]:
    issues: list[str] = []
    required = (
        trial_dir / "trial_manifest.json",
        trial_dir / "summary" / "summary.json",
        trial_dir / "raw" / "samples.csv",
    )
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            issues.append(f"missing artifact: {path.relative_to(trial_dir)}")
    if summary is None:
        issues.append("summary is unavailable")
    if not raw_samples:
        issues.append("raw sample telemetry is empty")
    if not log_paths:
        issues.append("launch logs are missing")
    if not framework_samples:
        issues.append("framework-v2 telemetry is missing")
    elif any(
        later.sim_time_s <= earlier.sim_time_s
        for earlier, later in zip(framework_samples, framework_samples[1:])
    ):
        issues.append("framework-v2 telemetry time is not strictly increasing")

    invalid_markers: list[str] = []
    for path in log_paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        invalid_markers.extend(
            match.group(0) for match in _RUNTIME_INVALID_PATTERN.finditer(text)
        )
        if _INVALID_STATE_PATTERN.search(text):
            invalid_markers.append("invalid virtual-sheet numeric state")
        if summary is None or summary.get("metadata", {}).get("status") != "success":
            invalid_markers.extend(
                match.group(0) for match in _JSON_SENSOR_PATTERN.finditer(text)
            )
    if invalid_markers:
        issues.append(
            "runtime-invalid log markers: "
            + ", ".join(sorted(set(invalid_markers)))
        )
    if summary is not None and (
        summary.get("metadata", {}).get("failure_reason")
        == "runtime_exception"
    ):
        issues.append("summary reports runtime_exception")

    return {
        "valid": not issues,
        "issues": issues,
        "invalid_log_markers": sorted(set(invalid_markers)),
    }


def _fixed_objective(
    *,
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    raw_samples: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = dict(summary.get("metrics", {}))
    termination = dict(manifest.get("termination", {}))
    no_go_angle_deg = float(
        contract["evaluation"]["evaluation_no_go_angle_deg"]
    )
    sailing_metrics = compute_sailing_behavior_metrics(
        raw_samples,
        {
            "no_go_angle_deg": no_go_angle_deg,
            "no_go_grace_s": 3.0,
            "tack_min_hold_s": 3.0,
            "min_course_speed_mps": max(
                0.05,
                0.5 * float(termination.get("min_speed_mps", 0.25)),
            ),
        },
    )
    metrics.update(sailing_metrics)
    constraints = dict(summary.get("constraints", {}))
    objective = compute_objective(metrics, constraints, termination)
    objective["command_activity_term"] = objective["control_term"]
    objective["control_term_interpretation"] = "command_activity_term"
    objective["evaluation_no_go_angle_deg"] = no_go_angle_deg
    return metrics, objective


def _assert_trial_identity(
    *,
    manifest: Mapping[str, Any],
    expected_scenario_id: str,
    expected_repeat_idx: int,
    expected_params: Mapping[str, float],
) -> None:
    if manifest.get("scenario_id") != expected_scenario_id:
        raise ContractError("trial scenario ID does not match expected slot")
    if int(manifest.get("repeat_idx", -1)) != expected_repeat_idx:
        raise ContractError("trial repeat index does not match expected slot")
    actual_params = manifest.get("params", {})
    if set(actual_params) != set(expected_params):
        raise ContractError("trial parameter names do not match candidate")
    for name, expected in expected_params.items():
        actual = float(actual_params[name])
        if not math.isclose(
            actual, float(expected), rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ContractError(
                f"trial parameter mismatch: {name}={actual} != {expected}"
            )


def evaluate_trial(
    *,
    contract: Mapping[str, Any],
    trial_dir: Path,
    expected_scenario_id: str,
    expected_repeat_idx: int,
    expected_params: Mapping[str, float],
) -> dict[str, Any]:
    """Evaluate exactly one existing trial and persist one compact artifact."""

    manifest_path = trial_dir / "trial_manifest.json"
    summary_path = trial_dir / "summary" / "summary.json"
    samples_path = trial_dir / "raw" / "samples.csv"
    log_paths = sorted((trial_dir / "logs").glob("launch.*.log"))

    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    if manifest:
        _assert_trial_identity(
            manifest=manifest,
            expected_scenario_id=expected_scenario_id,
            expected_repeat_idx=expected_repeat_idx,
            expected_params=expected_params,
        )
    summary = _read_json(summary_path) if summary_path.is_file() else None
    raw_samples = _read_csv(samples_path) if samples_path.is_file() else []
    framework_samples = parse_virtual_sheet_logs(log_paths)
    sheet_taut_states = parse_sheet_taut_states(
        log_paths, framework_samples
    )
    runtime = _runtime_validity(
        trial_dir,
        summary=summary,
        raw_samples=raw_samples,
        framework_samples=framework_samples,
        log_paths=log_paths,
    )

    framework_failure_cost = float(
        contract["penalty_policy"]["framework_failure_cost"]
    )
    runtime_invalid_cost = float(
        contract["penalty_policy"][
            "runtime_invalid_or_missing_artifact_cost"
        ]
    )

    payload: dict[str, Any] = {
        "contract_id": CONTRACT_ID,
        "contract_patch_revision": _patch_revision(contract),
        "scenario_id": expected_scenario_id,
        "repeat_idx": expected_repeat_idx,
        "trial_dir": str(trial_dir),
        "runtime": runtime,
        "layer1": {"status": "NOT_EVALUATED", "issues": []},
        "layer2": {
            "classification": "NOT_EVALUATED",
            "framework_version": contract["framework"]["version"],
        },
        "layer3": {"status": "NOT_EVALUATED", "issues": []},
        "bo_feasible": False,
        "failure_count": 1,
        "effective_cost": runtime_invalid_cost,
        "objective": None,
        "artifact_sha256": {},
    }

    for path in (manifest_path, summary_path, samples_path, *log_paths):
        if path.is_file():
            payload["artifact_sha256"][
                str(path.relative_to(trial_dir))
            ] = sha256_file(path)

    if runtime["valid"]:
        framework = contract["framework"]
        reference_rad = math.radians(
            float(framework["penetration_reference_deg"])
        )
        layer1 = evaluate_layer1(
            framework_samples,
            maximum_allowance_rad=math.radians(
                float(framework["maximum_allowance_deg"])
            ),
            mechanical_limit_rad=math.radians(
                float(framework["mechanical_limit_deg"])
            ),
            reference_rad=reference_rad,
            sheet_taut_states=sheet_taut_states,
        )
        compliance = evaluate_compliance(
            framework_samples,
            framework_version=str(framework["version"]),
            reference_rad=reference_rad,
            maximum_allowance_rad=math.radians(
                float(framework["maximum_allowance_deg"])
            ),
        )
        layer2 = _framework_serializable(compliance)
        layer3 = evaluate_layer3(summary or {})
        _, objective = _fixed_objective(
            contract=contract,
            manifest=manifest,
            summary=summary or {},
            raw_samples=raw_samples,
        )
        feasible = (
            layer1["status"] == framework["required_layer1"]
            and compliance.classification == framework["required_layer2"]
            and layer3["status"] == framework["required_layer3"]
        )
        payload.update(
            {
                "layer1": layer1,
                "layer2": layer2,
                "layer3": layer3,
                "bo_feasible": feasible,
                "failure_count": 0 if feasible else 1,
                "effective_cost": (
                    float(objective["scenario_cost"])
                    if feasible
                    else framework_failure_cost
                ),
                "objective": objective,
            }
        )

    output_path = trial_dir / "physics_diagnostics" / (
        _trial_evaluation_filename(contract)
    )
    write_json(output_path, payload)
    payload["evaluation_artifact"] = str(output_path)
    return payload


def _discover_trials(
    candidate_dir: Path,
) -> dict[tuple[str, int], Path]:
    discovered: dict[tuple[str, int], Path] = {}
    for manifest_path in sorted(candidate_dir.glob("*/trial_manifest.json")):
        manifest = _read_json(manifest_path)
        key = (
            str(manifest.get("scenario_id")),
            int(manifest.get("repeat_idx", -1)),
        )
        if key in discovered:
            raise ContractError(f"duplicate trial slot: {key!r}")
        discovered[key] = manifest_path.parent
    return discovered


def evaluate_candidate(
    *,
    contract_path: str | Path,
    scenario_path: str | Path,
    baseline_params_path: str | Path,
    params_path: str | Path,
    candidate_dir: str | Path,
    runtime_image_id: str,
    source_commit: str | None = None,
    iteration: int | None = None,
    batch_returncode: int = 0,
) -> dict[str, Any]:
    """Evaluate and aggregate all six expected trials into one BO observation."""

    provenance = validate_provenance(
        contract_path=contract_path,
        scenario_path=scenario_path,
        baseline_params_path=baseline_params_path,
        runtime_image_id=runtime_image_id,
        source_commit=source_commit,
    )
    contract_path = Path(contract_path).resolve()
    contract = load_contract(contract_path)
    params_path = Path(params_path).resolve()
    candidate_dir = Path(candidate_dir).resolve()
    params = _read_json(params_path)
    expected_names = {
        item["name"] for item in contract["search_space"]
    }
    if set(params) != expected_names:
        raise ContractError("candidate parameter names differ from search space")

    discovered = _discover_trials(candidate_dir)
    expected_slots = [
        (scenario_id, repeat_idx)
        for scenario_id in contract["scenario"]["scenario_ids"]
        for repeat_idx in range(
            int(contract["scenario"]["repeats_per_scenario"])
        )
    ]
    unexpected = sorted(set(discovered) - set(expected_slots))
    if unexpected:
        raise ContractError(f"unexpected trial slots: {unexpected!r}")

    runtime_invalid_cost = float(
        contract["penalty_policy"][
            "runtime_invalid_or_missing_artifact_cost"
        ]
    )
    trials: list[dict[str, Any]] = []
    for scenario_id, repeat_idx in expected_slots:
        trial_dir = discovered.get((scenario_id, repeat_idx))
        if trial_dir is None:
            trials.append(
                {
                    "contract_id": CONTRACT_ID,
                    "contract_patch_revision": _patch_revision(contract),
                    "scenario_id": scenario_id,
                    "repeat_idx": repeat_idx,
                    "trial_dir": None,
                    "runtime": {
                        "valid": False,
                        "issues": ["missing trial/evaluator artifact"],
                    },
                    "layer1": {"status": "NOT_EVALUATED", "issues": []},
                    "layer2": {
                        "classification": "NOT_EVALUATED",
                        "framework_version": contract["framework"]["version"],
                    },
                    "layer3": {
                        "status": "NOT_EVALUATED",
                        "issues": [],
                    },
                    "bo_feasible": False,
                    "failure_count": 1,
                    "effective_cost": runtime_invalid_cost,
                    "objective": None,
                    "artifact_sha256": {},
                    "evaluation_artifact": None,
                }
            )
            continue
        trials.append(
            evaluate_trial(
                contract=contract,
                trial_dir=trial_dir,
                expected_scenario_id=scenario_id,
                expected_repeat_idx=repeat_idx,
                expected_params=params,
            )
        )

    scenario_costs: dict[str, list[float]] = {}
    for trial in trials:
        scenario_costs.setdefault(trial["scenario_id"], []).append(
            float(trial["effective_cost"])
        )
    costs = [float(trial["effective_cost"]) for trial in trials]
    mean_cost = sum(costs) / len(costs)
    worst_scenario_cost = max(
        sum(values) / len(values) for values in scenario_costs.values()
    )
    worst_weight = float(
        contract["optimization"]["worst_case_weight"]
    )
    aggregate_objective = (
        (1.0 - worst_weight) * mean_cost
        + worst_weight * worst_scenario_cost
    )
    failure_count = sum(
        min(1, int(trial["failure_count"])) for trial in trials
    )

    result = {
        "contract_id": CONTRACT_ID,
        "contract_patch_revision": _patch_revision(contract),
        "contract_sha256": provenance["contract_sha256"],
        "iteration": iteration,
        "candidate_dir": str(candidate_dir),
        "params_file": str(params_path),
        "params_sha256": sha256_file(params_path),
        "params": params,
        "provenance": provenance,
        "evaluation_no_go_angle_deg": float(
            contract["evaluation"]["evaluation_no_go_angle_deg"]
        ),
        "control_term_interpretation": "command_activity_term",
        "candidate_feasible": failure_count == 0,
        "objective": aggregate_objective,
        "mean_scenario_cost": mean_cost,
        "worst_scenario_cost": worst_scenario_cost,
        "failure_count": failure_count,
        "trial_count": len(discovered),
        "expected_trial_count": len(expected_slots),
        "batch_returncode": int(batch_returncode),
        "trials": trials,
    }

    candidate_dir.mkdir(parents=True, exist_ok=True)
    output_path = candidate_dir / _candidate_evaluation_filename(contract)
    write_json(output_path, result)
    shutil.copyfile(contract_path, candidate_dir / contract_path.name)
    (candidate_dir / f"{contract_path.name}.sha256").write_text(
        provenance["contract_sha256"] + "\n",
        encoding="utf-8",
    )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BO campaign v1 provenance and post-trial evaluator"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--contract", type=Path, required=True)
    preflight.add_argument("--scenario", type=Path, required=True)
    preflight.add_argument("--baseline-params", type=Path, required=True)
    preflight.add_argument("--runtime-image-id", required=True)
    preflight.add_argument("--source-commit")
    preflight.add_argument("--bo-dir", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate-candidate")
    evaluate.add_argument("--contract", type=Path, required=True)
    evaluate.add_argument("--scenario", type=Path, required=True)
    evaluate.add_argument("--baseline-params", type=Path, required=True)
    evaluate.add_argument("--params-file", type=Path, required=True)
    evaluate.add_argument("--candidate-dir", type=Path, required=True)
    evaluate.add_argument("--runtime-image-id", required=True)
    evaluate.add_argument("--source-commit")
    evaluate.add_argument("--iteration", type=int)
    evaluate.add_argument("--batch-returncode", type=int, default=0)
    evaluate.add_argument("--json-out", type=Path)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "preflight":
        provenance = validate_campaign_preflight(
            contract_path=args.contract,
            scenario_path=args.scenario,
            baseline_params_path=args.baseline_params,
            runtime_image_id=args.runtime_image_id,
            source_commit=args.source_commit,
            bo_dir=args.bo_dir,
        )
        print(json.dumps(provenance, indent=2, sort_keys=True))
        return 0

    result = evaluate_candidate(
        contract_path=args.contract,
        scenario_path=args.scenario,
        baseline_params_path=args.baseline_params,
        params_path=args.params_file,
        candidate_dir=args.candidate_dir,
        runtime_image_id=args.runtime_image_id,
        source_commit=args.source_commit,
        iteration=args.iteration,
        batch_returncode=args.batch_returncode,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_out is not None:
        write_json(args.json_out, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
