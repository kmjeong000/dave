"""Freeze and validate BO campaign state rooted in corrected patch1 iteration 0.

This module deliberately does not evaluate trials or propose parameters.  It
turns one already-evaluated patch1 reference artifact into the only admissible
root for a resumable campaign, and validates that root before ``run_bo.py`` may
continue at iteration 1 or later.

The original v1 result tree remains read-only.  A frozen state contains a
snapshot of the corrected candidate artifact and an immutable manifest that
binds it to the accepted source, runtime image, contract, framework evaluator,
post-trial evaluator, scenario, score implementation, and iter_008 parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .bo_campaign_evaluator import (
        CONTRACT_ID,
        PATCH1_CANDIDATE_EVALUATION_FILENAME,
        PATCH1_REVISION,
        ContractError,
        load_contract,
        sha256_file,
        validate_provenance,
        write_json,
    )
except ImportError:
    from bo_campaign_evaluator import (
        CONTRACT_ID,
        PATCH1_CANDIDATE_EVALUATION_FILENAME,
        PATCH1_REVISION,
        ContractError,
        load_contract,
        sha256_file,
        validate_provenance,
        write_json,
    )


STATE_FORMAT = "bo_campaign_contract_v1_patch1_state_v1"
STATE_MANIFEST_FILENAME = "campaign_state.json"
STATE_MANIFEST_SHA_FILENAME = "campaign_state.json.sha256"
DEFAULT_INITIAL_RANDOM = 12
DEFAULT_LOCAL_WARMUP = 6
DEFAULT_LOCAL_RADIUS = 0.15
DEFAULT_CANDIDATE_POOL = 1024


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ContractError(
                f"expected JSON object at {path}:{line_number}"
            )
        results.append(payload)
    return results


def _read_history(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _numbers_match(left: Any, right: Any, *, tolerance: float = 1.0e-12) -> bool:
    try:
        return math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=tolerance
        )
    except (TypeError, ValueError):
        return False


def _assert_numeric_mapping_equal(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    context: str,
) -> None:
    if set(actual) != set(expected):
        raise ContractError(f"{context} parameter names differ")
    for name, expected_value in expected.items():
        if not _numbers_match(actual[name], expected_value):
            raise ContractError(
                f"{context} parameter mismatch: "
                f"{name}={actual[name]!r} != {expected_value!r}"
            )


def _search_space_names(contract: Mapping[str, Any]) -> list[str]:
    return [str(item["name"]) for item in contract["search_space"]]


def _iteration_result_from_candidate(
    candidate: Mapping[str, Any], source_summary_csv: Path
) -> dict[str, Any]:
    return {
        "iteration": 0,
        "params": dict(candidate["params"]),
        "objective": float(candidate["objective"]),
        "mean_scenario_cost": float(candidate["mean_scenario_cost"]),
        "worst_scenario_cost": float(candidate["worst_scenario_cost"]),
        "trial_count": int(candidate["trial_count"]),
        "expected_trial_count": int(candidate["expected_trial_count"]),
        "failure_count": int(candidate["failure_count"]),
        "returncode": int(candidate["batch_returncode"]),
        "summary_csv": str(source_summary_csv.resolve()),
    }


def _write_history(
    path: Path,
    result: Mapping[str, Any],
    parameter_names: Sequence[str],
) -> None:
    fieldnames = [
        "iteration",
        "objective",
        "mean_scenario_cost",
        "worst_scenario_cost",
        "trial_count",
        "expected_trial_count",
        "failure_count",
        "returncode",
        "summary_csv",
    ] + [f"param__{name}" for name in parameter_names]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        row = {name: result[name] for name in fieldnames if name in result}
        for name in parameter_names:
            row[f"param__{name}"] = result["params"][name]
        writer.writerow(row)


def _candidate_contract_checks(
    *,
    candidate: Mapping[str, Any],
    contract: Mapping[str, Any],
    provenance: Mapping[str, Any],
    baseline_params: Mapping[str, Any],
    baseline_params_path: Path,
) -> None:
    expected_trials = len(contract["scenario"]["scenario_ids"]) * int(
        contract["scenario"]["repeats_per_scenario"]
    )
    if candidate.get("contract_id") != CONTRACT_ID:
        raise ContractError("iteration-0 artifact contract ID mismatch")
    if int(candidate.get("contract_patch_revision", -1)) != PATCH1_REVISION:
        raise ContractError("iteration-0 artifact is not patch1")
    if candidate.get("contract_sha256") != provenance["contract_sha256"]:
        raise ContractError("iteration-0 artifact contract SHA mismatch")
    if int(candidate.get("iteration", -1)) != 0:
        raise ContractError("iteration-0 artifact has a different iteration")
    if not bool(candidate.get("candidate_feasible")):
        raise ContractError("iteration-0 corrected candidate is not feasible")
    if int(candidate.get("failure_count", -1)) != 0:
        raise ContractError("iteration-0 corrected failure_count is not zero")
    if int(candidate.get("trial_count", -1)) != expected_trials:
        raise ContractError("iteration-0 corrected trial_count mismatch")
    if int(candidate.get("expected_trial_count", -1)) != expected_trials:
        raise ContractError("iteration-0 expected trial count mismatch")
    if len(candidate.get("trials", [])) != expected_trials:
        raise ContractError("iteration-0 trial artifact count mismatch")
    if candidate.get("params_sha256") != sha256_file(baseline_params_path):
        raise ContractError("iteration-0 params SHA is not iter_008")
    _assert_numeric_mapping_equal(
        candidate.get("params", {}),
        baseline_params,
        context="iteration-0",
    )

    candidate_provenance = candidate.get("provenance", {})
    for field in (
        "contract_sha256",
        "source_commit",
        "runtime_image_id",
        "scenario_sha256",
        "framework_evaluator_sha256",
        "post_trial_evaluator_sha256",
        "score_sha256",
        "baseline_params_sha256",
    ):
        if candidate_provenance.get(field) != provenance.get(field):
            raise ContractError(
                f"iteration-0 provenance mismatch: {field}"
            )

    framework = contract["framework"]
    for index, trial in enumerate(candidate["trials"]):
        if not bool(trial.get("runtime", {}).get("valid")):
            raise ContractError(f"iteration-0 trial {index} runtime invalid")
        if trial.get("layer1", {}).get("status") != framework["required_layer1"]:
            raise ContractError(f"iteration-0 trial {index} Layer 1 mismatch")
        if (
            trial.get("layer2", {}).get("classification")
            != framework["required_layer2"]
        ):
            raise ContractError(f"iteration-0 trial {index} Layer 2 mismatch")
        if trial.get("layer3", {}).get("status") != framework["required_layer3"]:
            raise ContractError(f"iteration-0 trial {index} Layer 3 mismatch")
        if not bool(trial.get("bo_feasible")):
            raise ContractError(f"iteration-0 trial {index} is infeasible")
        if int(trial.get("failure_count", -1)) != 0:
            raise ContractError(f"iteration-0 trial {index} failure_count mismatch")


def freeze_iteration0_state(
    *,
    contract_path: str | Path,
    scenario_path: str | Path,
    baseline_params_path: str | Path,
    candidate_evaluation_path: str | Path,
    bo_dir: str | Path,
    runtime_image_id: str,
    source_commit: str | None = None,
    initial_random: int = DEFAULT_INITIAL_RANDOM,
    local_warmup: int = DEFAULT_LOCAL_WARMUP,
    local_radius: float = DEFAULT_LOCAL_RADIUS,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
) -> dict[str, Any]:
    """Create a fresh, resumable patch1 campaign rooted only in iteration 0."""

    contract_path = Path(contract_path).resolve()
    scenario_path = Path(scenario_path).resolve()
    baseline_params_path = Path(baseline_params_path).resolve()
    candidate_evaluation_path = Path(candidate_evaluation_path).resolve()
    bo_dir = Path(bo_dir).resolve()

    if bo_dir.exists() and any(bo_dir.iterdir()):
        raise ContractError(f"patch1 campaign directory is not empty: {bo_dir}")
    if candidate_evaluation_path.name != PATCH1_CANDIDATE_EVALUATION_FILENAME:
        raise ContractError(
            "iteration-0 source must be candidate_evaluation_v1_patch1.json"
        )

    contract = load_contract(contract_path)
    if int(contract.get("contract", {}).get("patch_revision", 0)) != PATCH1_REVISION:
        raise ContractError("frozen continuation requires patch1 contract")
    provenance = validate_provenance(
        contract_path=contract_path,
        scenario_path=scenario_path,
        baseline_params_path=baseline_params_path,
        runtime_image_id=runtime_image_id,
        source_commit=source_commit,
    )
    candidate = _read_json(candidate_evaluation_path)
    baseline_params = _read_json(baseline_params_path)
    _candidate_contract_checks(
        candidate=candidate,
        contract=contract,
        provenance=provenance,
        baseline_params=baseline_params,
        baseline_params_path=baseline_params_path,
    )

    source_summary_csv = candidate_evaluation_path.parent / "summary.csv"
    if not source_summary_csv.is_file():
        raise ContractError(
            f"iteration-0 source summary is missing: {source_summary_csv}"
        )

    parameter_names = _search_space_names(contract)
    iteration_result = _iteration_result_from_candidate(
        candidate, source_summary_csv
    )
    optimizer_settings = {
        "initial_random": int(initial_random),
        "local_warmup": int(local_warmup),
        "local_radius": float(local_radius),
        "candidate_pool": int(candidate_pool),
        "repeats": int(contract["scenario"]["repeats_per_scenario"]),
        "seed": int(contract["optimization"]["seed"]),
        "objective_column": str(contract["optimization"]["objective_column"]),
        "worst_case_weight": float(
            contract["optimization"]["worst_case_weight"]
        ),
        "no_baseline": False,
        "trial_dry_run": False,
        "stop_on_failure": False,
    }

    bo_dir.mkdir(parents=True, exist_ok=True)
    iteration_dir = bo_dir / "iter_000"
    iteration_dir.mkdir()
    contract_snapshot = bo_dir / contract_path.name
    shutil.copyfile(contract_path, contract_snapshot)
    (bo_dir / f"{contract_path.name}.sha256").write_text(
        provenance["contract_sha256"] + "\n", encoding="utf-8"
    )
    shutil.copyfile(baseline_params_path, iteration_dir / "params.json")
    candidate_snapshot = (
        iteration_dir / PATCH1_CANDIDATE_EVALUATION_FILENAME
    )
    shutil.copyfile(candidate_evaluation_path, candidate_snapshot)
    write_json(
        iteration_dir / "proposal.json",
        {
            "iteration": 0,
            "source": "fresh_iter_008_reference_from_patch1_artifact",
            "seed": optimizer_settings["seed"],
            "params": baseline_params,
            "gp_proposal_generated": False,
        },
    )
    write_json(iteration_dir / "iteration_result.json", iteration_result)
    _write_history(
        bo_dir / "history.csv", iteration_result, parameter_names
    )
    (bo_dir / "observations.jsonl").write_text(
        json.dumps(iteration_result, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_json(
        bo_dir / "best_params.json",
        {
            "iteration": 0,
            "objective": iteration_result["objective"],
            "failure_count": 0,
            "params": baseline_params,
        },
    )
    write_json(
        bo_dir / "bo_config.json",
        {
            "campaign_state_format": STATE_FORMAT,
            "campaign_contract": str(contract_snapshot),
            "scenario": str(scenario_path),
            "scenario_ids": list(contract["scenario"]["scenario_ids"]),
            "split": "train",
            "baseline_policy": "fresh_iteration_0_reference",
            "baseline_name": str(contract["optimization"]["baseline_name"]),
            "baseline_params_file": str(baseline_params_path),
            "baseline_params": baseline_params,
            "search_space": provenance["search_space"],
            "optimizer_settings": optimizer_settings,
            "provenance": provenance,
            "continuation": {
                "external_or_historical_history_resume_allowed": False,
                "same_frozen_patch1_state_resume_allowed": True,
                "next_iteration": 1,
            },
        },
    )

    immutable_relative_paths = [
        contract_path.name,
        f"{contract_path.name}.sha256",
        "bo_config.json",
        "iter_000/params.json",
        "iter_000/proposal.json",
        "iter_000/iteration_result.json",
        f"iter_000/{PATCH1_CANDIDATE_EVALUATION_FILENAME}",
    ]
    manifest = {
        "format": STATE_FORMAT,
        "status": "FROZEN_ITERATION_0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "id": CONTRACT_ID,
            "patch_revision": PATCH1_REVISION,
            "sha256": provenance["contract_sha256"],
        },
        "accepted_parent": {
            "source_commit": provenance["source_commit"],
            "runtime_image_id": provenance["runtime_image_id"],
        },
        "provenance": provenance,
        "iteration_0": {
            "source_candidate_evaluation": str(candidate_evaluation_path),
            "source_candidate_evaluation_sha256": sha256_file(
                candidate_evaluation_path
            ),
            "snapshot": str(
                Path("iter_000") / PATCH1_CANDIDATE_EVALUATION_FILENAME
            ),
            "params_sha256": sha256_file(baseline_params_path),
            "objective": iteration_result["objective"],
            "failure_count": 0,
            "candidate_feasible": True,
            "trial_count": iteration_result["trial_count"],
            "expected_trial_count": iteration_result[
                "expected_trial_count"
            ],
        },
        "continuation": {
            "external_or_historical_history_resume_allowed": False,
            "same_frozen_patch1_state_resume_allowed": True,
            "required_history_root_iteration": 0,
            "next_iteration": 1,
        },
        "immutable_files": {
            relative: sha256_file(bo_dir / relative)
            for relative in immutable_relative_paths
        },
    }
    manifest_path = bo_dir / STATE_MANIFEST_FILENAME
    write_json(manifest_path, manifest)
    manifest_sha = sha256_file(manifest_path)
    (bo_dir / STATE_MANIFEST_SHA_FILENAME).write_text(
        manifest_sha + "\n", encoding="utf-8"
    )
    manifest["manifest_sha256"] = manifest_sha
    manifest["campaign_dir"] = str(bo_dir)
    return manifest


def _assert_iteration_record_matches(
    *,
    history_row: Mapping[str, Any],
    observation: Mapping[str, Any],
    iteration_result: Mapping[str, Any],
    parameter_names: Sequence[str],
) -> None:
    iteration = int(iteration_result["iteration"])
    if int(float(history_row["iteration"])) != iteration:
        raise ContractError(f"history iteration mismatch at {iteration}")
    if int(observation["iteration"]) != iteration:
        raise ContractError(f"observation iteration mismatch at {iteration}")
    scalar_fields = (
        "objective",
        "mean_scenario_cost",
        "worst_scenario_cost",
        "trial_count",
        "expected_trial_count",
        "failure_count",
        "returncode",
    )
    for field in scalar_fields:
        expected = iteration_result[field]
        if not _numbers_match(history_row[field], expected):
            raise ContractError(
                f"history {field} mismatch at iteration {iteration}"
            )
        if not _numbers_match(observation[field], expected):
            raise ContractError(
                f"observation {field} mismatch at iteration {iteration}"
            )
    if history_row["summary_csv"] != str(iteration_result["summary_csv"]):
        raise ContractError(
            f"history summary path mismatch at iteration {iteration}"
        )
    if observation["summary_csv"] != iteration_result["summary_csv"]:
        raise ContractError(
            f"observation summary path mismatch at iteration {iteration}"
        )
    _assert_numeric_mapping_equal(
        observation.get("params", {}),
        iteration_result.get("params", {}),
        context=f"iteration {iteration} observation",
    )
    for name in parameter_names:
        if not _numbers_match(
            history_row[f"param__{name}"],
            iteration_result["params"][name],
        ):
            raise ContractError(
                f"history parameter mismatch at iteration {iteration}: {name}"
            )


def _validate_best(
    *,
    best: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
) -> None:
    expected = min(
        observations,
        key=lambda item: (
            int(item["failure_count"]),
            float(item["objective"]),
            int(item["iteration"]),
        ),
    )
    if int(best.get("iteration", -1)) != int(expected["iteration"]):
        raise ContractError("best_params iteration does not match history")
    if int(best.get("failure_count", -1)) != int(expected["failure_count"]):
        raise ContractError("best_params failure_count does not match history")
    if not _numbers_match(best.get("objective"), expected["objective"]):
        raise ContractError("best_params objective does not match history")
    _assert_numeric_mapping_equal(
        best.get("params", {}),
        expected.get("params", {}),
        context="best_params",
    )


def validate_frozen_campaign_state(
    *,
    contract_path: str | Path,
    scenario_path: str | Path,
    baseline_params_path: str | Path,
    runtime_image_id: str,
    bo_dir: str | Path,
    source_commit: str | None = None,
    expected_optimizer_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate that continuation belongs to this exact frozen patch1 state."""

    bo_dir = Path(bo_dir).resolve()
    manifest_path = bo_dir / STATE_MANIFEST_FILENAME
    manifest_sha_path = bo_dir / STATE_MANIFEST_SHA_FILENAME
    if not manifest_path.is_file() or not manifest_sha_path.is_file():
        raise ContractError(
            "external/historical campaign resume forbidden: "
            "frozen patch1 campaign_state manifest is missing"
        )
    expected_manifest_sha = manifest_sha_path.read_text(
        encoding="utf-8"
    ).strip()
    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != expected_manifest_sha:
        raise ContractError("campaign_state manifest SHA mismatch")
    manifest = _read_json(manifest_path)
    if manifest.get("format") != STATE_FORMAT:
        raise ContractError("unsupported campaign_state format")
    if manifest.get("status") != "FROZEN_ITERATION_0":
        raise ContractError("campaign_state is not a frozen iteration-0 root")

    contract = load_contract(contract_path)
    if int(contract.get("contract", {}).get("patch_revision", 0)) != PATCH1_REVISION:
        raise ContractError("only patch1 frozen campaign state may continue")
    provenance = validate_provenance(
        contract_path=contract_path,
        scenario_path=scenario_path,
        baseline_params_path=baseline_params_path,
        runtime_image_id=runtime_image_id,
        source_commit=source_commit,
    )
    for field in (
        "contract_sha256",
        "source_commit",
        "runtime_image_id",
        "scenario_sha256",
        "framework_evaluator_sha256",
        "post_trial_evaluator_sha256",
        "score_sha256",
        "baseline_params_sha256",
    ):
        if manifest.get("provenance", {}).get(field) != provenance.get(field):
            raise ContractError(f"campaign_state provenance mismatch: {field}")

    for relative, expected_sha in manifest.get("immutable_files", {}).items():
        path = bo_dir / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ContractError(f"immutable campaign file mismatch: {relative}")

    config = _read_json(bo_dir / "bo_config.json")
    frozen_settings = config.get("optimizer_settings", {})
    if expected_optimizer_settings is not None:
        for field, expected in expected_optimizer_settings.items():
            actual = frozen_settings.get(field)
            if isinstance(expected, (int, float)) and not isinstance(
                expected, bool
            ):
                matches = _numbers_match(actual, expected)
            else:
                matches = actual == expected
            if not matches:
                raise ContractError(
                    f"continuation optimizer setting mismatch: "
                    f"{field}={actual!r} != {expected!r}"
                )

    history = _read_history(bo_dir / "history.csv")
    observations = _read_jsonl(bo_dir / "observations.jsonl")
    if not history or len(history) != len(observations):
        raise ContractError("campaign history/observation length mismatch")
    expected_iterations = list(range(len(observations)))
    actual_iterations = [int(item.get("iteration", -1)) for item in observations]
    if actual_iterations != expected_iterations:
        raise ContractError("campaign observations are not contiguous from iteration 0")

    parameter_names = _search_space_names(contract)
    for iteration, (history_row, observation) in enumerate(
        zip(history, observations)
    ):
        result_path = bo_dir / f"iter_{iteration:03d}" / "iteration_result.json"
        if not result_path.is_file():
            raise ContractError(
                f"missing iteration_result for iteration {iteration}"
            )
        iteration_result = _read_json(result_path)
        _assert_iteration_record_matches(
            history_row=history_row,
            observation=observation,
            iteration_result=iteration_result,
            parameter_names=parameter_names,
        )

        if iteration == 0:
            candidate_path = (
                bo_dir
                / "iter_000"
                / PATCH1_CANDIDATE_EVALUATION_FILENAME
            )
        else:
            candidate_path = (
                bo_dir
                / f"iter_{iteration:03d}"
                / "trials"
                / PATCH1_CANDIDATE_EVALUATION_FILENAME
            )
        if not candidate_path.is_file():
            raise ContractError(
                f"missing patch1 candidate artifact for iteration {iteration}"
            )
        candidate = _read_json(candidate_path)
        if int(candidate.get("iteration", -1)) != iteration:
            raise ContractError(
                f"candidate artifact iteration mismatch at {iteration}"
            )
        if candidate.get("contract_sha256") != provenance["contract_sha256"]:
            raise ContractError(
                f"candidate contract mismatch at iteration {iteration}"
            )
        for field in (
            "objective",
            "mean_scenario_cost",
            "worst_scenario_cost",
            "failure_count",
            "trial_count",
            "expected_trial_count",
        ):
            if not _numbers_match(candidate.get(field), iteration_result[field]):
                raise ContractError(
                    f"candidate {field} mismatch at iteration {iteration}"
                )
        _assert_numeric_mapping_equal(
            candidate.get("params", {}),
            iteration_result.get("params", {}),
            context=f"iteration {iteration} candidate",
        )

    initial = manifest["iteration_0"]
    first = observations[0]
    if int(first["failure_count"]) != 0:
        raise ContractError("frozen iteration 0 failure_count is not zero")
    if not _numbers_match(first["objective"], initial["objective"]):
        raise ContractError("frozen iteration 0 objective changed")
    if sha256_file(
        bo_dir / "iter_000" / PATCH1_CANDIDATE_EVALUATION_FILENAME
    ) != initial["source_candidate_evaluation_sha256"]:
        raise ContractError("frozen iteration-0 candidate artifact changed")

    _validate_best(
        best=_read_json(bo_dir / "best_params.json"),
        observations=observations,
    )
    return {
        "state_format": STATE_FORMAT,
        "manifest_sha256": actual_manifest_sha,
        "provenance": provenance,
        "observation_count": len(observations),
        "next_iteration": len(observations),
        "iteration_0": dict(initial),
        "same_frozen_state_continuation_allowed": True,
        "external_or_historical_resume_allowed": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze or validate a patch1 BO campaign state"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--contract", type=Path, required=True)
    common.add_argument("--scenario", type=Path, required=True)
    common.add_argument("--baseline-params", type=Path, required=True)
    common.add_argument("--runtime-image-id", required=True)
    common.add_argument("--source-commit")
    common.add_argument("--bo-dir", type=Path, required=True)

    freeze = subparsers.add_parser("freeze-iteration0", parents=[common])
    freeze.add_argument("--candidate-evaluation", type=Path, required=True)
    freeze.add_argument("--initial-random", type=int, default=DEFAULT_INITIAL_RANDOM)
    freeze.add_argument("--local-warmup", type=int, default=DEFAULT_LOCAL_WARMUP)
    freeze.add_argument("--local-radius", type=float, default=DEFAULT_LOCAL_RADIUS)
    freeze.add_argument("--candidate-pool", type=int, default=DEFAULT_CANDIDATE_POOL)

    subparsers.add_parser("validate", parents=[common])
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    kwargs = {
        "contract_path": args.contract,
        "scenario_path": args.scenario,
        "baseline_params_path": args.baseline_params,
        "runtime_image_id": args.runtime_image_id,
        "source_commit": args.source_commit,
        "bo_dir": args.bo_dir,
    }
    if args.command == "freeze-iteration0":
        result = freeze_iteration0_state(
            **kwargs,
            candidate_evaluation_path=args.candidate_evaluation,
            initial_random=args.initial_random,
            local_warmup=args.local_warmup,
            local_radius=args.local_radius,
            candidate_pool=args.candidate_pool,
        )
    else:
        result = validate_frozen_campaign_state(**kwargs)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
