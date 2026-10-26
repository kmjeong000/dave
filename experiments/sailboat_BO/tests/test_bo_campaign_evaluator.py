from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.sailboat_BO import bo_campaign_evaluator as evaluator
from experiments.sailboat_BO import run_bo
from experiments.sailboat_physics.virtual_sheet_validation import (
    VirtualSheetSample,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = (
    REPO_ROOT
    / "experiments"
    / "sailboat_BO"
    / "bo_campaign_contract_v1.yaml"
)
PATCH1_CONTRACT_PATH = (
    REPO_ROOT
    / "experiments"
    / "sailboat_BO"
    / "bo_campaign_contract_v1_patch1.yaml"
)


class BoCampaignEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.contract = evaluator.load_contract(CONTRACT_PATH)
        self.params = {
            item["name"]: 0.5 * (float(item["low"]) + float(item["high"]))
            for item in self.contract["search_space"]
        }

    def _sample(self, **updates):
        values = {
            "sim_time_s": 1.0,
            "raw_allowance_rad": 0.70,
            "effective_allowance_rad": 0.70,
            "allowance_lag_rad": 0.0,
            "boom_angle_rad": 0.50,
            "boom_velocity_rad_s": 0.0,
            "penetration_rad": 0.0,
            "requested_torque_nm": 0.0,
            "applied_torque_nm": 0.0,
            "sheet_out_release": False,
            "tightening": False,
            "torque_capped": False,
        }
        values.update(updates)
        return VirtualSheetSample(**values)

    def _write_trial(
        self,
        root: Path,
        *,
        scenario_id: str = "train_crosswind_straight",
        repeat_idx: int = 0,
        hard_stop: bool = False,
    ) -> Path:
        trial = root / f"trial__{scenario_id}__r{repeat_idx:02d}"
        (trial / "summary").mkdir(parents=True)
        (trial / "raw").mkdir()
        (trial / "logs").mkdir()

        manifest = {
            "trial_id": trial.name,
            "scenario_id": scenario_id,
            "repeat_idx": repeat_idx,
            "params": self.params,
            "termination": {
                "timeout_s": 240.0,
                "min_speed_mps": 0.25,
            },
        }
        (trial / "trial_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        summary = {
            "metadata": {
                "status": "success",
                "failure_reason": "",
            },
            "metrics": {
                "mission_time_s": 20.0,
                "xte_rms_m": 0.5,
                "progress_ratio": 1.0,
                "rudder_total_variation_rad": 1.0,
                "sail_total_variation_rad": 1.0,
                "max_abs_roll_deg": 10.0,
                "max_abs_roll_deg_after_grace": 10.0,
                "upwind_sailing_ratio": 0.0,
            },
            "constraints": {
                "constraint_mission_complete": True,
                "constraint_timeout": False,
                "constraint_stuck": False,
                "constraint_no_progress": False,
                "constraint_excessive_roll": False,
            },
            "objective": {},
        }
        (trial / "summary" / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )

        fieldnames = [
            "sim_time_s",
            "x_m",
            "y_m",
            "wind_direction_rad",
            "target_bearing_rad",
        ]
        with (trial / "raw" / "samples.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "sim_time_s": 0.0,
                    "x_m": 0.0,
                    "y_m": 0.0,
                    "wind_direction_rad": 0.0,
                    "target_bearing_rad": math.pi,
                }
            )
            writer.writerow(
                {
                    "sim_time_s": 1.0,
                    "x_m": 0.0,
                    "y_m": -1.0,
                    "wind_direction_rad": 0.0,
                    "target_bearing_rad": math.pi,
                }
            )

        boom = 1.5708 if hard_stop else 0.50
        penetration = max(abs(boom) - 0.70, 0.0)
        requested = -1.0 if hard_stop else 0.0
        applied = requested
        lines = []
        for index, time_s in enumerate((1.000, 1.001, 1.002)):
            lines.append(
                "[VirtualSheetActuator][physics] "
                f"simTimeS={time_s:.3f} "
                "rawAllowanceRad=0.700000 "
                "effectiveAllowanceRad=0.700000 "
                "allowanceLagRad=0.000000 "
                f"boomAngleRad={boom:.6f} "
                "boomVelocityRadS=0.000000 "
                f"effectivePenetrationRad={penetration:.6f} "
                f"requestedTorqueNm={requested:.6f} "
                f"appliedTorqueNm={applied:.6f} "
                f"sheetTaut={'true' if penetration > 0.0 else 'false'} "
                "allowanceSheetOutRelease=false "
                "allowanceTightening=false "
                "torqueCapped=false\n"
            )
        (trial / "logs" / "launch.stdout.log").write_text(
            "".join(lines), encoding="utf-8"
        )
        return trial

    def test_contract_pins_accepted_parent_search_space_and_fixed_no_go(self):
        self.assertEqual(
            self.contract["accepted_parent"]["source_commit"],
            "4ec49538912fe88ca3be569c1a787dba4c07994b",
        )
        self.assertEqual(len(self.contract["search_space"]), 6)
        self.assertEqual(
            self.contract["evaluation"]["evaluation_no_go_angle_deg"],
            60.0,
        )
        self.assertTrue(
            self.contract["evaluation"][
                "optimized_parameter_affects_controller_only"
            ]
        )

    def test_patch1_contract_pins_evaluator_and_separate_artifacts(self):
        contract = evaluator.load_contract(PATCH1_CONTRACT_PATH)
        identity = contract["contract"]
        post_trial = contract["post_trial_evaluator"]

        self.assertEqual(identity["patch_revision"], 1)
        self.assertEqual(
            identity["parent_contract_sha256"],
            evaluator.sha256_file(CONTRACT_PATH),
        )
        self.assertEqual(
            post_trial["sha256"],
            evaluator.sha256_file(Path(evaluator.__file__)),
        )
        self.assertEqual(
            evaluator._trial_evaluation_filename(contract),
            evaluator.PATCH1_TRIAL_EVALUATION_FILENAME,
        )
        self.assertEqual(
            evaluator._candidate_evaluation_filename(contract),
            evaluator.PATCH1_CANDIDATE_EVALUATION_FILENAME,
        )

    def test_layer1_accepts_slack_contract_and_rejects_qeff_below_raw(self):
        valid = [
            self._sample(sim_time_s=1.0),
            self._sample(sim_time_s=1.001),
            self._sample(sim_time_s=1.002),
        ]
        result = evaluator.evaluate_layer1(
            valid,
            maximum_allowance_rad=0.7854,
            mechanical_limit_rad=math.pi / 2.0,
            reference_rad=math.radians(15.0),
        )
        self.assertEqual(result["status"], "PASS")

        invalid = list(valid)
        invalid[1] = self._sample(
            sim_time_s=1.001,
            raw_allowance_rad=0.70,
            effective_allowance_rad=0.60,
            allowance_lag_rad=-0.10,
        )
        result = evaluator.evaluate_layer1(
            invalid,
            maximum_allowance_rad=0.7854,
            mechanical_limit_rad=math.pi / 2.0,
            reference_rad=math.radians(15.0),
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any("q_eff < q_raw" in issue for issue in result["issues"])
        )

    def test_layer1_uses_plant_sheet_taut_semantics_at_tiny_penetration(self):
        tiny_taut = self._sample(
            boom_angle_rad=-6.8e-7,
            boom_velocity_rad_s=-0.035,
            raw_allowance_rad=0.0,
            effective_allowance_rad=0.0,
            penetration_rad=6.8e-7,
            requested_torque_nm=1.54,
            applied_torque_nm=1.54,
        )

        authoritative = evaluator.evaluate_layer1(
            [tiny_taut],
            maximum_allowance_rad=0.7854,
            mechanical_limit_rad=math.pi / 2.0,
            reference_rad=math.radians(15.0),
            sheet_taut_states=[True],
        )
        self.assertEqual(authoritative["status"], "PASS")
        self.assertEqual(
            authoritative["sheet_taut_authoritative_sample_count"], 1
        )

        fallback = evaluator.evaluate_layer1(
            [tiny_taut],
            maximum_allowance_rad=0.7854,
            mechanical_limit_rad=math.pi / 2.0,
            reference_rad=math.radians(15.0),
        )
        self.assertEqual(fallback["status"], "PASS")
        self.assertEqual(fallback["sheet_taut_fallback_sample_count"], 1)

    def test_layer1_rejects_nonzero_torque_in_authoritative_slack_state(self):
        slack_with_torque = self._sample(
            boom_angle_rad=0.0,
            raw_allowance_rad=0.0,
            effective_allowance_rad=0.0,
            penetration_rad=0.0,
            requested_torque_nm=0.25,
            applied_torque_nm=0.25,
        )
        result = evaluator.evaluate_layer1(
            [slack_with_torque],
            maximum_allowance_rad=0.7854,
            mechanical_limit_rad=math.pi / 2.0,
            reference_rad=math.radians(15.0),
            sheet_taut_states=[False],
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertIn(
            "nonzero sheet torque while slack at sample 0",
            result["issues"],
        )

    def test_feasible_trial_uses_fixed_no_go_and_preserves_control_field(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trial = self._write_trial(Path(temp_dir))
            result = evaluator.evaluate_trial(
                contract=self.contract,
                trial_dir=trial,
                expected_scenario_id="train_crosswind_straight",
                expected_repeat_idx=0,
                expected_params=self.params,
            )

            self.assertTrue(result["runtime"]["valid"])
            self.assertEqual(result["layer1"]["status"], "PASS")
            self.assertEqual(
                result["layer2"]["classification"], "BOUNDED TRANSIENT"
            )
            self.assertEqual(result["layer3"]["status"], "PASS")
            self.assertTrue(result["bo_feasible"])
            self.assertEqual(result["failure_count"], 0)
            self.assertEqual(
                result["objective"]["evaluation_no_go_angle_deg"], 60.0
            )
            self.assertEqual(
                result["objective"]["control_term"],
                result["objective"]["command_activity_term"],
            )
            self.assertTrue(
                (
                    trial
                    / "physics_diagnostics"
                    / evaluator.TRIAL_EVALUATION_FILENAME
                ).is_file()
            )
            self.assertTrue((trial / "raw" / "samples.csv").is_file())

    def test_patch1_trial_evaluation_does_not_overwrite_v1_artifact(self):
        patch_contract = evaluator.load_contract(PATCH1_CONTRACT_PATH)
        with tempfile.TemporaryDirectory() as temp_dir:
            trial = self._write_trial(Path(temp_dir))
            diagnostics = trial / "physics_diagnostics"
            diagnostics.mkdir()
            v1_artifact = diagnostics / evaluator.TRIAL_EVALUATION_FILENAME
            v1_artifact.write_text("frozen-v1\n", encoding="utf-8")

            result = evaluator.evaluate_trial(
                contract=patch_contract,
                trial_dir=trial,
                expected_scenario_id="train_crosswind_straight",
                expected_repeat_idx=0,
                expected_params=self.params,
            )

            self.assertTrue(result["bo_feasible"])
            self.assertEqual(
                v1_artifact.read_text(encoding="utf-8"), "frozen-v1\n"
            )
            self.assertTrue(
                (
                    diagnostics
                    / evaluator.PATCH1_TRIAL_EVALUATION_FILENAME
                ).is_file()
            )

    def test_optimized_no_go_parameter_cannot_change_evaluation_reference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            objectives = []
            for suffix, controller_no_go in (("low", 40.0), ("high", 70.0)):
                self.params["SAIL_NO_GO_ANGLE"] = controller_no_go
                trial = self._write_trial(root / suffix)
                result = evaluator.evaluate_trial(
                    contract=self.contract,
                    trial_dir=trial,
                    expected_scenario_id="train_crosswind_straight",
                    expected_repeat_idx=0,
                    expected_params=self.params,
                )
                objectives.append(result["objective"])

        self.assertEqual(
            objectives[0]["evaluation_no_go_angle_deg"], 60.0
        )
        self.assertEqual(
            objectives[1]["evaluation_no_go_angle_deg"], 60.0
        )
        self.assertEqual(
            objectives[0]["sailing_term"], objectives[1]["sailing_term"]
        )
        self.assertEqual(
            objectives[0]["scenario_cost"], objectives[1]["scenario_cost"]
        )

    def test_framework_failure_has_exact_cost_five_and_one_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trial = self._write_trial(Path(temp_dir), hard_stop=True)
            result = evaluator.evaluate_trial(
                contract=self.contract,
                trial_dir=trial,
                expected_scenario_id="train_crosswind_straight",
                expected_repeat_idx=0,
                expected_params=self.params,
            )

        self.assertTrue(result["runtime"]["valid"])
        self.assertFalse(result["bo_feasible"])
        self.assertEqual(result["effective_cost"], 5.0)
        self.assertEqual(result["failure_count"], 1)

    def test_missing_runtime_artifacts_have_cost_ten_and_one_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trial = Path(temp_dir) / "missing_trial"
            trial.mkdir()
            result = evaluator.evaluate_trial(
                contract=self.contract,
                trial_dir=trial,
                expected_scenario_id="train_crosswind_straight",
                expected_repeat_idx=0,
                expected_params=self.params,
            )

        self.assertFalse(result["runtime"]["valid"])
        self.assertEqual(result["effective_cost"], 10.0)
        self.assertEqual(result["failure_count"], 1)

    def test_missing_six_trial_candidate_aggregates_to_six_failures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            params_path = root / "params.json"
            params_path.write_text(json.dumps(self.params), encoding="utf-8")
            candidate_dir = root / "candidate"
            candidate_dir.mkdir()
            fake_provenance = {
                "contract_sha256": evaluator.sha256_file(CONTRACT_PATH),
                "source_commit": self.contract["accepted_parent"][
                    "source_commit"
                ],
                "runtime_image_id": self.contract["accepted_parent"][
                    "runtime_image_id"
                ],
            }
            with mock.patch.object(
                evaluator,
                "validate_provenance",
                return_value=fake_provenance,
            ):
                result = evaluator.evaluate_candidate(
                    contract_path=CONTRACT_PATH,
                    scenario_path=REPO_ROOT
                    / "experiments"
                    / "sailboat_BO"
                    / "scenario.yaml",
                    baseline_params_path=params_path,
                    params_path=params_path,
                    candidate_dir=candidate_dir,
                    runtime_image_id=self.contract["accepted_parent"][
                        "runtime_image_id"
                    ],
                    source_commit=self.contract["accepted_parent"][
                        "source_commit"
                    ],
                    iteration=3,
                )

        self.assertEqual(result["failure_count"], 6)
        self.assertEqual(result["expected_trial_count"], 6)
        self.assertEqual(result["trial_count"], 0)
        self.assertEqual(result["objective"], 10.0)
        self.assertFalse(result["candidate_feasible"])
        self.assertTrue(
            all(trial["failure_count"] == 1 for trial in result["trials"])
        )

    def test_campaign_settings_require_fresh_two_by_three_policy(self):
        evaluator.validate_run_settings(
            contract=self.contract,
            scenario_ids=self.contract["scenario"]["scenario_ids"],
            repeats=2,
            seed=42,
            objective_column="scenario_cost",
            worst_case_weight=0.0,
            resume=False,
            no_baseline=False,
            trial_dry_run=False,
            stop_on_failure=False,
        )
        with self.assertRaises(evaluator.ContractError):
            evaluator.validate_run_settings(
                contract=self.contract,
                scenario_ids=self.contract["scenario"]["scenario_ids"],
                repeats=1,
                seed=42,
                objective_column="scenario_cost",
                worst_case_weight=0.0,
                resume=False,
                no_baseline=False,
                trial_dry_run=False,
                stop_on_failure=False,
            )

    def test_run_bo_iteration_consumes_single_campaign_aggregate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            aggregate = {
                "objective": 0.75,
                "mean_scenario_cost": 0.75,
                "worst_scenario_cost": 0.90,
                "failure_count": 0,
                "trial_count": 6,
            }
            with mock.patch.object(
                run_bo.subprocess,
                "run",
                return_value=mock.Mock(returncode=0),
            ), mock.patch.object(
                run_bo,
                "evaluate_candidate",
                return_value=aggregate,
            ) as post_trial:
                result = run_bo.run_iteration(
                    iteration=0,
                    params=self.params,
                    specs=[
                        run_bo.ParamSpec(
                            item["name"],
                            item["type"],
                            float(item["low"]),
                            float(item["high"]),
                        )
                        for item in self.contract["search_space"]
                    ],
                    bo_dir=Path(temp_dir),
                    scenario_file="scenario.yaml",
                    scenario_ids=self.contract["scenario"]["scenario_ids"],
                    repeats=2,
                    objective_column="scenario_cost",
                    worst_case_weight=0.0,
                    execution_backend="local",
                    docker_container=None,
                    container_repo_root=None,
                    trial_dry_run=False,
                    keep_going=True,
                    campaign_contract=str(CONTRACT_PATH),
                    baseline_params_file="iter_008.json",
                    runtime_image_id=self.contract["accepted_parent"][
                        "runtime_image_id"
                    ],
                    source_commit=self.contract["accepted_parent"][
                        "source_commit"
                    ],
                )

        post_trial.assert_called_once()
        self.assertEqual(result.trial_count, 6)
        self.assertEqual(result.failure_count, 0)
        self.assertEqual(result.objective, 0.75)


if __name__ == "__main__":
    unittest.main()
