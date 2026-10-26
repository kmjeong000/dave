from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.sailboat_BO import bo_campaign_evaluator as evaluator
from experiments.sailboat_BO import bo_campaign_state as state


REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = (
    REPO_ROOT
    / "experiments"
    / "sailboat_BO"
    / "bo_campaign_contract_v1_patch1.yaml"
)
SCENARIO_PATH = (
    REPO_ROOT / "experiments" / "sailboat_BO" / "scenario.yaml"
)


class BoCampaignStateTests(unittest.TestCase):
    def setUp(self):
        self.contract = evaluator.load_contract(CONTRACT_PATH)
        self.params = {
            item["name"]: 0.5 * (float(item["low"]) + float(item["high"]))
            for item in self.contract["search_space"]
        }

    def _provenance(self, baseline_path: Path) -> dict[str, object]:
        return {
            "contract_id": evaluator.CONTRACT_ID,
            "contract_patch_revision": 1,
            "contract_sha256": evaluator.sha256_file(CONTRACT_PATH),
            "source_commit": self.contract["accepted_parent"]["source_commit"],
            "runtime_image_id": self.contract["accepted_parent"][
                "runtime_image_id"
            ],
            "scenario_sha256": self.contract["scenario"]["sha256"],
            "framework_evaluator_sha256": self.contract["framework"][
                "evaluator_sha256"
            ],
            "post_trial_evaluator_sha256": self.contract[
                "post_trial_evaluator"
            ]["sha256"],
            "score_sha256": self.contract["evaluation"]["score_sha256"],
            "baseline_params_sha256": evaluator.sha256_file(baseline_path),
            "search_space": [
                {
                    "name": str(item["name"]),
                    "type": str(item["type"]),
                    "low": float(item["low"]),
                    "high": float(item["high"]),
                }
                for item in self.contract["search_space"]
            ],
        }

    def _source_artifact(
        self, root: Path, baseline_path: Path, provenance: dict[str, object]
    ) -> Path:
        candidate_dir = root / "source" / "trials"
        candidate_dir.mkdir(parents=True)
        (candidate_dir / "summary.csv").write_text(
            "scenario_id,status,scenario_cost\n",
            encoding="utf-8",
        )
        trials = []
        for scenario_id in self.contract["scenario"]["scenario_ids"]:
            for repeat_idx in range(2):
                trials.append(
                    {
                        "scenario_id": scenario_id,
                        "repeat_idx": repeat_idx,
                        "runtime": {"valid": True, "issues": []},
                        "layer1": {"status": "PASS", "issues": []},
                        "layer2": {"classification": "BOUNDED TRANSIENT"},
                        "layer3": {"status": "PASS", "issues": []},
                        "bo_feasible": True,
                        "failure_count": 0,
                        "effective_cost": 0.4,
                    }
                )
        payload = {
            "contract_id": evaluator.CONTRACT_ID,
            "contract_patch_revision": 1,
            "contract_sha256": provenance["contract_sha256"],
            "iteration": 0,
            "candidate_feasible": True,
            "objective": 0.3953604710399666,
            "mean_scenario_cost": 0.3953604710399666,
            "worst_scenario_cost": 0.5133847373454723,
            "failure_count": 0,
            "trial_count": 6,
            "expected_trial_count": 6,
            "batch_returncode": 0,
            "params": self.params,
            "params_sha256": evaluator.sha256_file(baseline_path),
            "provenance": provenance,
            "trials": trials,
        }
        path = candidate_dir / evaluator.PATCH1_CANDIDATE_EVALUATION_FILENAME
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _freeze(self, root: Path) -> tuple[Path, dict[str, object]]:
        baseline = root / "iter_008.json"
        baseline.write_text(json.dumps(self.params), encoding="utf-8")
        provenance = self._provenance(baseline)
        candidate = self._source_artifact(root, baseline, provenance)
        bo_dir = root / "patch1_campaign"
        with mock.patch.object(
            state, "validate_provenance", return_value=provenance
        ):
            state.freeze_iteration0_state(
                contract_path=CONTRACT_PATH,
                scenario_path=SCENARIO_PATH,
                baseline_params_path=baseline,
                candidate_evaluation_path=candidate,
                bo_dir=bo_dir,
                runtime_image_id=str(provenance["runtime_image_id"]),
                source_commit=str(provenance["source_commit"]),
            )
        return bo_dir, provenance

    def test_freeze_creates_one_corrected_observation_and_validates_resume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bo_dir, provenance = self._freeze(root)
            baseline = root / "iter_008.json"

            history = list(
                csv.DictReader(
                    (bo_dir / "history.csv").open(encoding="utf-8")
                )
            )
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["iteration"], "0")
            self.assertEqual(history[0]["failure_count"], "0")
            self.assertAlmostEqual(
                float(history[0]["objective"]), 0.3953604710399666
            )
            self.assertTrue((bo_dir / state.STATE_MANIFEST_FILENAME).is_file())
            self.assertTrue(
                (
                    bo_dir
                    / "iter_000"
                    / evaluator.PATCH1_CANDIDATE_EVALUATION_FILENAME
                ).is_file()
            )

            with mock.patch.object(
                state, "validate_provenance", return_value=provenance
            ):
                validated = state.validate_frozen_campaign_state(
                    contract_path=CONTRACT_PATH,
                    scenario_path=SCENARIO_PATH,
                    baseline_params_path=baseline,
                    runtime_image_id=str(provenance["runtime_image_id"]),
                    source_commit=str(provenance["source_commit"]),
                    bo_dir=bo_dir,
                    expected_optimizer_settings={
                        "seed": 42,
                        "repeats": 2,
                        "initial_random": 12,
                    },
                )

            self.assertEqual(validated["observation_count"], 1)
            self.assertEqual(validated["next_iteration"], 1)
            self.assertTrue(validated["same_frozen_state_continuation_allowed"])
            self.assertFalse(validated["external_or_historical_resume_allowed"])

    def test_external_or_old_history_without_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "history.csv").write_text(
                "iteration,objective,failure_count\n0,2.7207,3\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                evaluator.ContractError, "external/historical campaign resume"
            ):
                state.validate_frozen_campaign_state(
                    contract_path=CONTRACT_PATH,
                    scenario_path=SCENARIO_PATH,
                    baseline_params_path=root / "missing.json",
                    runtime_image_id=self.contract["accepted_parent"][
                        "runtime_image_id"
                    ],
                    source_commit=self.contract["accepted_parent"][
                        "source_commit"
                    ],
                    bo_dir=root,
                )

    def test_tampering_iteration0_with_old_v1_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bo_dir, provenance = self._freeze(root)
            history_path = bo_dir / "history.csv"
            history = history_path.read_text(encoding="utf-8")
            history_path.write_text(
                history.replace("0.3953604710399666", "2.7207").replace(
                    ",0,0,", ",3,0,"
                ),
                encoding="utf-8",
            )

            with mock.patch.object(
                state, "validate_provenance", return_value=provenance
            ), self.assertRaises(evaluator.ContractError):
                state.validate_frozen_campaign_state(
                    contract_path=CONTRACT_PATH,
                    scenario_path=SCENARIO_PATH,
                    baseline_params_path=root / "iter_008.json",
                    runtime_image_id=str(provenance["runtime_image_id"]),
                    source_commit=str(provenance["source_commit"]),
                    bo_dir=bo_dir,
                )

    def test_optimizer_setting_drift_blocks_same_state_resume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bo_dir, provenance = self._freeze(root)
            with mock.patch.object(
                state, "validate_provenance", return_value=provenance
            ), self.assertRaisesRegex(
                evaluator.ContractError, "optimizer setting mismatch"
            ):
                state.validate_frozen_campaign_state(
                    contract_path=CONTRACT_PATH,
                    scenario_path=SCENARIO_PATH,
                    baseline_params_path=root / "iter_008.json",
                    runtime_image_id=str(provenance["runtime_image_id"]),
                    source_commit=str(provenance["source_commit"]),
                    bo_dir=bo_dir,
                    expected_optimizer_settings={"seed": 43},
                )


if __name__ == "__main__":
    unittest.main()
