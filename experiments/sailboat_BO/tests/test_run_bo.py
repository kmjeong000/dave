from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from experiments.sailboat_BO import run_bo


class RunBoTests(unittest.TestCase):
    def test_parse_search_space_and_select_train_scenarios(self):
        config = {
            "search_space": [
                {"name": "A", "type": "float", "low": 0.0, "high": 1.0},
                {"name": "B", "type": "int", "low": 1, "high": 5},
            ],
            "scenarios": [
                {"id": "train_a", "split": "train"},
                {"id": "eval_b", "split": "eval"},
            ],
        }

        specs = run_bo.parse_search_space(config)
        selected = run_bo.select_scenario_ids(config, "train", None)

        self.assertEqual([spec.name for spec in specs], ["A", "B"])
        self.assertEqual(selected, ["train_a"])

    def test_aggregate_objective_penalizes_missing_trials(self):
        rows = [
            {"status": "success", "scenario_cost": "0.5"},
            {"status": "failure", "scenario_cost": "5.5"},
        ]

        objective, mean_cost, worst_cost, failure_count = run_bo.aggregate_objective(
            rows,
            objective_column="scenario_cost",
            expected_trial_count=3,
            missing_trial_penalty=10.0,
        )

        self.assertAlmostEqual(objective, (0.5 + 5.5 + 10.0) / 3.0)
        self.assertAlmostEqual(mean_cost, objective)
        self.assertEqual(worst_cost, 10.0)
        self.assertEqual(failure_count, 2)

    def test_aggregate_objective_blends_mean_and_worst_scenario(self):
        rows = [
            {"scenario_id": "easy", "status": "success", "scenario_cost": "0.2"},
            {"scenario_id": "easy", "status": "success", "scenario_cost": "0.4"},
            {"scenario_id": "hard", "status": "success", "scenario_cost": "0.8"},
            {"scenario_id": "hard", "status": "success", "scenario_cost": "1.0"},
        ]

        objective, mean_cost, worst_cost, failure_count = run_bo.aggregate_objective(
            rows,
            objective_column="scenario_cost",
            expected_trial_count=4,
            worst_case_weight=0.5,
        )

        self.assertAlmostEqual(mean_cost, 0.6)
        self.assertAlmostEqual(worst_cost, 0.9)
        self.assertAlmostEqual(objective, 0.75)
        self.assertEqual(failure_count, 0)

    def test_suggest_params_stays_inside_bounds(self):
        specs = [
            run_bo.ParamSpec("A", "float", 0.0, 1.0),
            run_bo.ParamSpec("B", "float", 10.0, 20.0),
        ]
        observations = [
            run_bo.Observation(0, {"A": 0.1, "B": 11.0}, 1.0),
            run_bo.Observation(1, {"A": 0.8, "B": 19.0}, 2.0),
        ]

        params = run_bo.suggest_params(
            observations,
            specs,
            random.Random(42),
            initial_random=1,
            candidate_pool=32,
        )

        self.assertGreaterEqual(params["A"], 0.0)
        self.assertLessEqual(params["A"], 1.0)
        self.assertGreaterEqual(params["B"], 10.0)
        self.assertLessEqual(params["B"], 20.0)

    def test_best_observation_prefers_fewer_failures_before_objective(self):
        observations = [
            run_bo.Observation(0, {"A": 0.1}, 0.2, failure_count=1),
            run_bo.Observation(1, {"A": 0.9}, 1.5, failure_count=0),
        ]

        best = run_bo.select_best_observation(observations)

        self.assertEqual(best.iteration, 1)

    def test_best_observation_uses_objective_when_failures_match(self):
        observations = [
            run_bo.Observation(0, {"A": 0.1}, 1.5, failure_count=0),
            run_bo.Observation(1, {"A": 0.9}, 0.2, failure_count=0),
        ]

        best = run_bo.select_best_observation(observations)

        self.assertEqual(best.iteration, 1)

    def test_load_history_restores_failure_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = Path(temp_dir) / "history.csv"
            history_path.write_text(
                "iteration,objective,failure_count,param__A\n"
                "3,0.25,2,0.75\n",
                encoding="utf-8",
            )

            observations = run_bo.load_history(
                history_path,
                [run_bo.ParamSpec("A", "float", 0.0, 1.0)],
            )

        self.assertEqual(
            observations,
            [
                run_bo.Observation(
                    iteration=3,
                    params={"A": 0.75},
                    objective=0.25,
                    failure_count=2,
                )
            ],
        )

    def test_load_params_for_specs_requires_complete_in_bounds_mapping(self):
        specs = [
            run_bo.ParamSpec("A", "float", 0.0, 1.0),
            run_bo.ParamSpec("B", "int", 1.0, 5.0),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "baseline.json"
            path.write_text('{"A": 0.25, "B": 3}', encoding="utf-8")
            params = run_bo.load_params_for_specs(path, specs)

        self.assertEqual(params, {"A": 0.25, "B": 3.0})

    def test_proposal_schedule_starts_with_baseline_then_local_and_global(self):
        specs = [run_bo.ParamSpec("A", "float", 0.0, 1.0)]
        baseline = {"A": 0.5}
        rng = random.Random(42)
        observations: list[run_bo.Observation] = []

        params, source = run_bo.propose_iteration_params(
            iteration=0,
            observations=observations,
            baseline_params=baseline,
            specs=specs,
            rng=rng,
            initial_random=3,
            local_warmup=2,
            local_radius=0.1,
            candidate_pool=8,
        )
        self.assertEqual((params, source), (baseline, "baseline"))
        observations.append(run_bo.Observation(0, params, 0.4))

        params, source = run_bo.propose_iteration_params(
            iteration=1,
            observations=observations,
            baseline_params=baseline,
            specs=specs,
            rng=rng,
            initial_random=3,
            local_warmup=2,
            local_radius=0.1,
            candidate_pool=8,
        )
        self.assertEqual(source, "local_warmup")
        self.assertGreaterEqual(params["A"], 0.4)
        self.assertLessEqual(params["A"], 0.6)
        observations.append(run_bo.Observation(1, params, 0.3))

        _, source = run_bo.propose_iteration_params(
            iteration=3,
            observations=observations,
            baseline_params=baseline,
            specs=specs,
            rng=rng,
            initial_random=3,
            local_warmup=2,
            local_radius=0.1,
            candidate_pool=8,
        )
        self.assertEqual(source, "global_warmup")

    def test_no_baseline_schedule_starts_global_then_uses_gp_ei(self):
        specs = [run_bo.ParamSpec("A", "float", 0.0, 1.0)]
        rng = random.Random(42)
        observations: list[run_bo.Observation] = []

        params, source = run_bo.propose_iteration_params(
            iteration=0,
            observations=observations,
            baseline_params=None,
            specs=specs,
            rng=rng,
            initial_random=2,
            local_warmup=0,
            local_radius=0.1,
            candidate_pool=8,
        )
        self.assertEqual(source, "global_warmup")
        self.assertGreaterEqual(params["A"], 0.0)
        self.assertLessEqual(params["A"], 1.0)
        observations.append(run_bo.Observation(0, params, 0.4))

        params, source = run_bo.propose_iteration_params(
            iteration=1,
            observations=observations,
            baseline_params=None,
            specs=specs,
            rng=rng,
            initial_random=2,
            local_warmup=0,
            local_radius=0.1,
            candidate_pool=8,
        )
        self.assertEqual(source, "global_warmup")
        observations.append(run_bo.Observation(1, params, 0.3))

        _, source = run_bo.propose_iteration_params(
            iteration=2,
            observations=observations,
            baseline_params=None,
            specs=specs,
            rng=rng,
            initial_random=2,
            local_warmup=0,
            local_radius=0.1,
            candidate_pool=8,
        )
        self.assertEqual(source, "gp_ei")


if __name__ == "__main__":
    unittest.main()
