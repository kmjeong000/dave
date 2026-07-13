from __future__ import annotations

import random
import unittest

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

        objective, failure_count = run_bo.aggregate_objective(
            rows,
            objective_column="scenario_cost",
            expected_trial_count=3,
            missing_trial_penalty=10.0,
        )

        self.assertAlmostEqual(objective, (0.5 + 5.5 + 10.0) / 3.0)
        self.assertEqual(failure_count, 2)

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


if __name__ == "__main__":
    unittest.main()
