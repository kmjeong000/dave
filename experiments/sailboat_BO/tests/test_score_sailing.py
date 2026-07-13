from __future__ import annotations

import math
import unittest

from experiments.sailboat_BO import score


def sample(
    sim_time_s: float,
    x_m: float,
    y_m: float,
    *,
    wind_direction_rad: float = 0.0,
    target_bearing_rad: float = math.pi,
) -> dict[str, float]:
    return {
        "sim_time_s": sim_time_s,
        "x_m": x_m,
        "y_m": y_m,
        "wind_direction_rad": wind_direction_rad,
        "target_bearing_rad": target_bearing_rad,
    }


class SailingBehaviorMetricTests(unittest.TestCase):
    def test_upwind_zigzag_counts_tacks_without_no_go_violation(self):
        samples = [
            sample(0.0, 0.0, 0.0),
            sample(10.0, 8.66, -5.0),
            sample(20.0, -8.66, -15.0),
            sample(30.0, 8.66, -25.0),
        ]

        metrics = score.compute_sailing_behavior_metrics(
            samples,
            {
                "no_go_angle_deg": 45.0,
                "no_go_grace_s": 3.0,
                "tack_min_hold_s": 3.0,
                "min_course_speed_mps": 0.05,
                "min_course_delta_m": 0.01,
            },
        )

        self.assertEqual(metrics["tack_count"], 2.0)
        self.assertEqual(metrics["upwind_tack_count"], 2.0)
        self.assertAlmostEqual(metrics["upwind_sailing_time_s"], 30.0)
        self.assertAlmostEqual(metrics["upwind_sailing_ratio"], 1.0)
        self.assertAlmostEqual(metrics["no_go_violation_time_s"], 0.0)
        self.assertAlmostEqual(metrics["upwind_no_go_violation_ratio"], 0.0)

    def test_straight_upwind_course_records_no_go_violation_time(self):
        samples = [
            sample(0.0, 0.0, 0.0),
            sample(10.0, 0.0, -10.0),
        ]

        metrics = score.compute_sailing_behavior_metrics(
            samples,
            {
                "no_go_angle_deg": 45.0,
                "no_go_grace_s": 3.0,
                "tack_min_hold_s": 3.0,
                "min_course_speed_mps": 0.05,
                "min_course_delta_m": 0.01,
            },
        )

        self.assertEqual(metrics["tack_count"], 0.0)
        self.assertAlmostEqual(metrics["moving_course_time_s"], 10.0)
        self.assertAlmostEqual(metrics["no_go_violation_time_s"], 7.0)
        self.assertAlmostEqual(metrics["no_go_violation_ratio"], 0.7)
        self.assertAlmostEqual(metrics["upwind_no_go_violation_time_s"], 7.0)

    def test_short_no_go_crossing_inside_grace_is_not_counted_as_violation(self):
        samples = [
            sample(0.0, 0.0, 0.0),
            sample(2.0, 0.0, -2.0),
        ]

        metrics = score.compute_sailing_behavior_metrics(
            samples,
            {
                "no_go_angle_deg": 45.0,
                "no_go_grace_s": 3.0,
                "min_course_speed_mps": 0.05,
                "min_course_delta_m": 0.01,
            },
        )

        self.assertAlmostEqual(metrics["no_go_violation_time_s"], 0.0)
        self.assertAlmostEqual(metrics["upwind_no_go_violation_time_s"], 0.0)

    def test_tack_count_requires_new_side_to_hold_long_enough(self):
        samples = [
            sample(0.0, 0.0, 0.0),
            sample(10.0, 8.66, -5.0),
            sample(11.0, 6.93, -6.0),
            sample(20.0, 14.72, -10.5),
        ]

        metrics = score.compute_sailing_behavior_metrics(
            samples,
            {
                "no_go_angle_deg": 45.0,
                "tack_min_hold_s": 3.0,
                "min_course_speed_mps": 0.05,
                "min_course_delta_m": 0.01,
            },
        )

        self.assertEqual(metrics["tack_count"], 0.0)
        self.assertEqual(metrics["upwind_tack_count"], 0.0)

    def test_compute_metrics_includes_sailing_behavior_columns(self):
        samples = [
            sample(0.0, 0.0, 0.0),
            sample(10.0, 8.66, -5.0),
            sample(20.0, -8.66, -15.0),
        ]

        metrics = score.compute_metrics(
            samples,
            sailing_cfg={
                "no_go_angle_deg": 45.0,
                "no_go_grace_s": 3.0,
                "tack_min_hold_s": 3.0,
                "min_course_speed_mps": 0.05,
                "min_course_delta_m": 0.01,
            },
        )

        self.assertIn("upwind_tack_count", metrics)
        self.assertEqual(metrics["upwind_tack_count"], 1.0)

    def test_sailing_objective_does_not_penalize_stable_upwind_tacking(self):
        metrics = {
            "upwind_sailing_ratio": 0.72,
            "upwind_tack_count": 6.0,
            "upwind_no_go_violation_ratio": 0.12,
        }

        sailing_term = score.compute_sailing_objective_term(metrics, {})

        self.assertAlmostEqual(sailing_term, 0.0)

    def test_sailing_objective_penalizes_missing_upwind_tacks(self):
        metrics = {
            "upwind_sailing_ratio": 0.72,
            "upwind_tack_count": 2.0,
            "upwind_no_go_violation_ratio": 0.10,
        }

        sailing_term = score.compute_sailing_objective_term(metrics, {})

        self.assertAlmostEqual(sailing_term, 0.5)

    def test_sailing_objective_penalizes_excessive_upwind_no_go_time(self):
        metrics = {
            "upwind_sailing_ratio": 0.72,
            "upwind_tack_count": 6.0,
            "upwind_no_go_violation_ratio": 0.575,
        }

        sailing_term = score.compute_sailing_objective_term(metrics, {})

        self.assertAlmostEqual(sailing_term, 0.5)

    def test_sailing_objective_ignores_non_upwind_scenarios(self):
        metrics = {
            "upwind_sailing_ratio": 0.0,
            "upwind_tack_count": 0.0,
            "upwind_no_go_violation_ratio": 0.0,
        }

        sailing_term = score.compute_sailing_objective_term(metrics, {})

        self.assertAlmostEqual(sailing_term, 0.0)

    def test_compute_objective_includes_sailing_term(self):
        metrics = {
            "mission_time_s": 0.0,
            "xte_rms_m": 0.0,
            "progress_ratio": 1.0,
            "rudder_total_variation_rad": 0.0,
            "sail_total_variation_rad": 0.0,
            "max_abs_roll_deg_after_grace": 0.0,
            "upwind_sailing_ratio": 0.72,
            "upwind_tack_count": 0.0,
            "upwind_no_go_violation_ratio": 0.0,
        }
        constraints = {
            "constraint_mission_complete": True,
            "constraint_timeout": False,
            "constraint_stuck": False,
            "constraint_no_progress": False,
            "constraint_excessive_roll": False,
        }

        objective = score.compute_objective(metrics, constraints, {"timeout_s": 100.0})

        self.assertAlmostEqual(objective["sailing_term"], 1.0)
        self.assertAlmostEqual(objective["scenario_cost"], 0.15)


if __name__ == "__main__":
    unittest.main()
