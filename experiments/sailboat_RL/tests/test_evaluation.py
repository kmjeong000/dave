from experiments.sailboat_RL.evaluation import summarize_evaluations


def make_record(controller, pair_index, **overrides):
    record = {
        "evaluation_status": "complete",
        "controller": controller,
        "pair_index": pair_index,
        "reason": "mission_complete",
        "episode_reward": 100.0,
        "episode_steps": 300,
        "bo_mission_time_s": 140.0,
        "bo_progress_ratio": 0.99,
        "bo_final_distance_to_wp_m": 3.0,
        "bo_max_abs_roll_deg": 6.0,
        "rudder_abs_saturation_ratio": 0.01,
        "sail_abs_saturation_ratio": 0.02,
    }
    record.update(overrides)
    return record


def test_summarize_evaluations_builds_controller_and_paired_deltas():
    records = [
        make_record("zero", 0, episode_reward=100.0, bo_mission_time_s=140.0),
        make_record("policy", 0, episode_reward=110.0, bo_mission_time_s=130.0),
        make_record("zero", 1, episode_reward=90.0, bo_mission_time_s=150.0),
        make_record("policy", 1, episode_reward=95.0, bo_mission_time_s=145.0),
    ]

    summary = summarize_evaluations(records)

    assert summary["status"] == "complete"
    assert summary["controllers"]["zero"]["mission_success_rate"] == 1.0
    assert summary["controllers"]["policy"]["episode_reward"]["mean"] == 102.5
    assert summary["paired"]["complete_pair_count"] == 2
    assert summary["paired"]["episode_reward_delta"]["mean"] == 7.5
    assert summary["paired"]["mission_time_s_delta"]["mean"] == -7.5
    assert summary["paired"]["policy_better_reward_count"] == 2
    assert summary["paired"]["policy_faster_mission_count"] == 2


def test_summarize_evaluations_marks_errors_as_partial():
    records = [
        make_record("zero", 0),
        {
            "evaluation_status": "error",
            "controller": "policy",
            "pair_index": 0,
            "error_type": "RuntimeError",
        },
    ]

    summary = summarize_evaluations(records)

    assert summary["status"] == "partial"
    assert summary["error_count"] == 1
    assert summary["controllers"]["policy"]["episodes_error"] == 1
    assert summary["paired"]["complete_pair_count"] == 0
