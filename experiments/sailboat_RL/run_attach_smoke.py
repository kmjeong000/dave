from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.sailboat_BO.common import load_yaml
from experiments.sailboat_BO.run_trial import get_scenario
from experiments.sailboat_RL.core import EnvironmentConfig
from experiments.sailboat_RL.env import SailboatResidualEnv
from experiments.sailboat_RL.ros_backend import Ros2AttachBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach a Gymnasium residual environment to a running BO trial"
    )
    parser.add_argument(
        "--scenario",
        default="experiments/sailboat_BO/scenario.yaml",
        help="BO scenario YAML used by the running trial",
    )
    parser.add_argument(
        "--scenario-id",
        default="eval_long_oblique",
        help="Scenario id used by the running trial",
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--action",
        default="[0.0, 0.0]",
        help="Normalized JSON action [rudder, sail] in [-1, 1]",
    )
    parser.add_argument("--control-period-s", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(Path(args.scenario))
    scenario = get_scenario(config, args.scenario_id)
    action = np.asarray(json.loads(args.action), dtype=np.float32)
    if action.shape != (2,):
        raise SystemExit("--action must be a JSON two-vector")

    termination = dict(config.get("termination", {}))
    termination.update(scenario.get("termination", {}))
    backend = Ros2AttachBackend(
        waypoints=scenario["mission"]["waypoints"],
        namespace=str(config["study"].get("namespace", "sailboat")),
        wind_world_xyz_mps=scenario["world"]["wind_world_xyz_mps"],
        waypoint_capture_radius_m=float(
            termination.get("waypoint_capture_radius_m", 5.0)
        ),
    )
    env = SailboatResidualEnv(
        backend,
        EnvironmentConfig(
            success_radius_m=float(termination.get("success_radius_m", 5.0)),
            max_roll_deg=float(termination.get("max_roll_deg", 45.0)),
            episode_timeout_s=float(termination.get("timeout_s", 240.0)),
            control_period_s=float(args.control_period_s),
        ),
    )

    try:
        observation, info = env.reset()
        print("reset:", {"observation": observation.tolist(), "info": info})
        for index in range(max(0, args.steps)):
            observation, reward, terminated, truncated, info = env.step(action)
            print(
                "step:",
                {
                    "index": index,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "observation": observation.tolist(),
                    "info": info,
                },
            )
            if terminated or truncated:
                break
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
