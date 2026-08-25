# Sailboat residual RL environment

This package learns small rudder and sail corrections on top of the frozen BO
controller in `experiments/sailboat_BO/verified_incumbent_params.json`.

The action is a normalized two-vector:

```text
[rudder residual, sail residual] in [-1, 1]
```

It is scaled to the command adapter's configured residual limits before being
published. The BO command remains the base command.

## Runtime modes

`Ros2AttachBackend` attaches to a trial that is already launched, mission-ready,
armed, and in AUTO. Its `reset()` zeros residuals and resets only the local
waypoint tracker. By default it also sets the command adapter's
`residual_enabled` parameter to `true` on reset and returns it to `false` on
close.

`EpisodeLifecycleBackend` is the training-oriented mode. Every Gymnasium
`reset()` starts a fresh `experiments/sailboat_BO/run_trial.py` subprocess and
waits until that runner has uploaded the mission, armed the vehicle, entered
AUTO, and enabled sail force. It then creates a fresh `Ros2AttachBackend` for
the episode. The next reset or `close()` first zeros and disables residuals,
requests cooperative BO-runner shutdown, waits for Gazebo/SITL cleanup, and
only then starts another episode.

The BO runner remains the single owner of launch, mission, telemetry logging,
result writing, and cleanup. Atomic ready/stop/status marker files are used for
lifecycle coordination instead of parsing console text.

## Installation

Inside the simulation Python environment:

```bash
python3 -m pip install -r experiments/sailboat_RL/requirements.txt
```

## Minimal attach example

```python
from experiments.sailboat_RL.env import SailboatResidualEnv
from experiments.sailboat_RL.ros_backend import Ros2AttachBackend

backend = Ros2AttachBackend(
    waypoints=[[30, 50], [60, 100], [90, 140]],
    wind_world_xyz_mps=[0, 8, 0],
)
env = SailboatResidualEnv(backend)

observation, info = env.reset()
observation, reward, terminated, truncated, info = env.step([0.0, 0.0])
env.close()
```

For a short command-line check against an already running trial:

```bash
python3 -m experiments.sailboat_RL.run_attach_smoke \
  --scenario-id eval_long_oblique \
  --steps 10 \
  --action '[0.0, 0.0]'
```

## Automatic lifecycle smoke test

Run this inside the `dave:sailboat-rl` container with no BO trial already
running. Two short zero-residual episodes will be started and cleaned
automatically:

```bash
python3 -m experiments.sailboat_RL.run_lifecycle_smoke \
  --scenario experiments/sailboat_BO/scenario.yaml \
  --scenario-id eval_long_oblique \
  --params-file experiments/sailboat_BO/verified_incumbent_params.json \
  --results-dir experiments/sailboat_RL/results/lifecycle_smoke \
  --episodes 2 \
  --max-steps-per-episode 10 \
  --action '[0.0, 0.0]' \
  --execution-backend local
```

Use `--max-steps-per-episode 0` to let every episode run until the environment
reports mission completion, excessive roll, timeout, no progress, or another
runner terminal condition.

Per-episode BO artifacts remain under the selected results directory. Lifecycle
runner stdout, stderr, and coordination markers are stored under
`<results-dir>/_lifecycle/` for startup and cleanup diagnosis.
