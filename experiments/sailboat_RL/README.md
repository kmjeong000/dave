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
waypoint tracker; it does not restart Gazebo or ArduPilot. By default it also
sets the command adapter's `residual_enabled` parameter to `true` on reset and
returns it to `false` on close.

A full episode lifecycle backend will later reuse the existing
`experiments/sailboat_BO/run_trial.py` launch, mission-upload, arm, and cleanup
operations behind the same backend protocol.

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
