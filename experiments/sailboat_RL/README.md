# Sailboat residual RL environment

This package learns small rudder and sheet-allowance corrections on top of the
frozen BO controller in `experiments/sailboat_BO/optimized_baseline_params.json`.

The action is a normalized two-vector:

```text
[rudder residual, sail residual] in [-1, 1]
```

It is scaled to the command adapter's configured residual limits before being
published. The BO command remains the base command. The rudder residual is a
signed steering correction. The sail residual is a signed change to the
unsigned sheet allowance, so the final sheet command is clamped to `[0, 45]`
degrees; it is not a signed boom-side command.

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
lifecycle coordination instead of parsing console text. Local ROS waypoint
capture is used for observations and reward shaping, but lifecycle episodes do
not terminate until the BO runner has finalized its outcome and written the
trial summary. This prevents `env.close()` from converting a locally detected
mission completion into an `environment_close` BO failure.

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
  --params-file experiments/sailboat_BO/optimized_baseline_params.json \
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

### Per-step lifecycle telemetry

Each managed training, evaluation, or lifecycle-smoke trial also writes
`rl/rl_steps.csv` inside that trial's existing BO result directory. The file is
aligned to `env.step()` and records the normalized policy action, requested
physical residual, observed base command, actual final adapter output, state,
per-step reward components, and terminal reason.

`residual_*_rad` is the **actual adapter contribution**
(`final_*_rad - base_*_rad`); `requested_residual_*_rad` is the physical action
sent by the environment. This distinction preserves rate-limit and clamp
evidence. The recorded actuator invariant is therefore:

```text
final_rudder = clamp(base_rudder + residual_rudder, -0.7854, +0.7854)
final_sail   = clamp(base_sail + residual_sail,       0.0,    0.7854)
```

For a driver-limited smoke run, the final sampled step is retained with
`truncated=True, reason=environment_close`; naturally terminal trials retain
their runner-provided terminal reason.

## SAC environment check and training

`train_sac.py` uses Stable-Baselines3 SAC to learn only the bounded rudder and
sail residuals. It does not replace or update the frozen BO base controller.
Every SAC episode uses the same automatic lifecycle backend as the smoke test,
so a Gymnasium reset creates a fresh BO trial and the previous trial is cleaned
before the next one starts.

First run the live Stable-Baselines3 compatibility check. This starts real
simulation episodes, so no BO trial should already be running:

```bash
python3 -m experiments.sailboat_RL.train_sac \
  --scenario experiments/sailboat_BO/scenario_generalization_bo_eval.yaml \
  --scenario-id train_crosswind_straight \
  --params-file experiments/sailboat_BO/optimized_baseline_params.json \
  --run-dir experiments/sailboat_RL/results/sac_env_check \
  --check-env-only \
  --execution-backend local
```

Then run a short integration training check. The small values below verify that
SAC can collect transitions, update its networks, and save artifacts; they are
not intended to produce a useful policy:

```bash
python3 -m experiments.sailboat_RL.train_sac \
  --scenario experiments/sailboat_BO/scenario_generalization_bo_eval.yaml \
  --scenario-id train_crosswind_straight \
  --params-file experiments/sailboat_BO/optimized_baseline_params.json \
  --run-dir experiments/sailboat_RL/results/sac_smoke \
  --total-timesteps 20 \
  --learning-starts 5 \
  --buffer-size 1000 \
  --batch-size 16 \
  --checkpoint-freq 10 \
  --skip-env-check \
  --execution-backend local
```

For an initial longer run, omit `--run-dir` to create a timestamped result
directory and use the defaults (`10000` steps, `1000` random warm-up steps).
Each run stores `training_config.json`, `training_summary.json`, Gymnasium
monitor data, BO trial artifacts, TensorBoard events, periodic checkpoints, the
final model, and its replay buffer beneath one run directory.
TensorBoard rollout scalars are emitted after every completed episode.

```bash
python3 -m experiments.sailboat_RL.train_sac \
  --scenario experiments/sailboat_BO/scenario_generalization_bo_eval.yaml \
  --scenario-id train_crosswind_straight \
  --total-timesteps 10000 \
  --execution-backend local
```

Training is rejected unless the selected scenario has `split: train`.
`eval_long_oblique` and `eval_shifted_wind_loop` are holdouts and must only be
used with `evaluate_sac.py`.

Inspect learning curves from another shell in the same container:

```bash
tensorboard --logdir experiments/sailboat_RL/results/sac --bind_all
```

## Deterministic paired evaluation

Do not judge a policy from its training reward alone. `evaluate_sac.py` runs
the frozen BO controller with zero residual and the saved SAC policy under the
same seed and repeat index. In compare mode the execution order alternates
between pairs to reduce systematic time/order bias. SAC actions are
deterministic by default.

Five pairs mean ten real simulation trials:

```bash
python3 -m experiments.sailboat_RL.evaluate_sac \
  --model experiments/sailboat_RL/results/sac/<run>/models/sac_final.zip \
  --mode compare \
  --episodes 5 \
  --scenario experiments/sailboat_BO/scenario_generalization_bo_eval.yaml \
  --scenario-id eval_long_oblique \
  --params-file experiments/sailboat_BO/optimized_baseline_params.json \
  --execution-backend local
```

Each pair uses the same scenario, BO parameters, Gymnasium seed, and repeat
index for the two controllers. The current scenario does not expose a separate
Gazebo random seed, so this pairing controls configured inputs and identifiers
but does not claim to eliminate all simulator timing noise. The output
directory contains the complete episode records in JSON and CSV, separate BO
trial artifacts under `zero/` and `policy/`, and an
`evaluation_summary.json` with controller aggregates and `policy_minus_zero`
paired deltas for reward, mission time, progress, final distance, and roll.

To test only the zero-residual reference without loading a model:

```bash
python3 -m experiments.sailboat_RL.evaluate_sac \
  --mode zero \
  --episodes 1 \
  --execution-backend local
```
