# Sailboat physical-frame diagnostics

This package checks the coordinate contract underneath the BO and residual-RL
pipelines. It does not tune parameters or train a policy.

## Frame convention under test

The Gazebo world is ENU. The current sailboat mesh and aerodynamic surfaces use
model `+Y` as the bow, model `+X` as starboard, and model `-Z` as down. The
diagnostic converts the complete Gazebo model quaternion into a conventional
body FRD / world NED attitude before comparing it with MAVLink `ATTITUDE`.

This is important because the ordinary Gazebo Euler `roll` about model `+X` is
physical pitch for a `+Y`-forward hull. Physical heel/roll is rotation about
model `+Y`.

## What the command runs

`diagnose_frames` runs eight independent, unarmed, zero-wind cases by default:

- compass headings 0, 90, 180 and 270 degrees;
- body-FRD roll -10 and +10 degrees;
- body-FRD pitch -10 and +10 degrees.

Sail force remains disabled. After MAVLink connects, the requested pose is held
with Gazebo's `/world/waves/set_pose` service while Gazebo odometry quaternion
and MAVLink `ATTITUDE` are sampled together. Each case receives a fresh
Gazebo/SITL process and is cleaned up before the next case.

Run this only inside the `dave:sailboat-rl` container, with no other BO trial,
Gazebo instance, or SITL process running:

```bash
cd /home/docker/sailboat_ws/src/dave

python3 -m experiments.sailboat_physics.diagnose_frames \
  --label pre_fix \
  --scenario experiments/sailboat_BO/scenario.yaml \
  --base-scenario-id train_crosswind_straight \
  --params-file experiments/sailboat_BO/baseline_params.json \
  --results-dir /home/docker/sailboat_rl_results/physics/frame_checks \
  --execution-backend local
```

Do not add `--require-pass` to the pre-fix run: a frame-gate failure is expected
to be useful diagnostic output. Runtime/data failures still return exit code 1.

After a future SDF coordinate fix, run the same command with `--label post_fix`
and `--require-pass`. A complete but misaligned result then returns exit code 2.

```bash
python3 -m experiments.sailboat_physics.diagnose_frames \
  --label post_fix \
  --scenario experiments/sailboat_BO/scenario.yaml \
  --base-scenario-id train_crosswind_straight \
  --params-file experiments/sailboat_BO/baseline_params.json \
  --results-dir /home/docker/sailboat_rl_results/physics/frame_checks \
  --execution-backend local \
  --require-pass
```

## Outputs and gates

The command prints its timestamped `run_dir`. Inspect:

- `frame_check_summary.json`: complete nested results and individual gates;
- `frame_check_summary.csv`: one compact row per case;
- `cases/<trial_id>/raw/frame_samples.csv`: time-aligned raw attitudes;
- `cases/<trial_id>/logs/launch.stdout.log` and `launch.stderr.log`: runtime logs.

`runtime_complete` is true only when all cases produced usable data.
`all_frame_gates_passed` is true only when every case also satisfies:

- at least 10 paired samples;
- requested vs Gazebo physical heading error <= 3 degrees;
- requested vs MAVLink heading error <= 3 degrees;
- MAVLink vs Gazebo heading error <= 3 degrees;
- the equivalent requested/source-alignment gates for roll and pitch <= 3 degrees.

For the pre-fix run, `runtime_complete: true` is the immediate success criterion.
The individual failed frame gates identify the SDF transform that must be fixed.

## Unit tests

```bash
python3 -m pytest -q \
  experiments/sailboat_physics/tests \
  experiments/sailboat_BO/tests/test_run_trial_mission.py
```
