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

## Wind contract diagnostic

After the attitude-frame check passes, run the wind diagnostic. It launches six
fresh, unarmed held-pose cases:

- north-, east-, south-, and west-going static wind at heading 0 degrees;
- north-going wind at heading 90 degrees, which distinguishes world-frame from
  sensor-frame anemometer output;
- a north-going wind with sinusoidal magnitude, which checks live updates in a
  single running process.

Sail force remains disabled. `SailLiftDragSystem` now reports the wind selected
by `PreUpdate()` even in that gated state, so the diagnostic does not need to
move or arm the boat to inspect the aerodynamic input.

```bash
python3 -m experiments.sailboat_physics.diagnose_wind \
  --label post_frame_fix \
  --scenario experiments/sailboat_BO/scenario.yaml \
  --base-scenario-id train_crosswind_straight \
  --params-file experiments/sailboat_BO/baseline_params.json \
  --results-dir /home/docker/sailboat_rl_results/physics/wind_checks \
  --execution-backend local \
  --require-pass
```

Inspect the printed timestamped `run_dir`:

- `wind_check_summary.json`: full gates and inferred anemometer frame contract;
- `wind_check_summary.csv`: one compact row per case;
- `cases/<trial_id>/raw/wind_samples.csv`: raw anemometer and MAVLink `WIND`;
- `cases/<trial_id>/raw/sail_plugin_wind.csv`: actual wind source and value read
  by the sail-force plugin;
- `cases/<trial_id>/logs/launch.*.log`: original runtime evidence.

The run passes only when all cases complete, the installed anemometer contract
is consistent, the static directions/speeds match, the sail plugin reads the
Gazebo world component, and all three paths change during the dynamic case.
`wind_check: RUNTIME_INCOMPLETE` means data collection or cleanup failed;
`COMPLETED_WITH_GATE_FAILURES` means the run completed but exposed a coordinate,
source-selection, or live-update mismatch.

## Sail-wrench tacking diagnostic

After an armed `train_upwind_tack` trial, inspect whether the actual sail force
creates opposite hull-roll moments on the two boom sides.  The plugin emits a
rate-limited record containing boom angle, raw/applied force, center of
pressure, lever arm, world moment, hull roll axis, and projected roll moment.

```bash
python3 -m experiments.sailboat_physics.diagnose_tacking_wrench \
  --trial-dir "$TRIAL_DIR" \
  --require-pass
```

Inspect `physics_diagnostics/sail_wrench_summary.json` and
`physics_diagnostics/sail_wrench_samples.csv`.  The diagnostic passes only
when both boom sides contain active-force samples, the logged wrench agrees
with `lever x force`, and the mean applied roll moment reverses sign between
the two sides.  A failed sign-reversal gate isolates the problem to the sail
force direction or application geometry; a passed gate with one-sided vehicle
roll points downstream to hull/hydrodynamic response instead.

## Rudder command / hydrodynamic response diagnostic

Do not change `SERVO1_REVERSED` or the SDF rudder multiplier while diagnosing
the steering sign. The rudder foil plugin now records the final command seen
on the JointPositionController topic, the actual rudder angle, water force,
lever arm, hull yaw axis and projected yaw moment. The command adapter also
has a diagnostic override which is disabled by default and affects only the
rudder when explicitly enabled.

Start a fresh `train_upwind_tack` trial in terminal 1. Immediately after the
vehicle arms and sail force is enabled, use terminal 2 to hold both rudder
sides long enough to collect rate-limited force samples:

```bash
ros2 param set /sailboat_command_adapter diagnostic_rudder_command_rad 0.349066
ros2 param set /sailboat_command_adapter diagnostic_rudder_override_enabled true
sleep 8
ros2 param set /sailboat_command_adapter diagnostic_rudder_command_rad -0.349066
sleep 8
ros2 param set /sailboat_command_adapter diagnostic_rudder_command_rad 0.0
ros2 param set /sailboat_command_adapter diagnostic_rudder_override_enabled false
```

The default `false` value preserves normal BO and RL commands exactly. The
override is bounded by the same rudder limits and bypasses residual mixing so
the two diagnostic inputs remain unambiguous.

After the trial exits, analyze the saved trial directory:

```bash
python3 -m experiments.sailboat_physics.diagnose_rudder_response \
  --trial-dir "$TRIAL_DIR" \
  --require-pass
```

Inspect `physics_diagnostics/rudder_response_summary.json` and
`physics_diagnostics/rudder_response_samples.csv`. Passing requires:

- at least three active-water samples for each command sign;
- actual rudder angle matching the final command sign in at least 80% of
  active samples;
- logged moment matching `lever x force` and its yaw-axis projection;
- mean hydrodynamic yaw moment reversing sign between the two rudder sides.

Mean yaw rate and its correlation with rudder yaw moment are also reported,
but are not pass gates because simultaneous sail and hull moments can delay or
temporarily dominate whole-vehicle rotation. If the four gates pass, another
servo/SDF sign flip is not justified; the next isolation target is the
competing sail or hull moment.
