# Sailboat performance characterization

This package measures the sailing-performance envelope of the frozen
BO-optimized baseline controller.  It is separate from the BO objective and
from residual RL.

The first vertical slice runs exactly one polar cell:

- true wind speed: 8 m/s;
- desired navigation heading: 0 deg (North);
- signed true-wind angle: +90 deg (wind from starboard / East);
- parameter source: `experiments/sailboat_BO/optimized_baseline_params.json`;
- no waypoint mission, no AUTO mode, no residual action, no motor propulsion.

The runner reuses the validated startup-state hold and sail-force gate.  After
arming it enters GUIDED and refreshes a `SET_POSITION_TARGET_LOCAL_NED` command
whose active fields are `vx`, `vy`, and `yaw`.  The positive desired speed keeps
normal sailboat control active; static guards require the only ArduPilotPlugin
actuators to be steering and mainsail.  The live smoke must still verify that
the vehicle remains wind-propelled and does not invoke automatic tacking.

## Coordinate contract

- Gazebo world: ENU (`x=east`, `y=north`, `z=up`).
- Navigation heading: compass/NED bearing (`0=north`, clockwise positive).
- Gazebo wind vector: wind-to direction.
- `WIND.direction` and signed TWA: wind-from direction.
- `signed_TWA = wrap180(wind_from_bearing - heading)`.
- positive upwind VMG points toward the measured wind source.

The 3-D Gazebo OdometryPublisher supplies model/body-frame twist.  The runner
rotates it by the model quaternion before computing world ENU velocity, course,
leeway, and VMG.

## Telemetry contract

| Data | Source | Frame / semantics | Unit | Nominal rate | Stale gate |
|---|---|---|---|---:|---:|
| Pose, attitude | ROS `/model/sailboat/odometry` | ENU position; FRD attitude | m, deg | 20 Hz | 0.75 s |
| Velocity | same odometry | body/model twist converted to world ENU | m/s | 20 Hz | 0.75 s |
| Heading error / derived yaw rate | consecutive odometry headings | target minus measured heading; wrapped finite-difference rate, clockwise positive | deg, deg/s | 20 Hz | odometry gate |
| Gazebo angular yaw rate | odometry `twist.angular.z` | Gazebo's internal pose-derived body rate; model +Z-up converted to body-FRD +Z-down | deg/s | 20 Hz | odometry gate |
| ArduPilot yaw / yaw rate | MAVLink `ATTITUDE` | NED/FRD, clockwise positive | deg, deg/s | requested 10 Hz | 0.75 s |
| GUIDED target | transmitted MAVLink setpoint | NED heading and north/east velocity | deg, m/s | 2 Hz refresh | constant provenance |
| True wind | MAVLink `WIND` | wind-from compass bearing | deg, m/s | requested 5 Hz | 0.75 s |
| Rudder and sheet allowance | ROS base/final command topics | signed rudder; unsigned sheet allowance | rad | adapter rate | 0.75 s |
| Physical rudder and boom | bridged Gazebo joint state | signed `rudder_joint` / `sail_joint` position and velocity | rad, rad/s | physics publisher | 0.75 s |

Sheet allowance and actual physical boom angle are intentionally separate.
Rudder tracking error is `actual_rudder_angle_rad - final_rudder_rad` in the
same signed SDF joint coordinate. Gazebo / MAVLink heading differences are
wrapped across 0/360 degrees before they are summarized.
The Gazebo angular rate is computed inside `OdometryPublisher` from successive
pose quaternions. It is independent of this runner's published-heading finite
difference, but it is not a direct physics-engine angular-velocity component.
The summary keeps measurement-only quality metrics unchanged and adds separate
raw-acquisition ratios plus a compact diagnostic for the final steady-state-sized
window, so a `not_steady` run remains interpretable.

## Steady-state gate

The configurable detector uses a 20 s sliding window and then requires 10 s of
continuous qualification.  It checks heading RMSE and circular dispersion,
speed trend and coefficient of variation, roll trend and dispersion, source
freshness, wind settlement, sail-force ramp completion, and absence of a tack.
After qualification, a separate 30 s measurement window is written.

Outputs are:

```text
<results>/<trial_id>/
  trial_manifest.json
  trial.parm
  logs/
  raw/samples.csv
  summary/summary.json
```

## Unit tests

```bash
python3 -m pytest -q experiments/sailboat_performance/tests
```

## Live single-cell smoke

Run only inside the rebuilt `dave:sailboat-rl` container:

```bash
python3 -m experiments.sailboat_performance.run_polar_trial \
  --config experiments/sailboat_performance/config/polar_smoke.yaml \
  --params-file experiments/sailboat_BO/optimized_baseline_params.json \
  --results-dir /home/docker/sailboat_rl_results/performance/polar_smoke \
  --execution-backend local
```

This command is one live validation, not a polar sweep.  Do not proceed to the
planned `-90, -60, +60, +90 deg` symmetry trials until its lifecycle, guidance,
propulsion, telemetry, steady-state, measurement, and cleanup gates have been
reviewed.

## Unloaded fixed-rudder isolation

The first rudder isolation test removes the ArduPilot steering loop and uses
the diagnostic-only `rudder_calm` world. That world keeps the normal hull
buoyancy / hydrodynamics path but replaces the production FFT wavefield with
`waves_low` (`wind_speed=0`, `steepness=0`), loads no ocean-current plugin, and
sets air wind and all wind variation to zero. The command adapter's existing
diagnostic override is the sole final-command owner; it does not add another
publisher. The lifecycle is:

```text
startup hold ON -> sail force OFF -> arm -> residual OFF
-> diagnostic override ON at 0 deg -> startup hold OFF
-> adaptive low-speed settling -> fixed setpoints
```

After hold release, measurement starts only after speed remains at or below
0.25 m/s for 2.0 continuous simulation seconds.  A separate 30 s wall-clock
timeout prevents an infinite wait.  The diagnostic override stays enabled at
zero throughout settling, and settling samples are recorded only in the JSON
summary rather than in the fixed-step `samples.csv` metrics.

The setpoints are `0, +5, -5, +10, -10, 0 deg`, held for 8 simulation seconds
each. The raw CSV records target, final command, physical angle and velocity,
tracking errors, source ages, and boat speed. The summary reports rise and
settling time, steady error, overshoot, maximum velocity, and fraction of
samples at the SDF 3 rad/s joint limit. A pass requires fresh telemetry, final
command ownership, boat speed no greater than 0.25 m/s, final 2 s tracking
within 1 deg, no more than 2 deg overshoot, at least 2 s continuously settled,
and no more than 5% of samples at 99% of the velocity limit.

Run only inside the rebuilt simulation container:

```bash
python3 -m experiments.sailboat_performance.run_rudder_tracking \
  --config experiments/sailboat_performance/config/rudder_unloaded.yaml \
  --params-file experiments/sailboat_BO/optimized_baseline_params.json \
  --results-dir /home/docker/sailboat_rl_results/performance/rudder_unloaded \
  --execution-backend local \
  --require-pass
```

After the 1 m/s condition passes, `rudder_loaded_2mps.yaml` repeats the same
single `+5 deg` diagnostic with only the rudder-relative water velocity raised
to world ENU `[0, -2, 0] m/s`. It uses the same isolation, settling, tracking,
and hydrodynamic-load gates; it is not a sweep.

Do not run a hydrodynamically loaded fixed-rudder test until this unloaded
contract has been reviewed. Controller gains, joint limits, foil coefficients,
and the frozen BO parameter set remain unchanged.

## Rudder-only hydrodynamic load isolation

After the strict-unloaded gate passes, `rudder_loaded_1mps.yaml` reuses the
same calm world, startup settling, diagnostic command ownership, and tracking
limits for one `+5 deg` step. The runner creates a trial-local copy of the
production sailboat SDF. In that copy, the rudder's normal `water_velocity`
remains zero and a diagnostic target of world ENU `[0, -1, 0] m/s` starts
disabled in `RudderDiagnosticLoadSystem`, which reuses the production
`LiftDragModel` equations and parameters. Startup, arming, startup-hold release,
and calm settling therefore run without the imposed test flow. Immediately
before the `+5 deg` step, the runner captures the current odometry heading,
publishes the diagnostic-flow enable command, and then sends the rudder
command. The daggerboard stays at zero; the installed production SDF, normal
`FoilLiftDragSystem`, and production world are not edited. Since the sailboat
model points forward along `+Y`, the enabled target is initially a longitudinal
incoming relative flow of approximately 1 m/s.

For the bounded yaw-origin checkpoint, the runner also writes
`raw/yaw_origin_samples.csv` continuously from startup through measurement.
Each row records lifecycle phase, simulation time, heading, both derived and
Gazebo angular yaw rate, position, speed, actual rudder angle, requested hold
state, and requested diagnostic-flow state. This is evidence-only telemetry:
it does not add a pass threshold or automatically classify the yaw cause.

The loaded verdict reuses the normal telemetry, ownership, steady error,
overshoot, rise, settling, and joint-velocity gates. The unloaded boat-speed
gate is intentionally reported but excluded after the `+5 deg` load begins.
Separate gates require the foil log to show the configured relative-water
speed/direction, a nonzero force and yaw moment, the expected negative yaw
moment for this `+5 deg`/`-Y` model convention, and consistent `r x F` math. The
`0.1 N m` nonzero threshold is above the maximum `0.0793 N m` observed over
the settled strict-unloaded measurement window (the stationary final window
was exactly zero). Per-sample relative velocity, AoA, force, and yaw moment
are written beside the summary.

The yaw-sign reference is captured from odometry immediately before
the `+5 deg` diagnostic command is sent. The sign contract is evaluated only
while the free-yaw hull remains within 15 deg of that pre-step heading, so the
later large hull rotation is not treated as the original rudder operating
point. This heading envelope and the foil's aerodynamic angle-of-attack stall
threshold are distinct physical quantities. At least five foil samples must
have both a heading inside the envelope and a yaw moment above the existing
0.1 N m threshold; otherwise the sign gate fails explicitly. The trial-local
rudder plugin logs at 0.1 s intervals to resolve this early response without
changing the installed production SDF. The summary records the reference
heading and capture time, matched and accepted sample counts, heading-deviation
ranges, heading-alignment time deltas, and the plugin-confirmed flow activation
simulation time.

```bash
python3 -m experiments.sailboat_performance.run_rudder_tracking \
  --config experiments/sailboat_performance/config/rudder_loaded_1mps.yaml \
  --params-file experiments/sailboat_BO/optimized_baseline_params.json \
  --results-dir /home/docker/sailboat_rl_results/performance/rudder_loaded_1mps \
  --execution-backend local \
  --require-pass
```
