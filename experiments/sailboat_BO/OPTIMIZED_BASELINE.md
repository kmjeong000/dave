# BO-Optimized Sailboat Baseline

## 1. Overview

This document records the Bayesian Optimization (BO) procedure used to determine and validate the final baseline controller parameters for the autonomous sailboat simulation.

The optimization was performed after stabilizing the sailboat physics, coordinate frames, actuator behavior, wind handling, and startup initial conditions.

The final optimized parameter set was selected and validated through the following procedure:

1. Run Bayesian Optimization on three core training scenarios.
2. Select the top three failure-free BO candidates.
3. Re-evaluate the top three candidates and the original baseline using five repeated trials per training scenario.
4. Select the best candidate according to the predefined mean scenario cost.
5. Evaluate the selected candidate on a held-out scenario.
6. Evaluate the frozen candidate on an additional generalization scenario.
7. Evaluate the frozen candidate on two out-of-BO transfer scenarios.
8. Fix the validated candidate as the final BO-optimized baseline.

No parameter tuning was performed after the held-out or generalization/transfer evaluations.

---

## 2. Experiment Environment

### Repository baseline

Physics and startup stabilization baseline:

```text
Git commit: 521811c
```

This commit represents the stabilized simulation state used before BO optimization.

The stabilized startup sequence was:

```text
spawn
→ startup hold ON
→ sail force OFF
→ ArduPilot / EKF initialization
→ arm
→ AUTO
→ startup hold OFF
→ sail force ON
→ mission execution
```

The same stabilized physics and startup behavior were retained throughout the BO and subsequent validation experiments.

### BO run

```text
validated_ros2_521811c_core3_seed42_20260901T074459Z
```

WSL result directory:

```text
~/sailboat_rl_results/bo/
validated_ros2_521811c_core3_seed42_20260901T074459Z/
```

### Optimization configuration

```text
Iterations          : 40
Initial random      : 12
Local warmup        : 6
Candidate pool      : 4096
Repeats             : 2
Training scenarios  : 3
Random seed         : 42
Worst-case weight   : 0.0
```

Optimizer:

```text
Gaussian Process with RBF kernel
Expected Improvement acquisition
```

Training scenarios:

```text
train_crosswind_straight
train_upwind_tack
train_dogleg
```

Total BO evaluation trials:

```text
40 candidates × 3 scenarios × 2 repeats = 240 trials
```

The BO objective was minimized.

Candidate selection priority:

```text
1. Fewer failures
2. Lower mean objective
3. Earlier iteration if otherwise tied
```

The BO search space was:

| Parameter          |  Low | High |
| ------------------ | ---: | ---: |
| `SAIL_ANGLE_IDEAL` | 20.0 | 50.0 |
| `SAIL_NO_GO_ANGLE` | 40.0 | 70.0 |
| `SAIL_XTRACK_MAX`  |  4.0 | 20.0 |
| `ATC_SAIL_P`       | 0.03 | 0.40 |
| `ATC_STR_ANG_P`    |  0.8 |  4.0 |
| `ATC_STR_RAT_FF`   | 0.10 | 1.20 |

---

## 3. Optimized Parameters

Final selected BO candidate:

```text
iter_008
```

Final parameter file:

```text
experiments/sailboat_BO/optimized_baseline_params.json
```

Parameters:

| Parameter          |  Baseline | Optimized |
| ------------------ | --------: | --------: |
| `ATC_SAIL_P`       |  0.100000 |  0.175176 |
| `ATC_STR_ANG_P`    |  2.000000 |  3.554299 |
| `ATC_STR_RAT_FF`   |  0.500000 |  0.842664 |
| `SAIL_ANGLE_IDEAL` | 35.000000 | 36.006823 |
| `SAIL_NO_GO_ANGLE` | 60.000000 | 56.349306 |
| `SAIL_XTRACK_MAX`  | 10.000000 |  4.915016 |

Exact optimized values:

```json
{
  "ATC_SAIL_P": 0.1751756098336152,
  "ATC_STR_ANG_P": 3.554298809695749,
  "ATC_STR_RAT_FF": 0.8426644623135695,
  "SAIL_ANGLE_IDEAL": 36.006823456092405,
  "SAIL_NO_GO_ANGLE": 56.349306124137385,
  "SAIL_XTRACK_MAX": 4.915015755559315
}
```

---

## 4. Bayesian Optimization Result

The original baseline corresponded to iteration 0.

```text
Baseline objective : 0.386743
Best BO objective  : 0.302544
Best iteration     : iter_008
Failure count      : 0
```

Relative objective improvement:

```text
approximately 21.77%
```

Top failure-free BO candidates:

| Rank |  Iteration | BO objective |
| ---: | ---------: | -----------: |
|    1 | `iter_008` |     0.302544 |
|    2 | `iter_018` |     0.311732 |
|    3 | `iter_036` |     0.320535 |
|    4 | `iter_025` |     0.321869 |
|    5 | `iter_014` |     0.334513 |

The top three candidates were selected for repeated revalidation.

Across the full BO run:

```text
Total trials      : 240
Successful trials : 216
Failed trials     : 24
```

`iter_008` itself completed all six BO trials successfully.

Because `iter_008` was found relatively early in the BO run, the result should not be interpreted as evidence of strong Gaussian-process convergence.

The appropriate interpretation is:

```text
The 40-candidate search identified a stable superior candidate.
```

---

## 5. Top-3 Revalidation

To verify that the BO ranking was not caused by two-repeat simulation variation, the following four parameter sets were evaluated again:

```text
baseline
iter_008
iter_018
iter_036
```

Each parameter set was tested on:

```text
3 training scenarios × 5 repeats = 15 trials
```

Total revalidation trials:

```text
4 parameter sets × 15 trials = 60 trials
```

WSL result directory:

```text
~/sailboat_rl_results/bo_validation/
```

Subdirectories:

```text
baseline_core3_r5/
iter008_core3_r5/
iter018_core3_r5/
iter036_core3_r5/
```

### Revalidation result

All four parameter sets completed all trials successfully.

| Parameter set | Success | Mean scenario cost | Worst scenario mean |
| ------------- | ------: | -----------------: | ------------------: |
| `iter_008`    |   15/15 |       **0.302345** |            0.458265 |
| `iter_018`    |   15/15 |           0.312119 |        **0.436800** |
| `iter_036`    |   15/15 |           0.320000 |            0.457552 |
| baseline      |   15/15 |           0.386873 |            0.512053 |

### Per-scenario result

#### Baseline

| Scenario    | Mean cost | Mission time | Progress | XTE RMS |
| ----------- | --------: | -----------: | -------: | ------: |
| Crosswind   |  0.247066 |      70.40 s |   1.0000 |  5.66 m |
| Upwind tack |  0.512053 |     161.00 s |   0.9822 |  8.30 m |
| Dogleg      |  0.401501 |      85.78 s |   1.0000 |  6.95 m |

#### iter_008

| Scenario    | Mean cost | Mission time | Progress | XTE RMS |
| ----------- | --------: | -----------: | -------: | ------: |
| Crosswind   |  0.201619 |      66.56 s |   0.9891 |  3.42 m |
| Upwind tack |  0.458265 |     198.12 s |   1.0000 |  3.95 m |
| Dogleg      |  0.247151 |      94.20 s |   0.9915 |  3.29 m |

### Selection rationale

`iter_008` was selected because it produced the lowest mean scenario cost while maintaining:

```text
15/15 successful trials
```

Compared with the original baseline:

```text
Baseline mean cost : 0.386873
iter_008 mean cost : 0.302345
Improvement        : approximately 21.85%
```

`iter_018` produced a lower worst-scenario mean cost than `iter_008`, but the BO run was configured with:

```text
worst-case-weight = 0.0
```

Therefore, the predefined optimization criterion was mean scenario cost rather than worst-case cost.

Changing the selection rule after observing the results would alter the original experimental criterion, so `iter_008` was retained as the final BO candidate.

---

## 6. Training-Scenario Trade-off

The optimized controller did not improve every individual metric.

In particular, for the upwind scenario:

```text
Baseline mission time : 161.00 s
iter_008 mission time : 198.12 s
```

However, path-tracking performance improved substantially:

```text
Baseline XTE RMS : 8.30 m
iter_008 XTE RMS : 3.95 m
```

The optimized controller also achieved full progress in the upwind scenario:

```text
Baseline progress : 0.9822
iter_008 progress : 1.0000
```

Therefore, the BO result should not be interpreted as simply increasing sailing speed.

The optimized parameters improved the composite waypoint-tracking objective, especially cross-track tracking performance, while maintaining successful waypoint completion and acceptable roll behavior.

The result should therefore be described as an improvement in the combined navigation objective rather than as universally faster navigation.

---

## 7. Held-Out Validation

After selecting `iter_008`, its parameters were frozen and evaluated without further tuning.

Held-out scenario:

```text
eval_long_oblique
```

This scenario was not used in the three-scenario BO training objective.

Both the optimized controller and original baseline were evaluated for five repeats.

WSL result directory:

```text
~/sailboat_rl_results/bo_holdout/
```

Subdirectories:

```text
iter008_eval_long_oblique_r5/
baseline_eval_long_oblique_r5/
```

### Held-out result

| Metric                  |            Baseline |               Optimized |
| ----------------------- | ------------------: | ----------------------: |
| Success                 |                 5/5 |                     5/5 |
| Scenario cost           | 0.232300 ± 0.004117 | **0.196933 ± 0.000886** |
| XTE RMS                 |             4.687 m |             **3.296 m** |
| Maximum roll            |             23.963° |             **23.602°** |
| Mission time            |             73.08 s |             **72.60 s** |
| Progress ratio          |          **0.9990** |                  0.9949 |
| Final waypoint distance |             4.115 m |             **3.172 m** |

Held-out scenario cost improvement:

```text
0.232300 → 0.196933
approximately 15.2% reduction
```

XTE RMS improvement:

```text
4.687 m → 3.296 m
approximately 29.7% reduction
```

Final waypoint distance improvement:

```text
4.115 m → 3.172 m
approximately 22.9% reduction
```

The small decrease in progress ratio was not associated with mission failure; all five optimized trials completed successfully.

The roll result is best interpreted as:

```text
Tracking performance improved without worsening roll behavior.
```

No parameter tuning was performed using the held-out result.

---

## 8. Generalization and Transfer Validation

After the held-out validation, the frozen `iter_008` parameter set was evaluated on three additional scenarios.

No optimized parameter was changed during this stage.

WSL result directory:

```text
~/sailboat_rl_results/bo_generalization/
```

### Evaluation-only scenario configuration

The original generalization configuration contained narrower parameter-validation ranges than the BO search space used to obtain `iter_008`.

As a result, the first attempt to execute `eval_shifted_wind_loop` was rejected during parameter validation before the simulation started.

This rejection was not a controller or sailing failure.

A separate evaluation-only configuration was therefore created:

```text
experiments/sailboat_BO/scenario_generalization_bo_eval.yaml
```

The only changes relative to the original generalization configuration were the six allowed parameter ranges:

```text
SAIL_ANGLE_IDEAL : 20.0 – 50.0
SAIL_NO_GO_ANGLE : 40.0 – 70.0
SAIL_XTRACK_MAX  : 4.0 – 20.0
ATC_SAIL_P       : 0.03 – 0.40
ATC_STR_ANG_P    : 0.8 – 4.0
ATC_STR_RAT_FF   : 0.10 – 1.20
```

These ranges match the BO search space.

The following were not modified:

```text
optimized parameter values
scenario waypoints
wind conditions
termination criteria
scoring configuration
simulation physics
startup sequence
```

Therefore, the evaluation configuration allows the frozen BO candidate to be tested without altering the controller or evaluation scenarios.

---

### 8.1 eval_shifted_wind_loop

`eval_shifted_wind_loop` was evaluated as an additional generalization scenario.

Both the original baseline and `iter_008` were tested for five repeats.

| Metric                  |            Baseline |               Optimized |
| ----------------------- | ------------------: | ----------------------: |
| Success                 |                 5/5 |                     5/5 |
| Scenario cost           | 0.528326 ± 0.080673 | **0.358071 ± 0.013795** |
| XTE RMS                 |            10.272 m |             **6.280 m** |
| Progress ratio          |          **0.9988** |                  0.9860 |
| Final waypoint distance |             5.040 m |             **3.185 m** |
| Mission time            |            127.66 s |            **106.04 s** |
| Mean maximum roll       |             36.442° |             **30.367°** |
| Peak roll               |             40.317° |             **32.762°** |

Scenario cost improvement:

```text
0.528326 → 0.358071
approximately 32.2% reduction
```

XTE RMS improvement:

```text
10.272 m → 6.280 m
approximately 38.9% reduction
```

Final waypoint distance improvement:

```text
5.040 m → 3.185 m
approximately 36.8% reduction
```

Mission-time improvement:

```text
127.66 s → 106.04 s
approximately 16.9% reduction
```

The optimized controller maintained 5/5 mission success while improving the composite scenario cost, XTE, final waypoint distance, mission time, and roll metrics.

The progress ratio was slightly lower than the original baseline.

---

### 8.2 train_oblique_zigzag

`train_oblique_zigzag` was not included in the BO core-3 training objective.

However, because this scenario is labeled as `train` in the generalization scenario file, it is reported as an:

```text
out-of-BO transfer test
```

rather than as a strict held-out test.

Both controllers were evaluated for five repeats.

| Metric                  |            Baseline |               Optimized |
| ----------------------- | ------------------: | ----------------------: |
| Success                 |                 5/5 |                     5/5 |
| Scenario cost           | 0.271441 ± 0.004096 | **0.185126 ± 0.000423** |
| XTE RMS                 |             4.664 m |             **2.110 m** |
| Progress ratio          |          **0.9890** |                  0.9815 |
| Final waypoint distance |         **2.046 m** |                 3.046 m |
| Mission time            |             87.96 s |             **78.92 s** |
| Mean maximum roll       |             23.482° |             **20.013°** |
| Peak roll               |             24.157° |             **20.758°** |

Scenario cost improvement:

```text
0.271441 → 0.185126
approximately 31.8% reduction
```

XTE RMS improvement:

```text
4.664 m → 2.110 m
approximately 54.8% reduction
```

Mission-time improvement:

```text
87.96 s → 78.92 s
approximately 10.3% reduction
```

The optimized controller substantially reduced XTE and mission time while maintaining 5/5 mission success.

The original baseline achieved a slightly higher progress ratio and a smaller final waypoint distance.

Therefore, this result again demonstrates that the optimized controller does not improve every individual metric, but substantially improves the combined navigation objective.

---

### 8.3 train_reverse_oblique_variable_wind

`train_reverse_oblique_variable_wind` was also not included in the BO core-3 training objective.

Because it is labeled as `train` in the scenario configuration, it is also reported as an:

```text
out-of-BO transfer test
```

Both controllers were evaluated for five repeats.

| Metric                  |            Baseline |               Optimized |
| ----------------------- | ------------------: | ----------------------: |
| Success                 |                 5/5 |                     5/5 |
| Scenario cost           | 0.302059 ± 0.002915 | **0.226479 ± 0.000945** |
| XTE RMS                 |             5.760 m |             **2.968 m** |
| Progress ratio          |              0.9881 |              **0.9903** |
| Final waypoint distance |             2.815 m |             **1.493 m** |
| Mission time            |             90.96 s |             **86.48 s** |
| Mean maximum roll       |         **20.713°** |                 24.861° |
| Peak roll               |         **24.946°** |                 25.189° |

Scenario cost improvement:

```text
0.302059 → 0.226479
approximately 25.0% reduction
```

XTE RMS improvement:

```text
5.760 m → 2.968 m
approximately 48.5% reduction
```

Final waypoint distance improvement:

```text
2.815 m → 1.493 m
approximately 47.0% reduction
```

Mission-time improvement:

```text
90.96 s → 86.48 s
approximately 4.9% reduction
```

The optimized controller improved composite cost, XTE, progress ratio, final waypoint distance, and mission time while maintaining 5/5 mission success.

Its mean maximum roll was higher than that of the original baseline:

```text
Baseline mean maximum roll : 20.713°
iter_008 mean maximum roll : 24.861°
```

However, the peak roll values were similar:

```text
Baseline peak roll : 24.946°
iter_008 peak roll : 25.189°
```

No roll-safety violation occurred.

Therefore, the result should not be described as an improvement in roll stability.

Instead, it demonstrates improved tracking and navigation performance without introducing a roll-safety failure.

---

### 8.4 Generalization summary

Across the three additional scenarios:

| Scenario                              | Baseline cost | Optimized cost | Cost reduction | Classification            |
| ------------------------------------- | ------------: | -------------: | -------------: | ------------------------- |
| `eval_shifted_wind_loop`              |      0.528326 |   **0.358071** |      **32.2%** | Generalization evaluation |
| `train_oblique_zigzag`                |      0.271441 |   **0.185126** |      **31.8%** | Out-of-BO transfer        |
| `train_reverse_oblique_variable_wind` |      0.302059 |   **0.226479** |      **25.0%** | Out-of-BO transfer        |

The optimized controller completed:

```text
3 scenarios × 5 repeats = 15/15 successful trials
```

The corresponding original baseline also completed all 15 trials successfully.

Across all three scenarios, `iter_008` achieved a lower mean scenario cost than the original baseline.

The additional tests therefore provide evidence that the BO improvement was not limited to the three core optimization scenarios.

However, these results should be interpreted as simulation-based generalization and transfer evidence within the tested scenario set, not as proof of universal controller robustness.

No parameters were changed based on these evaluation results.

---

## 9. Final Decision

The final BO-optimized baseline controller is:

```text
iter_008
```

Final parameter file:

```text
experiments/sailboat_BO/optimized_baseline_params.json
```

`iter_008` was accepted because:

* it achieved the lowest mean objective according to the predefined BO selection criterion;
* it completed all six trials associated with its original BO evaluation successfully;
* it remained the best candidate after independent five-repeat core-scenario revalidation;
* all 15 core revalidation trials succeeded;
* it reduced the revalidated mean scenario cost from 0.386873 to 0.302345;
* all five `eval_long_oblique` held-out trials succeeded;
* held-out scenario cost and XTE improved relative to the original baseline;
* all five `eval_shifted_wind_loop` generalization trials succeeded;
* all ten out-of-BO transfer trials succeeded;
* the optimized controller achieved lower mean scenario cost than the original baseline in all three additional generalization/transfer scenarios;
* no additional parameter tuning was performed after candidate selection.

The complete validation sequence therefore supports fixing `iter_008` as the frozen BO-optimized baseline.

The BO result should be described as:

```text
A 40-candidate search identified a stable superior candidate.
```

It should not be described as proof of Gaussian-process convergence.

The validated baseline should also not be described as improving every metric under every scenario.

Observed trade-offs include:

```text
longer mission time in the core upwind scenario
slightly lower progress in some evaluation scenarios
larger final waypoint distance in train_oblique_zigzag
higher mean maximum roll in train_reverse_oblique_variable_wind
```

Despite these trade-offs, `iter_008` consistently improved the composite waypoint-tracking objective and completed all repeated validation trials successfully.

Therefore:

```text
iter_008 is fixed as the frozen BO-optimized baseline controller
for subsequent experiments.
```

Any future residual reinforcement-learning stage should use this frozen optimized baseline only after the residual-action interface and sail-actuator semantics have been verified.

---

## 10. Result Provenance

### Raw BO results

```text
~/sailboat_rl_results/bo/
validated_ros2_521811c_core3_seed42_20260901T074459Z/
```

### Top-3 revalidation

```text
~/sailboat_rl_results/bo_validation/
```

Subdirectories:

```text
baseline_core3_r5/
iter008_core3_r5/
iter018_core3_r5/
iter036_core3_r5/
```

### Held-out validation

```text
~/sailboat_rl_results/bo_holdout/
```

Subdirectories:

```text
baseline_eval_long_oblique_r5/
iter008_eval_long_oblique_r5/
```

### Generalization and transfer validation

```text
~/sailboat_rl_results/bo_generalization/
```

Relevant subdirectories:

```text
baseline_eval_shifted_wind_loop_r5/
iter008_eval_shifted_wind_loop_r5/

baseline_train_oblique_zigzag_r5/
iter008_train_oblique_zigzag_r5/

baseline_train_reverse_oblique_variable_wind_r5/
iter008_train_reverse_oblique_variable_wind_r5/
```

Evaluation configuration:

```text
experiments/sailboat_BO/scenario_generalization_bo_eval.yaml
```

### Final result snapshot

```text
~/sailboat_rl_results/bo/final_optimized_baseline/
```

The main snapshot contains:

```text
bo_history.csv
core3_revalidation_summary.csv
holdout_baseline_summary.csv
holdout_optimized_summary.csv
iter_008_original_params.json
optimized_baseline_params.json
generalization/
```

The generalization snapshot contains:

```text
generalization/
├── scenario_generalization_bo_eval.yaml
├── baseline_eval_shifted_wind_loop_summary.csv
├── iter008_eval_shifted_wind_loop_summary.csv
├── baseline_oblique_zigzag_summary.csv
├── iter008_oblique_zigzag_summary.csv
├── baseline_reverse_variable_wind_summary.csv
└── iter008_reverse_variable_wind_summary.csv
```

The raw result directories should be retained because they contain individual trial summaries, parameter files, generated worlds, telemetry data, and simulation logs required to reproduce or audit the final baseline selection.

The final optimized parameter set should remain frozen unless a new optimization experiment is explicitly defined and evaluated independently.

---
