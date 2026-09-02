# BO-Optimized Sailboat Baseline

## 1. Overview

This document records the Bayesian Optimization (BO) procedure used to determine the final baseline controller parameters for the autonomous sailboat simulation.

The optimization was performed after stabilizing the sailboat physics, coordinate frames, actuator behavior, wind handling, and startup initial conditions.

The final optimized parameter set was selected through the following procedure:

1. Run Bayesian Optimization on three training scenarios.
2. Select the top three failure-free BO candidates.
3. Re-evaluate the top three candidates and the original baseline using five repeated trials per training scenario.
4. Select the best candidate based on the predefined mean scenario cost.
5. Evaluate the selected candidate on a held-out scenario.
6. Fix the validated candidate as the final optimized baseline.

---

## 2. Experiment Environment

### Repository baseline

Physics and startup stabilization baseline:

```text
Git commit: 521811c
```

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
Best iteration      : iter_008
Failure count       : 0
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

Note that `iter_008` was found relatively early in the BO run. Therefore, the result is interpreted as a successful 40-candidate parameter search rather than evidence of strong Gaussian-process convergence.

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

Therefore, the predefined optimization criterion was mean scenario cost rather than worst-case cost. Changing the selection rule after optimization would change the original experimental criterion, so `iter_008` was retained as the final candidate.

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

Therefore, the BO result should not be interpreted as simply increasing sailing speed.

The optimized parameters improved the combined navigation objective, especially cross-track tracking performance, while maintaining successful waypoint completion and acceptable roll behavior.

---

## 7. Held-Out Validation

After selecting `iter_008`, its parameters were fixed and evaluated without further tuning.

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

---

## 8. Final Decision

The final baseline controller for subsequent experiments is:

```text
iter_008
```

Final parameter file:

```text
experiments/sailboat_BO/optimized_baseline_params.json
```

The candidate was accepted because:

* it achieved the lowest mean cost in the BO search;
* it remained the best candidate after independent five-repeat revalidation;
* all 15 core revalidation trials succeeded;
* all five held-out trials succeeded;
* held-out scenario cost and XTE were improved relative to the original baseline;
* no additional parameter tuning was performed using the held-out result.

Therefore, `iter_008` is fixed as the optimized baseline controller for subsequent residual reinforcement-learning experiments.

---

## 9. Result Provenance

Raw BO results:

```text
~/sailboat_rl_results/bo/
validated_ros2_521811c_core3_seed42_20260901T074459Z/
```

Top-3 revalidation:

```text
~/sailboat_rl_results/bo_validation/
```

Held-out validation:

```text
~/sailboat_rl_results/bo_holdout/
```

Final result snapshot:

```text
~/sailboat_rl_results/bo/final_optimized_baseline/
```

The snapshot contains:

```text
bo_history.csv
core3_revalidation_summary.csv
holdout_baseline_summary.csv
holdout_optimized_summary.csv
iter_008_original_params.json
optimized_baseline_params.json
```

The raw directories above should be retained because they contain individual trial summaries and simulation logs required to reproduce or audit the final selection.

---
