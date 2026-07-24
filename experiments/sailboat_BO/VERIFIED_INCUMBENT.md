# Verified BO incumbent

This file fixes the current parameter incumbent before the reinforcement-learning stage.

## Parameter artifact

- File: `experiments/sailboat_BO/verified_incumbent_params.json`
- Source BO: `results/bo/generalization_v3_20260721`
- Source iteration: `iter_019`
- Search type: baseline-seeded BO
- BO seed: `42`
- BO training objective: `0.32555198345814584`
- BO training failure count: `0`

| Parameter | Value |
| --- | ---: |
| `SAIL_ANGLE_IDEAL` | 30.545678146440927 |
| `SAIL_NO_GO_ANGLE` | 60.48161820397466 |
| `SAIL_XTRACK_MAX` | 6.105825999530261 |
| `ATC_SAIL_P` | 0.05260268861899456 |
| `ATC_STR_ANG_P` | 2.6238889881381717 |
| `ATC_STR_RAT_FF` | 0.7859412892077355 |

## Repeated validation

Generalization train validation used five scenarios with five repeats each.

| Candidate | Success | Mean cost | Mean mission time (s) | Mean final distance (m) |
| --- | ---: | ---: | ---: | ---: |
| v3 iteration 19 | 25/25 | 0.3338363337 | 101.05596 | 3.96429 |
| baseline | 24/25 | 0.5913581446 | 102.18400 | 4.05257 |

Corrected final evaluation used the corrected 8% wind-variation scenario with five repeats.

| Candidate | Success | Mean cost | Mean mission time (s) | Mean XTE RMS (m) | Mean final distance (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 iteration 19 | 5/5 | 0.4076523492 | 130.2400 | 6.11612 | 2.16495 |
| baseline | 5/5 | 0.4761605804 | 137.6598 | 8.45785 | 5.47973 |

On corrected final evaluation, the incumbent reduced mean cost by 14.4%, mission time by 5.4%, XTE RMS by 27.7%, and final waypoint distance by 60.5% relative to the baseline.

## Interpretation and limitation

The incumbent is the best currently verified static ArduPilot parameter set and should be used as the fixed controller baseline for RL experiments. It was discovered by a baseline-seeded search, so it is not evidence that an unseeded BO search converges to the same solution. The v3 BO training run also included the earlier mis-scaled variable-wind scenario; the clean evidence for retaining it is the later corrected final evaluation above.

The partially started unseeded run under `results/bo/corrected_unseeded_20260723` is preserved but is not part of this incumbent decision.

## BO-to-RL handoff archive

- Archive: `experiments/sailboat_BO/archive/bo_rl_handoff_20260723.tar.gz`
- SHA-256 manifest: `experiments/sailboat_BO/archive/bo_rl_handoff_20260723.sha256`
- Archive contents: the complete host-side BO results tree, scenarios, baseline and incumbent parameter files, BO execution/scoring code, tests, and this provenance document
- Snapshot date: 2026-07-23

Verify the archive before restoring it:

```bash
cd experiments/sailboat_BO/archive
sha256sum -c bo_rl_handoff_20260723.sha256
```

## Reproduction command

```bash
python3 experiments/sailboat_BO/run_batch.py \
  --scenario experiments/sailboat_BO/scenario_generalization.yaml \
  --params-file experiments/sailboat_BO/verified_incumbent_params.json \
  --split eval \
  --repeats 5 \
  --results-dir experiments/sailboat_BO/results/validation/verified_incumbent_eval_r5 \
  --execution-backend local \
  --keep-going
```
