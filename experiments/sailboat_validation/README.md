# Independent BO incumbent validation v1

This package does not change BO, production physics, controller or framework.
Only the explicit `run` action starts trials. `prepare` freezes a fresh sidecar;
`preflight`, `status`, `run --dry-run` and `compare` never run a simulation.

## Frozen experiment

- A: original iter_008 JSON bytes, equal to this BO campaign iteration 0.
- B: original completed campaign `iter_014/params.json`, not best_params wrapper.
- Immutable accepted source `4ec49538912fe88ca3be569c1a787dba4c07994b` and image
  `sha256:c7dd2ca9507f72a6937b694f7318560a1f6c923035e561a9884d2b68458c16bc`.
- Core-3: 3 scenarios x 2 arms x 2 repeats = 12 maximum trials.
- Held-out: long-oblique / shifted-wind loop x 2 arms x 2 repeats = 8 maximum.
- Per scenario: r00 A, B; r01 B, A. No parameter search/proposal/RNG changes.
- YAML is a deep copy of accepted scenario.yaml, appending only the existing
  shifted-wind scenario from scenario_generalization.yaml. The existing
  long-oblique definition and every global setting/search range stay unchanged.
  The old generalization YAML's narrow bounds are NOT copied. Exact original
  A/B file SHA is enforced before every trial; parameters are never clamped.
- The original contract and evaluators are not revised. The new validation
  manifest explicitly authorizes these five scenarios and a different external
  aggregation scope; it does not pretend to satisfy the BO-only scenario hash.
  Every trial uses unchanged patch1 evaluate_trial(), fixed evaluation no-go
  60 degrees, frozen Layer1/2/3, framework penalty 5, runtime-invalid penalty 10.
- Effective cost is averaged separately in Core-3 and held-out, with identical
  scenario/repeat pairs for A and B. Objective weights are not renormalized.
  Raw run_trial summary.objective is not used for scoring (controller no-go).
- Proceed to held-out only after all Core-3 runtime checks pass, B is 6/6
  feasible, and mean effective cost B < A. Final held-out requires B 4/4 feasible
  and mean effective cost B <= A. A feasibility is reported, not silently forced.
- Runtime-invalid/process error/partial slot stops immediately. B infeasible
  also stops; artifacts/penalties are preserved. There is no retry or recovery
  switch. Reissuing a completed phase never reruns a trial. A partial slot blocks
  all subsequent execution, including unrelated slots.
- Two repeats support descriptive comparisons only, not significance or broad
  robustness. Seed 42 remains study metadata: run_trial has no explicit Gazebo
  FFT/SITL RNG seed injection, so identical stochastic realizations are NOT
  claimed. Known pitch/daggerboard limitations remain deferred.

## Output and isolation

`validation_manifest.json` + `.sha256` bind A/B, source/image, scenarios, seed,
repeats, tooling, contract, BO provenance and the full 20-slot schedule.
`inputs/` holds exact-byte provenance snapshots; execution reads original A/B
paths mounted read-only. `slots/<slot>/` holds a single fresh trial, console,
command and completion receipt with all raw/artifact SHA hashes. Compact
navigation and framework results are included in phase comparisons.

`core3_comparison.json`, `heldout_comparison.json`, `final_comparison.json`
are separate, no-clobber artifacts. STOP.json records any execution abort.
Original BO history and all original results are mounted read-only. Only this
validation output is writable, with private generated-world output. No whole
repository or diagnostic physics overlay is mounted. All tooling/evaluators
are read-only mounts; the accepted image is not rebuilt or edited.

## User steps (WSL, individually)

From the production-candidate worktree set `VAL` to the prepared sidecar path.
Always inspect PASS/output before proceeding. Do not use shell `set -u` when
sourcing ROS/colcon setup scripts (COLCON_TRACE may be unset).

1. `python3 -B -m experiments.sailboat_validation.validate_ab preflight --output "$VAL" --docker`
2. `python3 -B -m experiments.sailboat_validation.validate_ab shell --output "$VAL"`
3. Inside container: `source /home/docker/sailboat_ws/install/setup.bash`
4. `python3 -B -m experiments.sailboat_validation.validate_ab preflight --runtime`
5. `python3 -B -m experiments.sailboat_validation.validate_ab run --phase core3 --dry-run`
6. **Starts simulations (maximum 12)**: `python3 -B -m experiments.sailboat_validation.validate_ab run --phase core3`
7. Inspect `core3_comparison.json`; PASS required. No automatic held-out launch.
8. **Starts simulations (maximum 8)**: `python3 -B -m experiments.sailboat_validation.validate_ab run --phase heldout`
9. `python3 -B -m experiments.sailboat_validation.validate_ab compare`

On failure/interrupt, stop. Do not delete slots/STOP or create a new attempt.
`status` is a read-only inventory for later audit, not a recovery command.
No promotion, commit, new BO observation, RL or physics modification follows
automatically, even if validation passes.
