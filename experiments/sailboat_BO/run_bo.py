from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    from .common import DEFAULT_PARAMS_PATH, DEFAULT_RESULTS_DIR, DEFAULT_SCENARIO_PATH, load_yaml
except ImportError:
    from common import DEFAULT_PARAMS_PATH, DEFAULT_RESULTS_DIR, DEFAULT_SCENARIO_PATH, load_yaml


DEFAULT_OBJECTIVE_COLUMN = "scenario_cost"
MISSING_TRIAL_PENALTY = 10.0


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type_name: str
    low: float
    high: float


@dataclass(frozen=True)
class Observation:
    iteration: int
    params: dict[str, float]
    objective: float
    failure_count: int = 0


@dataclass(frozen=True)
class IterationResult:
    iteration: int
    params: dict[str, float]
    objective: float
    mean_scenario_cost: float
    worst_scenario_cost: float
    trial_count: int
    expected_trial_count: int
    failure_count: int
    returncode: int
    summary_csv: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dependency-free BO over sailboat trial parameters")
    parser.add_argument("--scenario", default=str(DEFAULT_SCENARIO_PATH), help="Path to scenario.yaml")
    parser.add_argument("--split", choices=["train", "eval", "all"], default="train")
    parser.add_argument("--scenario-ids", nargs="*", help="Optional explicit scenario ids")
    parser.add_argument("--iterations", type=int, default=20, help="Total BO iterations to run")
    parser.add_argument(
        "--baseline-params",
        default=str(DEFAULT_PARAMS_PATH),
        help="Flat JSON parameter file evaluated as iteration 0 for a new BO run",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help=(
            "Start without evaluating or seeding from baseline parameters. "
            "The first --initial-random iterations are sampled globally before GP/EI."
        ),
    )
    parser.add_argument(
        "--initial-random",
        type=int,
        help=(
            "Warmup candidates after the baseline, or total global-random warmup "
            "iterations with --no-baseline, before GP/EI suggestions"
        ),
    )
    parser.add_argument(
        "--local-warmup",
        type=int,
        default=6,
        help="Number of warmup candidates sampled near the baseline",
    )
    parser.add_argument(
        "--local-radius",
        type=float,
        default=0.15,
        help="Half-width of local sampling as a fraction of each search-space span",
    )
    parser.add_argument("--candidate-pool", type=int, default=1024, help="Random candidates scored by EI")
    parser.add_argument("--repeats", type=int, default=1, help="Repeats per scenario for each parameter set")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Base results directory")
    parser.add_argument("--bo-dir", help="Directory for BO history. Defaults under results/bo/")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing --bo-dir")
    parser.add_argument("--seed", type=int, help="Random seed. Defaults to study.seed in scenario.yaml")
    parser.add_argument("--objective-column", default=DEFAULT_OBJECTIVE_COLUMN)
    parser.add_argument(
        "--worst-case-weight",
        type=float,
        default=0.0,
        help=(
            "Blend the mean trial cost with the worst per-scenario mean cost. "
            "0 preserves mean-only scoring; 1 uses only the worst scenario."
        ),
    )
    parser.add_argument("--execution-backend", choices=["local", "docker-exec"])
    parser.add_argument("--docker-container")
    parser.add_argument("--container-repo-root")
    parser.add_argument("--trial-dry-run", action="store_true", help="Forward --dry-run to run_batch.py")
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop a batch iteration on the first failed trial instead of collecting all scenario outcomes",
    )
    return parser.parse_args()


def parse_search_space(config: dict[str, Any]) -> list[ParamSpec]:
    specs: list[ParamSpec] = []
    for item in config.get("search_space", []):
        type_name = str(item.get("type", "float")).lower()
        if type_name not in {"float", "int"}:
            raise ValueError(f"Unsupported parameter type for BO: {item.get('name')}={type_name}")
        specs.append(
            ParamSpec(
                name=str(item["name"]),
                type_name=type_name,
                low=float(item["low"]),
                high=float(item["high"]),
            )
        )
    if not specs:
        raise ValueError("scenario.yaml search_space is empty")
    return specs


def select_scenario_ids(config: dict[str, Any], split: str, scenario_ids: list[str] | None) -> list[str]:
    scenarios = config.get("scenarios", [])
    if scenario_ids:
        available = {str(scenario.get("id")) for scenario in scenarios}
        missing = sorted(set(scenario_ids) - available)
        if missing:
            raise KeyError(f"Unknown scenario ids: {', '.join(missing)}")
        return list(scenario_ids)
    if split == "all":
        return [str(scenario["id"]) for scenario in scenarios]
    selected = [str(scenario["id"]) for scenario in scenarios if scenario.get("split") == split]
    if not selected:
        raise ValueError(f"No scenarios selected for split={split}")
    return selected


def default_bo_dir(results_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return results_dir / "bo" / stamp


def encode_params(params: dict[str, float], specs: list[ParamSpec]) -> list[float]:
    encoded: list[float] = []
    for spec in specs:
        span = max(spec.high - spec.low, 1e-12)
        encoded.append((float(params[spec.name]) - spec.low) / span)
    return encoded


def decode_vector(vector: Iterable[float], specs: list[ParamSpec]) -> dict[str, float]:
    params: dict[str, float] = {}
    for value, spec in zip(vector, specs):
        clamped = max(0.0, min(1.0, float(value)))
        raw = spec.low + clamped * (spec.high - spec.low)
        if spec.type_name == "int":
            raw = float(round(raw))
        params[spec.name] = raw
    return params


def sample_random_params(specs: list[ParamSpec], rng: random.Random) -> dict[str, float]:
    return decode_vector((rng.random() for _ in specs), specs)


def sample_local_params(
    center_params: dict[str, float],
    specs: list[ParamSpec],
    rng: random.Random,
    *,
    radius: float,
) -> dict[str, float]:
    if not 0.0 < radius <= 1.0:
        raise ValueError("local radius must be in (0, 1]")
    center = encode_params(center_params, specs)
    return decode_vector(
        (value + rng.uniform(-radius, radius) for value in center),
        specs,
    )


def params_key(params: dict[str, float], specs: list[ParamSpec]) -> tuple[float, ...]:
    return tuple(round(float(params[spec.name]), 8) for spec in specs)


def load_params_for_specs(path: str | Path, specs: list[ParamSpec]) -> dict[str, float]:
    parsed = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON mapping at {path}")
    params: dict[str, float] = {}
    for spec in specs:
        if spec.name not in parsed:
            raise ValueError(f"Baseline parameter file is missing {spec.name}: {path}")
        value = safe_float(parsed[spec.name])
        if not math.isfinite(value):
            raise ValueError(f"Baseline parameter is not numeric: {spec.name}={parsed[spec.name]!r}")
        if value < spec.low or value > spec.high:
            raise ValueError(
                f"Baseline parameter is outside the search space: "
                f"{spec.name}={value} not in [{spec.low}, {spec.high}]"
            )
        if spec.type_name == "int":
            value = float(round(value))
        params[spec.name] = value
    return params


def sample_unique_params(
    sampler: Any,
    observations: list[Observation],
    specs: list[ParamSpec],
) -> dict[str, float]:
    tried = {params_key(observation.params, specs) for observation in observations}
    for _ in range(1000):
        params = sampler()
        if params_key(params, specs) not in tried:
            return params
    raise RuntimeError("Unable to sample an untried parameter set")


def propose_iteration_params(
    *,
    iteration: int,
    observations: list[Observation],
    baseline_params: Optional[dict[str, float]],
    specs: list[ParamSpec],
    rng: random.Random,
    initial_random: int,
    local_warmup: int,
    local_radius: float,
    candidate_pool: int,
) -> tuple[dict[str, float], str]:
    if baseline_params is None:
        if iteration < initial_random:
            params = sample_unique_params(
                lambda: sample_random_params(specs, rng),
                observations,
                specs,
            )
            return params, "global_warmup"
        return (
            suggest_params(
                observations,
                specs,
                rng,
                initial_random=0,
                candidate_pool=candidate_pool,
            ),
            "gp_ei",
        )

    if iteration == 0 and not observations:
        return dict(baseline_params), "baseline"

    warmup_index = iteration - 1
    local_count = min(local_warmup, initial_random)
    if 0 <= warmup_index < local_count:
        params = sample_unique_params(
            lambda: sample_local_params(
                baseline_params,
                specs,
                rng,
                radius=local_radius,
            ),
            observations,
            specs,
        )
        return params, "local_warmup"
    if 0 <= warmup_index < initial_random:
        params = sample_unique_params(
            lambda: sample_random_params(specs, rng),
            observations,
            specs,
        )
        return params, "global_warmup"
    return (
        suggest_params(
            observations,
            specs,
            rng,
            initial_random=0,
            candidate_pool=candidate_pool,
        ),
        "gp_ei",
    )


def rbf_kernel(a: list[float], b: list[float], *, length_scale: float = 0.35) -> float:
    sqdist = sum((x - y) ** 2 for x, y in zip(a, b))
    return math.exp(-0.5 * sqdist / max(length_scale * length_scale, 1e-12))


def cholesky(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    lower = [[0.0 for _ in range(size)] for _ in range(size)]
    for row in range(size):
        for col in range(row + 1):
            total = sum(lower[row][k] * lower[col][k] for k in range(col))
            if row == col:
                lower[row][col] = math.sqrt(max(matrix[row][row] - total, 1e-12))
            else:
                lower[row][col] = (matrix[row][col] - total) / max(lower[col][col], 1e-12)
    return lower


def solve_lower(lower: list[list[float]], values: list[float]) -> list[float]:
    result: list[float] = []
    for row, value in enumerate(values):
        total = sum(lower[row][col] * result[col] for col in range(row))
        result.append((value - total) / max(lower[row][row], 1e-12))
    return result


def solve_upper_from_lower(lower: list[list[float]], values: list[float]) -> list[float]:
    size = len(lower)
    result = [0.0 for _ in range(size)]
    for row in range(size - 1, -1, -1):
        total = sum(lower[col][row] * result[col] for col in range(row + 1, size))
        result[row] = (values[row] - total) / max(lower[row][row], 1e-12)
    return result


def normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def expected_improvement(best: float, mean: float, sigma: float) -> float:
    if sigma <= 1e-12:
        return max(0.0, best - mean)
    z_score = (best - mean) / sigma
    return (best - mean) * normal_cdf(z_score) + sigma * normal_pdf(z_score)


def observation_rank(observation: Observation) -> tuple[int, float, int]:
    """Prefer fewer failures, then a lower objective, then an earlier iteration."""
    return observation.failure_count, observation.objective, observation.iteration


def select_best_observation(observations: list[Observation]) -> Observation:
    if not observations:
        raise ValueError("Cannot select a best observation from an empty list")
    return min(observations, key=observation_rank)


def predict_gp(
    observations: list[Observation],
    specs: list[ParamSpec],
    candidate: dict[str, float],
) -> tuple[float, float]:
    x_train = [encode_params(observation.params, specs) for observation in observations]
    y_values = [observation.objective for observation in observations]
    y_mean = sum(y_values) / len(y_values)
    y_scale = max(
        math.sqrt(sum((value - y_mean) ** 2 for value in y_values) / len(y_values)),
        1e-6,
    )
    y_train = [(value - y_mean) / y_scale for value in y_values]

    size = len(observations)
    kernel_matrix = [
        [
            rbf_kernel(x_train[row], x_train[col]) + (1e-6 if row == col else 0.0)
            for col in range(size)
        ]
        for row in range(size)
    ]
    lower = cholesky(kernel_matrix)
    alpha = solve_upper_from_lower(lower, solve_lower(lower, y_train))
    x_star = encode_params(candidate, specs)
    k_star = [rbf_kernel(x, x_star) for x in x_train]
    mean_std = sum(k * a for k, a in zip(k_star, alpha))
    v_vec = solve_lower(lower, k_star)
    var_std = max(1.0 - sum(v * v for v in v_vec), 1e-12)
    return y_mean + y_scale * mean_std, y_scale * math.sqrt(var_std)


def suggest_params(
    observations: list[Observation],
    specs: list[ParamSpec],
    rng: random.Random,
    *,
    initial_random: int,
    candidate_pool: int,
) -> dict[str, float]:
    tried = {params_key(observation.params, specs) for observation in observations}
    if not observations or len(observations) < initial_random:
        for _ in range(1000):
            params = sample_random_params(specs, rng)
            if params_key(params, specs) not in tried:
                return params
        return sample_random_params(specs, rng)

    incumbent = select_best_observation(observations)
    best_value = incumbent.objective
    best_params = incumbent.params
    best_candidate: dict[str, float] | None = None
    best_ei = -1.0
    for idx in range(max(candidate_pool, 1)):
        if idx < candidate_pool // 4:
            center = encode_params(best_params, specs)
            vector = [min(1.0, max(0.0, value + rng.gauss(0.0, 0.15))) for value in center]
            candidate = decode_vector(vector, specs)
        else:
            candidate = sample_random_params(specs, rng)
        if params_key(candidate, specs) in tried:
            continue
        mean, sigma = predict_gp(observations, specs, candidate)
        ei = expected_improvement(best_value, mean, sigma)
        if ei > best_ei:
            best_ei = ei
            best_candidate = candidate
    return best_candidate if best_candidate is not None else sample_random_params(specs, rng)


def read_summary_rows(summary_csv: Path) -> list[dict[str, str]]:
    if not summary_csv.exists():
        return []
    with summary_csv.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def aggregate_objective(
    rows: list[dict[str, str]],
    *,
    objective_column: str,
    expected_trial_count: int,
    worst_case_weight: float = 0.0,
    missing_trial_penalty: float = MISSING_TRIAL_PENALTY,
) -> tuple[float, float, float, int]:
    if not 0.0 <= worst_case_weight <= 1.0:
        raise ValueError("worst_case_weight must be in [0, 1]")

    costs: list[float] = []
    costs_by_scenario: dict[str, list[float]] = {}
    for index, row in enumerate(rows):
        cost = safe_float(row.get(objective_column), missing_trial_penalty)
        costs.append(cost)
        scenario_id = row.get("scenario_id") or f"__trial_{index}"
        costs_by_scenario.setdefault(scenario_id, []).append(cost)

    failure_count = sum(1 for row in rows if row.get("status") != "success")
    missing = max(0, expected_trial_count - len(rows))
    costs.extend([missing_trial_penalty] * missing)
    if missing:
        costs_by_scenario["__missing__"] = [missing_trial_penalty] * missing
    failure_count += missing
    if not costs:
        return (
            missing_trial_penalty,
            missing_trial_penalty,
            missing_trial_penalty,
            max(1, expected_trial_count),
        )

    mean_cost = sum(costs) / len(costs)
    worst_scenario_cost = max(
        sum(scenario_costs) / len(scenario_costs)
        for scenario_costs in costs_by_scenario.values()
    )
    objective = (
        (1.0 - worst_case_weight) * mean_cost
        + worst_case_weight * worst_scenario_cost
    )
    return objective, mean_cost, worst_scenario_cost, failure_count


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_history_csv(path: Path, result: IterationResult, specs: list[ParamSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "iteration",
        "objective",
        "mean_scenario_cost",
        "worst_scenario_cost",
        "trial_count",
        "expected_trial_count",
        "failure_count",
        "returncode",
        "summary_csv",
    ] + [f"param__{spec.name}" for spec in specs]
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        row = {
            "iteration": result.iteration,
            "objective": result.objective,
            "mean_scenario_cost": result.mean_scenario_cost,
            "worst_scenario_cost": result.worst_scenario_cost,
            "trial_count": result.trial_count,
            "expected_trial_count": result.expected_trial_count,
            "failure_count": result.failure_count,
            "returncode": result.returncode,
            "summary_csv": result.summary_csv,
        }
        for spec in specs:
            row[f"param__{spec.name}"] = result.params[spec.name]
        writer.writerow(row)


def load_history(path: Path, specs: list[ParamSpec]) -> list[Observation]:
    if not path.exists():
        return []
    observations: list[Observation] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            params = {spec.name: safe_float(row.get(f"param__{spec.name}")) for spec in specs}
            observations.append(
                Observation(
                    iteration=int(safe_float(row.get("iteration"), 0.0)),
                    params=params,
                    objective=safe_float(row.get("objective"), MISSING_TRIAL_PENALTY),
                    failure_count=int(safe_float(row.get("failure_count"), 0.0)),
                )
            )
    return observations


def build_batch_command(
    *,
    scenario_file: str,
    scenario_ids: list[str],
    params_file: Path,
    results_dir: Path,
    repeats: int,
    execution_backend: str | None,
    docker_container: str | None,
    container_repo_root: str | None,
    trial_dry_run: bool,
    keep_going: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "experiments/sailboat_BO/run_batch.py",
        "--scenario",
        scenario_file,
        "--params-file",
        str(params_file),
        "--scenario-ids",
        *scenario_ids,
        "--repeats",
        str(repeats),
        "--results-dir",
        str(results_dir),
    ]
    if execution_backend:
        cmd.extend(["--execution-backend", execution_backend])
    if docker_container:
        cmd.extend(["--docker-container", docker_container])
    if container_repo_root:
        cmd.extend(["--container-repo-root", container_repo_root])
    if trial_dry_run:
        cmd.append("--dry-run")
    if keep_going:
        cmd.append("--keep-going")
    return cmd


def run_iteration(
    *,
    iteration: int,
    params: dict[str, float],
    specs: list[ParamSpec],
    bo_dir: Path,
    scenario_file: str,
    scenario_ids: list[str],
    repeats: int,
    objective_column: str,
    worst_case_weight: float,
    execution_backend: str | None,
    docker_container: str | None,
    container_repo_root: str | None,
    trial_dry_run: bool,
    keep_going: bool,
) -> IterationResult:
    iter_dir = bo_dir / f"iter_{iteration:03d}"
    trials_dir = iter_dir / "trials"
    params_file = iter_dir / "params.json"
    write_json(params_file, params)
    command = build_batch_command(
        scenario_file=scenario_file,
        scenario_ids=scenario_ids,
        params_file=params_file,
        results_dir=trials_dir,
        repeats=repeats,
        execution_backend=execution_backend,
        docker_container=docker_container,
        container_repo_root=container_repo_root,
        trial_dry_run=trial_dry_run,
        keep_going=keep_going,
    )
    write_json(iter_dir / "command.json", command)
    print(f"[run_bo] iteration {iteration}: launching {' '.join(command)}", flush=True)
    completed = subprocess.run(command, check=False)
    summary_csv = trials_dir / "summary.csv"
    rows = read_summary_rows(summary_csv)
    expected_trial_count = len(scenario_ids) * repeats
    objective, mean_scenario_cost, worst_scenario_cost, failure_count = aggregate_objective(
        rows,
        objective_column=objective_column,
        expected_trial_count=expected_trial_count,
        worst_case_weight=worst_case_weight,
    )
    result = IterationResult(
        iteration=iteration,
        params=params,
        objective=objective,
        mean_scenario_cost=mean_scenario_cost,
        worst_scenario_cost=worst_scenario_cost,
        trial_count=len(rows),
        expected_trial_count=expected_trial_count,
        failure_count=failure_count,
        returncode=completed.returncode,
        summary_csv=str(summary_csv),
    )
    write_json(iter_dir / "iteration_result.json", result.__dict__)
    append_history_csv(bo_dir / "history.csv", result, specs)
    with (bo_dir / "observations.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result.__dict__, sort_keys=True) + "\n")
    return result


def main() -> int:
    args = parse_args()
    config = load_yaml(args.scenario)
    specs = parse_search_space(config)
    scenario_ids = select_scenario_ids(config, args.split, args.scenario_ids)
    results_dir = Path(args.results_dir)
    bo_dir = Path(args.bo_dir) if args.bo_dir else default_bo_dir(results_dir)
    bo_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed if args.seed is not None else int(config.get("study", {}).get("seed", 0))
    initial_random = args.initial_random if args.initial_random is not None else max(len(specs) + 1, 2 * len(specs))
    if args.iterations < 1:
        raise SystemExit("--iterations must be at least 1")
    if initial_random < 0:
        raise SystemExit("--initial-random cannot be negative")
    if args.local_warmup < 0:
        raise SystemExit("--local-warmup cannot be negative")
    if not 0.0 < args.local_radius <= 1.0:
        raise SystemExit("--local-radius must be in (0, 1]")
    if not 0.0 <= args.worst_case_weight <= 1.0:
        raise SystemExit("--worst-case-weight must be in [0, 1]")
    baseline_params = None if args.no_baseline else load_params_for_specs(args.baseline_params, specs)
    local_warmup = 0 if args.no_baseline else args.local_warmup
    history_csv = bo_dir / "history.csv"
    observations = load_history(history_csv, specs) if args.resume else []
    start_iteration = len(observations)

    write_json(
        bo_dir / "bo_config.json",
        {
            "scenario": args.scenario,
            "scenario_ids": scenario_ids,
            "split": args.split,
            "iterations": args.iterations,
            "no_baseline": args.no_baseline,
            "baseline_params_file": None if args.no_baseline else args.baseline_params,
            "baseline_params": baseline_params,
            "initial_random": initial_random,
            "local_warmup": local_warmup,
            "local_radius": args.local_radius,
            "candidate_pool": args.candidate_pool,
            "repeats": args.repeats,
            "seed": seed,
            "objective_column": args.objective_column,
            "worst_case_weight": args.worst_case_weight,
            "search_space": [spec.__dict__ for spec in specs],
        },
    )

    print(f"[run_bo] bo_dir={bo_dir}", flush=True)
    print(f"[run_bo] optimizing scenarios={','.join(scenario_ids)} repeats={args.repeats}", flush=True)
    for iteration in range(start_iteration, args.iterations):
        iteration_rng = random.Random(f"{seed}:{iteration}")
        params, proposal_source = propose_iteration_params(
            iteration=iteration,
            observations=observations,
            baseline_params=baseline_params,
            specs=specs,
            rng=iteration_rng,
            initial_random=initial_random,
            local_warmup=local_warmup,
            local_radius=args.local_radius,
            candidate_pool=args.candidate_pool,
        )
        print(
            f"[run_bo] iteration {iteration} proposal_source={proposal_source}",
            flush=True,
        )
        write_json(
            bo_dir / f"iter_{iteration:03d}" / "proposal.json",
            {
                "iteration": iteration,
                "source": proposal_source,
                "seed": seed,
                "params": params,
            },
        )
        result = run_iteration(
            iteration=iteration,
            params=params,
            specs=specs,
            bo_dir=bo_dir,
            scenario_file=args.scenario,
            scenario_ids=scenario_ids,
            repeats=args.repeats,
            objective_column=args.objective_column,
            worst_case_weight=args.worst_case_weight,
            execution_backend=args.execution_backend,
            docker_container=args.docker_container,
            container_repo_root=args.container_repo_root,
            trial_dry_run=args.trial_dry_run,
            keep_going=not args.stop_on_failure,
        )
        observations.append(
            Observation(
                iteration=iteration,
                params=params,
                objective=result.objective,
                failure_count=result.failure_count,
            )
        )
        best = select_best_observation(observations)
        write_json(
            bo_dir / "best_params.json",
            {
                "iteration": best.iteration,
                "objective": best.objective,
                "failure_count": best.failure_count,
                "params": best.params,
            },
        )
        print(
            "[run_bo] iteration "
            f"{iteration} objective={result.objective:.4f} "
            f"mean={result.mean_scenario_cost:.4f} "
            f"worst={result.worst_scenario_cost:.4f} "
            f"failures={result.failure_count}/"
            f"{result.expected_trial_count}; best failures={best.failure_count}, "
            f"objective={best.objective:.4f} at iteration {best.iteration}",
            flush=True,
        )

    best = select_best_observation(observations)
    print(
        f"[run_bo] complete: best failures={best.failure_count}, "
        f"objective={best.objective:.4f} iteration={best.iteration}",
        flush=True,
    )
    print(f"[run_bo] best params: {json.dumps(best.params, sort_keys=True)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
