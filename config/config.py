from dataclasses import dataclass
from pathlib import Path
import tomllib

from race_model.models import Athlete


@dataclass(frozen=True)
class ReferenceCheck:
    n_hotdogs: int
    distance_m: float


@dataclass(frozen=True)
class SimulationSettings:
    athlete_a: str
    athlete_b: str
    min_distance_m: float
    max_distance_m: float
    distance_samples: int
    max_hotdogs: int
    monte_carlo_samples: int
    random_seed: int
    reference_checks: tuple[ReferenceCheck, ...]


@dataclass(frozen=True)
class AppConfig:
    sprint_threshold_distance_m: float
    sprint_reference_distance_m: float
    reference_distance_m: float
    max_modeled_distance_m: float
    runner_types: dict[str, float]
    athletes: dict[str, Athlete]
    simulation: SimulationSettings

    def get_athlete(self, runner_name: str) -> Athlete:
        athlete = self.athletes.get(runner_name)
        if athlete is None:
            raise ValueError(f"Unknown runner '{runner_name}'")
        return athlete


def load_app_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path)
    with path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)

    running_model = _require_table(raw_config, "running_model", "root")
    sprint_threshold_distance_m = _require_positive_float(
        running_model,
        "sprint_threshold_distance",
        "running_model",
    )
    sprint_reference_distance_m = _require_positive_float(
        running_model,
        "sprint_reference_distance",
        "running_model",
    )
    reference_distance_m = _require_positive_float(
        running_model,
        "reference_distance",
        "running_model",
    )
    max_modeled_distance_m = _require_positive_float(
        running_model,
        "max_modeled_distance",
        "running_model",
    )

    if sprint_reference_distance_m >= sprint_threshold_distance_m:
        raise ValueError(
            "running_model.sprint_reference_distance must be < "
            "running_model.sprint_threshold_distance"
        )
    if reference_distance_m <= sprint_threshold_distance_m:
        raise ValueError(
            "running_model.reference_distance must be > "
            "running_model.sprint_threshold_distance"
        )
    if max_modeled_distance_m < reference_distance_m:
        raise ValueError(
            "running_model.max_modeled_distance must be >= "
            "running_model.reference_distance"
        )

    runner_types_table = _require_table(
        running_model,
        "runner_types",
        "running_model",
    )
    runner_types = {
        runner_type_name: _require_positive_float(
            _require_table(
                runner_types_table,
                runner_type_name,
                "running_model.runner_types",
            ),
            "k",
            f"running_model.runner_types.{runner_type_name}",
        )
        for runner_type_name in runner_types_table
    }
    if not runner_types:
        raise ValueError("running_model.runner_types must not be empty")

    runners_table = _require_table(running_model, "runners", "running_model")
    athletes: dict[str, Athlete] = {}
    for runner_name in runners_table:
        runner_table = _require_table(
            runners_table,
            runner_name,
            "running_model.runners",
        )
        runner_type_name = _require_string(
            runner_table,
            "type",
            f"running_model.runners.{runner_name}",
        )
        if runner_type_name not in runner_types:
            raise ValueError(
                f"running_model.runners.{runner_name}.type references unknown "
                f"runner type '{runner_type_name}'"
            )

        athletes[runner_name] = Athlete(
            name=runner_name,
            run_sprint_threshold_distance_m=sprint_threshold_distance_m,
            run_sprint_reference_distance_m=sprint_reference_distance_m,
            run_sprint_reference_time_sec=_require_positive_float(
                runner_table,
                "sprint_ref_time",
                f"running_model.runners.{runner_name}",
            ),
            run_reference_distance_m=reference_distance_m,
            run_reference_time_sec=_require_positive_float(
                runner_table,
                "ref_time",
                f"running_model.runners.{runner_name}",
            ),
            run_endurance_exponent=runner_types[runner_type_name],
            run_max_modeled_distance_m=max_modeled_distance_m,
            eat_base_sec=_require_positive_float(
                runner_table,
                "eat_base_sec",
                f"running_model.runners.{runner_name}",
            ),
            eat_slowdown_sec=_require_non_negative_float(
                runner_table,
                "eat_slowdown_sec",
                f"running_model.runners.{runner_name}",
            ),
            post_eat_linear=_require_non_negative_float(
                runner_table,
                "post_eat_linear",
                f"running_model.runners.{runner_name}",
            ),
            post_eat_quadratic=_require_non_negative_float(
                runner_table,
                "post_eat_quadratic",
                f"running_model.runners.{runner_name}",
            ),
            eat_noise_sec=_require_non_negative_float(
                runner_table,
                "eat_noise_sec",
                f"running_model.runners.{runner_name}",
            ),
            run_sprint_reference_time_rel_std=_require_non_negative_float(
                runner_table,
                "run_sprint_ref_time_rel_std",
                f"running_model.runners.{runner_name}",
            ),
            run_reference_time_rel_std=_require_non_negative_float(
                runner_table,
                "run_ref_time_rel_std",
                f"running_model.runners.{runner_name}",
            ),
            run_endurance_rel_std=_require_non_negative_float(
                runner_table,
                "run_endurance_rel_std",
                f"running_model.runners.{runner_name}",
            ),
            post_eat_rel_std=_require_non_negative_float(
                runner_table,
                "post_eat_rel_std",
                f"running_model.runners.{runner_name}",
            ),
        )

        threshold_time = athletes[runner_name].run_reference_time_sec * (
            athletes[runner_name].run_sprint_threshold_distance_m
            / athletes[runner_name].run_reference_distance_m
        ) ** athletes[runner_name].run_endurance_exponent
        if athletes[runner_name].run_sprint_reference_time_sec >= threshold_time:
            raise ValueError(
                f"running_model.runners.{runner_name}.sprint_ref_time must be < "
                "the modeled time at running_model.sprint_threshold_distance"
            )

    if not athletes:
        raise ValueError("running_model.runners must not be empty")

    simulation_table = _require_table(raw_config, "simulation", "root")
    simulation = SimulationSettings(
        athlete_a=_require_string(simulation_table, "athlete_a", "simulation"),
        athlete_b=_require_string(simulation_table, "athlete_b", "simulation"),
        min_distance_m=_require_positive_float(
            simulation_table,
            "min_distance_m",
            "simulation",
        ),
        max_distance_m=_require_positive_float(
            simulation_table,
            "max_distance_m",
            "simulation",
        ),
        distance_samples=_require_positive_int(
            simulation_table,
            "distance_samples",
            "simulation",
        ),
        max_hotdogs=_require_non_negative_int(
            simulation_table,
            "max_hotdogs",
            "simulation",
        ),
        monte_carlo_samples=_require_positive_int(
            simulation_table,
            "monte_carlo_samples",
            "simulation",
        ),
        random_seed=_require_non_negative_int(
            simulation_table,
            "random_seed",
            "simulation",
        ),
        reference_checks=_load_reference_checks(simulation_table),
    )

    if simulation.min_distance_m > simulation.max_distance_m:
        raise ValueError("simulation.min_distance_m must be <= simulation.max_distance_m")
    if simulation.max_distance_m > max_modeled_distance_m:
        raise ValueError(
            "simulation.max_distance_m must be <= running_model.max_modeled_distance"
        )
    for reference_check in simulation.reference_checks:
        if reference_check.distance_m > max_modeled_distance_m:
            raise ValueError(
                "simulation.reference_checks distance must be <= "
                "running_model.max_modeled_distance"
            )

    if simulation.athlete_a not in athletes:
        raise ValueError(
            f"simulation.athlete_a references unknown runner '{simulation.athlete_a}'"
        )
    if simulation.athlete_b not in athletes:
        raise ValueError(
            f"simulation.athlete_b references unknown runner '{simulation.athlete_b}'"
        )

    return AppConfig(
        sprint_threshold_distance_m=sprint_threshold_distance_m,
        sprint_reference_distance_m=sprint_reference_distance_m,
        reference_distance_m=reference_distance_m,
        max_modeled_distance_m=max_modeled_distance_m,
        runner_types=runner_types,
        athletes=athletes,
        simulation=simulation,
    )


def _load_reference_checks(simulation_table: dict[str, object]) -> tuple[ReferenceCheck, ...]:
    raw_checks = simulation_table.get("reference_checks", [])
    if not isinstance(raw_checks, list):
        raise ValueError("simulation.reference_checks must be an array of tables")

    checks: list[ReferenceCheck] = []
    for index, raw_check in enumerate(raw_checks):
        if not isinstance(raw_check, dict):
            raise ValueError(
                f"simulation.reference_checks[{index}] must be a table"
            )
        checks.append(
            ReferenceCheck(
                n_hotdogs=_require_non_negative_int(
                    raw_check,
                    "n_hotdogs",
                    f"simulation.reference_checks[{index}]",
                ),
                distance_m=_require_positive_float(
                    raw_check,
                    "distance_m",
                    f"simulation.reference_checks[{index}]",
                ),
            )
        )

    return tuple(checks)


def _require_table(
    parent: dict[str, object],
    key: str,
    context: str,
) -> dict[str, object]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid table '{context}.{key}'")
    return value


def _require_string(parent: dict[str, object], key: str, context: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing or invalid string '{context}.{key}'")
    return value


def _require_positive_float(parent: dict[str, object], key: str, context: str) -> float:
    value = _require_number(parent, key, context)
    if value <= 0.0:
        raise ValueError(f"'{context}.{key}' must be > 0")
    return value


def _require_non_negative_float(
    parent: dict[str, object],
    key: str,
    context: str,
) -> float:
    value = _require_number(parent, key, context)
    if value < 0.0:
        raise ValueError(f"'{context}.{key}' must be >= 0")
    return value


def _require_positive_int(parent: dict[str, object], key: str, context: str) -> int:
    value = parent.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"'{context}.{key}' must be a positive integer")
    return value


def _require_non_negative_int(parent: dict[str, object], key: str, context: str) -> int:
    value = parent.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"'{context}.{key}' must be a non-negative integer")
    return value


def _require_number(parent: dict[str, object], key: str, context: str) -> float:
    value = parent.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Missing or invalid numeric value '{context}.{key}'")
    return float(value)
