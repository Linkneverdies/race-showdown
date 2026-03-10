from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Athlete:
    name: str

    # Mean running model
    run_sprint_threshold_distance_m: float
    run_sprint_reference_distance_m: float
    run_sprint_reference_time_sec: float
    run_reference_distance_m: float
    run_reference_time_sec: float
    run_endurance_exponent: float
    run_max_modeled_distance_m: float

    # Mean eating / post-eating model
    eat_base_sec: float
    eat_slowdown_sec: float
    post_eat_linear: float
    post_eat_quadratic: float

    # Noise model
    eat_noise_sec: float
    run_sprint_reference_time_rel_std: float
    run_reference_time_rel_std: float
    run_endurance_rel_std: float
    post_eat_rel_std: float


def riegel_running_time(
    distance_m: float,
    reference_distance_m: float,
    reference_time_sec: float,
    endurance_exponent: float,
) -> float:
    """
    Riegel power-law running model:
        R(d) = T_ref * (d / d_ref) ** k
    """
    distance = np.asarray(distance_m, dtype=float)
    reference_distance = np.asarray(reference_distance_m, dtype=float)
    reference_time = np.asarray(reference_time_sec, dtype=float)
    endurance = np.asarray(endurance_exponent, dtype=float)

    if np.any(distance <= 0):
        raise ValueError("distance_m must be positive")
    if np.any(reference_distance <= 0):
        raise ValueError("reference_distance_m must be positive")
    if np.any(reference_time <= 0):
        raise ValueError("reference_time_sec must be positive")
    if np.any(endurance <= 0):
        raise ValueError("endurance_exponent must be positive")

    running_time = reference_time * np.power(distance / reference_distance, endurance)
    if running_time.ndim == 0:
        return float(running_time)
    return running_time


def sprint_running_time(
    distance_m: float,
    sprint_reference_distance_m: float,
    sprint_reference_time_sec: float,
    sprint_threshold_distance_m: float,
    threshold_time_sec: float,
) -> float:
    """
    Sprint regime for d <= d_s.

    A single Riegel curve is too rigid for sprint specialists because the same
    exponent has to explain both a 100 m dash and multi-kilometer racing. Here
    sprint ability is controlled by sprint_reference_time_sec, while endurance
    still controls the time at the transition point d_s.
    """
    distance = np.asarray(distance_m, dtype=float)
    sprint_reference_distance = np.asarray(sprint_reference_distance_m, dtype=float)
    sprint_reference_time = np.asarray(sprint_reference_time_sec, dtype=float)
    sprint_threshold_distance = np.asarray(sprint_threshold_distance_m, dtype=float)
    threshold_time = np.asarray(threshold_time_sec, dtype=float)

    if np.any(distance <= 0):
        raise ValueError("distance_m must be positive")
    if np.any(sprint_reference_distance <= 0):
        raise ValueError("sprint_reference_distance_m must be positive")
    if np.any(sprint_reference_time <= 0):
        raise ValueError("sprint_reference_time_sec must be positive")
    if np.any(sprint_threshold_distance <= sprint_reference_distance):
        raise ValueError(
            "sprint_threshold_distance_m must be greater than sprint_reference_distance_m"
        )
    if np.any(threshold_time <= sprint_reference_time):
        raise ValueError(
            "threshold_time_sec must be greater than sprint_reference_time_sec"
        )

    sprint_exponent = np.log(threshold_time / sprint_reference_time) / np.log(
        sprint_threshold_distance / sprint_reference_distance
    )
    running_time = sprint_reference_time * np.power(
        distance / sprint_reference_distance,
        sprint_exponent,
    )

    if running_time.ndim == 0:
        return float(running_time)
    return running_time


def two_regime_running_time(
    distance_m: float,
    sprint_threshold_distance_m: float,
    sprint_reference_distance_m: float,
    sprint_reference_time_sec: float,
    reference_distance_m: float,
    reference_time_sec: float,
    endurance_exponent: float,
    max_modeled_distance_m: float | None = None,
) -> float:
    """
    Two-regime running model:
    - d <= d_s uses a sprint-focused curve anchored by sprint_reference_time_sec
    - d > d_s uses the endurance Riegel curve

    The sprint curve is solved to match the endurance curve exactly at d_s, so
    the transition is continuous and configuration-driven rather than athlete-
    specific code. sprint_reference_time_sec controls short-distance ability,
    while reference_time_sec and endurance_exponent control longer-distance
    performance and slowdown.
    """
    distance = np.asarray(distance_m, dtype=float)
    sprint_threshold_distance = np.asarray(sprint_threshold_distance_m, dtype=float)

    if np.any(distance <= 0):
        raise ValueError("distance_m must be positive")
    if np.any(sprint_threshold_distance <= 0):
        raise ValueError("sprint_threshold_distance_m must be positive")
    if np.any(reference_distance_m <= sprint_threshold_distance_m):
        raise ValueError(
            "reference_distance_m must be greater than sprint_threshold_distance_m"
        )
    if max_modeled_distance_m is not None and np.any(distance > max_modeled_distance_m):
        raise ValueError(
            f"distance_m must be <= max_modeled_distance_m ({max_modeled_distance_m})"
        )

    threshold_time = riegel_running_time(
        distance_m=sprint_threshold_distance,
        reference_distance_m=reference_distance_m,
        reference_time_sec=reference_time_sec,
        endurance_exponent=endurance_exponent,
    )
    sprint_time = sprint_running_time(
        distance_m=distance,
        sprint_reference_distance_m=sprint_reference_distance_m,
        sprint_reference_time_sec=sprint_reference_time_sec,
        sprint_threshold_distance_m=sprint_threshold_distance,
        threshold_time_sec=threshold_time,
    )
    endurance_time = riegel_running_time(
        distance_m=distance,
        reference_distance_m=reference_distance_m,
        reference_time_sec=reference_time_sec,
        endurance_exponent=endurance_exponent,
    )
    running_time = np.where(distance <= sprint_threshold_distance, sprint_time, endurance_time)

    if running_time.ndim == 0:
        return float(running_time)
    return running_time


def mean_cumulative_eat_time(n_hotdogs: int, athlete: Athlete) -> float:
    """
    Mean total time to eat n_hotdogs.
    The i-th hot dog takes:
        athlete.eat_base_sec + athlete.eat_slowdown_sec * i
    """
    if n_hotdogs < 0:
        raise ValueError("n_hotdogs must be non-negative")

    if n_hotdogs == 0:
        return 0.0

    i = np.arange(1, n_hotdogs + 1)
    return float(np.sum(athlete.eat_base_sec + athlete.eat_slowdown_sec * i))


def mean_post_eating_multiplier(n_hotdogs: int, athlete: Athlete) -> float:
    """
    Mean multiplier applied to running time after eating n_hotdogs.
    """
    if n_hotdogs < 0:
        raise ValueError("n_hotdogs must be non-negative")

    n = n_hotdogs
    return 1.0 + athlete.post_eat_linear * n + athlete.post_eat_quadratic * (n ** 2)


def mean_running_time(distance_m: float, n_hotdogs: int, athlete: Athlete) -> float:
    """
    Mean running time:
        two_regime_run_time * post_eating_penalty
    """
    if distance_m <= 0:
        raise ValueError("distance_m must be positive")

    base_run = two_regime_running_time(
        distance_m=distance_m,
        sprint_threshold_distance_m=athlete.run_sprint_threshold_distance_m,
        sprint_reference_distance_m=athlete.run_sprint_reference_distance_m,
        sprint_reference_time_sec=athlete.run_sprint_reference_time_sec,
        reference_distance_m=athlete.run_reference_distance_m,
        reference_time_sec=athlete.run_reference_time_sec,
        endurance_exponent=athlete.run_endurance_exponent,
        max_modeled_distance_m=athlete.run_max_modeled_distance_m,
    )
    return base_run * mean_post_eating_multiplier(n_hotdogs, athlete)


def mean_total_time(distance_m: float, n_hotdogs: int, athlete: Athlete) -> float:
    """
    Mean total race time = eat first, then run.
    """
    return (
        mean_cumulative_eat_time(n_hotdogs, athlete)
        + mean_running_time(distance_m, n_hotdogs, athlete)
    )
