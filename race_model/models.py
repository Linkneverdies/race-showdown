from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Athlete:
    name: str

    # Mean performance model
    base_speed_mps: float
    run_fatigue_coef: float
    eat_base_sec: float
    eat_slowdown_sec: float
    post_eat_linear: float
    post_eat_quadratic: float

    # Noise model
    eat_noise_sec: float
    run_speed_rel_std: float
    run_fatigue_rel_std: float
    post_eat_rel_std: float


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
        base_run_time * post_eating_penalty
    where
        base_run_time = (distance / speed) * (1 + fatigue * distance)
    """
    if distance_m < 0:
        raise ValueError("distance_m must be non-negative")

    base_run = (distance_m / athlete.base_speed_mps) * (
            1.0 + athlete.run_fatigue_coef * distance_m
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