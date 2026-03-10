import numpy as np

from models import (
    Athlete,
    mean_cumulative_eat_time,
    mean_post_eating_multiplier,
    mean_total_time,
)


def sample_total_times(
    athlete: Athlete,
    n_hotdogs: int,
    distance_m: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample total race times for one athlete.

    Randomness sources:
    - eating time noise
    - running speed noise
    - running fatigue noise
    - post-eating penalty noise
    """
    if n_hotdogs < 0:
        raise ValueError("n_hotdogs must be non-negative")
    if distance_m < 0:
        raise ValueError("distance_m must be non-negative")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    # Eating component
    eat_mean = mean_cumulative_eat_time(n_hotdogs, athlete)
    eat_std = athlete.eat_noise_sec * np.sqrt(n_hotdogs) if n_hotdogs > 0 else 0.0

    eat_time = rng.normal(loc=eat_mean, scale=eat_std, size=n_samples)
    eat_time = np.maximum(eat_time, 0.0)

    # Positive multiplicative perturbations
    speed_mult = np.maximum(
        rng.normal(loc=1.0, scale=athlete.run_speed_rel_std, size=n_samples),
        0.2,
    )
    fatigue_mult = np.maximum(
        rng.normal(loc=1.0, scale=athlete.run_fatigue_rel_std, size=n_samples),
        0.1,
    )
    post_mult_noise = np.maximum(
        rng.normal(loc=1.0, scale=athlete.post_eat_rel_std, size=n_samples),
        0.1,
    )

    speed = athlete.base_speed_mps * speed_mult
    fatigue = athlete.run_fatigue_coef * fatigue_mult
    post_eat = mean_post_eating_multiplier(n_hotdogs, athlete) * post_mult_noise

    run_time = (distance_m / speed) * (1.0 + fatigue * distance_m) * post_eat
    run_time = np.maximum(run_time, 0.0)

    return eat_time + run_time


def win_probability(
    athlete_a: Athlete,
    athlete_b: Athlete,
    n_hotdogs: int,
    distance_m: float,
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    """
    Estimate P(athlete_a wins) via Monte Carlo simulation.
    """
    a_times = sample_total_times(
        athlete=athlete_a,
        n_hotdogs=n_hotdogs,
        distance_m=distance_m,
        n_samples=n_samples,
        rng=rng,
    )
    b_times = sample_total_times(
        athlete=athlete_b,
        n_hotdogs=n_hotdogs,
        distance_m=distance_m,
        n_samples=n_samples,
        rng=rng,
    )
    return float(np.mean(a_times < b_times))


def estimate_probability_grid(
    athlete_a: Athlete,
    athlete_b: Athlete,
    hotdog_counts: np.ndarray,
    distances_m: np.ndarray,
    n_samples: int,
    seed: int = 7,
) -> np.ndarray:
    """
    Returns:
        P[i, j] = P(athlete_a wins | distance=distances_m[i], hotdogs=hotdog_counts[j])
    """
    rng = np.random.default_rng(seed)
    probabilities = np.empty((len(distances_m), len(hotdog_counts)), dtype=float)

    for j, n_hotdogs in enumerate(hotdog_counts):
        for i, distance_m in enumerate(distances_m):
            probabilities[i, j] = win_probability(
                athlete_a=athlete_a,
                athlete_b=athlete_b,
                n_hotdogs=int(n_hotdogs),
                distance_m=float(distance_m),
                n_samples=n_samples,
                rng=rng,
            )

    return probabilities


def mean_time_difference_grid(
    athlete_a: Athlete,
    athlete_b: Athlete,
    hotdog_counts: np.ndarray,
    distances_m: np.ndarray,
) -> np.ndarray:
    """
    Deterministic reference grid:
        D[i, j] = mean_time_a - mean_time_b
    """
    diff = np.empty((len(distances_m), len(hotdog_counts)), dtype=float)

    for j, n_hotdogs in enumerate(hotdog_counts):
        for i, distance_m in enumerate(distances_m):
            diff[i, j] = (
                mean_total_time(float(distance_m), int(n_hotdogs), athlete_a)
                - mean_total_time(float(distance_m), int(n_hotdogs), athlete_b)
            )

    return diff


def deterministic_outcome_grid(
    athlete_a: Athlete,
    athlete_b: Athlete,
    hotdog_counts: np.ndarray,
    distances_m: np.ndarray,
) -> np.ndarray:
    """
    Returns:
        Z[i, j] = mean_time_a - mean_time_b
    """
    return mean_time_difference_grid(
        athlete_a=athlete_a,
        athlete_b=athlete_b,
        hotdog_counts=hotdog_counts,
        distances_m=distances_m,
    )


def find_deterministic_tie_distance(
    n_hotdogs: int,
    athlete_a: Athlete,
    athlete_b: Athlete,
    max_distance_m: float,
    resolution_m: float = 1.0,
) -> float | None:
    """
    Find the first distance where the deterministic mean model ties.
    """
    distances = np.arange(0.0, max_distance_m + resolution_m, resolution_m)
    values = np.array([
        mean_total_time(float(d), n_hotdogs, athlete_a)
        - mean_total_time(float(d), n_hotdogs, athlete_b)
        for d in distances
    ])

    exact_idx = np.where(np.isclose(values, 0.0))[0]
    if len(exact_idx) > 0:
        return float(distances[exact_idx[0]])

    signs = np.sign(values)
    change_idx = np.where(signs[:-1] != signs[1:])[0]
    if len(change_idx) == 0:
        return None

    i = int(change_idx[0])
    d0, d1 = distances[i], distances[i + 1]
    v0, v1 = values[i], values[i + 1]

    if np.isclose(v1, v0):
        return float(d0)

    return float(d0 - v0 * (d1 - d0) / (v1 - v0))


def compute_deterministic_tie_curve(
    hotdog_counts: np.ndarray,
    athlete_a: Athlete,
    athlete_b: Athlete,
    max_distance_m: float,
    resolution_m: float = 1.0,
) -> list[float | None]:
    """
    Compute deterministic tie distance for each integer hot-dog count.
    """
    return [
        find_deterministic_tie_distance(
            n_hotdogs=int(n),
            athlete_a=athlete_a,
            athlete_b=athlete_b,
            max_distance_m=max_distance_m,
            resolution_m=resolution_m,
        )
        for n in hotdog_counts
    ]
