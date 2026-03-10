import numpy as np
import matplotlib.pyplot as plt
from models import Athlete
from simulation import (
    compute_deterministic_tie_curve,
    deterministic_outcome_grid,
    estimate_probability_grid,
    win_probability,
)
from plotting import plot_deterministic_phase_diagram, plot_probability_phase_diagram


def build_athletes() -> tuple[Athlete, Athlete]:
    chestnut = Athlete(
        name="Chestnut",
        base_speed_mps=200 / 32,
        run_fatigue_coef=0.00025,
        eat_base_sec=3.0,
        eat_slowdown_sec=0.15,
        post_eat_linear=0.010,
        post_eat_quadratic=0.00035,
        eat_noise_sec=0.20,
        run_speed_rel_std=0.03,
        run_fatigue_rel_std=0.10,
        post_eat_rel_std=0.08,
    )

    bolt = Athlete(
        name="Bolt",
        base_speed_mps=200 / 19.19,
        run_fatigue_coef=0.00015,
        eat_base_sec=9.0,
        eat_slowdown_sec=0.25,
        post_eat_linear=0.025,
        post_eat_quadratic=0.0012,
        eat_noise_sec=0.35,
        run_speed_rel_std=0.02,
        run_fatigue_rel_std=0.08,
        post_eat_rel_std=0.12,
    )

    return chestnut, bolt


def main() -> None:
    chestnut, bolt = build_athletes()

    hotdog_counts = np.arange(0, 31, 1)
    distances_m = np.linspace(0, 2500, 100)

    probabilities = estimate_probability_grid(
        athlete_a=chestnut,
        athlete_b=bolt,
        hotdog_counts=hotdog_counts,
        distances_m=distances_m,
        n_samples=10000,
        seed=7,
    )

    plot_probability_phase_diagram(
        probabilities=probabilities,
        hotdog_counts=hotdog_counts,
        distances_m=distances_m,
        athlete_a_name=chestnut.name,
        athlete_b_name=bolt.name,
    )

    deterministic_grid = deterministic_outcome_grid(
        athlete_a=chestnut,
        athlete_b=bolt,
        hotdog_counts=hotdog_counts,
        distances_m=distances_m,
    )

    tie_distances = compute_deterministic_tie_curve(
        hotdog_counts=hotdog_counts,
        athlete_a=chestnut,
        athlete_b=bolt,
        max_distance_m=float(distances_m.max()),
        resolution_m=1.0,
    )

    plot_deterministic_phase_diagram(
        outcome_grid=deterministic_grid,
        hotdog_counts=hotdog_counts,
        distances_m=distances_m,
        athlete_a_name=chestnut.name,
        athlete_b_name=bolt.name,
        tie_distances=tie_distances,
    )

    plt.show()
    print("Reference probabilities:")
    for n, d in [(1, 200), (5, 400), (10, 1000), (20, 1800)]:
        p = win_probability(
            athlete_a=chestnut,
            athlete_b=bolt,
            n_hotdogs=n,
            distance_m=d,
            n_samples=3000,
            rng=np.random.default_rng(123),
        )
        print(f"P({chestnut.name} wins | {n} hot dogs, {d} m) = {p:.3f}")


if __name__ == "__main__":
    main()
