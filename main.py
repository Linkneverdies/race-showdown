import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config.config import load_app_config
from race_model.simulation import (
    compute_deterministic_tie_curve,
    deterministic_outcome_grid,
    estimate_probability_grid,
    win_probability,
)
from race_model.plotting import plot_deterministic_phase_diagram, plot_probability_phase_diagram


def main() -> None:
    config = load_app_config(
        Path(__file__).resolve().parent / "config" / "config.toml"
    )
    athlete_a = config.get_athlete(config.simulation.athlete_a)
    athlete_b = config.get_athlete(config.simulation.athlete_b)

    hotdog_counts = np.arange(0, config.simulation.max_hotdogs + 1, 1)
    distances_m = np.linspace(
        config.simulation.min_distance_m,
        config.simulation.max_distance_m,
        config.simulation.distance_samples,
    )

    probabilities = estimate_probability_grid(
        athlete_a=athlete_a,
        athlete_b=athlete_b,
        hotdog_counts=hotdog_counts,
        distances_m=distances_m,
        n_samples=config.simulation.monte_carlo_samples,
        seed=config.simulation.random_seed,
    )

    plot_probability_phase_diagram(
        probabilities=probabilities,
        hotdog_counts=hotdog_counts,
        distances_m=distances_m,
        athlete_a_name=athlete_a.name,
        athlete_b_name=athlete_b.name,
    )

    deterministic_grid = deterministic_outcome_grid(
        athlete_a=athlete_a,
        athlete_b=athlete_b,
        hotdog_counts=hotdog_counts,
        distances_m=distances_m,
    )

    tie_distances = compute_deterministic_tie_curve(
        hotdog_counts=hotdog_counts,
        athlete_a=athlete_a,
        athlete_b=athlete_b,
        min_distance_m=float(distances_m.min()),
        max_distance_m=float(distances_m.max()),
        resolution_m=1.0,
    )

    plot_deterministic_phase_diagram(
        outcome_grid=deterministic_grid,
        hotdog_counts=hotdog_counts,
        distances_m=distances_m,
        athlete_a_name=athlete_a.name,
        athlete_b_name=athlete_b.name,
        tie_distances=tie_distances,
    )

    plt.show()
    print("Reference probabilities:")
    for reference_check in config.simulation.reference_checks:
        p = win_probability(
            athlete_a=athlete_a,
            athlete_b=athlete_b,
            n_hotdogs=reference_check.n_hotdogs,
            distance_m=reference_check.distance_m,
            n_samples=3000,
            rng=np.random.default_rng(123),
        )
        print(
            f"P({athlete_a.name} wins | {reference_check.n_hotdogs} hot dogs, "
            f"{reference_check.distance_m:.0f} m) = {p:.3f}"
        )


if __name__ == "__main__":
    main()
