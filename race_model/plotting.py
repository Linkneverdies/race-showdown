import numpy as np
import matplotlib.pyplot as plt


def _build_y_edges(distances_m: np.ndarray) -> np.ndarray:
    if len(distances_m) < 2:
        raise ValueError("Need at least 2 distance values")

    dy = distances_m[1] - distances_m[0]
    return np.concatenate((
        [distances_m[0] - dy / 2],
        (distances_m[:-1] + distances_m[1:]) / 2,
        [distances_m[-1] + dy / 2],
    ))


def _build_x_edges_centered(hotdog_counts: np.ndarray) -> np.ndarray:
    """
    Discrete-state plotting:
    each integer count is centered in its own column.
    """
    return np.arange(hotdog_counts.min() - 0.5, hotdog_counts.max() + 1.5, 1.0)


def plot_probability_phase_diagram(
        probabilities: np.ndarray,
        hotdog_counts: np.ndarray,
        distances_m: np.ndarray,
        athlete_a_name: str,
        athlete_b_name: str,
) -> None:
    """
    Plot P(athlete_a wins).
    """
    x_edges = _build_x_edges_centered(hotdog_counts)
    y_edges = _build_y_edges(distances_m)

    _, ax = plt.subplots(figsize=(9, 6))

    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        probabilities,
        shading="flat",
        vmin=0.0,
        vmax=1.0,
    )
    ax.figure.colorbar(mesh, ax=ax, label=f"P({athlete_a_name} wins)")

    # 50% contour: probabilistic tie boundary
    X, Y = np.meshgrid(hotdog_counts, distances_m)
    ax.contour(X, Y, probabilities, levels=[0.5], linewidths=2)

    ax.set_xlim(hotdog_counts.min() - 0.5, hotdog_counts.max() + 0.5)
    ax.set_ylim(0.0, distances_m.max())
    ax.set_xticks(hotdog_counts)

    ax.set_xlabel("Number of Hot Dogs")
    ax.set_ylabel("Running Distance (meters)")
    ax.set_title(f"Probabilistic Phase Diagram: {athlete_a_name} vs {athlete_b_name}")

def plot_deterministic_phase_diagram(
        outcome_grid: np.ndarray,
        hotdog_counts: np.ndarray,
        distances_m: np.ndarray,
        athlete_a_name: str,
        athlete_b_name: str,
        tie_distances: list[float | None],
) -> None:
    """
    Plot deterministic winner regions and the tie boundary.
    """
    x_edges = _build_x_edges_centered(hotdog_counts)
    y_edges = _build_y_edges(distances_m)

    _, ax = plt.subplots(figsize=(9, 6))

    # Match the probabilistic colormap direction:
    # athlete_a wins -> 1 (lighter), athlete_b wins -> 0 (darker)
    region = np.where(outcome_grid < 0, 1.0, 0.0)

    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        region,
        shading="flat",
        alpha=0.35,
        vmin=0.0,
        vmax=1.0,
    )
    ax.figure.colorbar(
        mesh,
        ax=ax,
        ticks=[0.0, 1.0],
        label="Winner region",
    )
    colorbar = mesh.colorbar
    if colorbar is not None:
        colorbar.ax.set_yticklabels([athlete_b_name, athlete_a_name])

    boundary_x = []
    boundary_y = []
    for n_hotdogs, tie_distance in zip(hotdog_counts, tie_distances):
        if tie_distance is not None:
            boundary_x.append(n_hotdogs)
            boundary_y.append(tie_distance)

    if boundary_x:
        ax.plot(boundary_x, boundary_y, linewidth=2)

    ax.set_xlim(hotdog_counts.min() - 0.5, hotdog_counts.max() + 0.5)
    ax.set_ylim(0.0, distances_m.max())
    ax.set_xticks(hotdog_counts)

    ax.set_xlabel("Number of Hot Dogs")
    ax.set_ylabel("Running Distance (meters)")
    ax.set_title(f"Deterministic Phase Diagram: {athlete_a_name} vs {athlete_b_name}")
    ax.text(2, distances_m.max() * 0.75, f"{athlete_b_name} wins", fontsize=11)
    ax.text(hotdog_counts.max() * 0.6, distances_m.max() * 0.2, f"{athlete_a_name} wins", fontsize=11)
