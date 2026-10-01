import marimo

__generated_with = "0.24.2"
app = marimo.App()

with app.setup:
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    if 'SSE' not in globals():
        from optimizer_module import SSE
    if 'MSE' not in globals():
        from optimizer_module import MSE
    if 'RMSE' not in globals():
        from optimizer_module import RMSE


@app.function
def plot_history(
    history,
    rho_true,
    phi_true,
    x_array,
    y_array,
    weights,
    loss_func,
    filename
):
    rho_start = history[0, 0]
    phi_start = history[0, 1]
    rho_final = history[-1, 0]
    phi_final = history[-1, 1]

    rho_grid = np.linspace(
        min(rho_start, rho_true, np.min(history[:, 0])) - 1,
        max(rho_start, rho_true, np.max(history[:, 0])) + 1,
        160,
    )
    phi_grid = np.linspace(
        min(-np.pi / 2, phi_true, np.min(history[:, 1])) - 0.05,
        max(np.pi / 2, phi_true, np.max(history[:, 1])) + 0.05,
        160,
    )
    loss_grid = np.array(
        [[loss_func(rho, phi, x_array, y_array, weights) for rho in rho_grid] for phi in phi_grid]
    )
    rho_edges = np.linspace(
        rho_grid[0] - (rho_grid[1] - rho_grid[0]) / 2,
        rho_grid[-1] + (rho_grid[-1] - rho_grid[-2]) / 2,
        len(rho_grid) + 1,
    )
    phi_edges = np.linspace(
        phi_grid[0] - (phi_grid[1] - phi_grid[0]) / 2,
        phi_grid[-1] + (phi_grid[-1] - phi_grid[-2]) / 2,
        len(phi_grid) + 1,
    )

    optimization_fig, optimization_ax = plt.subplots(figsize=(10, 7))
    heatmap = optimization_ax.pcolormesh(
        rho_edges,
        phi_edges,
        loss_grid,
        shading="flat",
        cmap="gray",
    )
    contour_levels = np.linspace(np.min(loss_grid), np.max(loss_grid), 12)[1:-1]
    contours = optimization_ax.contour(
        rho_grid,
        phi_grid,
        loss_grid,
        levels=contour_levels,
        colors="black",
        linewidths=0.7,
        alpha=0.75,
    )
    optimization_ax.plot(
        history[:, 0],
        history[:, 1],
        color="white",
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="История оптимизации",
    )
    optimization_ax.scatter(
        rho_start,
        phi_start,
        color="tab:orange",
        s=70,
        label="Начальная точка",
        zorder=3,
    )
    optimization_ax.scatter(
        rho_final,
        phi_final,
        color="tab:red",
        s=70,
        label="Финальная точка",
        zorder=3,
    )
    optimization_ax.scatter(
        rho_true,
        phi_true,
        color="cyan",
        marker="x",
        s=90,
        label="Истинные параметры",
        zorder=3,
    )
    optimization_ax.set_xlabel(r"$\rho$")
    optimization_ax.set_ylabel(r"$\phi$, рад")
    optimization_ax.set_title("Оптимизация на фазовой плоскости")
    optimization_ax.grid(True, alpha=0.25)
    optimization_ax.legend()
    optimization_fig.colorbar(heatmap, ax=optimization_ax, label=loss_func.__name__)
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/fig_{filename}.png")
    plt.show()

    print(f"Начальный SSE: {SSE(rho_start, phi_start, x_array, y_array, weights):.4f}")
    print(f"Финальный SSE: {SSE(rho_final, phi_final, x_array, y_array, weights):.4f}")
    print(f"Начальный RMSE: {RMSE(rho_start, phi_start, x_array, y_array, weights):.4f}")
    print(f"Финальный RMSE: {RMSE(rho_final, phi_final, x_array, y_array, weights):.4f}")
    print(f"Оценка: rho={rho_final:.4f}, phi={phi_final:.4f}")


if __name__ == "__main__":
    app.run()
