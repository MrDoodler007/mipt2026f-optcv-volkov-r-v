import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from generator_module import read_pc, read_raster, write_pc, write_raster

    return np, os, plt, read_pc


@app.cell
def _(np, weights, x_array, y_array):
    def normal_residuals(rho, phi, x_arr, y_arr):
        x = np.asarray(x_arr, dtype=float)
        y = np.asarray(y_arr, dtype=float)
        return x * np.cos(phi) + y * np.sin(phi) - rho


    def SSE(rho, phi, x_arr, y_arr, weights=None):
        """Sum of squared deviations in the normal direction"""
        residuals = normal_residuals(rho, phi, x_arr, y_arr)
        if weights is None:
            return np.sum(residuals ** 2)
        return np.sum(np.asarray(weights) * residuals ** 2)


    def MSE(rho, phi, x_arr, y_arr, weights=None):
        """Mean squared deviation in the normal direction"""
        residuals = normal_residuals(rho, phi, x_arr, y_arr)
        if weights is None:
            return np.mean(residuals ** 2)
        return np.average(residuals ** 2, weights=np.asarray(weights))


    def RMSE(rho, phi, x_arr, y_arr, weights=None):
        return MSE(rho, phi, x_arr, y_arr, weights=None) ** 0.5


    def L1(rho, phi, x_arr, y_arr, weights=None):
        """Sum of deviations in the normal direction"""
        residuals = normal_residuals(rho, phi, x_arr, y_arr)
        if weights is None:
            return np.sum(np.abs(residuals))
        return np.sum(np.asarray(weights) * np.abs(residuals))


    def gradient_descent_step(
        loss_func, rho, phi, rho_step, phi_step, gradient_func=None
    ):
        """Perform one gradient descent step with separate parameter rates"""
        gradient_rho, gradient_phi = gradient_func(rho, phi, x_array, y_array, weights)
        next_rho = rho - rho_step * gradient_rho
        next_phi = phi - phi_step * gradient_phi
        return next_rho, next_phi, loss_func(next_rho, next_phi, x_array, y_array, weights)


    OPTIMIZATION_ALGORITHMS = {
        "gradient_descent": gradient_descent_step,
    }


    def optimize(
        loss_func,
        metric_func,
        x_discr,
        y_discr,
        weights,
        rho,
        phi,
        iterations,
        rho_step,
        phi_step,
        algorithm="gradient_descent",
        gradient_func=None,
    ):
        """Save the optimization history, including the initial point"""
        try:
            optimization_step = OPTIMIZATION_ALGORITHMS[algorithm]
        except KeyError as error:
            available = ", ".join(OPTIMIZATION_ALGORITHMS)
            raise ValueError(
                f"Unknown optimization algorithm {algorithm!r}. Available: {available}"
            ) from error

        history = [(rho, phi)]

        for _ in range(iterations):
            rho, phi, loss = optimization_step(
                loss_func, rho, phi, rho_step, phi_step, gradient_func
            )
            history.append((rho, phi))

        return np.asarray(history)


    def gradient_func(rho, phi, x_array, y_array, weights):
        residuals = normal_residuals(rho, phi, x_array, y_array)
        weights_array = np.ones_like(residuals) if weights is None else weights
        scale = np.sum(weights_array)
        gradient_rho = -2 * np.sum(weights_array * residuals) / scale
        tangent_projection = -x_array * np.sin(phi) + y_array * np.cos(phi)
        gradient_phi = 2 * np.sum(weights_array * residuals * tangent_projection) / scale
        return gradient_rho, gradient_phi

    return RMSE, SSE, gradient_func, optimize


@app.cell
def _(read_pc):
    data_num = 2
    filename = f"pc_rv_{data_num:04d}"
    PATH = f"mipt2026f-optcv/data/point_cloud/roman/{filename}.csv"
    rho_true, phi_true, data = read_pc(PATH)
    x_array, y_array, weights = data[:,0], data[:,1], data[:,2]
    return filename, phi_true, rho_true, weights, x_array, y_array


@app.cell
def _(
    RMSE,
    SSE,
    filename,
    gradient_func,
    np,
    optimize,
    os,
    phi_true,
    plt,
    rho_true,
    weights,
    x_array,
    y_array,
):
    def loss_func(rho, phi, x_array, y_array, weights):
        return SSE(rho, phi, x_array, y_array, weights)


    rho_start = -2.0
    phi_start = -0.3
    iterations = 30
    rho_step = 0.1
    phi_step = 0.01
    optimization_algorithm = "gradient_descent"

    history = optimize(
        loss_func,
        SSE,
        x_array,
        y_array,
        weights,
        rho_start,
        phi_start,
        iterations,
        rho_step,
        phi_step,
        optimization_algorithm,
        gradient_func,
    )

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
    sse_grid = np.array(
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
        sse_grid,
        shading="flat",
        cmap="gray",
    )
    contour_levels = np.linspace(np.min(sse_grid), np.max(sse_grid), 12)[1:-1]
    contours = optimization_ax.contour(
        rho_grid,
        phi_grid,
        sse_grid,
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
        history[-1, 0],
        history[-1, 1],
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
    optimization_fig.colorbar(heatmap, ax=optimization_ax, label="SSE")
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/fig_{filename}.png")
    plt.show()

    print(f"Начальный SSE: {SSE(history[0, 0], history[0, 1], x_array, y_array, weights):.4f}")
    print(f"Финальный SSE: {SSE(history[-1, 0], history[-1, 1], x_array, y_array, weights):.4f}")
    print(f"Начальный RMSE: {RMSE(history[0, 0], history[0, 1], x_array, y_array, weights):.4f}")
    print(f"Финальный RMSE: {RMSE(history[-1, 0], history[-1, 1], x_array, y_array, weights):.4f}")
    print(f"Оценка: rho={history[-1, 0]:.4f}, phi={history[-1, 1]:.4f}")
    return


if __name__ == "__main__":
    app.run()
