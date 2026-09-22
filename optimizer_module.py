import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return np, plt


@app.cell
def _(np):
    def generate_data(rho, phi, length, sigma, measurements):
        """Generate measurements with additive noise in normal direction"""
        normal = np.array([np.cos(phi), np.sin(phi)])
        tangent = np.array([-np.sin(phi), np.cos(phi)])
        point_on_line = rho * normal

        positions = np.linspace(-length / 2, length / 2, measurements)
        points = point_on_line[:, None] + tangent[:, None] * positions
        noise = np.random.normal(0.0, sigma, measurements)
        points += normal[:, None] * noise

        return points[0], points[1]


    def line_points(rho, phi, length):
        """Return two endpoints of a line segment in normal form"""
        normal = np.array([np.cos(phi), np.sin(phi)])
        tangent = np.array([-np.sin(phi), np.cos(phi)])
        point_on_line = rho * normal
        positions = np.array([-length / 2, length / 2])
        points = point_on_line[:, None] + tangent[:, None] * positions
        return points[0], points[1]


    def generate_matrix(x_arr, y_arr, cell_size):
        """Produce a matrix and preserve the coordinates of its cells"""
        x = np.asarray(x_arr, dtype=float)
        y = np.asarray(y_arr, dtype=float)

        x_min = np.floor(np.min(x) / cell_size) * cell_size
        x_max = np.ceil(np.max(x) / cell_size) * cell_size
        y_min = np.floor(np.min(y) / cell_size) * cell_size
        y_max = np.ceil(np.max(y) / cell_size) * cell_size

        if x_max == x_min:
            x_max += cell_size
        if y_max == y_min:
            y_max += cell_size

        x_edges = np.arange(x_min, x_max + cell_size / 2, cell_size)
        y_edges = np.arange(y_min, y_max + cell_size / 2, cell_size)
        matrix, _, _ = np.histogram2d(y, x, bins=(y_edges, x_edges))

        return matrix.astype(int), x_edges, y_edges


    def matrix2data(matrix, x_edges, y_edges):
        """Produce cell-center points and their weights from a matrix"""
        y_indices, x_indices = np.nonzero(matrix)
        x_arr = (x_edges[x_indices] + x_edges[x_indices + 1]) / 2
        y_arr = (y_edges[y_indices] + y_edges[y_indices + 1]) / 2
        weights = matrix[y_indices, x_indices]

        return x_arr, y_arr, weights


    def clone_points(x, y, weights):
        """Expand weighted points into an ordinary point cloud"""
        weights = np.asarray(weights, dtype=int)
        return np.repeat(x, weights), np.repeat(y, weights)


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


    def coordinate_search_step(loss_func, rho, phi, rho_step, phi_step, gradient_func=None):
        """Perform one numerical optimization step on a 3x3 neighborhood"""
        rho_candidates = rho + np.array([-rho_step, 0.0, rho_step])
        phi_candidates = phi + np.array([-phi_step, 0.0, phi_step])
        candidates = [
            (candidate_rho, candidate_phi)
            for candidate_rho in rho_candidates
            for candidate_phi in phi_candidates
        ]
        losses = np.array(
            [loss_func(candidate_rho, candidate_phi) for candidate_rho, candidate_phi in candidates]
        )
        best = int(np.argmin(losses))

        return (*candidates[best], losses[best])


    def gradient_descent_step(
        loss_func, rho, phi, rho_step, phi_step, gradient_func=None
    ):
        """Perform one gradient descent step with separate parameter rates"""
        if gradient_func is None:
            raise ValueError("gradient_func is required for gradient descent")

        gradient_rho, gradient_phi = gradient_func(rho, phi)
        next_rho = rho - rho_step * gradient_rho
        next_phi = phi - phi_step * gradient_phi
        return next_rho, next_phi, loss_func(next_rho, next_phi)


    OPTIMIZATION_ALGORITHMS = {
        "coordinate_search": coordinate_search_step,
        "gradient_descent": gradient_descent_step,
    }


    def optimize(
        loss_func,
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

        history = [(rho, phi, loss_func(rho, phi))]

        for _ in range(iterations):
            rho, phi, loss = optimization_step(
                loss_func, rho, phi, rho_step, phi_step, gradient_func
            )
            history.append((rho, phi, loss))

        return np.asarray(history)


    return (
        MSE,
        SSE,
        generate_data,
        generate_matrix,
        line_points,
        matrix2data,
        normal_residuals,
        optimize,
    )


@app.cell
def _(generate_data, generate_matrix, matrix2data, np):
    np.random.seed(2)

    rho_true = 0.0
    phi_true = np.pi / 6
    line_length = 20.0
    sigma = 1.0
    measurements = 50
    cell_size = 2.0

    x_array, y_array = generate_data(
        rho_true, phi_true, line_length, sigma, measurements
    )
    matrix, x_edges, y_edges = generate_matrix(x_array, y_array, cell_size)
    x_discr, y_discr, weights = matrix2data(matrix, x_edges, y_edges)
    return (
        line_length,
        matrix,
        phi_true,
        rho_true,
        weights,
        x_array,
        x_discr,
        x_edges,
        y_array,
        y_discr,
        y_edges,
    )


@app.cell
def _(
    line_length,
    line_points,
    matrix,
    phi_true,
    plt,
    rho_true,
    x_array,
    x_edges,
    y_array,
    y_edges,
):
    line_x, line_y = line_points(rho_true, phi_true, line_length)

    data_fig, data_ax = plt.subplots(figsize=(9, 7))
    image = data_ax.pcolormesh(
        x_edges,
        y_edges,
        matrix,
        shading="flat",
        alpha=0.45,
        cmap="gray",
    )
    data_ax.set_aspect("equal")
    data_ax.scatter(x_array, y_array, color="tab:blue", label="Измерения")
    data_ax.plot(line_x, line_y, color="tab:red", linewidth=2, label="Истинная прямая")
    data_ax.set_xlabel("x")
    data_ax.set_ylabel("y")
    data_ax.set_title("Наблюдаемые точки и породившая их матрица")
    data_ax.set_xticks(x_edges)
    data_ax.set_yticks(y_edges)
    data_ax.grid(True, which="major", alpha=0.35)
    data_ax.legend()
    data_fig.colorbar(image, ax=data_ax, label="Вес ячейки")
    plt.show()
    return


@app.cell
def _(
    MSE,
    SSE,
    normal_residuals,
    np,
    optimize,
    phi_true,
    plt,
    rho_true,
    weights,
    x_discr,
    y_discr,
):
    def loss_func(rho, phi):
        return SSE(rho, phi, x_discr, y_discr, weights)


    def gradient_func(rho, phi):
        residuals = normal_residuals(rho, phi, x_discr, y_discr)
        weights_array = np.ones_like(residuals) if weights is None else weights
        scale = np.sum(weights_array)
        gradient_rho = -2 * np.sum(weights_array * residuals) / scale
        tangent_projection = -x_discr * np.sin(phi) + y_discr * np.cos(phi)
        gradient_phi = 2 * np.sum(weights_array * residuals * tangent_projection) / scale
        return gradient_rho, gradient_phi


    rho_start = -2.0
    phi_start = -0.3
    iterations = 30
    rho_step = 0.1
    phi_step = 0.01
    optimization_algorithm = "gradient_descent"

    history = optimize(
        loss_func,
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
        [[loss_func(rho, phi) for rho in rho_grid] for phi in phi_grid]
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
    optimization_ax.set_title("Оптимизация SSE на фазовой плоскости")
    optimization_ax.grid(True, alpha=0.25)
    optimization_ax.legend()
    optimization_fig.colorbar(heatmap, ax=optimization_ax, label="SSE")
    plt.show()

    print(f"Начальный SSE: {history[0, 2]:.4f}")
    print(f"Финальный SSE: {history[-1, 2]:.4f}")
    print(f"Финальный MSE: {MSE(history[-1, 0], history[-1, 1], x_discr, y_discr, weights):.4f}")
    print(f"Оценка: rho={history[-1, 0]:.4f}, phi={history[-1, 1]:.4f}")
    return


if __name__ == "__main__":
    app.run()
