import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from scipy.optimize import approx_fprime


@app.function
def normal_residuals(rho, phi, x_arr, y_arr):
    x = np.asarray(x_arr, dtype=float)
    y = np.asarray(y_arr, dtype=float)
    return x * np.cos(phi) + y * np.sin(phi) - rho


@app.function
def SSE(rho, phi, x_arr, y_arr, weights=None):
    """Sum of squared deviations in the normal direction"""
    residuals = normal_residuals(rho, phi, x_arr, y_arr)
    if weights is None:
        return np.sum(residuals ** 2)
    return np.sum(np.asarray(weights) * residuals ** 2)


@app.function
def MSE(rho, phi, x_arr, y_arr, weights=None):
    """Mean squared deviation in the normal direction"""
    residuals = normal_residuals(rho, phi, x_arr, y_arr)
    if weights is None:
        return np.mean(residuals ** 2)
    return np.average(residuals ** 2, weights=np.asarray(weights))


@app.function
def RMSE(rho, phi, x_arr, y_arr, weights=None):
    residuals = normal_residuals(rho, phi, x_arr, y_arr)
    if weights is None:
        return np.mean(residuals ** 2)
    return np.average(residuals ** 2, weights=np.asarray(weights)) ** 0.5


@app.function
def L1(rho, phi, x_arr, y_arr, weights=None):
    """Sum of deviations in the normal direction"""
    residuals = normal_residuals(rho, phi, x_arr, y_arr)
    if weights is None:
        return np.sum(np.abs(residuals))
    return np.sum(np.asarray(weights) * np.abs(residuals))


@app.cell
def _():
    def gradient_func(rho, phi, x_arr, y_arr, _weights):
        residuals = normal_residuals(rho, phi, x_arr, y_arr)
        weights_array = np.ones_like(residuals) if _weights is None else _weights
        scale = np.sum(weights_array)
        gradient_rho = -2 * np.sum(weights_array * residuals) / scale
        tangent_projection = -x_arr * np.sin(phi) + y_arr * np.cos(phi)
        gradient_phi = 2 * np.sum(weights_array * residuals * tangent_projection) / scale
        return gradient_rho, gradient_phi


    def gradient_descent_step(
        loss_func, rho, phi, rho_step, phi_step, x_arr, y_arr, weights
    ):
        """Perform one gradient descent step with separate learning rates"""
        gradient_rho, gradient_phi = gradient_func(rho, phi, x_arr, y_arr, weights)
        next_rho = rho - rho_step * gradient_rho
        next_phi = phi - phi_step * gradient_phi
        return next_rho, next_phi, loss_func(next_rho, next_phi, x_arr, y_arr, weights)


    OPTIMIZATION_ALGORITHMS = {
        "gradient_descent": gradient_descent_step,
    }


    def optimize(
        loss_func,
        metric_func,
        x_arr,
        y_arr,
        weights,
        rho,
        phi,
        iterations,
        rho_step,
        phi_step,
        algorithm="gradient_descent"
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
                loss_func, rho, phi, rho_step, phi_step, x_arr, y_arr, weights
            )
            history.append((rho, phi))

        return np.asarray(history)

    return (optimize,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Чтение данных
    """)
    return


@app.cell
def _():
    from generator_module import read_pc, read_raster, write_pc, write_raster

    datas = []
    data_num_list = range(1, 11)
    for data_num in data_num_list:
        _filename = f"pc_rv_{data_num:04d}"
        path = f"mipt2026f-optcv/data/point_cloud/roman/{_filename}.csv"

        _rho_true, _phi_true, _xyw = read_pc(path)
        _x_arr, _y_arr, _weights = _xyw[:,0], _xyw[:,1], _xyw[:,2]
        datas.append((_rho_true, _phi_true, _x_arr, _y_arr, _weights, _filename))
    return (datas,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Оптимизация
    """)
    return


@app.cell
def _(datas, optimize):
    loss_func = SSE

    rho_start = -2.0
    phi_start = -0.3
    iterations = 30
    rho_step = 0.1
    phi_step = 0.01
    optimization_algorithm = "gradient_descent"

    logs = []
    for _data in datas:
        rho_true, phi_true, x_array, y_array, weights, filename = _data

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
        )
        log = {
            'history': history,
            'rho_true': rho_true,
            'phi_true': phi_true,
            'filename': filename
            }
        logs.append(log)
    return (
        filename,
        logs,
        loss_func,
        phi_true,
        rho_true,
        weights,
        x_array,
        y_array,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Визуализация
    """)
    return


@app.cell
def _(
    filename,
    logs,
    loss_func,
    phi_true,
    rho_true,
    weights,
    x_array,
    y_array,
):
    from visualize_module import plot_history
    if 1:
        for _log in logs:
            plot_history(
                _log['history'],
                rho_true,
                phi_true,
                x_array,
                y_array,
                weights,
                loss_func,
                filename,
            )
    return


@app.cell
def _(logs):
    from visualize_module import protocol_runs, protocol_stat
    protocol_runs(logs, "results/pc_runs.csv")
    protocol_stat("results/pc_runs.csv", "results/pc_stats.csv")
    return


if __name__ == "__main__":
    app.run()
