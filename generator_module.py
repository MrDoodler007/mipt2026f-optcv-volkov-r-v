import marimo

__generated_with = "0.24.2"
app = marimo.App()

with app.setup:
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import csv
    import os
    from pathlib import Path


@app.cell
def _():
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
        w_arr = matrix[y_indices, x_indices]

        return x_arr, y_arr, w_arr


    return generate_data, generate_matrix, line_points, matrix2data


@app.function
def read_pc(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        _rho_true, _phi_true, _ = next(reader)
        _data = np.array(list(reader), dtype=float)
    return float(_rho_true), float(_phi_true), _data


@app.function
def write_pc(path, rho_true, phi_true, x_arr, y_arr):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "w"])
        writer.writerow([rho_true, phi_true , ""])
        for row in zip(x_arr, y_arr, np.ones(x_arr.shape)):
            writer.writerow(row)


@app.function
def read_raster(path_csv, path_img):
    pass


@app.function
def write_raster(rho_true, phi_true, x_arr, y_arr, w_arr):
    pass


@app.cell
def _(generate_data, generate_matrix, matrix2data):
    N_DATASETS = 10
    n = 1

    for i in range(N_DATASETS):
        np.random.seed(i)

        rho_true = 0.0
        phi_true = np.pi / 6
        line_length = 20.0
        sigma = 1.0
        measurements = 50
        cell_size = 2.0

        x_array, y_array = generate_data(
            rho_true, phi_true, line_length, sigma, measurements
        )
        matrix, x_edges, y_edges = generate_matrix(
            x_array, y_array, cell_size
        )
        x_discr, y_discr, weights = matrix2data(
            matrix, x_edges, y_edges
        )

        RES_PATH = Path(f"mipt2026f-optcv/data/point_cloud/roman/pc_rv_{n:04d}.csv")
        os.makedirs(RES_PATH.parent, exist_ok=True)
        write_pc(RES_PATH, rho_true, phi_true, x_array, y_array)

        n += 1


    rho_true, phi_true, data = read_pc(RES_PATH)
    print(rho_true, phi_true)
    print(data.shape)
    return (
        line_length,
        matrix,
        phi_true,
        rho_true,
        x_array,
        x_edges,
        y_array,
        y_edges,
    )


@app.cell
def _(
    line_length,
    line_points,
    matrix,
    phi_true,
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
    data_ax.plot(line_x, line_y, color="tab:red", linewidth=2, label="Эталон")
    data_ax.set_xlabel("x")
    data_ax.set_ylabel("y")
    data_ax.set_title("Наблюдаемые точки и эталон")
    data_ax.set_xticks(x_edges)
    data_ax.set_yticks(y_edges)
    data_ax.grid(True, which="major", alpha=0.35)
    data_ax.legend()
    data_fig.colorbar(image, ax=data_ax, label="Вес ячейки")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
