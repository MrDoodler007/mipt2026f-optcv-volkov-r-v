import marimo

__generated_with = "0.25.1"
app = marimo.App()

with app.setup:
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import csv
    import os
    from pathlib import Path
    from PIL import Image


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

    return generate_data, generate_matrix, line_points


@app.function
def matrix2data(matrix, x_edges, y_edges):
    """Produce cell-center points and their weights from a matrix"""
    _y_indices, _x_indices = np.nonzero(matrix)
    _x_arr = (x_edges[_x_indices] + x_edges[_x_indices + 1]) / 2
    _y_arr = (y_edges[_y_indices] + y_edges[_y_indices + 1]) / 2
    _w_arr = matrix[_y_indices, _x_indices]

    return _x_arr, _y_arr, _w_arr


@app.function
def read_pc(path_csv, path_repo):
    with open(path_repo / path_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        _rho_true, _phi_true, _ = next(reader)
        _data = np.array(list(reader), dtype=float)
    return float(_rho_true), float(_phi_true), _data


@app.function
def write_pc(path_csv, path_repo, rho_true, phi_true, x_arr, y_arr):
    with open(path_repo / path_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "w"])
        writer.writerow([rho_true, phi_true , ""])
        for row in zip(x_arr, y_arr, np.ones(x_arr.shape)):
            writer.writerow(row)


@app.function
def read_raster(path_csv: Path, path_img: Path, path_repo: Path):
    _rho_true, _phi_true = None, None
    with open(path_repo / path_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    for row in rows:
        if row["image"] == str(path_img):
            _rho_true, _phi_true = row["rho"], row["theta"]

    _matrix = np.asarray(Image.open(path_repo / path_img))
    _x0, _y0, _cell_size = 0, 0, 2
    return float(_rho_true), float(_phi_true), _x0, _y0, _cell_size, _matrix


@app.function
def write_raster(
        path_csv: Path,
        path_img: Path,
        path_repo: Path,
        rho_true,
        phi_true,
        matrix,
        x_0,
        y_0,
        cell_size,
    ):
    path_csv = path_repo / path_csv
    if not path_csv.exists():
        with open(path_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "rho", "theta"])

    with open(path_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    # print(rows)
    for row in rows:
        if row["image"] == str(path_img):
            row["rho"] = rho_true
            row["theta"] = phi_true
            # print('replace', path_img)
            break
    else:
        # print('add', path_img)
        rows.append({"image": path_img, "rho": rho_true, "theta": phi_true})

    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # print(rows)
        writer.writerows(rows)

    # print()
    matrix = np.uint8(matrix / np.max(matrix) * 255)
    Image.fromarray(matrix).convert("L").save(path_repo / path_img)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Генерация данных
    """)
    return


@app.cell
def _(generate_data, generate_matrix):
    N_DATASETS = 10
    n = 1
    REPO_PATH = Path(f"mipt2026f-optcv/")

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

        # Point cloud
        PC_CSV_PATH = Path(f"data/point_cloud/roman/pc_rv_{n:04d}.csv")
        os.makedirs((REPO_PATH / PC_CSV_PATH).parent, exist_ok=True)
        write_pc(PC_CSV_PATH, REPO_PATH, rho_true, phi_true, x_array, y_array)

        # Raster images
        RASTER_CSV_PATH = Path(f"data/raster/roman/ground_truth.csv")
        os.makedirs((REPO_PATH / RASTER_CSV_PATH).parent, exist_ok=True)
        RASTER_IMG_PATH = Path(f"data/raster/roman/images/img_rv_{n:04d}.png")
        os.makedirs((REPO_PATH / RASTER_IMG_PATH).parent, exist_ok=True)
        write_raster(
            RASTER_CSV_PATH,
            RASTER_IMG_PATH,
            REPO_PATH,
            rho_true,
            phi_true,
            matrix,
            None,
            None,
            None,
        )

        n += 1


    rho_true, phi_true, data = read_pc(PC_CSV_PATH, REPO_PATH)
    print(rho_true, phi_true)
    print(data.shape)
    rho_true, phi_true, _x0, _y0, _cell_size, matrix = read_raster(RASTER_CSV_PATH, RASTER_IMG_PATH, REPO_PATH)
    print(rho_true, phi_true)
    print(matrix.shape)
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
