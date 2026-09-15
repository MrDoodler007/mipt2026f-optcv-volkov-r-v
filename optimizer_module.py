import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(2)
    return np, plt


@app.cell
def _(np):
    def generate_data(func, sigma, begin=-10, end=10, measurements=20):
        x_array = np.random.uniform(low=begin, high=end, size=measurements)
        y_array = func(x_array) + np.random.normal(loc=0, scale=sigma, size=measurements)
        return x_array, y_array


    def generate_matrix(x_array, y_array, cell_size=1, begin=-10, end=10):
        matrix = np.zeros(
            (
                (end-begin)//cell_size, (end-begin)//cell_size
            )
        )
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i, j] = np.sum((begin + i*cell_size <= y_array) & (y_array < begin + (i+1)*cell_size) & 
                                      (begin + j*cell_size <= x_array) & (x_array < begin + (j+1)*cell_size))
        return matrix


    def matrix2data(matrix, cell_size, begin, end):
        x_array, y_array = [], []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                for _ in range(int(matrix[i, j])):
                    y_array.append(begin + 1.5*i*cell_size)
                    x_array.append(begin + 1.5*j*cell_size)
        return np.array(x_array), np.array(y_array)

    return generate_data, generate_matrix, matrix2data


@app.function
def solve():
    pass


@app.cell
def _(generate_data, generate_matrix, matrix2data, np, plt):
    def line(x):
        return 1.5 * x + 3


    _begin = -10
    _end = 10
    _measurements = 50
    _sigma = 2
    _cell_size = 2

    # _x_array, _y_array = generate_data(line, _sigma, _begin, _end, _measurements)
    _x_array, _y_array = generate_data(line, _sigma, _begin, _end, _measurements)
    matrix = generate_matrix(_x_array, _y_array, _cell_size, _begin, _end)

    _x_discr, _y_discr = matrix2data(matrix, _cell_size, _begin, _end)
    k1_ls, k2_ls = np.polyfit(_x_array, _y_array, 1)
    print(f"Points: Least squares: k={k1_ls:.4f}, b={k2_ls:.4f}")
    k1_ls_discr, k2_ls_discr = np.polyfit(_x_discr, _y_discr, 1)
    print(f"Discr: Least squares: k={k1_ls:.4f}, b={k2_ls:.4f}")

    fig, ax = plt.subplots()
    _x = np.array([_begin, _end])
    ax.scatter(_x_array, _y_array)
    ax.plot(_x, line(_x))
    plt.plot(_x, k1_ls * _x + k2_ls)
    plt.plot(_x, k1_ls_discr * _x + k2_ls_discr)
    ax.legend(["Измерения", "Истина", "МНК точки", "МНК растр"])
    ax.grid(True)
    ax.set_xticks(np.arange(_begin, _end+1, _cell_size))
    ax.set_yticks(np.arange(_begin, _end+1, _cell_size))
    im = ax.imshow(matrix, extent=[_begin, _end, _begin, _end], aspect='auto', alpha=0.5, origin='lower', cmap='gray')
    cbar = fig.colorbar(im, ax=ax)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
