import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize_scalar, root_scalar
    from sklearn.linear_model import LinearRegression, RANSACRegressor

    return (
        LinearRegression,
        RANSACRegressor,
        minimize_scalar,
        mo,
        np,
        plt,
        root_scalar,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Рассматриваем две функции:

    1) $f(x) = (x + 2)^2$,

    2) $f(x) = -\text{sinc}(x + 2) = -\frac{\sin(x + 2)}{x + 2}$.
    """)
    return


@app.cell
def _(np):
    eps = 0.001

    # Quadratic function (min in x=-2)
    def f1(x):
        return (x + 2)**2

    def df1(x):
        return (f1(x+eps) - f1(x-eps)) / (2*eps)

    # Sinc function (min in x=-2)
    def f2(x):
        if x != -2.0:
            return -np.sin(x + 2.0) / (x + 2.0)
        else:
            return -1.0

    def df2(x):
        return (f2(x+eps) - f2(x-eps)) / (2*eps)

    return df1, df2, f1, f2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Осуществляем поиск минимума функции

    ### Используем 5 различных методов из модуля scipy.optimize:
    - метод Брента,
    - метод золотого сечения,
    - метод Ньютона (поиск стационарной точки),
    - метод секущих (поиск стационарной точки),
    - метод бинарного поиска (стационарной точки).

    Параметры алгоритмов по умолчанию, начальное приближение $x_0 = 0$, интервал для бинарного поиска $(-10, 10)$. Производная определяется численно через центральную конечную разность.

    Квадратичная функция является выпуклой вниз на всей области определения и содержит единственную стационарную точку-минимум — все методы должны найти минимум функции. Вторая функция имеет бесконечное количество стационарных точек и локальных экстремумов, из-за чего методы поиска стационарных точек будут непригодны для поиска минимума функции.
    """)
    return


@app.cell
def _(df1, df2, f1, f2, minimize_scalar, root_scalar):
    x_0 = 0

    res_brent_1  = minimize_scalar(f1, method='brent')
    res_golden_1 = minimize_scalar(f1, method='golden')
    res_newton_1 = root_scalar(df1, method="newton", x0=x_0)
    res_secant_1 = root_scalar(df1, method="secant", x0=x_0)
    res_bisect_1 = root_scalar(df1, method="bisect", x0=x_0, bracket=(-10, 10))

    print("Quadratic function")
    print(f"Brent:  x={res_brent_1.x: .6f}, nit={res_brent_1.nit:2d}, nfev={res_brent_1.nfev:2d}")
    print(f"Golden: x={res_golden_1.x: .6f}, nit={res_golden_1.nit:2d}, nfev={res_golden_1.nfev:2d}")
    print(f"Newton: x={res_newton_1.root: .6f}, nit={res_newton_1.iterations:2d}, nfev={res_newton_1.function_calls:2d}")
    print(f"Secant: x={res_secant_1.root: .6f}, nit={res_secant_1.iterations:2d}, nfev={res_secant_1.function_calls:2d}")
    print(f"Bisect: x={res_bisect_1.root: .6f}, nit={res_bisect_1.iterations:2d}, nfev={res_bisect_1.function_calls:2d}")
    print()

    res_brent_2  = minimize_scalar(f2, method='brent')
    res_golden_2 = minimize_scalar(f2, method='golden')
    res_newton_2 = root_scalar(df2, method="newton", x0=x_0)
    res_secant_2 = root_scalar(df2, method="secant", x0=x_0)
    # res_bisect_2 = root_scalar(df2, method="bisect", x0=x_0, bracket=(-10, 10))

    print("Sinc function")
    print(f"Brent:  x={res_brent_2.x: .6f}, nit={res_brent_2.nit:2d}, nfev={res_brent_2.nfev:2d}")
    print(f"Golden: x={res_golden_2.x: .6f}, nit={res_golden_2.nit:2d}, nfev={res_golden_2.nfev:2d}")
    print(f"Newton: x={res_newton_2.root: .6f}, nit={res_newton_2.iterations:2d}, nfev={res_newton_2.function_calls:2d}")
    print(f"Secant: x={res_secant_2.root: .6f}, nit={res_secant_2.iterations:2d}, nfev={res_secant_2.function_calls:2d}")
    # print(f"Bisect: x={res_bisect_2.root: .6f}, nit={res_bisect_2.iterations:2d}, nfev={res_bisect_2.function_calls:2d}")
    return (
        res_bisect_1,
        res_brent_1,
        res_brent_2,
        res_golden_1,
        res_golden_2,
        res_newton_1,
        res_newton_2,
        res_secant_1,
        res_secant_2,
        x_0,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Разные алгоритмы затратили различное количество итераций на достижение критерия остановы. Как и ожидалось, все методы верно нашли минимум квадратичной функции. Для второй функции методы Ньютона и секущей нашли один из локальных минимумов, но при другом начальном приближении результат будет иным. Иллюстрация найденных точек для второй функции приведена ниже:
    """)
    return


@app.cell
def _(f2, np, plt, res_brent_2, res_golden_2, res_newton_2, res_secant_2, x_0):
    _x_array = np.arange(-30, 5, 0.1)
    _f2_array = np.array([f2(x) for x in _x_array])

    plt.plot(_x_array, _f2_array)
    plt.scatter(x_0, f2(x_0))
    for p, m in zip([res_brent_2.x, res_golden_2.x, res_newton_2.root, res_secant_2.root],
                    ["s", "d", "v", "x"]):
        plt.scatter(p, f2(p), marker=m)

    plt.legend(["$-$sinc($x+2$)", "Начальное приближение", "Метод Брента", "Метод золотого сечения", "Метод Ньютона", "Метод секущей"])
    plt.grid(True)
    plt.xlabel("$x$")
    plt.ylabel("$-$sinc($x+2$)")
    plt.title("График функции $-$sinc($x+2$) и полученные точки")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Сравним количество совершенных итераций
    """)
    return


@app.cell
def _(
    plt,
    res_bisect_1,
    res_brent_1,
    res_brent_2,
    res_golden_1,
    res_golden_2,
    res_newton_1,
    res_newton_2,
    res_secant_1,
    res_secant_2,
):
    nit_1 = [res_brent_1.nit, res_golden_1.nit, res_newton_1.iterations, res_secant_1.iterations, res_bisect_1.iterations]
    nit_2 = [res_brent_2.nit, res_golden_2.nit, res_newton_2.iterations, res_secant_2.iterations]
    names = ["Метод Брента", "Метод золотого сечения", "Метод Ньютона", "Метод секущей", "Метод бинарного поиска"]

    plt.scatter(names, nit_1)
    plt.scatter(names[:-1], nit_2)

    plt.legend(["Парабола", "$-$sinc($x$)"])
    plt.grid(True)
    plt.ylabel("Количество итераций")
    plt.xticks(rotation=45, ha='right')
    # plt.title("")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Оценим параметры прямой в 2d по набору "наблюдаемых" зашумленных точек.

    Рассматриваем прямую $y = 1.5 \cdot x + 3$, измерения которой искажены аддитивным гауссовым шумом. Измерения по оси $x$ распределены случайно согласно равномерному распределению.

    Используем метод наименьших квадратов (МНК) и RANSAC из numpy и sklearn.linear_model.
    """)
    return


@app.cell
def _(LinearRegression, RANSACRegressor, np, plt):
    def line(x):
        return 1.5 * x + 3


    _begin = -10
    _end = 10
    _measurements = 20
    _sigma = 3
    np.random.seed(2)

    _x_array = np.random.uniform(low=_begin, high=_end, size=_measurements)
    _y_array = line(_x_array) + np.random.normal(loc=0, scale=_sigma, size=_measurements)

    k1_ls, k2_ls = np.polyfit(_x_array, _y_array, 1)
    print(f"Least squares: k={k1_ls:.4f}, b={k2_ls:.4f}")

    ransac = RANSACRegressor(estimator=LinearRegression())
    ransac.fit(_x_array.reshape(-1, 1), _y_array)
    k1_ransac, k2_ransac = ransac.estimator_.coef_[0], ransac.estimator_.intercept_
    print(f"RANSAC: k={k1_ransac:.4f}, b={k2_ransac:.4f}")

    _x = np.array([_begin, _end])
    plt.scatter(_x_array, _y_array)
    plt.plot(_x, line(_x))
    plt.plot(_x, k1_ls * _x + k2_ls)
    plt.plot(_x, k1_ransac * _x + k2_ransac)
    plt.legend(["Измерения", "Истина", "МНК", "RANSAC"])
    plt.grid(True)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
