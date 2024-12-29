import matplotlib.pyplot as plt
import numpy as np
from math import exp, cos


def Function1(x):
    def Chebyshov(x, n):
        if (n == 0):
            return 1
        if (n == 1):
            return x
        return 2*x*Chebyshov(x, n-1) - Chebyshov(x, n-2)

    #return np.sin(6*x)
    return Chebyshov(x, 5)

def Function2(x):
    return abs(np.cos(5*x))*(np.exp(-(np.power(x, 2))/2))

def interpolate_Lagrange(train_X, train_Y, test_X):
    """
    param:
            X: точки по которым проходит интерполяция
            F: заданная функция от X (поддерживающая в качестве аргумента массив)
            Y: интересующие нас точки, в которых мы хотим выяснить значение
    """

    L_part = train_Y / np.fromfunction(
        np.vectorize(lambda i: np.prod(train_X[int(i)] - train_X, where=np.fromfunction(lambda j: j != i, (train_X.size,)))),
        (train_X.size,)
    )

    def Lagrange(x):
        L_vec = np.fromfunction(
            np.vectorize(lambda i: np.prod((x - train_X), where=np.fromfunction(lambda j: j != i, (train_X.size,)))),
            (train_X.size,)
        )

        return np.sum(L_part * L_vec)

    return np.fromfunction(np.vectorize(lambda i: Lagrange(test_X[int(i)])), (test_X.size,))


def interpolate_Splain(train_X, train_Y, test_X):
    retval = np.zeros((test_X.size,))

    _h = np.fromfunction(np.vectorize(lambda i: train_X[int(i+1)] - train_X[int(i)]), (train_X.size - 1,))
    _F = 3 * np.fromfunction(np.vectorize(lambda i: (train_Y[int(i+2)] - train_Y[int(i+1)])/_h[int(i+1)] - \
                             (train_Y[int(i+1)] - train_Y[int(i)])/_h[int(i)]), (train_Y.size - 2,))
    _A = np.fromfunction(np.vectorize(lambda i: _h[int(i+1)]), (_h.size-1,))
    _B = np.fromfunction(np.vectorize(lambda i: _h[int(i+1)]), (_h.size-1,))
    _C = 2 * np.fromfunction(np.vectorize(lambda i: _h[int(i)] + _h[int(i+1)]), (_h.size-1,))

    """ Метод прогонки """

    solution = np.zeros((train_X.size-2,));
    alph_s, beta_s = np.zeros((train_X.size-2,)), np.zeros((train_X.size-2,))
    alph_s[0], beta_s[0] = -_B[0]/_C[0], _F[0]/_C[0]

    for i in range(1, solution.size):
        alph_s[i] = -_B[i] / (_A[i-1]*alph_s[i-1] + _C[i])
        beta_s[i] = (_F[i] - _A[i-1]*beta_s[i-1]) / (_A[i-1]*alph_s[i-1] + _C[i])

    solution = np.insert(solution, 0, 0)
    solution = np.insert(solution, 0, 0)

    for i in range(solution.size - 2, 0, -1):
        solution[i] = alph_s[i-1] * solution[i+1] + beta_s[i-1]

    """ Последовательное вычисление коэффицентов полиномов """

    j = 0
    for i in range(1, train_X.size):
        x_i, x_ii = train_X[i-1], train_X[i]
        y_i, y_ii = train_Y[i-1], train_Y[i]
        a_i = y_i
        b_i = (y_ii - y_i) / _h[i-1] -\
            (2 * solution[i-1] + solution[i]) * _h[i-1] / 3
        c_i = solution[i-1]
        d_i = (solution[i] - solution[i-1]) / (3 * _h[i-1])

        function = lambda x, a, b, c, d, x0: (
            a + b*(x - x0) + c*(x - x0)**2 + d*(x - x0)**3
        )

        while (test_X[j] <= x_ii):
            retval[j] = function(test_X[j], a_i, b_i, c_i, d_i, x_i)
            j += 1

            if (j == test_X.size):
                break

    return retval


def interpolate_Linear(train_X, train_Y, test_X):
    retval = np.zeros((test_X.size,))

    j = 0
    for i in range(1, train_X.size):
        x_i, x_ii = train_X[i-1], train_X[i]

        while test_X[j] <= x_ii:
            retval[j] = (test_X[j] - x_i) * (train_Y[i] - train_Y[i-1]) / (x_ii - x_i) + train_Y[i-1]
            j += 1

            if j == test_X.size:
                break
    return retval


if __name__ == "__main__":
#    X = np.linspace(-2, 0, 1000)
#    plt.plot(X, Function2(X), color="red")
#
#    plt.show()

    """
    p1 = 17; p = 100
    X = np.linspace(-2, 0, 1000)
    err = 0.001
    interpolated_X1 = np.linspace(-2, 0, p1)
    interpolated_X = np.linspace(-2, 0, p)

    fig, axd = plt.subplot_mosaic([["Func1", "Func2", "Func3", "Func4"],
                                   ["IFunc1", "IFunc2", "IFunc3", "IFunc4"],
                                   ["Func1Err", "Func2Err", "Func3Err", "Func4Err"]], layout="constrained")
    axd["Func1"].set_xlabel("Многочлен Чебышёва")
    axd["Func2"].set_xlabel("Вторая функция")

    F1_G = axd["Func1"].plot(X, Function1(X), color="red")
    F2_G = axd["Func2"].plot(X, Function2(X), color="blue")
    F3_G = axd["Func3"].plot(X, Function1(X), color="green")
    F4_G = axd["Func4"].plot(X, Function2(X), color="orange")

    IF1_Y = interpolate_Lagrange(interpolated_X1, Function1, X)
    IF2_Y = interpolate_Lagrange(interpolated_X1, Function2, X)

    # При интерполяции сплйнами p mod m = 0, где p - разбиение множества, а m - порядок сплайна
    IF3_Y = interpolate_Splain(interpolated_X, Function1, X)
    IF4_Y = interpolate_Splain(interpolated_X, Function2, X)

    axd["IFunc1"].plot(X, IF1_Y, color="red")
    axd["IFunc1"].set_xlabel("n = %d" % p1)
    axd["IFunc2"].plot(X, IF2_Y, color="blue")
    axd["IFunc2"].set_xlabel("n = %d" % p1)
    axd["IFunc3"].plot(X, IF3_Y, color="green")
    axd["IFunc4"].plot(X, IF4_Y, color="orange")

    axd["Func1Err"].plot(X, np.abs(IF1_Y - Function1(X)), color="red")
    axd["Func2Err"].plot(X, np.abs(IF2_Y - Function2(X)), color="blue")
    axd["Func3Err"].plot(X, np.abs(IF3_Y - Function1(X)), color="green")
    axd["Func4Err"].plot(X, np.abs(IF4_Y - Function2(X)), color="orange")

    axd["Func1Err"].set_xlabel("Error : %f" % (max(np.abs(IF1_Y - Function1(X)))))
    axd["Func2Err"].set_xlabel("Error : %f" % (max(np.abs(IF2_Y - Function2(X)))))
    axd["Func3Err"].set_xlabel("Error : %f" % (max(np.abs(IF3_Y - Function1(X)))))
    axd["Func4Err"].set_xlabel("Error : %f" % (max(np.abs(IF4_Y - Function2(X)))))

    axd["Func1Err"].set_ylim(-0.2, 2)
    axd["Func2Err"].set_ylim(-0.2, 2)
    axd["Func3Err"].set_ylim(-0.2, 2)
    axd["Func4Err"].set_ylim(-0.2, 2)
    """

    """
            Приближение многочленами Лагранжа
    """
    """
    params = [60]
    test_X = np.linspace(-2, 0, 1000)
    test_Y1 = Function1(test_X)
    test_Y2 = Function2(test_X)

    fig1, axd1 = plt.subplot_mosaic([["F_orig;param=%d" % p for p in params],
                                   ["F_inter;param=%d" % p for p in params],
                                   ["F_err;param=%d" % p for p in params]], layout="constrained")

    fig2, axd2 = plt.subplot_mosaic([["F_orig;param=%d" % p for p in params],
                                   ["F_inter;param=%d" % p for p in params],
                                   ["F_err;param=%d" % p for p in params]], layout="constrained")

    for param in params:
        train_X = np.linspace(-2, 0, param)
        train_Y1 = Function1(train_X)
        train_Y2 = Function2(train_X)

        pred_Y1 = interpolate_Lagrange(train_X, train_Y1, test_X)
        pred_Y2 = interpolate_Lagrange(train_X, train_Y2, test_X)

        err1 = np.abs(test_Y1 - pred_Y1)
        err2 = np.abs(test_Y2 - pred_Y2)

        axd1["F_orig;param=%d" % param].plot(test_X, test_Y1, color="red")
        axd1["F_inter;param=%d" % param].plot(test_X, pred_Y1, color="red")
        axd1["F_inter;param=%d" % param].set_xlabel("param = %d" % param)
        axd1["F_err;param=%d" % param].plot(test_X, err1, color="red")
        axd1["F_err;param=%d" % param].set_xlabel("metric = %f" % max(err1))

        axd2["F_orig;param=%d" % param].plot(test_X, test_Y2, color="blue")
        axd2["F_inter;param=%d" % param].plot(test_X, pred_Y2, color="blue")
        axd2["F_inter;param=%d" % param].set_xlabel("param = %d" % param)
        axd2["F_err;param=%d" % param].plot(test_X, err2, color="blue")
        axd2["F_err;param=%d" % param].set_xlabel("metric = %f" % max(err2))

    """

    def build_plot(params):
        #params = [11, 17]
        test_X = np.linspace(-2, 0, 1001)
        test_Y1 = Function1(test_X)
        test_Y2 = Function2(test_X)

        fig1, axd1 = plt.subplot_mosaic([["F_orig;param=%d" % p for p in params],
                                   ["F_inter;param=%d" % p for p in params],
                                   ["F_err;param=%d" % p for p in params]], layout="constrained")

        fig2, axd2 = plt.subplot_mosaic([["F_orig;param=%d" % p for p in params],
                                   ["F_inter;param=%d" % p for p in params],
                                   ["F_err;param=%d" % p for p in params]], layout="constrained")

        for param in params:
            train_X = np.linspace(-2, 0, param)
            train_Y1 = Function1(train_X)
            train_Y2 = Function2(train_X)

            pred_Y1 = interpolate_Splain(train_X, train_Y1, test_X)
            pred_Y2 = interpolate_Splain(train_X, train_Y2, test_X)

            err1 = np.abs(test_Y1 - pred_Y1)
            err2 = np.abs(test_Y2 - pred_Y2)

            axd1["F_orig;param=%d" % param].plot(test_X, test_Y1, color="red")
            axd1["F_inter;param=%d" % param].plot(test_X, pred_Y1, color="red")
            axd1["F_inter;param=%d" % param].set_xlabel("param = %d" % param)
            axd1["F_err;param=%d" % param].plot(test_X, err1, color="red")
            axd1["F_err;param=%d" % param].set_xlabel("metric = %f" % max(err1))

            axd2["F_orig;param=%d" % param].plot(test_X, test_Y2, color="blue")
            axd2["F_inter;param=%d" % param].plot(test_X, pred_Y2, color="blue")
            axd2["F_inter;param=%d" % param].set_xlabel("param = %d" % param)
            axd2["F_err;param=%d" % param].plot(test_X, err2, color="blue")
            axd2["F_err;param=%d" % param].set_xlabel("metric = %f" % max(err2))

        fig1.set_size_inches(4.8, 3.6)
        fig2.set_size_inches(4.8, 3.6)
        fig1.savefig("F1_p%dp%d_Splain.png" % (params[0], params[1]), dpi=200)
        fig2.savefig("F2_p%dp%d_Splain.png" % (params[0], params[1]), dpi=200)
        plt.show()

    build_plot([33, 65])

