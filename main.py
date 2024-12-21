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

    return Chebyshov(x, 5)

def Function2(x):
    return abs(np.cos(5*x))*(np.exp(-(np.power(x, 2))/2))

def interpolate_Lagrange(X, F, Y):
    """
    param:
            X: точки по которым проходит интерполяция
            F: заданная функция от X (поддерживающая в качестве аргумента массив)
            Y: интересующие нас точки, в которых мы хотим выяснить значение
    """

    F_x = F(X)

    L_part = F_x / np.fromfunction(
        np.vectorize(lambda i: np.prod(X[int(i)] - X, where=np.fromfunction(lambda j: j != i, (X.size,)))),
        (X.size,)
    )

    def Lagrange(x):
        L_vec = np.fromfunction(
            np.vectorize(lambda i: np.prod((x - X), where=np.fromfunction(lambda j: j != i, (X.size,)))),
            (X.size,)
        )

        return np.sum(L_part * L_vec)

    return np.fromfunction(np.vectorize(lambda i: Lagrange(Y[int(i)])), (Y.size,))


def interpolate_Splain(X, F, Y):
    result = np.zeros(Y.shape)
    m = 5

    def create_Splain(x):
        ret = interpolate_Lagrange(x, F, Y)
        for i, v in enumerate(Y):
            if (x[0] <= v and v <= x[-1]):
                result[i] = ret[i]

    for i in range(X.size // m):
        create_Splain(X[m*i : m*(i+1) + 1])
    return result


if __name__ == "__main__":
#    X = np.linspace(-2, 0, 1000)
#    plt.plot(X, Function2(X), color="red")
#
#    plt.show()

    p = 30
    X = np.linspace(-2, 0, 1000)
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

    IF1_Y = interpolate_Lagrange(interpolated_X, Function1, X)
    IF2_Y = interpolate_Lagrange(interpolated_X, Function2, X)

    # При интерполяции сплйнами p mod m = 0, где p - разбиение множества, а m - порядок сплайна
    IF3_Y = interpolate_Splain(interpolated_X, Function1, X)
    IF4_Y = interpolate_Splain(interpolated_X, Function2, X)

    axd["IFunc1"].plot(X, IF1_Y, color="red")
    axd["IFunc1"].set_xlabel("n = %d" % p)
    axd["IFunc2"].plot(X, IF2_Y, color="blue")
    axd["IFunc2"].set_xlabel("n = %d" % p)
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

    plt.show()

