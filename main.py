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

    L_part = np.array([
        np.prod([(X[i] - X[j]) for j in range(X.size) if i != j]) for i in range(X.size)
    ])

    def Lagrange(x):
        L_vec = np.array([
            np.prod([(x - X[j]) for j in range(X.size) if j != i]) for i in range(X.size)
        ])

        return np.sum(F_x * L_vec / L_part)

    return np.array([Lagrange(y_i) for y_i in Y])


if __name__ == "__main__":
#    X = np.linspace(-2, 0, 1000)
#    plt.plot(X, Function2(X), color="red")
#
#    plt.show()

    X = np.linspace(-2, 0, 1000)
    interpolated_X = np.linspace(-2, 0, 80)

    fig, axd = plt.subplot_mosaic([["Func1", "Func2"],
                                   ["IFunc1", "IFunc2"],
                                   ["Func1Err", "Func2Err"]], layout="constrained")
    axd["Func1"].set_xlabel("Многочлен Чебышёва")
    axd["Func2"].set_xlabel("Вторая функция")

    F1_G = axd["Func1"].plot(X, Function1(X), color="red")
    F2_G = axd["Func2"].plot(X, Function2(X), color="blue")

    IF1_Y = interpolate_Lagrange(interpolated_X, Function1, X)
    IF2_Y = interpolate_Lagrange(interpolated_X, Function2, X)
    axd["IFunc1"].plot(X, IF1_Y, color="red")
    axd["IFunc2"].plot(X, IF2_Y, color="blue")

    axd["Func1Err"].plot(X, np.abs(IF1_Y - Function1(X)), color="red")
    axd["Func2Err"].plot(X, np.abs(IF2_Y - Function2(X)), color="blue")

    axd["Func1Err"].set_ylim(-0.2, 2)
    axd["Func2Err"].set_ylim(-0.2, 2)

    plt.show()

