import numpy as np
import pylab


def f(x: float, b_arr):
    result = 0
    for i in range(len(b_arr)):
        result += float(b_arr[i] * (x ** i))
    return result


realX = [(i * 4) + 1 for i in range(16)]
realY = [312.89, 1612, 4225, 8043, 12900, 18560, 24740, 31070, 37160, 42510, 46600, 48820, 48510, 44960, 37370,
         24910]

bestB = []


def calculate():
    global transpose, MIN_CRITERION, bestB
    for n in range(len(realX) - 1):
        x_calculated = np.ones(((len(realX)), n + 1))
        for i in range(n):
            for j in range(len(realX)):
                x_calculated[j, i + 1] = realX[j] ** (i + 1)

        y_calculated = np.array([[realY[i]] for i in range(len(realX))])

        transpose = np.linalg.inv(x_calculated.transpose().dot(x_calculated))
        transpose = transpose.dot(x_calculated.transpose())
        b = transpose.dot(y_calculated)

        criterion = sum([(f(realX[i], b) - realY[i]) ** 2 for i in range(len(realX))])

        if criterion < MIN_CRITERION:
            MIN_CRITERION = criterion
            bestB = b


def show_results():
    print('squares:', float(MIN_CRITERION))
    print('b:', [round(float(bestB[i]), 6) for i in range(len(bestB))])
    print('calculated y:', [round((float(f(realX[i], bestB))), 3) for i in range(len(realX))])
    print('given y:', realY)
    pylab.plot(realX, realY, linewidth=10)
    calcY = [f(realX[i], bestB) for i in range(len(realX))]
    pylab.plot(realX, calcY, linewidth=2)
    pylab.show()


if __name__ == "__main__":
    calculate()
    show_results()
