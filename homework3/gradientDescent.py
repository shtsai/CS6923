#
#   CS6923 Machine Learning
#   Homework 3
#   Shang-Hung Tsai
#   03/30/2018
#

import matplotlib.pyplot as plt
import numpy as np
import csv

def f(x):
    return 16 * pow(x, 4) - 32 * pow(x, 3) - 8 * pow(x, 2) + 10 * x + 9

def f_derivative(x):
    return 64 * pow(x, 3) - 96 * pow(x, 2) - 16 * x + 10

def gradientDescent(initial, learning_rate, iteration):
    print("Initially, x = {0:f}".format(initial))

    x = initial
    for i in range(iteration):
        x = update(x, learning_rate)
        print("After iteration {0:d}, the x = {1:f}".format(i+1, x))
    print()

def update(x, learning_rate):
    return x - learning_rate * f_derivative(x)

def plotFunction():
#    interval = np.arange(-2., 4., 0.01)
    interval = np.linspace(-2.0, 3.0, num=500)
    print(interval)
    print([f(i) for i in interval])
    plt.plot(interval, [f(i) for i in interval], linewidth=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def main():
    # plotFunction()

    # 1. (b)
    print("x = -1, learning rate = 0.001, iteration = 5")
    gradientDescent(-1, 0.001, 5)
    print("x = -1, learning rate = 0.001, iteration = 1000")
    gradientDescent(-1, 0.001, 1000)

    # 1. (c)
    print("x = 2, learning rate = 0.001, iteration = 5")
    gradientDescent(2, 0.001, 5)
    print("x = 2, learning rate = 0.001, iteration = 1000")
    gradientDescent(2, 0.001, 1000)

    # 1. (d)
    print("x = -1, learning rate = 0.01, iteration = 1000")
    gradientDescent(-1, 0.01, 1000)

    # 1. (e)
    print("x = -1, learning rate = 0.05, iteration = 100")
    # gradientDescent(-1, 0.05, 100)


if __name__ == "__main__":
    main()