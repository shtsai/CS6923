#
#   CS6923 Machine Learning
#   Homework 2 Part II (3)
#   Shang-Hung Tsai
#   03/02/2018
#

from sklearn import linear_model
import matplotlib.pyplot as plt
from scipy.interpolate import spline
import csv
import copy
import numpy as np

def loadData(filePath):
    with open(filePath) as file:
        displacements = []
        mpgs = []
        data = csv.reader(file)
        # skip header
        next(data, None)
        for row in data:
            displacement = float(eval(row[0]))
            mpg = float(eval(row[2]))
            displacements.append([displacement])
            mpgs.append(mpg)
    return displacements, mpgs

# Generate polynomials to degree k
def transformInput(array, degree):
    newarray = copy.deepcopy(array)
    for i in range(2, degree + 1):
        for j in range(len(newarray)):
            newarray[j].append(pow(newarray[j][0], i))
    res = np.array(newarray)
    return res

def computeError(actualValues, predictedValues):
    error = 0.0
    size = len(actualValues)
    for i in range(size):
        error += pow(actualValues[i] - predictedValues[i], 2)
    return error / 2.0

def polynomialRegression(trainingInput, trainingOutput, testInput, testOutput, degree):
    print('Degree = {0:d}'.format(degree))

    # transform input to degree k
    trainingInput = transformInput(trainingInput, degree)
    testInput = transformInput(testInput, degree)

    # perform polynomial regression
    reg = linear_model.LinearRegression()
    reg.fit(trainingInput, trainingOutput)
    print('coefficients = ')
    print(reg.coef_)
    print('interception = ')
    print(reg.intercept_)

    # compute training error
    predictionOnTraining = reg.predict(trainingInput)
    trainingError = computeError(trainingOutput, predictionOnTraining)
    print('Training Error = {0:.4f}'.format(trainingError))

    # predict and compute test error
    predictionOnTest = reg.predict(testInput)
    testError = computeError(testOutput, predictionOnTest)
    print('Test Error = {0:.4f}'.format(testError))
    print('----------------------------------------')

    # return x and y axis of the plot
    xaxis = testInput[:,0]
    pairs = []
    for i in range(len(xaxis)):
        pairs.append((xaxis[i], predictionOnTest[i]))
    # sort values in acsending order of x value
    pairs.sort()
    xaxis = []
    yaxis = []
    for i in range(len(pairs)):
        xaxis.append(pairs[i][0])
        yaxis.append(pairs[i][1])

    return xaxis, yaxis

def main():
    trainingFilePath = "../auto_train.csv"
    testFilePath = "../auto_test.csv"
    trainingInput, trainingOutput = loadData(trainingFilePath)
    testInput, testOutput = loadData(testFilePath)

    x2, y2 = polynomialRegression(trainingInput, trainingOutput, testInput, testOutput, 2)
    x4, y4 = polynomialRegression(trainingInput, trainingOutput, testInput, testOutput, 4)
    x6, y6 = polynomialRegression(trainingInput, trainingOutput, testInput, testOutput, 6)

    # plot
    plt.scatter(np.array(testInput)[:,0], testOutput, color='black')
    plt.plot(x2, y2, color='green', label="2")
    plt.plot(x4, y4, color='blue', label="4")
    plt.plot(x6, y6, color='red', label="6")
    plt.xlabel('Displacement')
    plt.ylabel('MPG')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

if __name__ == "__main__":
    main()
