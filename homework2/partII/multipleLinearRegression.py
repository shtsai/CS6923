#
#   CS6923 Machine Learning
#   Homework 2 Part II (4)
#   Shang-Hung Tsai
#   03/02/2018
#

from sklearn import linear_model
import matplotlib.pyplot as plt
import csv

def loadData(filePath):
    input = []
    output = []
    with open(filePath) as file:
        data = csv.reader(file)
        # skip header
        next(data, None)
        for row in data:
            displacement = float(eval(row[0]))
            horsepower = float(eval(row[1]))
            mpg = float(eval(row[2]))
            input.append([displacement, horsepower])
            output.append(mpg)
    return input, output

def computeError(actualValues, predictedValues):
    error = 0.0
    size = len(actualValues)
    for i in range(size):
        error += pow(actualValues[i] - predictedValues[i], 2)
    return error / 2.0

def main():
    trainingFilePath = "../auto_train.csv"
    testFilePath = "../auto_test.csv"

    trainingInput, trainingOutput = loadData(trainingFilePath)
    testInput, testOutput = loadData(testFilePath)

    # perform linear regression
    reg = linear_model.LinearRegression()
    reg.fit(trainingInput, trainingOutput)
    print('The coefficient of linear regression function is:')
    print(reg.coef_)
    print(reg.intercept_)

    # compute training error
    predictionOnTraining = reg.predict(trainingInput)
    trainingError = computeError(trainingOutput, predictionOnTraining)
    print('Training Error = {0:.4f}'.format(trainingError))

    # predict and compute test error
    predictionOnTest = reg.predict(testInput)
    testError = computeError(testOutput, predictionOnTest)
    print('Test Error = {0:.4f}'.format(testError))

if __name__ == "__main__":
    main()
