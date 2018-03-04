#
#   CS6923 Machine Learning
#   Homework 2
#   Shang-Hung Tsai
#   03/02/2018
#

from sklearn import linear_model
import matplotlib.pyplot as plt
import csv

def loadData(filePath):
    outputArray = []
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
        outputArray.append(displacements)
        outputArray.append(mpgs)
    return outputArray

def computeError(actualValues, predictedValues):
    error = 0.0
    size = len(actualValues)
    for i in range(size):
        error += pow(actualValues[i] - predictedValues[i], 2)
    return error / 2.0

trainingFilePath = "../auto_train.csv"
testFilePath = "../auto_test.csv"

trainingData = loadData(trainingFilePath)
testData = loadData(testFilePath)

# perform linear regression
reg = linear_model.LinearRegression()
reg.fit(trainingData[0], trainingData[1])
print('The linear regression function is:')
print('y = {0:.4f}x + {1:.4f}'.format(reg.coef_[0], reg.intercept_))

# compute training error
predictionOnTraining = reg.predict(trainingData[0])
trainingError = computeError(trainingData[1], predictionOnTraining)
print('Training Error = {0:.4f}'.format(trainingError))

# predict and compute test error
predictionOnTest = reg.predict(testData[0])
testError = computeError(testData[1], predictionOnTest)
print('Test Error = {0:.4f}'.format(testError))

# plot
plt.scatter(testData[0], testData[1], color='black')
plt.plot(testData[0], predictionOnTest, color='blue', linewidth=3)
plt.xlabel('Displacement')
plt.ylabel('MPG')
plt.show()
