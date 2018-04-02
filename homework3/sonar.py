#
#   CS6923 Machine Learning
#   Homework 3 Part II (2)
#   Shang-Hung Tsai
#   04/02/2018
#

import matplotlib.pyplot as plt
import csv
import heapq
import math
import pandas as pd

class logisticRegressionForSonar(object):
    def __init__(self):
        self.data = self.loadData()

    # set number of iterations and the learning rate
    def setParams(self, iterations, learningRate):
        self.iterations = iterations
        self.learningRate = learningRate

        # Initialize all weights to be equal to 0.5
        self.weights = [0.5 for _ in range(self.data.shape[1] - 1)]
        self.weight0 = 0.5
        self.crossEntropies = []

    def loadData(self):
        datafile = "sonar.csv"
        data = pd.read_csv(datafile, header=None)

        # replace labels with 1 or 0
        for i in range(data.shape[0]):
            if data.at[i, 60] == "Mine":
                data.at[i, 60] = 1
            else:
                data.at[i, 60] = 0

        return data

    def train(self):
        self.crossEntropies = []
        for i in range(self.iterations):
            self.updateWeights()

    # use gradient descent to update weights of the prediction function
    def updateWeights(self):
        self.predictions = self.predict()

        # Update wi
        newWeights = []
        for wi in range(len(self.weights)):
            sum = 0
            for index, row in self.data.iterrows():
                sum += (row.iat[-1] - self.predictions[index]) * row.iat[wi]
            newW = self.weights[wi] + self.learningRate * sum
            newWeights.append(newW)
        self.weights = newWeights

        # Update w0
        sum = 0
        for index, row in self.data.iterrows():
            sum += (row.iat[-1] - self.predictions[index])
        self.weight0 += self.learningRate * sum

        self.crossEntropies.append(self.crossEntropy())

    # Make predictions on the data based on the current weights
    def predict(self):
        results = []
        for index, row in self.data.iterrows():
            prediction = self.weight0
            for i in range(len(self.weights)):
                prediction += self.weights[i] * row[i]
            results.append(prediction)

        return results

    def crossEntropy(self):
        result = 0.0

        for index, row in self.data.iterrows():
            # check the label of the data
            if row.iat[-1] == 0:
                y = 1 - self.predictions[index]
            else:
                y = self.predictions[index]

            # replace y with e^(-16) if y is too small
            if y < math.exp(-16):
                y = math.exp(-16)

            result += math.log(y)

        return -result

    def getIterationAndCrossEntropy(self):
        return [i for i in range(1, self.iterations + 1)], self.crossEntropies

    # Compute the percentage of training examples that are misclassified
    def classificationError(self):
        error = 0

        for index, row in self.data.iterrows():
            if self.predictions[index] > 0:
                p = 1
            else:
                p = 0
            if p != row.iat[-1]:
                error += 1

        return error / self.data.shape[0]

    def l2norm(self):
        result = 0.0
        for wi in self.weights:
            result += wi * wi
        return math.sqrt(result)

    def getSummary(self):
        print("------------------------------------------------")
        print("Learning rate = {0:f}".format(self.learningRate))
        print("Cross-entropy error = {0:f}".format(self.crossEntropy()))
        print("Classification error = {0:f}".format(self.classificationError()))
        print("||w||2 = {0:f}".format(self.l2norm()))

def main():
    sonar = logisticRegressionForSonar()
    for eta in (0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 1.5):
        sonar.setParams(50, eta)
        sonar.train()
        sonar.getSummary()
        i, c = sonar.getIterationAndCrossEntropy()
        plt.plot(i, c)
    # sonar.setParams(50, 0.001)
    # sonar.train()
    # i1, c1 = sonar.getIterationAndCrossEntropy()
    # plt.plot(i1, c1, linewidth=0.5)

    # sonar.setParams(50, 0.01)
    # sonar.train()
    # i2, c2 = sonar.getIterationAndCrossEntropy()
    # print(i2)
    # print(c2)
    # plt.plot(i2, c2)

    plt.xlabel("Iteration")
    plt.ylabel("Cross-entropy")
    plt.show()

    #
    # sonar.setParams(50, 0.05)
    # sonar.train()

if __name__ == "__main__":
    main()
