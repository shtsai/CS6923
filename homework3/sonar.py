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

    def loadData(self):
        datafile = "sonar.csv"
        data = pd.read_csv(datafile, header=None)

        # replace labels with 1 or 0
        for i in range(data.shape[0]):
            if data.at[i, 60] == "Mine":
                data.at[i, 60] = 1.0
            else:
                data.at[i, 60] = 0.0

        return data

    def train(self):
        for i in range(self.iterations):
            self.updateWeights()

    # use gradient descent to update weights of the prediction function
    def updateWeights(self):
        predictions = self.predict()

        # Update wi
        newWeights = []
        for wi in range(len(self.weights)):
            sum = 0
            for index, row in self.data.iterrows():
                sum += (row.iat[-1] - predictions[index]) * row.iat[wi]
            newW = self.weights[wi] + self.learningRate * sum
            newWeights.append(newW)
        self.weights = newWeights

        # Update w0
        sum = 0
        for index, row in self.data.iterrows():
            sum += (row.iat[-1] - predictions[index])
        self.weight0 += self.learningRate * sum

        print("New weights------------------------------")
        print(self.weights)
        print(self.weight0)
        print(self.crossEntropy())


    # Make predictions on the data based on the current weights
    def predict(self):
        results = []
        for index, row in self.data.iterrows():
            prediction = self.weight0
            for i in range(len(self.weights)):
                prediction += self.weights[i] + row[i]
            results.append(prediction)

        return results

    def crossEntropy(self):
        predictions = self.predict()
        result = 0.0

        for index, row in self.data.iterrows():
            # check the label of the data
            if row.iat[-1] == 0.0:
                y = 1 - predictions[index]
            else:
                y = predictions[index]

            # replace y with e^(-16) if y is too small
            if y < math.exp(-16):
                y = math.exp(-16)

            result += math.log(y)

        return -result


def main():
    sonar = logisticRegressionForSonar()
    sonar.setParams(50, 0.001)
    sonar.train()

    sonar.setParams(50, 0.01)
    sonar.train()

    sonar.setParams(50, 0.05)
    sonar.train()

if __name__ == "__main__":
    main()
