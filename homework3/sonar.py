#
#   CS6923 Machine Learning
#   Homework 3 Part II (2)
#   Shang-Hung Tsai
#   04/02/2018
#

import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np

class logisticRegressionForSonar(object):
    def __init__(self):
        self.originaldata = self.loadData()
        self.data = self.originaldata

    # set number of iterations and the learning rate
    def setParams(self, iterations, learningRate):
        self.iterations = iterations
        self.learningRate = learningRate


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
        # Initialize all weights to be equal to 0.5
        self.weights = [0.5 for _ in range(self.data.shape[1] - 1)]
        self.weight0 = 0.5

        self.predictions = self.predict()
        self.crossEntropies = [self.crossEntropy()]
        for i in range(self.iterations):
            self.updateWeights()

    # Make predictions on the data based on the current weights
    def predict(self):
        results = []
        for index, row in self.data.iterrows():
            prediction = self.weight0 + np.dot(self.weights, row[:-1])

            # The value of prediction might overflow
            try:
                prediction = 1.0 / (1.0 + math.exp(-prediction))
            except OverflowError:
                prediction = 0.0

            results.append(prediction)

        return results

    # use gradient descent to update weights of the prediction function
    def updateWeights(self):
        # Update wi
        newWeights = []

        # an array that contains all (r^t - y^t)
        diff = self.data.iloc[:, -1] - self.predictions

        for wi in range(len(self.weights)):
            sum = np.sum(diff * self.data.iloc[:,wi])
            newW = self.weights[wi] + self.learningRate * sum
            newWeights.append(newW)
        self.weights = newWeights

        # Update w0
        self.weight0 += self.learningRate * np.sum(diff)

        # Update predictions
        self.predictions = self.predict()
        self.crossEntropies.append(self.crossEntropy())

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
        return [i for i in range(0, self.iterations + 1)], self.crossEntropies

    # Compute the percentage of training examples that are misclassified
    def classificationError(self):
        error = 0

        for index, row in self.data.iterrows():
            if self.predictions[index] > 0.5:
                prediction = 1
            else:
                prediction = 0
            if prediction != row.iat[-1]:
                error += 1

        return error / self.data.shape[0]

    # compute the value of L2 norm of w
    def l2norm(self):
        return math.sqrt(np.sum(np.dot(self.weights, self.weights)))

    def getSummary(self):
        print("------------------------------------------------")
        print("Learning rate = {0:f}".format(self.learningRate))
        print("Cross-entropy error = {0:f}".format(self.crossEntropy()))
        print("Classification error = {0:f}".format(self.classificationError()))
        print("||w||2 = {0:f}".format(self.l2norm()))

def main():
    for eta in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 1.5]:
    # for eta in [0.001, 1.5]:
        sonar = logisticRegressionForSonar()
        sonar.setParams(50, eta)
        sonar.train()
        sonar.getSummary()
        i, c = sonar.getIterationAndCrossEntropy()
        plt.plot(i, c, label=str(eta))

    plt.xlabel("Iteration")
    plt.ylabel("Cross-entropy")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)
    plt.show()


if __name__ == "__main__":
    main()

