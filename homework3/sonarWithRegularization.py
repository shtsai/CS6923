#
#   CS6923 Machine Learning
#   Homework 3 Part II (2)
#   Shang-Hung Tsai
#   04/02/2018
#

from sonar import *

class logisticRegressionForSonarWithRegularization(logisticRegressionForSonar):
    # set number of iterations, learning rate and penalty
    def setParams(self, iterations, learningRate, penalty):
        self.iterations = iterations
        self.learningRate = learningRate
        self.penalty = penalty

        # Initialize all weights to be equal to 0.5
        self.weights = [0.5 for _ in range(self.data.shape[1] - 1)]
        self.weight0 = 0.5
        self.crossEntropies = []

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
            newW = self.weights[wi] + self.learningRate * (sum - self.penalty * self.weights[wi])
            newWeights.append(newW)
        self.weights = newWeights

        # Update w0
        sum = 0
        for index, row in self.data.iterrows():
            sum += (row.iat[-1] - self.predictions[index])
        self.weight0 += self.learningRate * sum

        self.crossEntropies.append(self.crossEntropy())

    def crossEntropy(self):
        result = 0.0

        for index, row in self.data.iterrows():
            # check the label of the data
            if row.iat[-1] == 0.0:
                y = 1 - self.predictions[index]
            else:
                y = self.predictions[index]

            # replace y with e^(-16) if y is too small
            if y < math.exp(-16):
                y = math.exp(-16)

            result += math.log(y)

        regularizedTerm = 0.0
        for wi in self.weights:
            regularizedTerm += wi * wi
        regularizedTerm *= self.penalty / 2.0

        return regularizedTerm - result

    def getSummary(self):
        print("------------------------------------------------")
        print("Penalty = {0:f}".format(self.penalty))
        print("Cross-entropy error = {0:f}".format(self.crossEntropy()))
        print("Classification error = {0:f}".format(self.classificationError()))
        print("||w||2 = {0:f}".format(self.l2norm()))

def main():
    sonarRegularized = logisticRegressionForSonarWithRegularization()
    for penalty in (0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        sonarRegularized.setParams(50, 0.1, penalty)
        sonarRegularized.train()
        sonarRegularized.getSummary()

if __name__ == "__main__":
    main()
