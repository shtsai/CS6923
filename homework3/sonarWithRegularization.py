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

    def train(self):
        self.predictions = self.predict()
        for i in range(self.iterations):
            self.updateWeights()

    # use gradient descent to update weights of the prediction function
    # regularized version
    def updateWeights(self):
        # Update wi
        newWeights = []

        # an array that contains all (r^t - y^t)
        diff = self.data.iloc[:, -1] - self.predictions

        for wi in range(len(self.weights)):
            sum = np.sum(diff * self.data.iloc[:,wi]) - self.penalty * self.weights[wi]
            newW = self.weights[wi] + self.learningRate * sum
            newWeights.append(newW)

        self.weights = newWeights

        # Update w0
        self.weight0 += self.learningRate * np.sum(diff)
        self.predictions = self.predict()

    # Compute the regularized version of the cross-entropy error
    def crossEntropy(self):
        original= super(logisticRegressionForSonarWithRegularization, self).crossEntropy()
        regularizedTerm = (self.penalty / 2.0) * np.sum(np.dot(self.weights, self.weights))
        # print("original = " + str(original))
        # print("regularized term = " + str(regularizedTerm))
        return original + regularizedTerm

    def getSummary(self):
        print("------------------------------------------------")
        print("Penalty = {0:f}".format(self.penalty))
        print("Cross-entropy error = {0:f}".format(self.crossEntropy()))
        print("Classification error = {0:f}".format(self.classificationError()))
        print("||w||2 = {0:f}".format(self.l2norm()))

def main():
    sonarRegularized = logisticRegressionForSonarWithRegularization()
    for penalty in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        sonarRegularized.setParams(50, 0.1, penalty)
        sonarRegularized.train()
        sonarRegularized.getSummary()

if __name__ == "__main__":
    main()
