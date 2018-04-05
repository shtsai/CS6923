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

    def trainAll(self):
        self.data = self.originaldata
        self.train()

    def train(self):
        # Initialize all weights to be equal to 0.5
        self.weights = [0.5 for _ in range(self.data.shape[1] - 1)]
        self.weight0 = 0.5

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


    # perform k-fold cross validation
    def crossValidation(self, k):
        self.splitData(k)
        self.errorCount = 0
        for i in range(k):
            trainingData = []
            for bi in range(k):
                if bi == i:
                    testData = self.dataBlocks[bi]
                else:
                    trainingData.append(self.dataBlocks[bi])

            # load training data
            self.data = pd.concat(trainingData)
            self.train()
            self.test(testData)

        return self.errorCount / self.originaldata.shape[0]

    def test(self, testData):
        testPrediction = self.predictTestData(testData)

        # count the number of errors
        index = 0
        for _, row in testData.iterrows():
            # make prediction
            if testPrediction[index] > 0.5:
                p = 1
            else:
                p = 0

            if p != row.iat[-1]:
                self.errorCount += 1

            index += 1

    def predictTestData(self, testData):
        results = []
        for index, row in testData.iterrows():
            prediction = self.weight0 + np.dot(self.weights, row[:-1])

            # The value of prediction might overflow
            try:
                prediction = 1.0 / (1.0 + math.exp(-prediction))
            except OverflowError:
                prediction = 0.0

            results.append(prediction)
        return results


    # split data into k blocks
    def splitData(self, k):
        dataBlocks = []
        rowEachBlock = int(self.originaldata.shape[0] / k)

        for i in range(k):
            currentBlock = self.originaldata.iloc[i * rowEachBlock : (i + 1) * rowEachBlock]
            dataBlocks.append(currentBlock)
        self.dataBlocks = dataBlocks

    def getSummary(self):
        print("------------------------------------------------")
        print("penalty = {0:f}".format(self.penalty))
        print("Learning rate = {0:f}".format(self.learningRate))
        print("Cross-entropy error = {0:f}".format(self.crossEntropy()))
        print("Classification error = {0:f}".format(self.classificationError()))
        print("||w||2 = {0:f}".format(self.l2norm()))
        print("Classification error (cross-validation) = {0:f}".format(self.crossValidation(5)))

def main():
    sonarRegularized = logisticRegressionForSonarWithRegularization()

    for penalty in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        sonarRegularized.setParams(50, 0.001, penalty)
        sonarRegularized.trainAll()
        sonarRegularized.getSummary()


if __name__ == "__main__":
    main()
