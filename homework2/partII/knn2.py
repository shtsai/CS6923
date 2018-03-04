#
#   CS6923 Machine Learning
#   Homework 2 part II (7)
#   Shang-Hung Tsai
#   03/02/2018
#

from knn import *

# Compute a weighted (by distance) average of neighbors
class KNearestNeighborWithWeightedAverage(KNearestNeighbor):
    def predict(self, testDataPoint):
        self.getKNN(testDataPoint)
        return self.computeWeightAverage()

    def computeWeightAverage(self):
        denominator = 0.0
        numerator = 0.0
        while self.heap:
            point = heapq.heappop(self.heap)
            distance = -point[0]
            if distance == 0.0:
                distance = 0.00001
            mpg = point[1].mpg
            denominator += (1.0 / distance) * mpg
            numerator += 1.0 / distance
        return denominator / numerator

class KNNExperimentWithWeightedAverage(KNNExperiment):
    def __init__(self, k, trainingFile, testFile):
        self.knn = KNearestNeighborWithWeightedAverage(k)
        self.trainingFile = trainingFile
        self.testFile = testFile


def main():
    trainingFile = "../auto_train.csv"
    testFile = "../auto_test.csv"
    knn1 = KNNExperimentWithWeightedAverage(1, trainingFile, testFile)
    knn1.train()
    knn1Err = knn1.test()
    print("K = 1, Err = {0:.2f}".format(knn1Err))

    knn3 = KNNExperimentWithWeightedAverage(3, trainingFile, testFile)
    knn3.train()
    knn3Err = knn3.test()
    print("K = 3, Err = {0:.2f}".format(knn3Err))

    knn20 = KNNExperimentWithWeightedAverage(20, trainingFile, testFile)
    knn20.train()
    knn20Err = knn20.test()
    print("K = 20, Err = {0:.2f}".format(knn20Err))

if __name__ == "__main__":
    main()

