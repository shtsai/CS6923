#
#   CS6923 Machine Learning
#   Homework 2 Part II (5)
#   Shang-Hung Tsai
#   03/02/2018
#

import matplotlib.pyplot as plt
import csv
import heapq
import math
import random

class DataPoint(object):
    def __init__(self, displacement, horsepower, mpg):
        self.displacement = displacement
        self.horsepower = horsepower
        self.mpg = mpg
        random.seed(0)    # use the same seed for the experiment

    # break tie randomly
    def __gt__(self, other):
        return random.uniform(0, 1) > 0.5

    def __str__(self):
        return "{0:f} {1:f} {2:f}".format(self.displacement, self.horsepower, self.mpg)

    def getEuclideanDistance(self, target):
        return math.sqrt(pow(self.displacement - target.displacement, 2) \
               + pow(self.horsepower - target.horsepower, 2))


class KNearestNeighbor(object):
    def __init__(self, k):
        self.k = k
        self.trainingdata = []
        self.heap = []

    def loadTrainingData(self, trainingFile):
        with open(trainingFile) as file:
            data = csv.reader(file)
            # Skip header
            next(data, None)

            for row in data:
                displacement = float(eval(row[0]))
                horsepower = float(eval(row[1]))
                mpg = float(eval(row[2]))
                self.trainingdata.append(DataPoint(displacement, horsepower, mpg))

    def printData(self):
        for d in self.trainingdata:
            print(d)

    # Given test data point, predict its mpg value using K Nearest Neighbor algorithm
    def predict(self, testDataPoint):
        self.getKNN(testDataPoint)
        return self.computeAverage()

    def getKNN(self, testDataPoint):
        for d in self.trainingdata:
            distance = d.getEuclideanDistance(testDataPoint)
            if len(self.heap) < self.k:
                heapq.heappush(self.heap, (-distance, d))
            elif self.heap[0][0] < -distance:
                heapq.heappop(self.heap)
                heapq.heappush(self.heap, (-distance, d))

    def computeAverage(self):
        sum = 0.0
        size = len(self.heap)
        while self.heap:
            point = heapq.heappop(self.heap)
            sum += point[1].mpg
        return sum / size


class KNNExperiment(object):
    def __init__(self, k, trainingFile, testFile):
        self.knn = KNearestNeighbor(k)
        self.trainingFile = trainingFile
        self.testFile = testFile

    def train(self):
        self.knn.loadTrainingData(self.trainingFile)

    def test(self):
        err = 0.0
        with open(self.testFile) as testData:
            data = csv.reader(testData)
            # skip header
            next(data, None)

            for row in data:
                displacement = float(eval(row[0]))
                horsepower = float(eval(row[1]))
                mpg = float(eval(row[2]))
                testDataPoint = DataPoint(displacement, horsepower, mpg)
                prediction = self.knn.predict(testDataPoint)
                err += pow(prediction - mpg, 2)
        return err / 2.0



def main():
    trainingFile = "../auto_train.csv"
    testFile = "../auto_test.csv"
    knn1 = KNNExperiment(1, trainingFile, testFile)
    knn1.train()
    knn1Err = knn1.test()
    print("K = 1, Err = {0:.2f}".format(knn1Err))

    knn3 = KNNExperiment(3, trainingFile, testFile)
    knn3.train()
    knn3Err = knn3.test()
    print("K = 3, Err = {0:.2f}".format(knn3Err))

    knn20 = KNNExperiment(20, trainingFile, testFile)
    knn20.train()
    knn20Err = knn20.test()
    print("K = 20, Err = {0:.2f}".format(knn20Err))

if __name__ == "__main__":
    main()

