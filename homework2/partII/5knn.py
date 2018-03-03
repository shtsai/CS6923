#
#   CS6923 Machine Learning
#   Homework 2
#   Shang-Hung Tsai
#   03/02/2018
#

import matplotlib.pyplot as plt
import csv
import heapq

class DataPoint(object):
    def __init__(self, displacement, horsepower, mpg):
        self.displacement = displacement
        self.horsepower = horsepower
        self.mpg = mpg

    def __gt__(self, other):
        return (pow(self.displacement, 2) + pow(self.horsepower, 2)) \
               > (pow(other.displacement, 2) + pow(other.horsepower,2))

    def __str__(self):
        return "{0:f} {1:f} {2:f}".format(self.displacement, self.horsepower, self.mpg)

    def getEuclideanDistance(self, target):
        return pow(self.displacement - target.displacement, 2) \
               + pow(self.horsepower - target.horsepower, 2)


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
        for d in self.trainingdata:
            distance = d.getEuclideanDistance(testDataPoint)
            if len(self.heap) < self.k:
                heapq.heappush(self.heap, (-distance, d))
            elif self.heap[0][0] < -distance:
                heapq.heappop(self.heap)
                heapq.heappush(self.heap, (-distance, d))

        sum = 0
        size = len(self.heap)
        while self.heap:
            point = heapq.heappop(self.heap)
            sum += point[1].mpg
            # print(str(point[0]) + " " + str(point[1]))
        return sum / size

knn = KNearestNeighbor(100)
knn.loadTrainingData("../auto_train.csv")
# knn.printData()

test = DataPoint(200, 200, 0)
print(knn.predict(test))
