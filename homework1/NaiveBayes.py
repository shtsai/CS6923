#
#   CS6923 Machine Learning
#   Homework 1
#   Shang-Hung Tsai
#   02/12/2018
#

import sys
import csv
import numpy
import math

class Attribute(object):
    def __init__(self):
        self.values = []

    def computeMean(self):
        self.mean = numpy.average(self.values)

    def computeStandardDeviation(self):
        self.std = numpy.std(self.values)

    def add(self, value):
        self.values.append(eval(value))

    def GaussianPDF(self, x):
        base = 1.0 / math.sqrt(2 * math.pi * self.std * self.std)
        exp = math.exp(-1.0 * (math.pow(x - self.mean, 2)) / (2 * self.std * self.std))
        return base * exp


    def __str__(self):
        res = "["
        for value in self.values:
            res += str(value) + ","
        res += "]"
        return res


class ClassInfo(object):
    def __init__(self, numAttributes):
        self.numAttributes = numAttributes
        self.attributes = [Attribute() for _ in range(numAttributes)]

    # Add a training example to this class
    def add(self, row):
        for i in range(1, self.numAttributes + 1):
            self.attributes[i-1].add(row[i])

    # compute mean and standard deviation for each attributes
    def compute(self):
        for attribute in self.attributes:
            attribute.computeMean()
            attribute.computeStandardDeviation()

    def getAttributeMean(self):
        res = "["
        for attribute in self.attributes:
            res += str(attribute.mean) + ","
        res += "]"
        return res

    def getAttributeSTD(self):
        res = "["
        for attribute in self.attributes:
            res += str(attribute.std) + ","
        res += "]"
        return res

    def __str__(self):
        res = ""
        for attribute in self.attributes:
            res += str(attribute) + "\n"
        return res


class Classifer(object):
    def __init__(self, numClass, numAttributes, classIndex):
        self.numClass = numClass
        self.numAttributes = numAttributes
        self.classIndex = classIndex
        self.classes = {}
        self.classCount = {}
        self.totalCount = 0
        self.classFrequency = {}

    # Take a training data, add it to its class
    def train(self, row):
        label = row[self.classIndex]
        if label not in self.classes:
            self.classes[label] = ClassInfo(self.numAttributes)
            self.classCount[label] = 0
        self.classes[label].add(row)
        self.classCount[label] += 1
        self.totalCount += 1

    # compute mean, standard deviation and frequency for each class
    def compute(self):
        for label in self.classes:
            self.classes[label].compute()

        # compute frequency estimate for each class
        for label in self.classes:
            self.classFrequency[label] = self.classCount[label] / self.totalCount

    def getMeans(self):
        res = ""
        if self.classes:
            for cls, classInfo in self.classes.items():
                res += str(cls) + classInfo.getAttributeMean() + "\n"
        return res

    def getSTDs(self):
        res = ""
        if self.classes:
            for cls, classInfo in self.classes.items():
                res += str(cls) + classInfo.getAttributeSTD() + "\n"
        return res

    def __str__(self):
        res = ""
        if self.classes:
            for key, value in self.classes.items():
                res += str(key) + str(value) + "\n"
        return res


class NaiveBayes(object):
    def __init__(self, inputFile, numClass, numAttributes, classIndex, numData, kFold):
        self.inputFile = inputFile
        self.classifier = Classifer(numClass, numAttributes, classIndex)

    def train(self):
        file = open(self.inputFile)
        data = csv.reader(file)

        # Load training data
        for row in data:
            self.classifier.train(row)
        self.classifier.compute()

    def __str__(self):
        return str(self.classifier)



NB = NaiveBayes("glasshw1.csv", 2, 9, 10, 200, 0)
# print(NB)
NB.train()
# print(NB)

#print(NB.classifier.getMeans())
#print(NB.classifier.getSTDs())



# print(GaussianPDF(9.2, 1.8, 9.9))
