#
#   CS6923 Machine Learning
#   Homework 1
#   Shang-Hung Tsai
#   02/12/2018
#

import sys
import csv
import math

class Attribute(object):
    def __init__(self):
        self.values = []

    def computeMean(self):
        sum = 0.0
        for v in self.values:
            sum += v
        self.mean = sum / len(self.values)

    def computeStandardDeviation(self):
        self.computeMean()
        mean = self.mean
        res = 0.0
        for v in self.values:
            res += math.pow(v - mean, 2)
        self.std = math.sqrt(res / (len(self.values) - 1))

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

    # Given a test example, compute its probability
    def classify(self, row, classFrequency):
        #probability = classFrequency
        #for i in range(1, self.numAttributes + 1):
        #    value = eval(row[i])
        #    probability *= self.attributes[i-1].GaussianPDF(value)

        # take natural log
        probability = math.log(classFrequency)
        for i in range(1, self.numAttributes + 1):
            value = eval(row[i])
            pdf = self.attributes[i-1].GaussianPDF(value)
            if pdf == 0:
                probability += -math.inf
            else:
                probability += math.log(pdf)
        return probability

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
            print("P({0:s}) = {1:f}".format(label, self.classFrequency[label]))

    # Take a test data, classify it
    def classify(self, row):
        res = None
        probability = 0
        for label in self.classes:
            p = self.classes[label].classify(row, self.classFrequency[label])
            if not res or p > probability:
                res = label
                probability = p
        print(row[0] + " => " + str(probability) + " => " + res)
        return res

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

    def clear(self):
        self.classes = {}
        self.classCount = {}
        self.totalCount = 0
        self.classFrequency = {}

    def __str__(self):
        res = ""
        if self.classes:
            for key, value in self.classes.items():
                res += str(key) + str(value) + "\n"
        return res


class NaiveBayes(object):
    '''
        inputFile: path of input csv file
        numClass: number of classes to predict
        numAttributes: number of attributes in each example
        classIndex: index of the class label in the data
        numData: number of examples
        kFold: value of k in k-fold cross validation
    '''
    def __init__(self, inputFile, numClass, numAttributes, classIndex, numData, kFold):
        self.inputFile = inputFile
        self.numAttributes = numAttributes
        self.classifier = Classifer(numClass, numAttributes, classIndex)
        self.numData = numData
        self.kFold = kFold
        if self.kFold > 1:
            self.splitData()

    def run(self):
        if self.kFold <= 1:
            # run entire data as training and test
            self.loadAllDataAsTrainingAndTest()
            correctCount, totalCount, correctRate = self.runTask()
        else:
            correctCount, totalCount, correctRate = self.runKFold()
        print("correct rate = {0:d} / {1:d} = {2:f}".format(correctCount, totalCount, correctRate))

    # This function starts classification task
    def runTask(self):
        self.loadTrainingData()
        self.classifier.compute()
        return self.loadTestData()

    def loadAllDataAsTrainingAndTest(self):
        file = open(self.inputFile)
        data = csv.reader(file)

        rows = []
        for row in data:
            rows.append(row)
        self.trainingData = rows
        self.testData = rows

    def loadTrainingData(self):
        for row in self.trainingData:
            self.classifier.train(row)

    def loadTestData(self):
        correctCount = 0
        totalCount = 0
        for row in self.testData:
            if self.classifier.classify(row) == row[-1]:
                correctCount += 1
            totalCount += 1

        return correctCount, totalCount, correctCount / totalCount

    # split data into k blocks
    def splitData(self):
        file = open(self.inputFile)
        data = csv.reader(file)

        dataBlocks = []
        rowEachBlock = self.numData / self.kFold
        count = 0
        currentBlock =  []
        for row in data:
            currentBlock.append(row)
            count += 1
            if count == rowEachBlock:
                dataBlocks.append(currentBlock)
                currentBlock = []
                count = 0
        self.dataBlocks = dataBlocks

    def runKFold(self):
        totalCorrectCount = 0
        totalCount = 0
        for i in range(self.kFold):
            self.classifier.clear()
            trainingData = []
            for blockIndex in range(self.kFold):
                if i == blockIndex:
                    self.testData = self.dataBlocks[blockIndex]
                else:
                    trainingData.extend(self.dataBlocks[blockIndex])
            self.trainingData = trainingData
            correctCount, count, _ = self.runTask()
            totalCorrectCount += correctCount
            totalCount += count
        return totalCorrectCount, totalCount, totalCorrectCount / totalCount


    def __str__(self):
        return str(self.classifier)



NB = NaiveBayes("glasshw1.csv", 2, 9, 10, 200, 0)
NB.run()
print(NB.classifier.getMeans())
print(NB.classifier.getSTDs())

NB5Fold = NaiveBayes("glasshw1.csv", 2, 9, 10, 200, 5)
NB5Fold.run()
