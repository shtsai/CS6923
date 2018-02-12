#
#   CS6923 Machine Learning
#   Homework 1
#   Shang-Hung Tsai
#   02/12/2018
#

import sys
import csv
import numpy


data_file = "glasshw1.csv"
file = open(data_file)
data = csv.reader(file)
attributes = [[] for _ in range(10)]
averages = [0.0 for _ in range(10)]
sds = [0.0 for _ in range(10)]

for row in data:
    for i in range(1,11):
        value = eval(row[i])
        attributes[i-1].append(value)

for i in range(len(attributes)):
    attr = attributes[i]
    #print(attr)
    averages[i] = numpy.average(attr)
    sds[i] = numpy.std(attr)

print(averages)
print(sds)



