#
#   CS6923 Machine Learning
#   Homework 2
#   Shang-Hung Tsai
#   03/02/2018
#

import matplotlib.pyplot as plt
import csv

MARGIN = 5
displacements = []
mpgs = []
displacementMax = float("-inf")
displacementMin = float("inf")
mpgMax = float("-inf")
mpgMin = float("inf")

with open("auto_train.csv") as file:
    data = csv.reader(file)

    # Skip header
    next(data, None)

    for row in data:
        displacement = float(eval(row[0]))
        mpg = float(eval(row[2]))
        displacements.append(displacement)
        displacementMax = max(displacementMax, displacement)
        displacementMin = min(displacementMin, displacement)
        mpgs.append(mpg)
        mpgMax = max(mpgMax, mpg)
        mpgMin = min(mpgMin, mpg)

plt.plot(displacements, mpgs, "ro")
plt.axis([displacementMin - MARGIN, displacementMax + MARGIN, mpgMin - MARGIN, mpgMax + MARGIN])
plt.xlabel("Displacement")
plt.ylabel("MPG")
plt.show()

