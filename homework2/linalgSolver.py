import numpy as np

a = np.array([[8, 9, 2], [43.5, 55, 9], [41, 43.5, 8]])
b = np.array([6.5, 24.5, 19])
x = np.linalg.solve(a, b)
print(x)