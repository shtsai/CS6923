import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_lfw_people
lfw_people = fetch_lfw_people(min_faces_per_person=70)
n_samples, h, w = lfw_people.images.shape
npix = h*w
fea = lfw_people.data

def plt_face(x):
    global h, w
    plt.imshow(x.reshape((h, w)), cmap=plt.cm.gray)
    plt.xticks([])

# plt.figure(figsize=(10, 20))
# nplt = 4
# for i in range(nplt):
#     plt.subplot(1, nplt, i+1)
#     plt_face(fea[i])

meanface = np.mean(fea, axis=0)
plt_face(meanface)
plt.show()