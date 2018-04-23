import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.decomposition as skd

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

# (a)
#plt_face(fea[4])

# (b)
meanface = np.mean(fea, axis=0)
#plt_face(meanface)

# (c)
pca = skd.PCA(n_components=6)
skd.PCA.fit(pca, fea)
newfea = pca.transform(fea)
print(newfea[4])

# (d)
# 6 principal components
# pca = skd.PCA(n_components=6)
# skd.PCA.fit(pca, fea)
# Z = pca.transform(fea)
# W = pca.components_
# X6 = W.T.dot(Z.T) + np.array([meanface]).T
# plt_face(X6.T[4])

# 100 principal components
# pca = skd.PCA(n_components=100)
# skd.PCA.fit(pca, fea)
# Z = pca.transform(fea)
# W = pca.components_
# X100 = W.T.dot(Z.T) + np.array([meanface]).T
# plt_face(X100.T[4])

plt.show()