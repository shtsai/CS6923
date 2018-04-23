import sklearn.decomposition as skd
import numpy as np

X = np.array([[4,1,3],[8,5,3],[6,0,1],[1,4,5]])
pca = skd.PCA(n_components=3)
skd.PCA.fit(pca, X)
W1 = pca.components_
W = W1.transpose()
Z = pca.transform(X)

print(Z)