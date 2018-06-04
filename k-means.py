from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import random

colors = ['darkorchid', 'turquoise', 'darkorange', 'crimson', 'green', 'dodgerblue', 'grey', 'greenyellow','navy']

def run_kmeans_and_plot(dataset):
    n_clusters = max(dataset[1]) + 1
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10).fit(dataset[0])
    clustering_labels = kmeans.fit_predict(dataset[0])
    true_label = dataset[1]
    for color, label in zip(colors, range(n_clusters)):
        plt.subplot(121)
        plt.scatter(dataset[0][clustering_labels == label, 0],
                    dataset[0][clustering_labels == label, 1], color=color,
                    label=label, marker='+')
        plt.title('K-means')
        plt.xticks(())
        plt.yticks(())
        # plt.suptitle("Concentric Circles")
        plt.subplot(122)
        plt.scatter(dataset[0][true_label == label, 0],
                    dataset[0][true_label == label, 1], color=color,
                    label=label, marker='+')
        plt.title('True Label')
        plt.xticks(())
        plt.yticks(())
    plt.show()
# plt.scatter(noisy_circles[0][:,0], noisy_circles[0][:,1])




n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,                                     noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# print aniso

for datatuple in [aniso]:
    run_kmeans_and_plot(datatuple)


def kernel_kmeans(datatuple):
    """
    :param datatuple: in format of(dataset, list of true labels).
    each data point is[x,y]
    :return:
    """
    npdataset = np.asarray(datatuple[0])
    k = max(datatuple[1]) + 1 # k: number of clusters
    n,_ = npdataset.shape # n: number of data points

    # Note: for matrix z and list c, rows and colums of index 0 is not being used
    # to be consistent with index in the algorithm
    z = np.asarray([[0 for j in range(k+1)] for i in range(n+1)]) #z: indicator matrix
    c = np.asarray([0 for i in range(k+1)]) # c: a list, size of each cluster

