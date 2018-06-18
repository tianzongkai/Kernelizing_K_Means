from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import random
import sklearn.metrics.pairwise as pairwisekernel
from sklearn.gaussian_process.kernels import RBF

colors = ['darkorchid', 'turquoise', 'darkorange', 'crimson', 'green', 'dodgerblue', 'grey', 'greenyellow','navy']


def kernel_kmeans(datatuple, max_iter = 100, kernel='linear'):
    """
    :param datatuple: in format of(dataset, list of true labels).
    each data point is[x,y]
    :return:
    """
    x = np.asarray(datatuple[0]) # dataset
    k = max(datatuple[1]) + 1 # k: number of clusters
    n,_ = x.shape # n: number of data points
    clusters_label = []

    # Note: for matrix z and list c, rows and colums of index 0 is not being used
    # to be consistent with index in the algorithm
    z = np.zeros((n+1,k+1), dtype=int) #z: indicator matrix
    c = np.zeros((k+1,), dtype=int) # c: a list, size of each cluster
    # kernel_product = np.zeros((n+1,n+1)) # matrix to store value of phi(x_i)*phi(x_j)
    if kernel == 'linear':
        kernel_product = pairwisekernel.linear_kernel(x)
        # print kernel_product.shape
    elif kernel == 'quadratic':
        kernel_product = pairwisekernel.polynomial_kernel(x,x, degree=2, gamma = 0.02, coef0=0.01)
    elif kernel == 'rbf':
        print 'rbf'
        kernel_product = pairwisekernel.rbf_kernel(x, gamma=60)
        # print kernel_product[:5,:5]

    def find_argmin(i):
        pt2center_dist = np.zeros(k+1, dtype=float)
        pt2center_dist[0] = float('inf')
        for j in range(1,k+1):
            if c[j] != 0:
                pts_in_cluster_j = (np.where(z[:,j] == 1))[0] # list of index of data points that in cluster j
                first_term = kernel_product[i-1,j-1]
                second_term = 0.0
                third_term = 0.0
                for l in pts_in_cluster_j:
                    third_term += kernel_product[i-1,l-1]
                    for m in pts_in_cluster_j:
                        second_term += kernel_product[l-1,m-1]
                second_term *= (1.0 / c[j]**2)
                third_term *= (2.0 / c[j])
                pt2center_dist[j] = first_term + second_term - third_term
            else:
                pt2center_dist[j] = float('inf')
        return np.argmin(pt2center_dist)

    for i in range(1,n+1):
        j = random.randint(1,k)
        z[i,j] = 1
        c[j] += 1
    for _ in range(max_iter):
        print c
        c_copy = c.copy()
        z_copy = z.copy()
        for i in range(1, n+1):
            l = (np.where(z[i] == 1))[0][0] # current cluster of point x_i
            q = find_argmin(i) # find a cluster center that closest to point x_i
            if l != q: # if the closest center is not the current center
                c_copy[l] -= 1
                c_copy[q] += 1
                z_copy[i,l] = 0
                z_copy[i,q] = 1
        c = c_copy
        z = z_copy
    for row in range(1,n+1):
        l = (np.where(z[row] == 1))[0][0]
        clusters_label.append(l-1)
    # print clusters_label
    return np.asarray(clusters_label)

def run_kmeans_and_plot(datatuple, clusteringAlgorithm='kmeans'):
    n_clusters = max(datatuple[1]) + 1
    true_labels = datatuple[1]
    kernels = ['linear', 'quadratic', 'rbf']
    kernel_option = kernels[0]
    if clusteringAlgorithm == 'kmeans':
        kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10).fit(datatuple[0])
        clustering_labels = kmeans.fit_predict(datatuple[0])
    elif clusteringAlgorithm == 'kernel kmeans':
        clustering_labels = kernel_kmeans(datatuple, max_iter = 10, kernel=kernel_option)
    # print (true_labels)
    # print (clustering_labels)
    for color, label in zip(colors, range(n_clusters)):
        plt.subplot(121)
        plt.scatter(datatuple[0][clustering_labels == label, 0],
                    datatuple[0][clustering_labels == label, 1], color=color,
                    label=label, marker='+')
        plt.title(kernel_option+' '+clusteringAlgorithm)
        # plt.xticks(())
        # plt.yticks(())
        # plt.suptitle("Concentric Circles")
        plt.subplot(122)
        plt.scatter(datatuple[0][true_labels == label, 0],
                    datatuple[0][true_labels == label, 1], color=color,
                    label=label, marker='+')
        plt.title('True Label')
        plt.xticks(())
        plt.yticks(())
    plt.show()
# plt.scatter(noisy_circles[0][:,0], noisy_circles[0][:,1])

n_samples = 200
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
moons = datasets.make_moons(n_samples=n_samples, random_state=123)

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# print aniso

for datatuple in [aniso]:
    r = 1
    if r == 1:
        run_kmeans_and_plot(datatuple, 'kernel kmeans')
    elif r==2:
        run_kmeans_and_plot(datatuple, 'kmeans')
    else:
        # print 'test'
        x = np.asarray(datatuple[0])
        kernel = 'rbf'

        if kernel == 'linear':
            kernel_product = pairwisekernel.linear_kernel(x, x)
            # print kernel_product.shape
        elif kernel == 'quadratic':
            kernel_product = pairwisekernel.polynomial_kernel(x, x, degree=2)
        elif kernel == 'rbf':
            # print 'test'
            rbf_kernel = RBF()
            print rbf_kernel.hyperparameters
            kernel_product = rbf_kernel.__call__(x, x)
            print kernel_product.shape