from typing import Tuple
import numpy as np


def wcss(X: np.ndarray, K: int, Z: np.ndarray, centroids: np.ndarray) -> float:
    """
    :param X: data for clustering, shape: (N, D), with N - number of data points, D - dimension of data points
    :param K: number of clusters
    :param Z: indicator variables for all data points, shape: (N, K)
    :param centroids: means of clusters, K vectors of dimension D, shape: (K, D)
    :return: objective function WCSS - a scalar value
    """

    # TODO: Calculate WCSS and return it
    wcss_value = 0.0

    for i in range(X.shape[0]):
        for k in range(K):
            # equation 1 (np.sum sums dimaensions)
            distance_squared = Z[i, k] * np.sum((X[i] - centroids[k]) * ((X[i] - centroids[k])))
            # sum up all distances
            wcss_value += distance_squared
            # This means, to each data point a one-hot vector of length K is assigned – there is exactly one 1 in the vector
            # at the index of the assigned cluster, and the remaining entries of the vector are zeros.
            if(Z[i, k] != 0):
                break  

    return wcss_value


def closest_centroid(sample: np.ndarray, centroids: np.ndarray) -> int:
    """
    :param sample: a data point x_n (of dimension D)
    :param centroids: means of clusters, K vectors of dimension D, shape: (K, D)
    :return: idx_closest_cluster, that is, the index of the closest cluster
    """

    # TODO: Calculate distance of the current sample to each centroid.
    #       Afterwards you should return the index of the closest centroid (int value from 0 to (K-1))
    distances = []
    for k in range (centroids.shape[0]):
        dist_temp = 0.0
        for i in range (centroids.shape[1]):
            dist_temp += (sample[i] - centroids[k][i]) * (sample[i] - centroids[k][i])
        distances.append(dist_temp)
    
    idx_closest_cluster = np.argmin(distances)
    
    return idx_closest_cluster


def compute_Z(X: np.ndarray, K: int, centroids: np.ndarray) -> np.ndarray:
    """
    :param X: data for clustering, shape: (N, D), with N - number of data points, D - dimension
    :param K: number of clusters
    :param centroids: means of clusters, K vectors of dimension D, shape: (K, D)
    :return: Z: indicator variables for all data points, shape: (N, K)
    """

    N = X.shape[0]
    # TODO: Compute Z matrix which holds the indicator variables for all data points (using `closest_centroid`).
    #       The indicator variables represent the cluster assignments of each data point.
    # Z MATRIX IS FILLED WITH ZEROES N - amount of datapoints, K amount of clusters
    Z = np.zeros((N, K))
    
    for i in range(N):
        # Finding index 
        row_of_assigned_cluster = closest_centroid(X[i], centroids)
        # [index of datapoint, index of cluster assigned to the point]
        Z[i, row_of_assigned_cluster] = 1

    assert len(np.unique(Z)) == 2 and np.min(Z) == 0 and np.max(Z) == 1, 'Z should be a matrix of zeros and ones'
    assert np.all(np.sum(Z, axis=1) == np.ones(Z.shape[0])), 'Each data point should be assigned to exactly 1 cluster'
    
    return Z


def recompute_centroids(X: np.ndarray, K: int, Z: np.ndarray) -> np.ndarray:
    """
    :param X: data for clustering, shape: (N, D), with N - number of data points, D - dimension
    :param K: number of clusters
    :param Z: indicator variables for all data points, shape: (N, K)
    :return: centroids - means of clusters, shape: (K, D)
    """

    D = X.shape[1]
    # TODO: Recompute centroids
    # At first filling centroid matrix with zeroes. K - amount of clusters, D - dimensions
    centroids = np.zeros((K, D))
    
    for k in range(K):
        points_assigned_to_cluster = []
        for i in range(X.shape[0]):
            if(Z[i][k] == 1):
                points_assigned_to_cluster.append(X[i])
        centroids[k] = np.mean(points_assigned_to_cluster, axis=0)
    
    return centroids


def kmeans(X: np.ndarray, K: int, max_iter: int, eps=1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param X: data for clustering, shape: (N, D), with N - number of data points, D - dimension
    :param K: number of clusters
    :param max_iter: maximum number of iterations for the K-means algorithm.
                     If the algorithm converges earlier, it should stop.
    :return: Z - indicator variables for all data points, shape: (N, K)
             centroids - means of clusters, shape: (K, D)
             wcss_list - list with values of the objective function J over iteration
    """

    N, D = X.shape

    # Init centroids
    rnd_points = np.random.choice(np.arange(N), size=K, replace=False)
    centroids = X[rnd_points, :]
    assert centroids.shape[0] == K and centroids.shape[1] == D
    print(f'Init: {centroids=}')

    wcss_list = []
    for it in range(max_iter):
        # Assign samples to the clusters (compute Z)
        Z = compute_Z(X, K, centroids) # TODO: function call to assign samples to clusters
        loss = wcss(X, K, Z, centroids) # TODO: function call to calculate WCSS
        wcss_list.append(loss)

        # Calculate new centroids from the clusters
        centroids = recompute_centroids(X, K, Z) # TODO: function call to recompute centroids
        loss = wcss(X, K, Z, centroids) # TODO: function call to calculate WCSS (again)
        wcss_list.append(loss)

        if it > 0 and np.abs(wcss_list[-1] - wcss_list[-2]) < eps:
            print(f'Algorithm converged at iteration {it}.')
            break

    print(f'Fitted parameters: {centroids=}')
    return Z, centroids, np.array(wcss_list)
