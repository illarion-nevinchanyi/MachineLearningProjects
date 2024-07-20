import numpy as np
from k_means import kmeans
from em import em
from utils import load_data, plot_original_data, plot_mickey_mouse, plot_objective_function


def task_kmeans(X):
    K = 3 # TODO: Choose the number of clusters
    max_iter = 300 # TODO: Choose the maximum number of iterations
    ind_samples_clusters, centroids, J = kmeans(X, K, max_iter)

    plot_objective_function(J)
    plot_mickey_mouse(X, K, ind_samples_clusters, centroids, cluster_method='K-means')


def task_em(X):
    K = 3 # TODO: Choose the number of clusters
    max_iter = 300 # TODO: Choose the maximum number of iterations
    means, soft_clusters, log_likelihood = em(X, K, max_iter)

    plot_objective_function(log_likelihood, ylabel='Log-likelihood')
    plot_mickey_mouse(X, K, soft_clusters, means, cluster_method='EM')


def main():
    x, y, data_labels = load_data(filename='data/mouse.txt')
    plot_original_data(x, y, data_labels)

    X_mouse = np.array([x, y]).T
    print('X_mouse shape:', X_mouse.shape)

    # ----- Task K-Means -----
    print('--- Task K-Means ---')
    task_kmeans(X_mouse)

    # ----- Task EM -----
    print('--- Task EM ---')
    task_em(X_mouse)


if __name__ == '__main__':
    main()
