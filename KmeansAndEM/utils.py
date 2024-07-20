import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip().split(' ') for x in content]

    lst_content = sum(content, [])
    x_str = lst_content[0::3]
    y_str = lst_content[1::3]
    data_labels = np.array(lst_content[2::3])

    x = np.array([float(i) for i in x_str])
    y = np.array([float(i) for i in y_str])
    return x, y, data_labels


def plot_original_data(x, y, data_labels):
    labels = ['Head', 'Ear_left', 'Ear_right', 'Noise']

    fig = plt.figure()
    fig.suptitle('Original data')
    ax = fig.add_subplot(111)

    ax.set(xlabel='x')
    ax.set(ylabel='y')

    for i in range(4):
        lbl = labels[i]
        x_ = x[np.where(data_labels == lbl)[0]]
        y_ = y[np.where(data_labels == lbl)[0]]

        ax.scatter(x_, y_)

    plt.show()


def plot_mickey_mouse(X, K, ind_samples_clusters, centroids, cluster_method):
    x, y = X[:, 0], X[:, 1]
    clusters = np.argmax(ind_samples_clusters, axis=1)

    fig = plt.figure()
    fig.suptitle(f'[{cluster_method}] Clustered data points')
    ax = fig.add_subplot(111)

    ax.set(xlabel='x')
    ax.set(ylabel='y')

    for i in range(K):
        x_ = x[np.where(clusters == i)[0]]
        y_ = y[np.where(clusters == i)[0]]

        ax.scatter(x_, y_)

    for i in range(K):
        ax.scatter(centroids[i, 0], centroids[i, 1], c='k', s=100)

    plt.show()


def plot_objective_function(objective_function_over_its, ylabel='Objective function'):
    plt.plot(objective_function_over_its)
    plt.title(f'{ylabel} vs. Iterations')
    plt.xlabel('Iterations')
    plt.ylabel(ylabel)
    plt.show()
