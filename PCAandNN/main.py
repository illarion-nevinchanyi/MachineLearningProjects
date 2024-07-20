import numpy as np
from sklearn.model_selection import train_test_split
from nn_classification_from_scratch import train_nn_own
from nn_classification_sklearn import reduce_dimension, train_nn, train_nn_with_regularization, \
    perform_grid_search, show_confusion_matrix_and_classification_report, plot_training_loss_curve

def task_1():
    # Load the 'data/sign_language_images.npy' and 'data/sign_language_targets.npy' using np.load.
    features = np.load('data/sign_language_images.npy')
    targets = np.load('data/sign_language_targets.npy')
    features = features.reshape((features.shape[0], -1))
    print(features.shape, targets.shape)

    X_train, X_test, y_train, y_test = train_test_split(features, targets,
                                                        test_size=0.2, random_state=42)
    print(features.shape)

    # PCA
    # Task 1.1.1
    print("----- Task 1.1.1 -----")
    # TODO: Choose the number of components such that 95% of the variance is retained
    n_components = 303  # sum(pca.explained_variance_ratio_): explained variance = 0.9500811803591205
    X_train_pca, pca = reduce_dimension(X_train, n_components)
    print('X_train_pca.shape: (n_samples, n_components) = {}'.format(X_train_pca.shape))

    # Task 1.1.2
    print("----- Task 1.1.2 -----")
    best_nn = train_nn(X_train_pca, y_train)
    best_nn.fit(X_train_pca, y_train)

    # Task 1.1.3 needs no code, only explanations in the report

    # Task 1.1.4
    print("----- Task 1.1.4 -----")
    best_reg_nn = train_nn_with_regularization(X_train_pca, y_train)
    best_reg_nn.fit(X_train_pca, y_train)

    # TODO: Choose the best model from the previous Tasks
    best_model_task_1_1 = best_reg_nn
    plot_training_loss_curve(best_model_task_1_1)

    # Task 1.2
    print("----- Task 1.2 -----")
    best_gs_nn = perform_grid_search(X_train_pca, y_train)
    best_gs_nn.fit(X_train_pca, y_train)

    # TODO: Choose the best model from *all* previous Tasks
    final_model = best_gs_nn

    X_test_pca = pca.transform(X_test)
    show_confusion_matrix_and_classification_report(final_model, X_test_pca, y_test)

def task_2():
    features = np.load('data/sign_language_images.npy')
    targets = np.load('data/sign_language_targets.npy')
    features = features.reshape((features.shape[0], -1))

    X_train, X_test, y_train, y_test = train_test_split(features, targets,
                                                        test_size=0.2, random_state=42)
    X_train_pca, pca = reduce_dimension(X_train, n_components=16) # Let's project the features to 16 dimensions

    nn = train_nn_own(X_train_pca, y_train)

    X_test_pca = pca.transform(X_test)
    test_acc = nn.score(X_test_pca, y_test)
    print(f'Test accuracy: {test_acc:.4f}.')


def task_3_bonus():
    features = np.load('data/sign_language_images.npy')
    targets = np.load('data/sign_language_targets.npy')
    features = features.reshape((features.shape[0], -1))

    # Create a binary classification dataset out of the multi-class dataset
    # (by keeping only the samples with targets 0 and 1)
    idxs = np.where(targets < 2)[0]
    features = features[idxs]
    targets = targets[idxs]

    X_train, X_test, y_train, y_test = train_test_split(features, targets,
                                                        test_size=0.2, random_state=42)
    X_train_pca, pca = reduce_dimension(X_train, n_components=16) # Let's project the features to 16 dimensions

    nn = train_nn_own(X_train_pca, y_train)

    X_test_pca = pca.transform(X_test)
    test_acc = nn.score(X_test_pca, y_test)
    print(f'Test accuracy: {test_acc:.4f}.')


def main():
    task_1()
    task_2()
    task_3_bonus()


if __name__ == '__main__':
    main()
