from typing import Tuple
import numpy as np
import sklearn.metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import warnings
# We will suppress ConvergenceWarnings in this task. In practice, you should take warnings more seriously.
warnings.filterwarnings("ignore")

def reduce_dimension(X_train: np.ndarray, n_components: int) -> Tuple[np.ndarray, PCA]:
    """
    :param X_train: Training data to reduce the dimensionality. Shape: (n_samples, n_features)
    :param n_components: Number of principal components
    :return: Data with reduced dimensionality, which has shape (n_samples, n_components), and the PCA object
    """

    # TODO: Create a PCA object and fit it using X_train
    #       Transform X_train using the PCA object.
    #       Print the explained variance ratio of the PCA object.
    #       Return both the transformed data and the PCA object.
    #

    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    print('Explained variance ratio of the PCA object: {}'.format(np.sum(pca.explained_variance_ratio_)))
    return X_train_pca, pca

def train_nn(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Train MLPClassifier with different number of neurons in one hidden layer.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The MLPClassifier you consider to be the best
    """

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=42)

    # TODO: Train MLPClassifier with different number of neurons in one hidden layer.
    #       Print the train accuracy, validation accuracy, and the training loss for each configuration.
    #       Return the MLPClassifier that you consider to be the best.

    print('n_hidden  | train accuracy       | validation accuracy  | training loss')
    hiddenNeuronsList = [2, 10, 100, 200, 500]

    for n_hidden in hiddenNeuronsList:
        mlp = MLPClassifier(hidden_layer_sizes=(n_hidden,), solver='adam',
                            max_iter=500, random_state=1).fit(X_train, y_train)

        pred_train = mlp.predict(X_train)
        pred_val = mlp.predict(X_val)

        train_acc = sklearn.metrics.accuracy_score(y_train, pred_train)
        val_acc = sklearn.metrics.accuracy_score(y_val, pred_val)

        training_loss = mlp.loss_

        print('{:>3}       | {:>20} | {:>20} | {:>20}'.format(n_hidden, train_acc, val_acc, training_loss))

    return MLPClassifier(hidden_layer_sizes=(100,), solver='adam', max_iter=500, random_state=1)


def train_nn_with_regularization(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Train MLPClassifier using regularization.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The MLPClassifier you consider to be the best
    """
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=42)

    # TODO: Use the code from the `train_nn` function, but add regularization to the MLPClassifier.
    #       Again, return the MLPClassifier that you consider to be the best.

    hiddenNeuronsList = [2, 10, 100, 200, 500]

    for alpha, bool_state in zip([0.1, 0.0001, 0.1], [False, True, True]):
        print('\nalpha  | early_stopping')
        print('{:>6} | {} \n'.format(alpha, bool_state))
        print('n_hidden  | train accuracy       | validation accuracy  | training loss')
        for n_hidden in hiddenNeuronsList:
            mlp = MLPClassifier(hidden_layer_sizes=(n_hidden,), solver='adam', alpha=alpha, # Default: alpha=0.0001
                                max_iter=500, random_state=1, early_stopping=bool_state).fit(X_train, y_train)

            pred_train = mlp.predict(X_train)
            pred_val = mlp.predict(X_val)

            train_acc = sklearn.metrics.accuracy_score(y_train, pred_train)
            val_acc = sklearn.metrics.accuracy_score(y_val, pred_val)

            training_loss = mlp.loss_

            print('{:>3}       | {:>20} | {:>20} | {:>20}'.format(n_hidden, train_acc, val_acc, training_loss))

    return MLPClassifier(hidden_layer_sizes=(200,), solver='adam', alpha=0.1,
                         max_iter=500, random_state=1, early_stopping=False)


def plot_training_loss_curve(nn: MLPClassifier) -> None:
    """
    Plot the training loss curve.

    :param nn: The trained MLPClassifier
    """
    # TODO: Plot the training loss curve of the MLPClassifier. Don't forget to label the axes.
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.plot(nn.loss_curve_, color='darkcyan', label='Loss with n_hidden = {}'.format(nn.hidden_layer_sizes))

    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)

    ax.set_xlabel('Number of Iterations', fontsize=25)
    ax.set_ylabel('Loss', fontsize=25)
    ax.set_title('Training Loss Over Iterations', size=35)

    plt.savefig('task_115_training_loss_curve.pdf')
    plt.show()


def show_confusion_matrix_and_classification_report(nn: MLPClassifier, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Plot confusion matrix and print classification report.

    :param nn: The trained MLPClassifier you want to evaluate
    :param X_test: Test features (PCA-projected)
    :param y_test: Test targets
    """
    # TODO: Use      to compute predictions onimport num
    #       Usert Tuple

    pred_test = nn.predict(X_test)
    conf_matrix = sklearn.metrics.confusion_matrix(y_test, pred_test)
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    report = classification_report(y_test, pred_test)

    display.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')

    plt.savefig('task_124_confusion_matrix.pdf')
    plt.show()

    with open('classification_report.txt', 'w') as f:
        f.write(report)

    print('Classification report: \n', report)

def perform_grid_search(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Perform GridSearch using GridSearchCV.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The best estimator (MLPClassifier) found by GridSearchCV
    """
    # TODO: Create parameter dictionary for GridSearchCV, as specified in the assignment sheet.
    #       Create an MLPClassifier with the specified default values.
    #       Run the grid search withiance raand (optionally)rt Tuple
    #       Print the best score (mean cross validation score) and the best parameter set.
    #       Return the best estimator found by GridSearchCV.

    param_grid = {
        'alpha': [0.0, 0.1, 1.0],
        'solver': ['lbfgs', 'adam'],
        'hidden_layer_sizes': [100, 200]
    }
    mlp = MLPClassifier(max_iter=100, random_state=42)
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, verbose=4).fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_

    print("Best Parameters:", best_params)
    print("Best Cross-validation Score:", best_score)

    return best_model