import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC
from utils.plotting import plot_dataset, plot_decision_boundary
from utils.datasets import get_toy_dataset
from knn import KNearestNeighborsClassifier
from svm import LinearSVM
from sklearn.ensemble import RandomForestClassifier


def grid_search_knn_and_plot_decision_boundary(X_train, y_train, X_test, y_test, dataset_name):
    knn = KNearestNeighborsClassifier()

    # TODO: Use the `GridSearchCV` meta-classifier and search over different values of `k`
    #       Include the `return_train_score=True` option to get the training accuracies
    param_grid = {'k': list(range(1, 101))}
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, return_train_score=True)
    grid_search.fit(X_train, y_train)

    # determine the best value of k, fit the clf and calculate the accuracy
    best_k_param = grid_search.best_params_['k']        # our chosen value of k
    knn = KNearestNeighborsClassifier(k=best_k_param)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)

    print('{:>9} | {:>6} | {}'.format(dataset_name, best_k_param, acc))

    # this plots the decision boundary
    plt.figure()
    plot_decision_boundary(X_train, grid_search)
    plot_dataset(X_train, y_train, X_test, y_test)
    plt.title(f"Decision boundary for dataset {dataset_name}\nwith k={grid_search.best_params_['k']}")

    # TODO you should use the plt.savefig(...) function to store your plots before calling plt.show()
    plt.savefig('Decision_bound_dataset_{}_k_{}.pdf'.format(dataset_name, grid_search.best_params_['k']))
    plt.show()

    # TODO: Create a plot that shows the mean training and validation scores (y axis)
    #       for each k \in {1,...,100} (x axis).
    #       Hint: Check the `cv_results_` attribute of the `GridSearchCV` object

    # estimate and plot the means
    cv_results = grid_search.cv_results_

    fig, ax = plt.subplots(figsize=(18, 12))

    ax.plot(list(range(1, 101)), cv_results['mean_train_score'], color='darkcyan', label='')
    ax.plot(list(range(1, 101)), cv_results['mean_test_score'], color='rebeccapurple', label='')

    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)

    ax.set_xlabel('Number of Neighbors (k)', fontsize=25)
    ax.set_ylabel('Accuracy', fontsize=25)
    ax.set_title('Mean Training and Validation Scores for KNN ' +
                 '\nDataset {} with k = {}'.format(dataset_name, grid_search.best_params_['k']), size=35)

    plt.savefig('mean_train_val_scores_knn_dataset_{}.pdf'.format(dataset_name))
    plt.show()

def task1_2():
    print('-' * 10, 'Task 1.2', '-' * 10)
    print('Dataset   | best k | Accuracy')
    for idx in [1, 2, 3]:
        X_train, X_test, y_train, y_test = get_toy_dataset(idx)
        grid_search_knn_and_plot_decision_boundary(X_train, y_train, X_test, y_test, dataset_name=idx)


def task1_4():
    print('-' * 10, 'Task 1.4', '-' * 10)
    dataset_name = '2 (noisy)'
    X_train, X_test, y_train, y_test = get_toy_dataset(2, apply_noise=True)
    print('Dataset   | k   | Accuracy')
    for k in [1, 30, 100]:
        # TODO: Fit your KNearestNeighborsClassifier with k in {1, 30, 100} and plot the decision boundaries.
        #       You can use the `cross_val_score` method to manually perform cross-validation.
        #       Report the mean cross-validated scores.
        knn = KNearestNeighborsClassifier(k=k)
        knn.fit(X_train, y_train)
        acc = knn.score(X_test, y_test)

        # estimate and plot the means
        cv_results_train = cross_val_score(estimator=knn, X=X_train, y=y_train, cv=5)
        cv_results_test = cross_val_score(estimator=knn, X=X_test, y=y_test, cv=5)

        print('{:>9} | {:>3} | {}'.format(dataset_name, k, acc))

        # This plots the decision boundaries without the test set
        # (we simply don't pass the test sets to `plot_dataset`).
        plt.figure()
        plt.title(f"Decision boundary for dataset {dataset_name}\nwith k={k}")
        plot_decision_boundary(X_train, knn)
        plot_dataset(X_train, y_train)
        plt.show()

        fig, ax = plt.subplots(figsize=(18, 12))

        ax.plot(range(1, len(cv_results_train) + 1), cv_results_train, color='darkcyan', label='Train Set')
        ax.plot(range(1, len(cv_results_test) + 1), cv_results_test, color='darkcyan', label='Test Set')

        ax.tick_params(axis='x', labelsize=25)
        ax.tick_params(axis='y', labelsize=25)

        ax.set_xlabel('Run of the cross validation', fontsize=25)
        ax.set_ylabel('Accuracy', fontsize=25)
        ax.set_title('Mean Training and Validation Scores for KNN ' +
                     '\nDataset {} with k = {}'.format(dataset_name, k), size=35)

        plt.savefig('cross_val_scores_dataset_{}_with_k_{}.pdf'.format(dataset_name, k))
        plt.legend(fontsize=15)
        plt.show()


    # This should find the best parameters for the noisy dataset.
    print('Dataset   | best k | Accuracy')
    grid_search_knn_and_plot_decision_boundary(X_train, y_train, X_test, y_test, dataset_name=dataset_name)


def task2_2():
    print('-' * 10, 'Task 2.2', '-' * 10)

    X_train, X_test, y_train, y_test = get_toy_dataset(1, remove_outlier=True)
    svm = LinearSVM()
    # TODO: Use grid search to find suitable parameters.
    param_grid = {'C': 10.**np.arange(-3, 3),    # e.g. [0.01, 0.1, 0.0, 1.0, 10.0, 100.0],
                  'eta': 10.**np.arange(-4, 0)   # e.g. [0.1, 0.01, 0.001, 0.0001]
                  }

    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    print(f'best_params: {grid_search.best_params_}')
    print(f'best_score: {grid_search.best_score_}')
    print(f'best_estimator: {grid_search.best_estimator_}')

    best_C = grid_search.best_params_['C']
    best_eta = grid_search.best_params_['eta']

    # TODO: Use the parameters you have found to instantiate a LinearSVM.
    #       The `fit` method returns a list of scores that you should plot in order to monitor the convergence.
    svm = LinearSVM(C=best_C, eta=best_eta)
    svm.fit(X_train, y_train)
    loss = svm.fit(X_train, y_train)
    print(loss)

    # This plots the decision boundary
    plt.figure()
    plot_dataset(X_train, y_train, X_test, y_test)
    plot_decision_boundary(X_train, svm)
    plt.title(f"[SVM] Decision boundary for dataset {1}\nwith params={grid_search.best_params_}")
    plt.savefig(f'SVM_decision_boundary_dataset_{1}_param_{grid_search.best_params_}.pdf')
    plt.show()

    fig, ax = plt.subplots(figsize=(18, 12))
    ax.plot(np.arange(0, len(loss)), loss, color='darkcyan')

    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)

    ax.set_xlabel('Number of Iterations', fontsize=25)
    ax.set_ylabel('Loss', fontsize=25)
    ax.set_title('[SVM] Training Loss Curve', size=35)

    plt.savefig('SVM_loss_curve.pdf')
    plt.show()


def task2_3():
    print('-' * 10, 'Task 2.3', '-' * 10)
    print('Dataset | Best Parameters             | Mean CV Accuracy')
    for idx in [1, 2, 3]:
        X_train, X_test, y_train, y_test = get_toy_dataset(idx)
        # svc = SVC(tol=1e-4)
        # TODO: Perform grid search, decide on suitable parameter ranges
        #       and state sensible parameter ranges in your report
        param_grid = {
            'linear': {'C': 10.**np.arange(-2, 4)},
            'rbf': {'C': 10.**np.arange(-2, 4), 'gamma': 10.**np.arange(-2, 4)}
        }
        for kernel, param_grid in param_grid.items():
            svc = SVC(kernel=kernel, tol=1e-4)
            grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5)
            grid_search.fit(X_train, y_train)

            print('{:>7} | {:>27} | {}'.format(idx, str(grid_search.best_params_), grid_search.best_score_))

            # TODO: Using the best parameter settings, report the score on the test dataset (X_test, y_test)

            # This plots the decision boundary
            plt.figure()
            plt.title(f"[SVM] Decision boundary for dataset {idx}\nwith params={grid_search.best_params_}")
            plot_dataset(X_train, y_train, X_test, y_test)
            plot_decision_boundary(X_train, grid_search)
            plt.savefig(f'dataset_{idx}_params_{grid_search.best_params_}.pdf')
            plt.show()

            if kernel == 'linear':
                svc = SVC(kernel=kernel, tol=1e-4, C=grid_search.best_params_['C'])
            else:
                kernel = 'rbf'
                svc = SVC(kernel=kernel, tol=1e-4, C=grid_search.best_params_['C'],
                          gamma=grid_search.best_params_['gamma'])

            svc.fit(X_train, y_train)
            acc = svc.score(X_test, y_test)
            print(f'Test Set Accuracy: {acc} \n')


def task3_1():
    print('-' * 10, 'Task 3.1', '-' * 10)
    n_estimators_list = [1, 100]
    max_depth_list = np.arange(1, 26)
    print(f'Dataset | n_estimators | Best max_depth | Best score')
    for idx in [1, 2, 3]:
        X_train, X_test, y_train, y_test = get_toy_dataset(idx)
        cv_val_accuracy = {}
        cv_train_accuracy = {}
        for n_estimators in n_estimators_list:
            # TODO: Instantiate a RandomForestClassifier with n_estimators and random_state=0
            #       and use GridSearchCV over max_depth_list to find the best max_depth.
            #       Use `return_train_score=True` to get the training accuracies during CV.
            rfc = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
            param_grid = {'max_depth': max_depth_list}
            grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, return_train_score=True)
            grid_search.fit(X_train, y_train)

            print('{:>7} | {:>12} | {:>14} | {}'.format(idx, n_estimators, grid_search.best_params_['max_depth'],
                                                        grid_search.best_score_))


            # TODO: Store `mean_test_score` and `mean_train_score` in cv_val_accuracy and cv_train_accuracy.
            #       The dictionary key should be the number of estimators.
            #       Hint: Check the `cv_results_` attribute of the `GridSearchCV` object

            mean_test_score = grid_search.cv_results_['mean_test_score']
            mean_train_score = grid_search.cv_results_['mean_train_score']

            cv_val_accuracy[n_estimators] = mean_test_score
            cv_train_accuracy[n_estimators] = mean_train_score

            # This plots the decision boundary with just the training dataset
            plt.figure()
            plot_decision_boundary(X_train, grid_search)
            plot_dataset(X_train, y_train)
            plt.title(f"Decision boundary for dataset {idx}\n"
                      f"n_estimators={n_estimators}, max_depth={grid_search.best_params_['max_depth']}")
            plt.savefig('[RFC]_decision_boundary_dataset_{}_{}_{}.pdf'.format(idx, n_estimators,
                                                                              grid_search.best_params_['max_depth']))
            plt.show()

        # TODO: Create a plot that shows the mean training and validation scores (y axis)
        #       for each max_depth in max_depth_list (x axis).
        #       Use different colors for each n_estimators and linestyle="--" for validation scores.

        fig, ax = plt.subplots(figsize=(14, 7))

        for n_estimators in n_estimators_list:
            ax.plot(max_depth_list, cv_train_accuracy[n_estimators],
                    label=f'Train Accuracy (n_estimators={n_estimators})')

            ax.plot(max_depth_list, cv_val_accuracy[n_estimators], linestyle='--',
                    label=f'Validation Accuracy (n_estimators={n_estimators})')

        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        ax.set_xlabel('Max Depth', fontsize=15)
        ax.set_ylabel('Accuracy', fontsize=15)
        ax.set_title('Training and Validation Accuracy over Max Depth (Dataset {})'.format(idx), size=20)
        ax.legend(fontsize=15)
        ax.grid(True)

        plt.savefig('train_val_acc_max_depth_dataset_{}.pdf'.format(idx))
        plt.show()


    # TODO: Instantiate a RandomForestClassifier with the best parameters for each dataset and
    #       report the test scores (using X_test, y_test) for each dataset.

    best_param = {1: {'n_estimators': [100], 'max_depth': [2]},
                  2: {'n_estimators': [100], 'max_depth': [11]},
                  3: {'n_estimators': [100], 'max_depth': [7]}
                  }

    print(f'Dataset | n_estimators | Best max_depth | Test Accuracy')
    for idx in [1, 2, 3]:
        X_train, X_test, y_train, y_test = get_toy_dataset(idx)
        for n_estimators, max_depth in zip(best_param[idx]['n_estimators'], best_param[idx]['max_depth']):
            rfc = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
            rfc.fit(X_train, y_train)
            acc = rfc.score(X_test, y_test)

            print('{:>7} | {:>12} | {:>14} | {}'.format(idx, n_estimators, max_depth, acc))



def task3_bonus():
    X_train, X_test, y_train, y_test = get_toy_dataset(4)

    # TODO: Find suitable parameters for an SVC and fit it.
    #       Report mean CV accuracy of the model you choose.

    # TODO: Fit a RandomForestClassifier with appropriate parameters.

    # TODO: Create a `barh` plot of the `feature_importances_` of the RF classifier.
    #       See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.barh.html

    # TODO: Use recursive feature elimination to automatically choose the best number of parameters.
    #       Set `scoring='accuracy'` to look for the feature subset with highest accuracy and fit the RFECV
    #       to the training dataset. You can pass the classifier from the previous step to RFECV.
    #       See https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html

    # TODO: Use the RFECV to transform the training dataset -- it automatically removes the least important
    #       feature columns from the datasets. You don't have to change y_train or y_test.
    #       Fit an SVC classifier with appropriate parameters on the new dataset and report the mean CV score.
    #       Do you see a difference in performance when compared to the previous dataset? Report your findings.

    # TODO: If the CV performance of this SVC is better, transform the test dataset as well and report the test score.
    #       If the performance is worse, report the test score of the previous SVC.


if __name__ == '__main__':
    # Task 1.1 consists of implementing the KNearestNeighborsClassifier class
    task1_2()
    # Task 1.3 does not need code to be answered
    task1_4()

    # Task 2.1 consists of a pen & paper exercise and the implementation of the LinearSVM class
    task2_2()
    task2_3()

    task3_1()
    # Task 3.2 is a theory question
    # task3_bonus()