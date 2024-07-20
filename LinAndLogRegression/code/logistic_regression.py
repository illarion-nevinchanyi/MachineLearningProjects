import numpy as np


def create_design_matrix_dataset_1(X_data: np.ndarray) -> np.ndarray:
    """
    Create the design matrix X for dataset 1.
    :param X_data: 2D numpy array with the data points
    :return: Design matrix X
    """
    # TODO: Create the design matrix X for dataset 1
    X = []
    x1 = X_data[:, 0]
    x2 = X_data[:, 1]
    x3 = []
    x4 = []
    # x2 = []
    for (index, elements) in enumerate(X_data):
        if X_data[index][0] >= 10 and X_data[index][1] <= 20:
            x3.append(1)
        else:
            x3.append(0)
    X = np.column_stack((x1, x2, x3))
    # X = np.array(X)
    assert X.shape[0] == X_data.shape[0], """The number of rows in the design matrix X should be the same as
                                             the number of data points."""
    assert X.shape[1] >= 2, "The design matrix X should have at least two columns (the original features)."
    return X


def create_design_matrix_dataset_2(X_data: np.ndarray) -> np.ndarray:
    """
    Create the design matrix X for dataset 2.
    :param X_data: 2D numpy array with the data points
    :return: Design matrix X
    """
    # TODO: Create the design matrix X for dataset 2
    X = []
    x1 = X_data[:, 0]
    x2 = X_data[:, 1]
    x3 = []
    x4 = []
    for (index, elements) in enumerate(X_data):
        x3.append(x2[index] ** 2)
        x4.append(x1[index] ** 2)
    X = np.column_stack((x1, x2, x3, x4))

    assert X.shape[0] == X_data.shape[0], """The number of rows in the design matrix X should be the same as
                                             the number of data points."""
    assert X.shape[1] >= 2, "The design matrix X should have at least two columns (the original features)."
    return X


def create_design_matrix_dataset_3(X_data: np.ndarray) -> np.ndarray:
    """
    Create the design matrix X for dataset 3.
    :param X_data: 2D numpy array with the data points
    :return: Design matrix X
    """
    # TODO: Create the design matrix X for dataset 3
    X = []
    x1 = X_data[:, 0]
    x2 = X_data[:, 1]
    x3 = []
    polynomial_function_features = [(X_data ** degree) for degree in range(1, 11)]
    X = np.hstack((X_data, np.column_stack(polynomial_function_features)))

    assert X.shape[0] == X_data.shape[0], """The number of rows in the design matrix X should be the same as
                                             the number of data points."""
    assert X.shape[1] >= 2, "The design matrix X should have at least two columns (the original features)."
    return X


def logistic_regression_params_sklearn():
    """
    :return: Return a dictionary with the parameters to be used in the LogisticRegression model from sklearn.
    Read the docs at https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """
    # TODO: Try different `penalty` parameters for the LogisticRegression model
    return {'penalty': None, 'dual': False, 'tol': 1e-4, 'C': 1.0, 'fit_intercept': True, 'intercept_scaling': 1,
            'class_weight': None, 'random_state': None, 'solver': 'lbfgs', 'max_iter': 50000, 'multi_class': 'auto',
            'verbose': 0, 'warm_start': False, 'n_jobs': None, 'l1_ratio': None}
