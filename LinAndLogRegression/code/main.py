# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from plot_utils import plot_model1, plot_model2, plot_logistic_regression, plot_datapoints, plot_function
from lin_reg_memristors import (model_to_use_for_fault_classification, fit_zero_intercept_lin_model,
                                fit_lin_model_with_intercept, bonus_fit_lin_model_with_intercept_using_pinv,
                                classify_memristor_fault_with_model1, classify_memristor_fault_with_model2)
from gradient_descent import ackley, gradient_ackley, gradient_descent
from logistic_regression import (create_design_matrix_dataset_1,
                                 create_design_matrix_dataset_2,
                                 create_design_matrix_dataset_3,
                                 logistic_regression_params_sklearn)


def task_1():
    print('---- Task 1.1 ----')

    # Load the data
    data = np.load('data/memristor_measurements.npy')
    print(data.shape)

    n_memristor = data.shape[0]

    ### --- Use Model 1 (zero-intercept lin. model, that is, fit the model using fit_zero_intercept_lin_model)
    estimated_params_per_memristor_model1 = np.zeros(n_memristor)
    for i in range(n_memristor):
        x, y = data[i][:, 0], data[i][:, 1]  # TODO: extract the corresponding entries from the data array
        theta = fit_zero_intercept_lin_model(x, y)  # TODO: call the appropriate function with x and y
        estimated_params_per_memristor_model1[i] = theta

    # Visualize the data and the best fit for each memristor
    plot_model1(data, estimated_params_per_memristor_model1)

    print('\nModel 1 (zero-intercept linear model).')
    print(f'Estimated theta per memristor: {estimated_params_per_memristor_model1}')

    ### --- Use Model 2 (lin. model with intercept, that is, fit the model using fit_lin_model_with_intercept)
    estimated_params_per_memristor_model2 = np.zeros((n_memristor, 2))
    for i in range(n_memristor):
        x, y = data[i][:, 0], data[i][:, 1]  # TODO: extract the corresponding entries from the data array
        # theta0, theta1 = fit_lin_model_with_intercept(x, y)  # TODO: call the appropriate function with x and y
        theta0, theta1 = bonus_fit_lin_model_with_intercept_using_pinv(x, y)
        # TODO: If you have implemented the bonus task, you can call the bonus function instead to check
        #  if the results are the same
        estimated_params_per_memristor_model2[i, :] = [theta0, theta1]

    # Visualize the data and the best fit for each memristor
    plot_model2(data, estimated_params_per_memristor_model2)

    print('\nModel 2 (linear model with intercept).')
    print(f"Estimated params (theta_0, theta_1) per memristor: {estimated_params_per_memristor_model2}")

    fault_types = []
    # # TODO: Use either Model 1 or Model 2 for the decision on memristor fault type.
    # #  Your choice should be reflected in the `model_to_use_for_fault_classification` function.
    error_comparison_model_1 = 0
    error_comparison_model_2 = 0
    for i in range(n_memristor):
        error_1 = 0.0
        error_2 = 0.0
        x, y = data[i][:, 0], data[i][:, 1]
        theta = estimated_params_per_memristor_model1[i]
        theta0 = estimated_params_per_memristor_model2[i][0]
        theta1 = estimated_params_per_memristor_model2[i][1]
        for delta_R_ideal, delta_R in zip(x, y):
            # Compute the error for the current measurement
            error_2 += (((theta0 + theta1 * delta_R_ideal) - delta_R) ** 2)
            error_1 += (((theta * delta_R_ideal) - delta_R) ** 2)
        error_1 *= 0.1
        error_2 *= 0.1
        print(f"Mean Error for Model 1, Memristor {i}: {error_1}")
        print(f"Mean Error for Model 2, Memristor {i}: {error_2}")
        if error_1 > error_2:
            error_comparison_model_2 += 1
        else:
            error_comparison_model_1 += 1
        error_2 = 0
        error_1 = 0
    if error_comparison_model_1 > error_comparison_model_2:
        print(
            f"Model 1 is better model because values of model 1's mean error is lesser then values of model 2's mean error")
    else:
        print(
            f"Model 2 is better model because values of model 1's mean error is greater then values of model 2's mean error")
        # Compute the error for the current measurement

    model_to_use = model_to_use_for_fault_classification()
    for i in range(n_memristor):
        if model_to_use == 1:
            fault_type = classify_memristor_fault_with_model1(estimated_params_per_memristor_model1[i])
        elif model_to_use == 2:
            fault_type = classify_memristor_fault_with_model2(estimated_params_per_memristor_model2[i, 0],
                                                              estimated_params_per_memristor_model2[i, 1])
        else:
            raise ValueError('Please choose either Model 1 or Model 2 for the decision on memristor fault type.')

        fault_types.append(fault_type)

    print(f'\nClassifications (based on Model {model_to_use})')
    for i, fault_type in enumerate(fault_types):
        print(f'Memristor {i + 1} is classified as {fault_type}.')


def task_2():
    print('\n---- Task 2 ----')

    for task in [1, 2, 3]:
        print(f'---- Logistic regression task {task} ----')
        if task == 1:
            # TODO: Load the data set 1 (X-1-data.npy and targets-dataset-1.npy)
            X_data = np.load('data/X-1-data.npy')  # TODO: change me
            y = np.load('data/targets-dataset-1.npy')  # TODO: change me
            create_design_matrix = create_design_matrix_dataset_1
        elif task == 2:
            # TODO: Load the data set 2 (X-1-data.npy and targets-dataset-2.npy)
            X_data = np.load('data/X-1-data.npy')  # TODO: change me
            y = np.load('data/targets-dataset-2.npy')  # TODO: change me
            create_design_matrix = create_design_matrix_dataset_2
        elif task == 3:
            # Load the data set 3 (X-2-data.npy and targets-dataset-3.npy)
            X_data = np.load('data/X-2-data.npy')  # TODO: change me
            y = np.load('data/targets-dataset-3.npy')  # TODO: change me
            create_design_matrix = create_design_matrix_dataset_3
        else:
            raise ValueError('Task not found.')

        X = create_design_matrix(X_data)

        # Plot the datapoints (just for visual inspection)
        plot_datapoints(X, y, f'Targets - Task {task}')

        # TODO: Split the dataset using the `train_test_split` function.
        #  The parameter `random_state` should be set to 0.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=0)
        print(
            f'Shapes of: X_train {X_train.shape}, X_test {X_test.shape}, y_train {y_train.shape}, y_test {y_test.shape}')

        # Train the classifier
        custom_params = logistic_regression_params_sklearn()
        clf = LogisticRegression(**custom_params)
        # TODO: Fit the model to the data using the `fit` method of the classifier `clf`
        clf.fit(X_train, y_train)
        clf.fit(X_test, y_test)
        acc_train, acc_test = clf.score(X_train, y_train), clf.score(X_test,
                                                                     y_test)  # TODO: Use the `score` method of the classifier `clf` to calculate accuracy

        print(f'Train accuracy: {acc_train * 100:.2f}%. Test accuracy: {100 * acc_test:.2f}%.')

        yhat_train = clf.predict_proba(X_train)  # TODO: Use the `predict_proba` method of the classifier `clf` to
        #  calculate the predicted probabilities on the training set
        yhat_test = clf.predict_proba(X_test)  # TODO: Use the `predict_proba` method of the classifier `clf` to
        #  calculate the predicted probabilities on the test set

        # TODO: Use the `log_loss` function to calculate the cross-entropy loss
        #  (once on the training set, once on the test set).
        #  You need to pass (1) the true binary labels and (2) the probability of the *positive* class to `log_loss`.
        #  Since the output of `predict_proba` is of shape (n_samples, n_classes), you need to select the probabilities
        #  of the positive class by indexing the second column (index 1).
        loss_train, loss_test = log_loss(y_train, yhat_train[:, 1]), log_loss(y_test, yhat_test[:, 1])
        print(f'Train loss: {loss_train}. Test loss: {loss_test}.')

        plot_logistic_regression(clf, create_design_matrix, X_train, f'(Dataset {task}) Train set predictions',
                                 figname=f'logreg_train{task}')
        plot_logistic_regression(clf, create_design_matrix, X_test, f'(Dataset {task}) Test set predictions',
                                 figname=f'logreg_test{task}')

        # TODO: Print theta vector (and also the bias term). Hint: Check the attributes of the classifier
        classifier_weights, classifier_bias = clf.coef_, clf.intercept_
        print(f'Parameters: {classifier_weights}, {classifier_bias}')


def task_3():
    print('\n---- Task 3 ----')

    # Plot the Function, to see how it looks like
    plot_function(ackley)

    # TODO: Choose a random starting point using samples from a standard normal distribution
    #np.random.seed(42)                      # Set numpy's random seed to a constant
    x0 = np.random.normal(loc=0, scale=1)   # loc is eq. to mu (mean)
    y0 = np.random.normal(loc=0, scale=1)   # scale is eq. to variance

    print(f'{x0:.4f}, {y0:.4f}')

    # TODO: Call the function `gradient_descent` with a chosen configuration of hyperparameters,
    #  i.e., learning_rate, lr_decay, and num_iters. Try out lr_decay=1 as well as values for lr_decay that are < 1.

    x, y, f_list = gradient_descent(ackley, gradient_ackley, x0, y0, 0.2, 0.6, 18)
    #0.2 06 18
    #0.4 0.7 50
    # Print the point that is found after `max_iter` solution
    print(f'{x:.4f}, {y:.4f}')

    # TODO: Use `f_list` to create a plot of the function over iteration.
    #  Do not forget to label the plot (xlabel, ylabel, title).
    def plot_gradient_descent():
        '''
        This function plots the array values over the number of values in the array.
        '''
        fig, ax = plt.subplots(figsize=(18, 12))
        x = np.linspace(0, len(f_list) - 1, len(f_list))
        ax.plot(x, f_list, color='darkcyan')
        ax.tick_params(axis='x', labelsize=25)
        ax.tick_params(axis='y', labelsize=25)
        ax.set_xlabel('Number of Iterations', fontsize=25)
        ax.set_ylabel(r'$f(x, y)$', fontsize=25)
        ax.set_title('Cost Change over iterations', size=35)
        plt.savefig('gradient_descent_ackley.png')
        plt.show()
    plot_gradient_descent()

    print(f'Solution found: f({x:.4f}, {y:.4f})= {ackley(x, y):.4f}')
    print(f'Global optimum: f(0, 0)= {ackley(0, 0):.4f}')

def main():
    np.random.seed(33761)

    task_1()
    task_2()
    task_3()

if __name__ == '__main__':
    main()