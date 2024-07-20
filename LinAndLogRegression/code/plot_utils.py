from typing import List
from matplotlib import pyplot as plt
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay


def plot_model1(data: np.ndarray, estimated_theta_per_memristor: np.ndarray,
                title='Model 1', figname='model_1_memristors'):
    """
    Plots the data points and the linear regression line for Model 1.

    :param data: Data points
    :param estimated_theta_per_memristor: Estimated slope of the linear model (per memristor)
    :param title: Title of the plot
    :param figname: Name of the file to save the plot
    :return:
    """

    x_lines, y_lines, labels = [], [], []
    for i, theta in enumerate(estimated_theta_per_memristor):
        x = data[i, :, 0]
        x_line = np.array([np.min(x), np.max(x)])
        y_line = theta * x_line
        x_lines.append(x_line)
        y_lines.append(y_line)
        labels.append(r'$\Delta R = $' + f'{theta:.2f}' + r' $\cdot \Delta R^{\text{ideal}}$')

    plot_linear_regression(data, x_lines, y_lines, title, figname, labels)


def plot_model2(data: np.ndarray, estimated_theta_per_memristor: np.ndarray,
                title='Model 2', figname='model_2_memristors'):
    """
    Plots the data points and the linear regression line for Model 2.

    :param data: Data points
    :param estimated_theta_per_memristor: Estimated slope and intercept of the linear model (per memristor)
    :param title: Title of the plot
    :param figname: Name of the file to save the plot
    :return:
    """

    x_lines, y_lines, labels = [], [], []
    for i, theta in enumerate(estimated_theta_per_memristor):
        x = data[i, :, 0]
        x_line = np.array([np.min(x), np.max(x)])
        y_line = theta[0] + theta[1] * x_line
        x_lines.append(x_line)
        y_lines.append(y_line)
        labels.append(r'$\Delta R = $' + f'{theta[0]:.2f} + {theta[1]:.2f}' + r' $\cdot \Delta R^{\text{ideal}}$')

    plot_linear_regression(data, x_lines, y_lines, title, figname, labels)


def plot_linear_regression(data: np.ndarray, x_lines: List[np.ndarray], y_lines: List[np.ndarray],
                           title: str, figname: str, labels: List[str]):
    """
    Plots the data points and the linear regression line for each memristor.

    :param data: Data points
    :param x_lines: List of x-coordinates of the individual lines
    :param y_lines: List of y-coordinates of the individual lines
    :param title: Suptitle of the plot
    :param figname: Name of the file to save the plot
    :param label: Label of the line (will be included in the subfigure title)
    :return:
    """

    # Create 8 subplots (2 rows and 4 columns)
    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    ax = ax.ravel()
    for i in range(8):
        x = data[i, :, 0]
        y = data[i, :, 1]
        ax[i].plot(x, y, 'ko')
        ax[i].plot(x_lines[i], y_lines[i])
        ax[i].set_title(f'Memristor {i+1} ({labels[i]})')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(title, fontsize=16)
    plt.savefig(f'plots/{figname}.pdf')
    plt.show()


def plot_logistic_regression(logreg_model, create_design_matrix, X, title, figname):
    """
    Plot the decision boundary of a logistic regression model.
    :param logreg_model: The logistic regression model
    :param create_design_matrix: Function to create the design matrix
    :param X: Data matrix
    :param title: Title of the plot
    :param figname: Filename to save the plot in the `plots` directory
    :return:
    """
    y = logreg_model.predict(X)
    xx0, xx1 = np.meshgrid(
        np.linspace(np.min(X[:, 0]), np.max(X[:, 0])),
        np.linspace(np.min(X[:, 1]), np.max(X[:, 1]))
    )
    x_grid = np.vstack([xx0.reshape(-1), xx1.reshape(-1)]).T
    x_grid = create_design_matrix(x_grid)
    y_grid = logreg_model.predict(x_grid).reshape(xx0.shape)
    display = DecisionBoundaryDisplay(xx0=xx0, xx1=xx1, response=y_grid)

    display.plot()
    p = display.ax_.scatter(
        X[:, 0], X[:, 1], c=y, edgecolor="black"
    )

    display.ax_.set_title(title)
    display.ax_.collections[0].set_cmap('coolwarm')
    display.ax_.figure.set_size_inches(5, 5)
    display.ax_.set_xlabel('x1')
    display.ax_.set_ylabel('x2')
    display.ax_.legend(*p.legend_elements(), loc='best', bbox_to_anchor=(1.02, 1.15))

    plt.savefig(f'plots/{figname}.pdf')
    plt.show()


def plot_datapoints(X, y, title):
    """
    Plot the data points in a scatter plot with color-coded classes.
    :param X: The data points
    :param y: The class labels
    :param title: Title of the plot
    :return:
    """
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    fig.suptitle(title, y=0.93)

    p = axs.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)

    axs.set_xlabel('x1')
    axs.set_ylabel('x2')
    axs.legend(*p.legend_elements(), loc='best', bbox_to_anchor=(0.96, 1.15))

    plt.show()


def plot_function(f):
    """
    Plotting the 3D surface for a given cost function f.
    :param f: The function to optimize
    :return:
    """
    n = 200
    bounds = [-2, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_ax = np.linspace(bounds[0], bounds[1], n)
    y_ax = np.linspace(bounds[0], bounds[1], n)
    XX, YY = np.meshgrid(x_ax, y_ax)

    ZZ = f(XX, YY)

    ax.plot_surface(XX, YY, ZZ, cmap='jet')
    plt.show()
