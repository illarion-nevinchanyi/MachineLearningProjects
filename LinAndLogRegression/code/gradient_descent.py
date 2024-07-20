import numpy as np


def gradient_descent(f, df, x0, y0, learning_rate, lr_decay, num_iters):
    """
    Find a local minimum of the function f(x) using gradient descent:
    Until the number of iteration is reached, decrease the current x and y points by the
    respective partial derivative times the learning_rate.
    In each iteration, record the current function value in the list f_list.
    The function should return the minimizing argument (x, y) and f_list.

    :param f: Function to minimize
    :param df: Gradient of f i.e, function that computes gradients
    :param x0: initial x0 point
    :param y0: initial y0 point
    :param learning_rate: Learning rate
    :param lr_decay: Learning rate decay
    :param num_iters: number of iterations
    :return: x, y (solution), f_list (array of function values over iterations)
    """
    f_list = np.zeros(num_iters)    # Array to store the function values over iterations
    x, y = x0, y0
    # TODO: Implement the gradient descent algorithm with a decaying learning rate
    eta = learning_rate

    if lr_decay <= 0 or lr_decay > 1:
        raise ValueError('Decaying learning rate lr_decay is out of boundaries (0, 1].')
    else:
        for epoch in range(0, num_iters):
            x -= eta * df(x, y)[0]
            y -= eta * df(x, y)[1]
            f_list[epoch] = f(x, y)
            eta *= lr_decay
    return x, y, f_list


def ackley(x, y):
    """
    Ackley function at point (x, y)
    :param x: X coordinate
    :param y: Y coordinate
    :return: f(x, y) where f is the Ackley function
    """
    # TODO: Implement the Ackley function (as specified in the Assignment 1 sheet)
    f = -20*np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2))) - np.exp(0.5*(np.cos(2*np.pi*x) + \
        np.cos(2*np.pi*y))) + np.exp(1) + 20
    return f


def gradient_ackley(x, y):
    """
    Compute the gradient of the Ackley function at point (x, y)
    :param x: X coordinate
    :param y: Y coordinate
    :return: \nabla f(x, y) where f is the Ackley function
    """
    # TODO: Implement partial derivatives of Ackley function w.r.t. x and y
    a = 20; b = 0.2; g = 0.5
    df_dx = a*b*np.sqrt(g) * x * 1/np.sqrt((x**2 + y**2)) * np.exp(-b * np.sqrt(g *(x**2 + y**2))) + \
            2*np.pi*g * np.sin(2*np.pi*x) * np.exp(g*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y)))

    df_dy = a*b*np.sqrt(g) * y * 1/np.sqrt((x**2 + y**2)) * np.exp(-b * np.sqrt(g *(x**2 + y**2))) + \
            2*np.pi*g * np.sin(2*np.pi*y) * np.exp(g*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y)))

    gradient = np.array([df_dx, df_dy])
    return gradient