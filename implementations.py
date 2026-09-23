import numpy as np
import helpers

########################### GLOBAL VARIABLES ###########################

MAX_ITER = 50
GAMMA = 0.1

########################### 6 MAIN functions ###########################

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using full-batch gradient descent.

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N, D)
        initial_w: numpy array of shape (D,), initial weights
        max_iters: int, number of GD steps
        gamma: float, step size

    Returns:
        (w, loss): final weight vector of shape (D,) and its MSE loss.
    """
    w = initial_w
    for _ in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w = w - gamma * grad
    loss = compute_loss_MSE(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent with
    mini-batch size 1.

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N, D)
        initial_w: numpy array of shape (D,), initial weights
        max_iters: int, number of SGD steps
        gamma: float, step size

    Returns:
        (w, loss): final weight vector of shape (D,) and its MSE loss
        (computed on the FULL dataset, not just the last mini-batch).
    """
    w = initial_w
    for _ in range(max_iters):
        for batch_y, batch_tx in batch_iter(y, tx, batch_size=1, num_batches=1):
            grad = compute_gradient(batch_y, batch_tx, w)
            w = w - gamma * grad
    loss = compute_loss_MSE(y, tx, w)
    return w, loss
    

def least_squares(y, tx):
    """Least squares regression using the normal equations (closed form).

    Solves (tx^T tx) w = tx^T y directly via np.linalg.solve 

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N, D)

    Returns:
        (w, loss): optimal weight vector of shape (D,) and its MSE loss.
    """
    a = np.dot(tx.T, tx)
    b = np.dot(tx.T, y)
    w = np.linalg.solve(a, b)
    loss = compute_loss_MSE(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using the normal equations (closed form).

    Solves (tx^T tx + 2 lambda_ N I) w = tx^T y.

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N, D)
        lambda_: float, regularization parameter

    Returns:
        (w, loss): optimal weight vector of shape (D,) and its MSE loss
        WITHOUT the penalty term.
    """
    N, D = tx.shape
    a = tx.T.dot(tx) + 2 * N * lambda_ * np.identity(D)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss_MSE(y, tx, w)  # no penalty term included, as required
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using full-batch gradient descent (y in {0,1}).

    Args:
        y: numpy array of shape (N,), entries in {0, 1}
        tx: numpy array of shape (N, D)
        initial_w: numpy array of shape (D,), initial weights
        max_iters: int, number of GD steps
        gamma: float, step size

    Returns:
        (w, loss): final weight vector of shape (D,) and its cross-entropy loss.
    """
    w = initial_w
    for _ in range(max_iters):
        grad = compute_gradient_logistic(y, tx, w)
        w = w - gamma * grad
    loss = compute_loss_cross_entropy(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using full-batch gradient descent
    (y in {0,1}), with L2 penalty term lambda_ * ||w||^2.

    Args:
        y: numpy array of shape (N,), entries in {0, 1}
        tx: numpy array of shape (N, D)
        lambda_: float, regularization parameter
        initial_w: numpy array of shape (D,), initial weights
        max_iters: int, number of GD steps
        gamma: float, step size

    Returns:
        (w, loss): final weight vector of shape (D,) and its cross-entropy
        loss WITHOUT the penalty term (the penalty is used only in the gradient/update step, 
        not in the reported loss).
    """
    w = initial_w
    for _ in range(max_iters):
        grad = compute_gradient_logistic(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
    loss = compute_loss_cross_entropy(y, tx, w)  # no penalty term included
    return w, loss


########################### HELPER FUNCTIONS ###########################

def compute_loss_MSE(y, tx, w):
    """Compute the mean squared error loss (with the 1/2 factor from the
    lecture notes: L(w) = 1/(2N) * sum((y - tx@w)^2)).

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N, D)
        w: numpy array of shape (D,)

    Returns:
        Scalar loss value.
    """
    N = len(tx)
    y_hat = np.dot(tx, w)
    e = y - y_hat
    loss = np.sum(e**2)/ (2*N)
    return loss


def compute_loss_MAE(y, tx, w):
    """Compute the mean absolute error loss (with the 1/2 factor from the
    lecture notes: L(w) = 1/(2N) * sum(|(y - tx@w)|)).

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N, D)
        w: numpy array of shape (D,)

    Returns:
        Scalar loss value.
    """
    N = len(tx)
    y_hat = np.dot(tx, w)
    e = y - y_hat
    loss = np.sum(np.abs(e)) / (2*N)
    return loss


def sigmoid(z):
    """Apply the sigmoid function element-wise.

    Args:
        z: numpy array of any shape.

    Returns:
        numpy array of the same shape as z, with values in (0, 1).
    """
    return 1/(1 + np.exp(-z))


def compute_loss_cross_entropy(y, tx, w, l=0):
    """Compute the logistic regression loss (negative log-likelihood / cross-entropy). 
    Can include the penalty term if l is set to anything else than 0.

    Args:
        y: numpy array of shape (N,), entries in {0, 1}
        tx: numpy array of shape (N, D)
        w: numpy array of shape (D,)
        l: penalty term lambda

    Returns:
        Scalar loss value.
    """
    N = len(tx)
    e = y*np.log(sigmoid(np.dot(tx, w))) + (1-y)*np.log(1-sigmoid(np.dot(tx.T, w))) + l*np.dot(w, w)
    loss = -1/N*np.sum(e)
    return loss


def compute_gradient(y, tx, w):
    """Compute the gradient of the MSE loss at w (full-batch).

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N, D)
        w: numpy array of shape (D,)

    Returns:
        numpy array of shape (D,), the gradient of the MSE loss at w.
    """
    N = len(tx)
    y_hat = np.dot(tx, w)
    e = y - y_hat
    grad = -1/N*np.dot(tx.T, e)
    return grad


def compute_gradient_logistic(y, tx, w, l=0):
    """Compute the gradient of the logistic regression loss.
    Can include the penalty term if l is set to anything else than 0.

    Args:
        y: numpy array of shape (N,), entries in {0, 1}
        tx: numpy array of shape (N, D)
        w: numpy array of shape (D,)
        l: penalty term lambda

    Returns:
        numpy array of shape (D,), the gradient at w.
    """
    N = len(tx)
    y_hat = sigmoid(np.dot(tx, w))
    e = y_hat - y
    grad = 1/N*np.dot(tx.T, e) + 2*l*w
    return grad


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Generate a minibatch iterator for a dataset (provided utility,
    used internally by mean_squared_error_sgd with batch_size=1 as required
    by the project spec).
    """
    data_size = len(y)
    batch_size = min(data_size, batch_size)
    max_batches = int(data_size / batch_size)
    remainder = data_size - max_batches * batch_size

    if shuffle:
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        end = start + batch_size
        yield y[start:end], tx[start:end]