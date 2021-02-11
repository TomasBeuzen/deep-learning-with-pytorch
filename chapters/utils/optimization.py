import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gradient(x, y, w):
    """MSE gradient."""
    y_hat = x @ w
    error = y - y_hat
    gradient = -(1.0 / len(x)) * 2 * x.T @ error
    mse = (error ** 2).mean()
    return gradient, mse


def gradient_descent(
    x,
    y,
    w,
    alpha,
    tolerance: float = 2e-5,
    max_iterations: int = 1000,
    verbose: bool = False,
    print_progress: int = 10,
    history: bool = False,
):
    """MSE gradient descent."""
    iterations = 1
    if verbose:
        print(f"Iteration 0.", "Weights:", [f"{_:.2f}" for _ in w])
    if history:
        ws = []
        mses = []
    while True:
        g, mse = gradient(x, y, w)
        if history:
            ws.append(list(w))
            mses.append(mse)
        w_new = w - alpha * g
        if sum(abs(w_new - w)) < tolerance:
            if verbose:
                print(f"Converged after {iterations} iterations!")
                print("Final weights:", [f"{_:.2f}" for _ in w_new])
            break
        if iterations % print_progress == 0:
            if verbose:
                print(
                    f"Iteration {iterations}.",
                    "Weights:",
                    [f"{_:.2f}" for _ in w_new],
                )
        iterations += 1
        if iterations > max_iterations:
            if verbose:
                print(f"Reached max iterations ({max_iterations})!")
                print("Final weights:", [f"{_:.2f}" for _ in w_new])
            break
        w = w_new
    if history:
        w = w_new
        _, mse = gradient(x, y, w)
        ws.append(list(w))
        mses.append(mse)
        return ws, mses


def stochastic_gradient_descent(
    x,
    y,
    w,
    alpha,
    tolerance: float = 2e-5,
    max_iterations: int = 1000,
    verbose: bool = False,
    print_progress: int = 10,
    history: bool = False,
    seed=None,
):
    """MSE stochastic gradient descent."""
    if seed is not None:
        np.random.seed(seed)
    iterations = 1
    if verbose:
        print(f"Iteration 0.", "Weights:", [f"{_:.2f}" for _ in w])
    if history:
        ws = []
        mses = []
    while True:
        i = np.random.randint(len(x))
        g, mse = gradient(x[i, None], y[i, None], w)
        if history:
            ws.append(list(w))
            mses.append(mse)
        w_new = w - alpha * g
        if sum(abs(w_new - w)) < tolerance:
            if verbose:
                print(f"Converged after {iterations} iterations!")
                print("Final weights:", [f"{_:.2f}" for _ in w_new])
            break
        if iterations % print_progress == 0:
            if verbose:
                print(
                    f"Iteration {iterations}.",
                    "Weights:",
                    [f"{_:.2f}" for _ in w_new],
                )
        iterations += 1
        if iterations > max_iterations:
            if verbose:
                print(f"Reached max iterations ({max_iterations})!")
                print("Final weights:", [f"{_:.2f}" for _ in w_new])
            break
        w = w_new
    if history:
        w = w_new
        _, mse = gradient(x, y, w)
        ws.append(list(w))
        mses.append(mse)
        return ws, mses


def minibatch_gradient_descent(
    x,
    y,
    w,
    alpha,
    batch_size,
    tolerance: float = 2e-5,
    max_iterations: int = 1000,
    verbose: bool = False,
    print_progress: int = 10,
    history: bool = False,
    seed=None,
):
    """MSE stochastic gradient descent."""
    if seed is not None:
        np.random.seed(seed)
    iterations = 1
    if verbose:
        print(f"Iteration 0.", "Weights:", [f"{_:.2f}" for _ in w])
    if history:
        ws = []
        mses = []
    while True:
        i = np.random.choice(
            range(len(x)), batch_size, replace=False
        )  # no replacement
        g, mse = gradient(x[i], y[i], w)
        if history:
            ws.append(list(w))
            mses.append(mse)
        w_new = w - alpha * g
        if sum(abs(w_new - w)) < tolerance:
            if verbose:
                print(f"Converged after {iterations} iterations!")
                print("Final weights:", [f"{_:.2f}" for _ in w_new])
            break
        if iterations % print_progress == 0:
            if verbose:
                print(
                    f"Iteration {iterations}.",
                    "Weights:",
                    [f"{_:.2f}" for _ in w_new],
                )
        iterations += 1
        if iterations > max_iterations:
            if verbose:
                print(f"Reached max iterations ({max_iterations})!")
                print("Final weights:", [f"{_:.2f}" for _ in w_new])
            break
        w = w_new
    if history:
        w = w_new
        _, mse = gradient(x, y, w)
        ws.append(list(w))
        mses.append(mse)
        return ws, mses