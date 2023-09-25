import numpy as np


def compute_residuals(df, beta):
    X = np.vstack(
        [
            np.power(df.epoch, 0.0),
            np.power(df.epoch, 1.0),
            0.5 * np.power(df.epoch, 2.0),
        ]
    ).T
    y = np.atleast_2d(df.transit_time.values).T
    y_hat = X @ beta
    residuals = (y_hat - y).flatten()
    return residuals


def compute_rss(df, beta):
    residuals = compute_residuals(df, beta)
    return np.power(residuals, 2.0).sum()


def log_likelihood(df, beta):
    N = len(df)
    X = np.vstack(
        [
            np.power(df.epoch, 0.0),
            np.power(df.epoch, 1.0),
            0.5 * np.power(df.epoch, 2.0),
        ]
    ).T
    y = np.atleast_2d(df.transit_time.values).T
    error = df.error.values
    inv_sigma = np.diag(1 / np.power(error, 2))

    y_hat = X @ beta
    residuals = y - y_hat
    ll = (
        -(0.5 * N * np.log(2 * np.pi))
        - (np.sum(np.log(error)))
        - (0.5 * residuals.T @ inv_sigma @ residuals)
    ).item()
    return ll


def bic(df, beta, k):
    n = len(df)
    return k * np.log(n) - 2 * log_likelihood(df, beta)
