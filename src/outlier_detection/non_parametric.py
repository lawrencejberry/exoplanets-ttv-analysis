import numpy as np
import scipy as sp


def non_parametric_with_error(df, threshold=1e-6):
    """
    A non-parametric form of outlier detection. Estimate the bivariate probability distribution
    using a gaussian KDE. Given the exact epoch of datapoints, remove datapoints whose integrated
    probability over the given 5-sigma error bound is less than 1 in a million.
    """
    kernel = sp.stats.gaussian_kde(np.vstack([df.epoch, df.transit_time]))

    # remove datapoints which fall in the most extreme 2 in a million according to the marginal
    # empirical cumulative distribution function along that epoch
    def is_accepted(row):
        prob_of_measurement = kernel.integrate_box(
            [row.epoch - 0.5, row.transit_time - 5 * row.error],
            [row.epoch + 0.5, row.transit_time + 5 * row.error],
        )
        return prob_of_measurement > threshold

    accepted = df.apply(is_accepted, axis=1)
    outliers = df[~accepted]
    survivors = df[accepted]
    return survivors, outliers


def non_parametric_without_error(df, threshold=1e-6):
    """
    A non-parametric form of outlier detection. Estimate the bivariate probability distribution
    using a gaussian KDE. Given the exact epoch of datapoints, remove datapoints whose transit
    times fall into the most extreme 2 in million of the estimated marginal distribution at those
    epochs.
    """
    df = df.copy()
    X = np.vstack(
        [
            np.power(df.epoch, 0.0),
            np.power(df.epoch, 1.0)
            # do not use a decay term
        ]
    ).T
    y = np.atleast_2d(df.transit_time.values).T
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = (X @ beta).flatten()
    df["residuals"] = y.flatten() - y_hat.flatten()

    kernel = sp.stats.gaussian_kde(np.vstack([df.epoch, df.residuals]))

    # remove datapoints which fall in the most extreme 2 in a million according to the marginal
    # empirical cumulative distribution function along that epoch
    def is_accepted(row):
        # using Bayes' theorem:
        marginal_prob_lt_residuals = kernel.integrate_box(
            [row.epoch - 0.5, -np.inf], [row.epoch + 0.5, row.residuals]
        ) / kernel.integrate_box([row.epoch - 0.5, -np.inf], [row.epoch + 0.5, np.inf])
        return threshold < marginal_prob_lt_residuals < (1 - threshold)

    accepted = df.apply(is_accepted, axis=1)
    outliers = df[~accepted]
    survivors = df[accepted]
    return survivors, outliers
