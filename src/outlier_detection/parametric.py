import numpy as np
import statsmodels as sm
import scipy as sp


def parametric_with_error(df, sigma_threshold=5):
    """
    A parametric form of outlier detection. Fit a linear constant period model and then
    remove any datapoints whose 5-sigma error does not agree with the fit i.e. 1 in 3.5
    million.
    """
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
    lower_bound = df.transit_time - (df.error * sigma_threshold)
    upper_bound = df.transit_time + (df.error * sigma_threshold)

    # remove datapoints where the model fit does not lie within the datapoint's 5 sigma error
    accepted = (lower_bound < y_hat) & (y_hat < upper_bound)
    outliers = df[~accepted]
    survivors = df[accepted]
    return survivors, outliers


def parametric_without_error(df, threshold=1e-6):
    """
    A parametric form of outlier detection. Fit a linear constant period model, then
    estimate the empirical cumulative distribution function and remove datapoints which
    are in the 2 * 1e-6 most extreme i.e. 2 in a million.
    """
    X = np.vstack(
        [
            np.power(df.epoch, 0.0),
            np.power(df.epoch, 1.0)
            # do not use a decay term
        ]
    ).T
    y = np.atleast_2d(df.transit_time.values).T
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    residuals = (y - y_hat).flatten()
    ecdf = sm.distributions.empirical_distribution.ECDF(residuals)
    cumulative_prob = ecdf(residuals)

    # remove datapoints whose values are in the most extreme 1e-6
    accepted = (threshold < cumulative_prob) & (cumulative_prob < (1 - threshold))
    outliers = df[~accepted]
    survivors = df[accepted]
    return survivors, outliers


def parametric_without_error_assuming_normality(df, sigma_threshold=5):
    """
    A parametric form of outlier detection. Fit a linear constant period model, then
    fit a normal distribution to the residuals and discard any residuals > 5 sigma away
    from the model fit.
    """
    X = np.vstack(
        [
            np.power(df.epoch, 0.0),
            np.power(df.epoch, 1.0)
            # do not use a decay term
        ]
    ).T
    y = np.atleast_2d(df.transit_time.values).T
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    residuals = y - y_hat
    _, std = sp.stats.norm.fit(residuals)

    # remove datapoints whose values are in the most extreme 1e-6
    threshold = sigma_threshold * std
    accepted = (-threshold < residuals) & (residuals < threshold)
    outliers = df[~accepted]
    survivors = df[accepted]
    return survivors, outliers
