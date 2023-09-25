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
    kernel = sp.stats.gaussian_kde(np.vstack([df.epoch, df.transit_time]))

    # remove datapoints which fall in the most extreme 2 in a million according to the marginal
    # empirical cumulative distribution function along that epoch
    def is_accepted(row):
        marginal_prob_lt_transit_time = kernel.integrate_box(
            [row.epoch - 0.5, -np.inf], [row.epoch + 0.5, row.transit_time]
        )
        return threshold < marginal_prob_lt_transit_time < (1 - threshold)

    accepted = df.apply(is_accepted, axis=1)
    outliers = df[~accepted]
    survivors = df[accepted]
    return survivors, outliers
