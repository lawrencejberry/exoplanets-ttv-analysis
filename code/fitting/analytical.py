import numpy as np
import scipy as sp


def bayesian_mvn_regression_fit(df, K0=None):
    """
    An analytical fit based on Bayesian multivariate regression but parameterising both
    the model parameters and observational errors.
    """
    # determine priors on T0 and P0 through a constant period least squares fit
    X = np.vstack(
        [
            np.power(df.epoch, 0.0),
            np.power(df.epoch, 1.0)
            # do not use a decay term
        ]
    ).T
    y = np.atleast_2d(df.transit_time.values).T
    constant_fit_beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    constant_fit_T0 = np.asscalar(constant_fit_beta[0])
    constant_fit_P0 = np.asscalar(constant_fit_beta[1])
    beta0 = np.atleast_2d([constant_fit_T0, constant_fit_P0, 0.0]).T

    # determine the precision of our prior on beta
    # 1.0 corresponds to a weighting of 50% in the posterior beta
    # precision on the T0 and P0 terms is equivalent to saying the 1 standard deviation = constant_fit_P0
    # precision on dP/dE is equivalent to saying the 1 standard deviation = 1 ms/epoch i.e. decay on the
    # order of more than a millisecond per epoch is highly unlikely
    ms_in_days = 1e-3 / 60 / 60 / 24
    if K0 is None:
        K0 = np.diag([1 / constant_fit_P0, 1 / constant_fit_P0, 1 / ms_in_days])
    K0_inv = np.linalg.inv(K0)
    I = np.identity(3)

    D = len(df)  # number of dimensions of the covariance matrix

    # we choose a prior degrees of freedom such that the expected value (aka mean) of the prior IW
    # distribution on Σ is S0, the prior covariance matrix given by the data
    v0 = D + 2

    # the prior scatter matrix which is in this case equivalent to prior covariance is just the
    # reported covariance we get from our datasets, adjusted for degrees of freedom
    S0 = np.diagflat(np.power(df.error.values, 2.0)) * (v0 + D + 1)

    X = np.vstack(
        [
            np.power(df.epoch, 0.0),
            np.power(df.epoch, 1.0),
            0.5 * np.power(df.epoch, 2.0),
        ]
    ).T
    y = np.atleast_2d(df.transit_time.values).T
    beta_hat, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    residuals = y - X @ beta_hat
    Se = residuals @ residuals.T  # empirical scatter matrix

    virtual_residuals = X @ (0.5 * np.linalg.inv(K0_inv + I)) @ (beta_hat - beta0)
    Sv = virtual_residuals @ virtual_residuals.T  # virtual scatter matrix

    beta = np.linalg.inv(K0 + I) @ ((K0 @ beta0) + beta_hat)
    S = S0 + Se + Sv
    v = (
        v0 + 1
    )  # add 1 to the prior degrees of freedom to account for the addition of a single set of observations
    K = (
        K0 + I
    )  # the new belief in our beta also increases by 1 to account for the addition of a single set of observations

    t_sigma = (
        np.linalg.inv(K) @ (np.linalg.inv(X.T @ np.linalg.inv(S) @ X)) / (v + 1 - 3)
    )
    t_dof = v + 1 - 3

    mu = beta[2, 0] * (24 * 60 * 60 * 1000)
    t_sd = np.sqrt(t_sigma[2, 2] * (24 * 60 * 60 * 1000) ** 2)
    sd = np.sqrt(t_sigma[2, 2] * (t_dof / (t_dof - 2)) * (24 * 60 * 60 * 1000) ** 2)
    prob_decay = sp.stats.t.cdf(0, t_dof, mu, t_sd)
    print(f"E[dP/dE] = {mu} ms/epoch SD[dP/dE] = {sd} Prob(dP/dE < 0) = {prob_decay}")
    posterior_observation_covariance = S / (
        v + len(df) + 1
    )  # the MAP of the IW distribution over Σ
    # Similarly: prior_observation_covariance = S0 / (v0 + len(df) + 1)
    error = np.power(np.diag(posterior_observation_covariance), 0.5)

    k = D + 3
    return (beta, t_sigma, error, k, t_dof, prob_decay)
