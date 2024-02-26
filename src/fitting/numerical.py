import numpy as np
import scipy as sp
import emcee


def ml_fit(df):
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

    def neg_log_likelihood(beta):
        beta = np.atleast_2d(beta).T
        y_hat = X @ beta
        residuals = y - y_hat
        log_likelihood = (
            -(0.5 * N * np.log(2 * np.pi))
            - (np.sum(np.log(error)))
            - (0.5 * residuals.T @ inv_sigma @ residuals)
        )
        return -log_likelihood

    res = sp.optimize.minimize(
        neg_log_likelihood,
        (0.0, 1.0, 0.0),
        method="BFGS",
        tol=1e-6,
        options={"disp": False, "maxiter": 1e4},
    )

    return (np.atleast_2d(res.x).T, error, 3)


def mcmc_fit(df):
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

    def log_likelihood(beta):
        beta = np.atleast_2d(beta).T
        y_hat = X @ beta
        residuals = y - y_hat
        log_likelihood = (
            -(0.5 * N * np.log(2 * np.pi))
            - (np.sum(np.log(error)))
            - (0.5 * residuals.T @ inv_sigma @ residuals)
        )
        return log_likelihood

    def log_prior(beta):
        _, P0, _ = beta
        if P0 < 0:
            return -np.inf  # can't have negative periods
        return 0.0

    def log_probability(beta):
        lp = log_prior(beta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(beta)

    beta_ml, *_ = ml_fit(df)
    pos = beta_ml.flatten() + 1e-4 * np.random.randn(32, 3)
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
    sampler.run_mcmc(pos, 5000, progress=True)

    flat_samples = sampler.get_chain(discard=100, thin=30, flat=True)
    beta = np.percentile(flat_samples, 50, axis=0)

    return (np.atleast_2d(beta).T, error, 3)
