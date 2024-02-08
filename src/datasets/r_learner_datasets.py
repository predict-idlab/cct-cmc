"""
Copied from the causalml package
see repo: https://github.com/uber/causalml

Note that this is an extended version of the original code
"""

import numpy as np  # type: ignore
from scipy.special import expit, logit  # type: ignore


def simulate_nuisance_and_easy_treatment(
    n=1000, p=5, sigma=1.0, adj=0.0, c=1.0, heteroscedastic=False
):
    """Synthetic data with a difficult nuisance components and an easy treatment effect
        From Setup A in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=5)
        sigma (float): standard deviation of the error term
        adj (float): adjustment term for the distribution of propensity, e. Higher values shift the distribution to 0.
        c (float): correlation between the error terms of the treatment and control groups.
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    """
    assert c <= 1 or c >= 0, "c must be between 0 and 1"
    X = np.random.uniform(size=n * p).reshape((n, -1))
    b = np.sin(np.pi * X[:, 0] * X[:, 1]) + 2 * (X[:, 2] - 0.5) ** 2 + X[:, 3] + 0.5 * X[:, 4]
    eta = 0.1
    e = np.maximum(
        np.repeat(eta, n),
        np.minimum(np.sin(np.pi * X[:, 0] * X[:, 1]), np.repeat(1 - eta, n)),
    )
    e = expit(logit(e) - adj)
    tau = (X[:, 0] + X[:, 1]) / 2

    w = np.random.binomial(1, e, size=n)
    if heteroscedastic:
        sigma = sigma * np.log1p(np.abs(X[:, 1]))
    epsilon_0 = sigma * np.random.normal(size=n)
    epsilon_1 = c * epsilon_0 + (1 - c) * sigma * np.random.normal(size=n)
    mu_0 = b - 0.5 * tau
    mu_1 = mu_0 + tau
    y0 = mu_0 + epsilon_0
    y1 = mu_1 + epsilon_1
    y = y0 * (1 - w) + y1 * w
    return y, X, w, tau, b, e, y0, y1


def simulate_randomized_trial(n=1000, p=5, sigma=1.0, adj=0.0, c=1.0, heteroscedastic=False):
    """Synthetic data of a randomized trial
        From Setup B in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=5)
        sigma (float): standard deviation of the error term
        adj (float): no effect. added for consistency
        c (float): correlation between the error terms of the treatment and control groups.
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    """
    assert c <= 1 or c >= 0, "c must be between 0 and 1"
    X = np.random.normal(size=n * p).reshape((n, -1))
    b = np.maximum.reduce([np.repeat(0.0, n), X[:, 0] + X[:, 1], X[:, 2]]) + np.maximum(
        np.repeat(0.0, n), X[:, 3] + X[:, 4]
    )
    e = np.repeat(0.5, n)
    tau = X[:, 0] + np.log1p(np.exp(X[:, 1]))

    w = np.random.binomial(1, e, size=n)
    if heteroscedastic:
        sigma = sigma * np.log1p(np.abs(X[:, 1]))
    epsilon_0 = sigma * np.random.normal(size=n)
    epsilon_1 = c * epsilon_0 + (1 - c) * sigma * np.random.normal(size=n)
    mu_0 = b - 0.5 * tau
    mu_1 = mu_0 + tau
    y0 = mu_0 + epsilon_0
    y1 = mu_1 + epsilon_1
    y = y0 * (1 - w) + y1 * w
    return y, X, w, tau, b, e, y0, y1


def simulate_easy_propensity_difficult_baseline(
    n=1000, p=5, sigma=1.0, adj=0.0, c=1.0, heteroscedastic=False
):
    """Synthetic data with easy propensity and a difficult baseline
        From Setup C in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=3)
        sigma (float): standard deviation of the error term
        adj (float): no effect. added for consistency
        c (float): correlation between the error terms of the treatment and control groups.
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    """
    assert c <= 1 or c >= 0, "c must be between 0 and 1"
    X = np.random.normal(size=n * p).reshape((n, -1))
    b = 2 * np.log1p(np.exp(X[:, 0] + X[:, 1] + X[:, 2]))
    e = 1 / (1 + np.exp(X[:, 1] + X[:, 2]))
    tau = np.repeat(1.0, n)

    w = np.random.binomial(1, e, size=n)
    if heteroscedastic:
        sigma = sigma * np.log1p(np.abs(X[:, 1]))
    epsilon_0 = sigma * np.random.normal(size=n)
    epsilon_1 = c * epsilon_0 + (1 - c) * sigma * np.random.normal(size=n)
    mu_0 = b - 0.5 * tau
    mu_1 = mu_0 + tau
    y0 = mu_0 + epsilon_0
    y1 = mu_1 + epsilon_1
    y = y0 * (1 - w) + y1 * w

    return y, X, w, tau, b, e, y0, y1


def simulate_unrelated_treatment_control(
    n=1000, p=5, sigma=1.0, adj=0.0, c=1.0, heteroscedastic=False
):
    """Synthetic data with unrelated treatment and control groups.
        From Setup D in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=3)
        sigma (float): standard deviation of the error term
        adj (float): adjustment term for the distribution of propensity, e. Higher values shift the distribution to 0.
        c (float): correlation between the error terms of the treatment and control groups.
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    """
    assert c <= 1 or c >= 0, "c must be between 0 and 1"
    X = np.random.normal(size=n * p).reshape((n, -1))
    b = (
        np.maximum(np.repeat(0.0, n), X[:, 0] + X[:, 1] + X[:, 2])
        + np.maximum(np.repeat(0.0, n), X[:, 3] + X[:, 4])
    ) / 2
    e = 1 / (1 + np.exp(-X[:, 0]) + np.exp(-X[:, 1]))
    e = expit(logit(e) - adj)
    tau = np.maximum(np.repeat(0.0, n), X[:, 0] + X[:, 1] + X[:, 2]) - np.maximum(
        np.repeat(0.0, n), X[:, 3] + X[:, 4]
    )

    w = np.random.binomial(1, e, size=n)
    if heteroscedastic:
        sigma = sigma * np.log1p(np.abs(X[:, 1]))
    epsilon_0 = sigma * np.random.normal(size=n)
    epsilon_1 = c * epsilon_0 + (1 - c) * sigma * np.random.normal(size=n)
    mu_0 = b - 0.5 * tau
    mu_1 = mu_0 + tau
    y0 = mu_0 + epsilon_0
    y1 = mu_1 + epsilon_1
    y = y0 * (1 - w) + y1 * w

    return y, X, w, tau, b, e, y0, y1
