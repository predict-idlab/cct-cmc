import numpy as np
import scipy
from scipy.integrate import quad
from scipy.stats import gaussian_kde, norm


def loglikelihood(y, y_hat, weights=None, return_average=True):
    """Partially copied from: https://github.com/toonvds/NOFLITE/blob/main/metrics.py"""
    # region_size = 0.5
    # EPS = 1e-12
    if weights is not None:
        weights = weights / np.sum(weights, axis=1)[:, None]
    lls = np.zeros(len(y))

    for i in range(len(y)):
        # TODO: check if this is correct, you get a problem here when kde is impossible to fit
        # Already made an adjustment for this like returning nan if one of the kde's fails,
        # and also return on average nan if one of the samples fails
        try:
            if weights is not None:
                kde = gaussian_kde(dataset=y_hat[i, :], weights=weights[i, :])
            else:
                kde = gaussian_kde(dataset=y_hat[i, :])
            ll = kde.logpdf(y[i])
        except:
            ll = np.nan

        lls[i] = ll

    if return_average:
        if np.isnan(lls).any():
            return np.nan
        return np.mean(lls)
    else:
        return lls


def crps(y, y_hat, return_average=True):
    """
    Calculate the Continuous Ranked Probability Score (CRPS) in a vectorized manner.

    CRPS(F, y) = E_F |Y - y| - 0.5 E_F |Y - Y'|, where Y ~ F and Y' ~ F.

    :y: The true values (n,)
    :y_hat: The predicted values (n, samples)
    :return_average: If True, return the average CRPS over all samples. If False, return CRPS for each sample.
    """
    # Ensure y is a column vector for broadcasting
    y = y[:, None]

    # Compute the first term: E_F |Y - y|
    term1 = np.mean(np.abs(y_hat - y), axis=1)

    # Compute the second term: 0.5 * E_F |Y - Y'|
    term2 = np.zeros(len(y))
    for i in range(len(y)):
        # Compute all pairwise differences for the i-th observation's samples
        diff = y_hat[i, :, None] - y_hat[i, :]  # Shape (samples, samples)
        # Take the mean of all absolute pairwise differences
        term2[i] = np.mean(np.abs(diff))
    term2 *= 0.5  # Apply the 0.5 factor once after the loop

    # Compute CRPS for each sample
    crps_results = term1 - term2

    if return_average:
        return np.mean(crps_results)
    else:
        return crps_results


def crps_normal(y, mu, sigma, return_average=True):
    """
    Calculate the Continuous Ranked Probability Score (CRPS) for a normal distribution.

    :y: The true values (n,)
    :mu: The mean of the normal distribution (n,)
    :sigma: The standard deviation of the normal distribution (n,)
    """
    # Ensure y is a column vector for broadcasting
    w = (y - mu) / sigma
    crps = sigma * (
        w * (2 * scipy.stats.norm.cdf(w) - 1) + 2 * scipy.stats.norm.pdf(w) - 1 / np.sqrt(np.pi)
    )
    if return_average:
        return np.mean(crps)
    return crps


def crps_weighted(y, y_hat, weights, return_average=True, batch_size=1000):
    """
    Calculate the Continuous Ranked Probability Score (CRPS) using weighted forecasts.

    Parameters:
    y (np.array): Observed values, shape (n_samples,)
    y_hat (np.array): Forecasted values, shape (n_samples, n_forecast)
    weights (np.array): Weights for each forecast, shape (n_samples, n_forecast)
    return_average (bool): If True, return the mean CRPS. Otherwise, return individual scores.
    batch_size (int): Batch size for processing to manage memory usage.

    Returns:
    float or np.array: CRPS score(s)
    """
    # Normalize weights to sum to 1 for each sample
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    y = y[:, None]  # Reshape y to (n_samples, 1)

    # Compute term1: sum(weights * |y_hat - y|)
    term1 = np.sum(weights * np.abs(y_hat - y), axis=1)

    term2 = np.zeros(len(y))
    n_forecast, n_samples = y_hat.shape

    if n_forecast > n_samples:
        # Process samples in batches
        for i in range(0, n_forecast, batch_size):
            end_idx = min(i + batch_size, n_forecast)
            batch_y_hat = y_hat[i:end_idx]
            batch_weights = weights[i:end_idx]
            diff = np.abs(batch_y_hat[:, :, None] - batch_y_hat[:, None, :])
            weight_prod = batch_weights[:, :, None] * batch_weights[:, None, :]
            term2[i:end_idx] = np.sum(diff * weight_prod, axis=(1, 2))
    else:
        # Process forecast points in batches with nested loops
        cum_weights = np.cumsum(weights, axis=1)
        cum_y = np.cumsum(weights * y_hat, axis=1)

        cum_weights_shifted = np.pad(cum_weights[:, :-1], ((0, 0), (1, 0)), mode="constant")
        cum_y_shifted = np.pad(cum_y[:, :-1], ((0, 0), (1, 0)), mode="constant")

        term2 = 2 * np.sum(
            y_hat * weights * cum_weights_shifted - cum_y_shifted * weights, axis=1
        )  # Times 2 for acounting for both sides of the distribution
    term2 *= 0.5
    crps_results = term1 - term2

    return np.mean(crps_results) if return_average else crps_results


def calculate_dispersion(y, dist, return_p_values=False):
    """
    Calculate the dispersion metric for a distribution.

    Parameters:
    y (np.array): Observed values, shape (n_samples,)
    dist (np.array): Forecasted sampled distribution, shape (n_samples, n_forecast)

    Returns:
    float: Dispersion metric
    """
    # Vectorized calculation of PIT values
    p_values = np.mean(dist <= y[:, None], axis=1)

    # Compute dispersion metric
    if return_p_values:
        return np.var(p_values), p_values
    return np.var(p_values)


def calculate_dispersion_normal(y, mu, sigma, return_p_values=False):
    """
    Calculate the dispersion metric assuming a normal distribution.

    Parameters:
    y (np.array): Observed values, shape (n_samples,)
    mu (np.array): Mean predictions, shape (n_samples,)
    sigma (np.array): Standard deviations, shape (n_samples,)

    Returns:
    float: Dispersion metric
    """
    # Compute PIT values assuming a normal distribution
    p_values = norm.cdf(y, loc=mu, scale=sigma)

    # Compute dispersion metric
    if return_p_values:
        return np.var(p_values), p_values
    return np.var(p_values)
