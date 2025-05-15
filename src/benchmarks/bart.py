import arviz as az
import numpy as np
import pymc as pm
import pymc_bart as pmb

from src.metrics import calculate_dispersion, crps, loglikelihood


class BART:

    def __init__(self, noise_dist: str = "normal", m=50, sigma=1, draws=500, chains=4, tune=500):
        """
        Class constructor.
        Initialize a BART object for causal inference.

        :noise_dist: The noise distribution. Default is normal
        :m: Number of trees in the BART model. Default is 50
        :sigma: Prior for prediction noise. Default is 1
        """
        if noise_dist not in ["normal"]:
            raise ValueError("Invalid noise distribution! Choose from 'normal'.")
        self.noise_dist = noise_dist
        self.m = m
        self.sigma = sigma
        self.draws = draws
        self.chains = chains
        self.tune = tune

    def fit(self, X, Y, W):
        """
        Fits the BART model.

        :X: The input covariates
        :Y: The corresponding outcomes
        :W: The treatment assignments
        """
        X_extended = np.concatenate([X, W.reshape((-1, 1))], axis=1)
        self.model = pm.Model()
        with self.model:
            self.X_input = pm.MutableData("X_input", X_extended)
            Y_input = Y
            mu = pmb.BART("mu", self.X_input, Y_input, m=self.m)
            y = pm.Normal("y", mu=mu, observed=Y_input, sigma=self.sigma, shape=mu.shape)
            self.idata_oos_regression = pm.sample(
                draws=self.draws,
                chains=self.chains,
                tune=self.tune,
                return_inferencedata=True,
            )

    def predict_dist_y0(self, X):
        """
        Predicts the distribution of the counterfactual outcome Y0.

        :X: The input covariates
        :samples: Number of samples to draw from the posterior
        """
        X_extended = np.concatenate([X, np.zeros((X.shape[0], 1))], axis=1)
        with self.model:
            self.X_input.set_value(X_extended)
            posterior_pred = pm.sample_posterior_predictive(
                trace=self.idata_oos_regression, var_names=["y"]
            )
        return np.transpose(posterior_pred.posterior_predictive["y"].values, (2, 0, 1)).reshape(
            len(X), -1
        )

    def predict_dist_y1(self, X):
        """
        Predicts the distribution of the counterfactual outcome Y1.

        :X: The input covariates
        :samples: Number of samples to draw from the posterior
        """
        X_extended = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        with self.model:
            self.X_input.set_value(X_extended)
            posterior_pred = pm.sample_posterior_predictive(
                trace=self.idata_oos_regression, var_names=["y"]
            )
        return np.transpose(posterior_pred.posterior_predictive["y"].values, (2, 0, 1)).reshape(
            len(X), -1
        )

    def predict_dist_ite(self, X):
        """
        Predicts the Individual Treatment Effect (ITE).

        :X: The input covariates
        :samples: Number of samples to draw from the posterior
        """
        y0 = self.predict_dist_y0(X)
        y1 = self.predict_dist_y1(X)
        return y1 - y0

    def evaluate(self, X, Y0, Y1, alpha=0.05, return_p_values=False):
        """
        Evaluates the BART model:
            - Point prediction related:
                - RMSE for Y0, Y1, and ITE
            - Interval prediction related:
                - Coverage for Y0, Y1, and ITE for a given alpha
                - Effiecency for Y0, Y1, and ITE for a given alpha
            - Distribution prediction related:
                - CRPS for Y0, Y1, and ITE
                - LL for Y0, Y1, and ITE

        :X: The input covariates
        :Y0: The counterfactual outcome Y0
        :Y1: The counterfactual outcome Y1
        :alpha: The significance level for the confidence interval
        """
        y0_pred = self.predict_dist_y0(X)
        y1_pred = self.predict_dist_y1(X)
        ite_pred = self.predict_dist_ite(X)
        # Point prediction
        rmse_y0 = np.sqrt(np.mean((y0_pred.mean(axis=1) - Y0) ** 2))
        rmse_y1 = np.sqrt(np.mean((y1_pred.mean(axis=1) - Y1) ** 2))
        rmse_ite = np.sqrt(np.mean((ite_pred.mean(axis=1) - (Y1 - Y0)) ** 2))
        # Interval prediction
        y0_upper = np.percentile(y0_pred, 100 * (1 - alpha / 2), axis=1)
        y0_lower = np.percentile(y0_pred, 100 * alpha / 2, axis=1)
        y1_upper = np.percentile(y1_pred, 100 * (1 - alpha / 2), axis=1)
        y1_lower = np.percentile(y1_pred, 100 * alpha / 2, axis=1)
        ite_upper = np.percentile(ite_pred, 100 * (1 - alpha / 2), axis=1)
        ite_lower = np.percentile(ite_pred, 100 * alpha / 2, axis=1)
        coverage_y0 = np.mean((Y0 >= y0_lower) & (Y0 <= y0_upper))
        coverage_y1 = np.mean((Y1 >= y1_lower) & (Y1 <= y1_upper))
        coverage_ite = np.mean(((Y1 - Y0) >= ite_lower) & ((Y1 - Y0) <= ite_upper))
        efficiency_y0 = np.mean((y0_upper - y0_lower))
        efficiency_y1 = np.mean((y1_upper - y1_lower))
        efficiency_ite = np.mean((ite_upper - ite_lower))
        # Distribution prediction
        crps_y0 = crps(Y0, y0_pred, return_average=True)
        crps_y1 = crps(Y1, y1_pred, return_average=True)
        crps_ite = crps(Y1 - Y0, ite_pred, return_average=True)
        ll_y0 = loglikelihood(Y0, y0_pred, return_average=True)
        ll_y1 = loglikelihood(Y1, y1_pred, return_average=True)
        ll_ite = loglikelihood(Y1 - Y0, ite_pred, return_average=True)

        # Dispersion
        dispersion_y0, p_values_y0 = calculate_dispersion(Y0, y0_pred, return_p_values=True)
        dispersion_y1, p_values_y1 = calculate_dispersion(Y1, y1_pred, return_p_values=True)
        dispersion_ite, p_values_ite = calculate_dispersion(Y1 - Y0, ite_pred, return_p_values=True)
        results = {
            "rmse_y0": rmse_y0,
            "rmse_y1": rmse_y1,
            "rmse_ite": rmse_ite,
            "coverage_y0": coverage_y0,
            "coverage_y1": coverage_y1,
            "coverage_ite": coverage_ite,
            "efficiency_y0": efficiency_y0,
            "efficiency_y1": efficiency_y1,
            "efficiency_ite": efficiency_ite,
            "crps_y0": crps_y0,
            "crps_y1": crps_y1,
            "crps_ite": crps_ite,
            "ll_y0": ll_y0,
            "ll_y1": ll_y1,
            "ll_ite": ll_ite,
            "dispersion_y0": dispersion_y0,
            "dispersion_y1": dispersion_y1,
            "dispersion_ite": dispersion_ite,
        }
        if return_p_values:
            results["p_values_y0"] = [p_values_y0]
            results["p_values_y1"] = [p_values_y1]
            results["p_values_ite"] = [p_values_ite]
        return results
