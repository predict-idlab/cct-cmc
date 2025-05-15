from copy import deepcopy

# import crepes
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from crepes_weighted import WrapProbabilisticRegressor, WrapRegressor  # type: ignore
from crepes_weighted.extras import DifficultyEstimator, binning  # type: ignore
from scipy.signal import convolve, fftconvolve  # type: ignore

# from crepes import WrapRegressor  # type: ignore
# from crepes.extras import DifficultyEstimator, binning  # type: ignore
from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore
from sklearn.model_selection import train_test_split

from src.benchmarks.fccn import FCCN
from src.benchmarks.noflite.noflite import NOFLITE
from src.metrics import crps_weighted, loglikelihood  # type: ignore


class OracleCPS:
    def __init__(
        self,
        y0_estimator,
        y1_estimator,
        ite_estimator,
        normalized_conformal=False,
        difficulty_estimator=None,
    ):
        self.y0_estimator = WrapRegressor(y0_estimator)
        self.y1_estimator = WrapRegressor(y1_estimator)
        self.ite_estimator = WrapRegressor(ite_estimator)
        self.normalized_conformal = normalized_conformal
        if self.normalized_conformal:
            if difficulty_estimator is None:
                self.de_y0 = DifficultyEstimator()
                self.de_y1 = DifficultyEstimator()
                self.de_ite = DifficultyEstimator()
            else:
                self.de_y0 = deepcopy(difficulty_estimator)
                self.de_y1 = deepcopy(difficulty_estimator)
                self.de_ite = deepcopy(difficulty_estimator)

    def fit(self, X, y0, y1):
        (
            X_train,
            X_cal,
            y0_train,
            y0_cal,
            y1_train,
            y1_cal,
        ) = train_test_split(X, y0, y1, test_size=0.5)
        ite_train = y1_train - y0_train
        ite_cal = y1_cal - y0_cal
        self.y0_estimator.fit(X_train, y0_train)
        self.y1_estimator.fit(X_train, y1_train)
        self.ite_estimator.fit(X_train, ite_train)
        if self.normalized_conformal:
            self.de_y0.fit(X_train, y0_train)
            self.de_y1.fit(X_train, y1_train)
            self.de_ite.fit(X_train, ite_train)
            sigmas0 = self.de_y0.apply(X_cal)
            sigmas1 = self.de_y1.apply(X_cal)
            sigmas_ite = self.de_ite.apply(X_cal)
        else:
            sigmas0 = None
            sigmas1 = None
            sigmas_ite = None
        self.y0_estimator.calibrate(X_cal, y0_cal, sigmas=sigmas0, cps=True)
        self.y1_estimator.calibrate(X_cal, y1_cal, sigmas=sigmas1, cps=True)
        self.ite_estimator.calibrate(X_cal, ite_cal, sigmas=sigmas_ite, cps=True)

    def predict(self, X):
        return self.y1_estimator.predict(X) - self.y0_estimator.predict(X)

    def predict_p_value(self, X, y):
        if self.normalized_conformal:
            sigmas = self.de_ite.apply(X)
        else:
            sigmas = None
        return self.ite_estimator.predict_cps(X, sigmas=sigmas, y=y, return_cpds=False)

    def predict_cps(self, X, return_cpds=True, lower_percentiles=None):
        if self.normalized_conformal:
            sigmas = self.de_ite.apply(X)
        else:
            sigmas = None
        return self.ite_estimator.predict_cps(
            X, sigmas=sigmas, return_cpds=return_cpds, lower_percentiles=lower_percentiles
        )

    def predict_int(self, X, confidence=0.95):
        if self.normalized_conformal:
            sigmas = self.de_ite.apply(X)
        else:
            sigmas = None
        return self.ite_estimator.predict_int(X, sigmas=sigmas, confidence=confidence)

    def predict_y0(self, X):
        return self.y0_estimator.predict(X)

    def predict_y1(self, X):
        return self.y1_estimator.predict(X)

    def predict_cps_y0(self, X, return_cpds=True, lower_percentiles=None):
        if self.normalized_conformal:
            sigmas = self.de_y0.apply(X)
        else:
            sigmas = None
        return self.y0_estimator.predict_cps(
            X, sigmas=sigmas, return_cpds=return_cpds, lower_percentiles=lower_percentiles
        )

    def predict_cps_y1(self, X, return_cpds=True, lower_percentiles=None):
        if self.normalized_conformal:
            sigmas = self.de_y1.apply(X)
        else:
            sigmas = None
        return self.y1_estimator.predict_cps(
            X, sigmas=sigmas, return_cpds=return_cpds, lower_percentiles=lower_percentiles
        )

    def predict_int_y0(self, X, confidence=0.95):
        if self.normalized_conformal:
            sigmas = self.de_y0.apply(X)
        else:
            sigmas = None
        return self.y0_estimator.predict_int(X, sigmas=sigmas, confidence=confidence)

    def predict_int_y1(self, X, confidence=0.95):
        if self.normalized_conformal:
            sigmas = self.de_y1.apply(X)
        else:
            sigmas = None
        return self.y1_estimator.predict_int(X, sigmas=sigmas, confidence=confidence)

    def predict_p_value_y0(self, X, y):
        if self.normalized_conformal:
            sigmas = self.de_y0.apply(X)
        else:
            sigmas = None
        return self.y0_estimator.predict_cps(X, sigmas=sigmas, y=y, return_cpds=False)

    def predict_p_value_y1(self, X, y):
        if self.normalized_conformal:
            sigmas = self.de_y1.apply(X)
        else:
            sigmas = None
        return self.y1_estimator.predict_cps(X, sigmas=sigmas, y=y, return_cpds=False)

    def predict_p_value_ite(self, X, y):
        if self.normalized_conformal:
            sigmas = self.de_ite.apply(X)
        else:
            sigmas = None
        return self.ite_estimator.predict_cps(X, sigmas=sigmas, y=y, return_cpds=False)

    def evaluate(self, X, Y0, Y1, alpha=0.05, return_p_values=False):
        """
        Evaluates model
        """
        # Point prediction - compute RMSE incrementally
        y0_sq_err_sum = 0
        y1_sq_err_sum = 0
        ite_sq_err_sum = 0
        n = len(X)

        # Process in batches to reduce memory usage
        batch_size = 1000
        for i in range(0, n, batch_size):
            batch_slice = slice(i, min(i + batch_size, n))
            X_batch = X[batch_slice]
            Y0_batch = Y0[batch_slice]
            Y1_batch = Y1[batch_slice]

            y0_pred_batch = self.predict_y0(X_batch)
            y1_pred_batch = self.predict_y1(X_batch)

            y0_sq_err_sum += np.sum((y0_pred_batch - Y0_batch) ** 2)
            y1_sq_err_sum += np.sum((y1_pred_batch - Y1_batch) ** 2)
            ite_sq_err_sum += np.sum((y1_pred_batch - y0_pred_batch - (Y1_batch - Y0_batch)) ** 2)

            # Free memory explicitly
            del y0_pred_batch, y1_pred_batch

        rmse_y0 = np.sqrt(y0_sq_err_sum / n)
        rmse_y1 = np.sqrt(y1_sq_err_sum / n)
        rmse_ite = np.sqrt(ite_sq_err_sum / n)

        # Interval prediction
        # Process in batches for memory efficiency
        batch_size = 1000
        coverage_y0_sum = 0
        coverage_y1_sum = 0
        coverage_ite_sum = 0
        efficiency_y0_sum = 0
        efficiency_y1_sum = 0
        efficiency_ite_sum = 0

        for i in range(0, len(X), batch_size):
            batch_slice = slice(i, min(i + batch_size, len(X)))
            X_batch = X[batch_slice]
            Y0_batch = Y0[batch_slice]
            Y1_batch = Y1[batch_slice]

            # Get intervals for this batch
            y0_int = self.predict_int_y0(X_batch, confidence=1 - alpha)
            y1_int = self.predict_int_y1(X_batch, confidence=1 - alpha)
            ite_int = self.predict_int(X_batch, confidence=1 - alpha)

            # Update running sums
            coverage_y0_sum += np.sum((Y0_batch >= y0_int[:, 0]) & (Y0_batch <= y0_int[:, 1]))
            coverage_y1_sum += np.sum((Y1_batch >= y1_int[:, 0]) & (Y1_batch <= y1_int[:, 1]))
            coverage_ite_sum += np.sum(
                ((Y1_batch - Y0_batch) >= ite_int[:, 0]) & ((Y1_batch - Y0_batch) <= ite_int[:, 1])
            )
            efficiency_y0_sum += np.sum(y0_int[:, 1] - y0_int[:, 0])
            efficiency_y1_sum += np.sum(y1_int[:, 1] - y1_int[:, 0])
            efficiency_ite_sum += np.sum(ite_int[:, 1] - ite_int[:, 0])

            # Free memory
            del y0_int, y1_int, ite_int

        coverage_y0 = coverage_y0_sum / n
        coverage_y1 = coverage_y1_sum / n
        coverage_ite = coverage_ite_sum / n
        efficiency_y0 = efficiency_y0_sum / n
        efficiency_y1 = efficiency_y1_sum / n
        efficiency_ite = efficiency_ite_sum / n
        # Distribution prediction
        batch_size = 1000
        crps_y0 = np.zeros(n)
        crps_y1 = np.zeros(n)
        crps_ite = np.zeros(n)
        ll_y0 = np.zeros(n)
        ll_y1 = np.zeros(n)
        ll_ite = np.zeros(n)

        for i in range(0, len(X), batch_size):
            batch_slice = slice(i, min(i + batch_size, len(X)))
            X_batch = X[batch_slice]
            Y0_batch = Y0[batch_slice]
            Y1_batch = Y1[batch_slice]
            y0_pred_cps = self.predict_cps_y0(X_batch)
            y1_pred_cps = self.predict_cps_y1(X_batch)
            ite_pred_cps = self.predict_cps(X_batch)
            crps_y0[batch_slice] = crps_weighted(
                Y0_batch, y0_pred_cps[:, :, 0], weights=y0_pred_cps[:, :, 1], return_average=False
            )
            crps_y1[batch_slice] = crps_weighted(
                Y1_batch, y1_pred_cps[:, :, 0], weights=y1_pred_cps[:, :, 1], return_average=False
            )
            crps_ite[batch_slice] = crps_weighted(
                Y1_batch - Y0_batch,
                ite_pred_cps[:, :, 0],
                weights=ite_pred_cps[:, :, 1],
                return_average=False,
            )
            ll_y0[batch_slice] = loglikelihood(
                Y0_batch, y0_pred_cps[:, :, 0], weights=y0_pred_cps[:, :, 1], return_average=False
            )
            ll_y1[batch_slice] = loglikelihood(
                Y1_batch, y1_pred_cps[:, :, 0], weights=y1_pred_cps[:, :, 1], return_average=False
            )
            ll_ite[batch_slice] = loglikelihood(
                Y1_batch - Y0_batch,
                ite_pred_cps[:, :, 0],
                weights=ite_pred_cps[:, :, 1],
                return_average=False,
            )
            # Free memory
            del y0_pred_cps, y1_pred_cps, ite_pred_cps
        crps_y0 = np.mean(crps_y0)
        crps_y1 = np.mean(crps_y1)
        crps_ite = np.mean(crps_ite)
        ll_y0 = np.mean(ll_y0)
        ll_y1 = np.mean(ll_y1)
        ll_ite = np.mean(ll_ite)

        p_values_y0 = self.predict_p_value_y0(X, Y0)
        p_values_y1 = self.predict_p_value_y1(X, Y1)
        p_values_ite = self.predict_p_value(X, Y1 - Y0)

        dispersion_y0 = np.var(
            p_values_y0
        )  # dispersion of p-values, ideally 1 / 12 for uniform distribution
        dispersion_y1 = np.var(p_values_y1)
        dispersion_ite = np.var(p_values_ite)
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


class CCT_Learner:
    def __init__(
        self, y0_estimator, y1_estimator, normalized_conformal=False, difficulty_estimator=None
    ):
        self.y0_estimator = WrapRegressor(y0_estimator)
        self.y1_estimator = WrapRegressor(y1_estimator)
        self.normalized_conformal = normalized_conformal
        if self.normalized_conformal:
            if difficulty_estimator is None:
                self.de_y0 = DifficultyEstimator()
                self.de_y1 = DifficultyEstimator()
            else:
                self.de_y0 = deepcopy(difficulty_estimator)
                self.de_y1 = deepcopy(difficulty_estimator)

    def fit(self, X, y, W, p=None):
        self.y_min = np.min(y)
        self.y_max = np.max(y)
        if p is None:
            p = 0.5 / np.ones_like(W)
        (
            X_train,
            X_cal,
            y_train,
            y_cal,
            W_train,
            W_cal,
            p_train,
            p_cal,
        ) = train_test_split(X, y, W, p, test_size=0.25)
        self.y0_estimator.fit(X_train[W_train == 0], y_train[W_train == 0])
        self.y1_estimator.fit(X_train[W_train == 1], y_train[W_train == 1])
        if self.normalized_conformal:
            self.de_y0.fit(X_train[W_train == 0], y_train[W_train == 0])
            self.de_y1.fit(X_train[W_train == 1], y_train[W_train == 1])
            sigmas0 = self.de_y0.apply(X_cal[W_cal == 0])
            sigmas1 = self.de_y1.apply(X_cal[W_cal == 1])
        else:
            sigmas0 = None
            sigmas1 = None
        self.y0_estimator.calibrate(
            X_cal[W_cal == 0],
            y_cal[W_cal == 0],
            sigmas=sigmas0,
            likelihood_ratios=1 / (1 - p_cal[W_cal == 0]),
            cps=True,
        )
        self.y1_estimator.calibrate(
            X_cal[W_cal == 1],
            y_cal[W_cal == 1],
            sigmas=sigmas1,
            likelihood_ratios=1 / p_cal[W_cal == 1],
            cps=True,
        )

    def predict(self, X, p=None, conformal_mean=False):
        if p is None:
            p = 0.5 / np.ones(len(X))
        if conformal_mean:
            cpds = self.predict_cps(X, p)
            return np.average(
                cpds[:, :, 0],
                weights=cpds[:, :, 1] / np.sum(cpds[:, :, 1], axis=-1)[:, :, None],
                axis=-1,
            )
        return self.y1_estimator.predict(X) - self.y0_estimator.predict(X)

    def predict_p_value(self, X, y, p=None):
        if p is None:
            p = 0.5 / np.ones_like(y)
        y = y.reshape(-1, 1)
        cpds = self.predict_cps(X, p)
        phi = np.random.uniform(low=0, high=1, size=len(y))
        p_w_test = 1 - np.sum(cpds[:, :, 1], axis=1)
        p_values = p_w_test * phi
        p_values += np.sum(cpds[:, :, 1] * (cpds[:, :, 0] < y), axis=1)
        p_values += np.sum(cpds[:, :, 1] * phi.reshape((-1, 1)) * (cpds[:, :, 0] == y), axis=1)
        return p_values

    def predict_cps(self, X, p=None):
        if p is None:
            p = 0.5 / np.ones(len(X))
        if self.normalized_conformal:
            sigmas0 = self.de_y0.apply(X)
            sigmas1 = self.de_y1.apply(X)
        else:
            sigmas0 = None
            sigmas1 = None
        y0_hat = self.y0_estimator.predict_cps(
            X, sigmas=sigmas0, likelihood_ratios=1 / 1 - p, return_cpds=True
        )
        y1_hat = self.y1_estimator.predict_cps(
            X, sigmas=sigmas1, likelihood_ratios=1 / p, return_cpds=True
        )

        ite_values = np.stack(
            [
                y1_hat[:, i, 0] - y0_hat[:, j, 0]
                for i in range(y1_hat.shape[1])
                for j in range(y0_hat.shape[1])
            ],
            axis=-1,
        )
        ite_prob = np.stack(
            [
                y1_hat[:, i, 1] * y0_hat[:, j, 1]
                for i in range(y1_hat.shape[1])
                for j in range(y0_hat.shape[1])
            ],
            axis=-1,
        )
        cpds = np.stack([ite_values, ite_prob], axis=-1)
        for i in range(len(X)):
            sort_idx = np.argsort(cpds[i, :, 0])
            cpds[i] = cpds[i, sort_idx]
        return cpds

    def predict_int(self, X, confidence=0.95, p=None):
        cpds = self.predict_cps(X, p)
        lower_bound = (1 - confidence) / 2
        upper_bound = 1 - lower_bound
        lower_indexes = (
            (
                np.cumsum(cpds[:, :, 1], axis=1)
                + (1 - np.sum(cpds[:, :, 1], axis=1)).reshape((-1, 1))
            )
            < lower_bound
        ).argmin(axis=1) - 1
        upper_indexes = (
            np.cumsum(
                np.concatenate(
                    (cpds[:, :, 1], (1 - np.sum(cpds[:, :, 1], axis=1)).reshape((-1, 1))),
                    axis=1,
                ),
                axis=1,
            )
            >= upper_bound
        ).argmax(axis=1)
        too_high_indexes, too_low_indexes = [], []
        if np.any(lower_indexes < 0):
            too_low_indexes = lower_indexes < 0
            lower_indexes[too_low_indexes] = 0
        if np.any(upper_indexes > (len(cpds[0]) - 1)):
            too_high_indexes = upper_indexes > (len(cpds[0]) - 1)
            upper_indexes[too_high_indexes] = len(cpds[0]) - 1
        lower_bounds = np.array([cpds[i, lower_indexes[i], 0] for i in range(len(X))])
        upper_bounds = np.array([cpds[i, upper_indexes[i], 0] for i in range(len(X))])
        if np.any(too_low_indexes):
            lower_bounds[too_low_indexes] = -np.inf
        if np.any(too_high_indexes):
            upper_bounds[too_high_indexes] = np.inf
        return np.stack([lower_bounds, upper_bounds], axis=1)

    def predict_y0(self, X, p=None, conformal_mean=False):
        if p is None:
            p = 0.5 / np.ones(len(X))
        if conformal_mean:
            cpds = self.predict_cps_y0(X, p)
            return np.average(
                cpds[:, :, 0],
                weights=cpds[:, :, 1] / np.sum(cpds[:, :, 1], axis=-1)[:, :, None],
                axis=-1,
            )
        return self.y0_estimator.predict(X)

    def predict_y1(self, X, p=None, conformal_mean=False):
        if p is None:
            p = 0.5 / np.ones(len(X))
        if conformal_mean:
            cpds = self.predict_cps_y1(X, p)
            return np.average(
                cpds[:, :, 0],
                weights=cpds[:, :, 1] / np.sum(cpds[:, :, 1], axis=-1)[:, :, None],
                axis=-1,
            )
        return self.y1_estimator.predict(X)

    def predict_p_value_y0(self, X, y, p=None):
        if p is None:
            p = 0.5 / np.ones_like(y)
        if self.normalized_conformal:
            sigmas = self.de_y0.apply(X)
        else:
            sigmas = None
        return self.y0_estimator.predict_cps(
            X, y=y, sigmas=sigmas, likelihood_ratios=1 / (1 - p), return_cpds=False
        )

    def predict_p_value_y1(self, X, y, p=None):
        if p is None:
            p = 0.5 / np.ones_like(y)
        if self.normalized_conformal:
            sigmas = self.de_y1.apply(X)
        else:
            sigmas = None
        return self.y1_estimator.predict_cps(
            X,
            y=y,
            sigmas=sigmas,
            likelihood_ratios=1 / p,
            return_cpds=False,
        )

    def predict_cps_y0(self, X, p=None):
        if p is None:
            p = 0.5 / np.ones(len(X))
        if self.normalized_conformal:
            sigmas = self.de_y0.apply(X)
        else:
            sigmas = None
        return self.y0_estimator.predict_cps(
            X, sigmas=sigmas, likelihood_ratios=1 / (1 - p), return_cpds=True
        )

    def predict_cps_y1(self, X, p=None):
        if p is None:
            p = 0.5 / np.ones(len(X))
        if self.normalized_conformal:
            sigmas = self.de_y1.apply(X)
        else:
            sigmas = None
        return self.y1_estimator.predict_cps(
            X, sigmas=sigmas, likelihood_ratios=1 / p, return_cpds=True
        )

    def predict_int_y0(self, X, confidence=0.95, p=None):
        if p is None:
            p = 0.5 / np.ones(len(X))
        if self.normalized_conformal:
            sigmas = self.de_y0.apply(X)
        else:
            sigmas = None
        return self.y0_estimator.predict_int(
            X, sigmas=sigmas, likelihood_ratios=1 / (1 - p), confidence=confidence
        )

    def predict_int_y1(self, X, confidence=0.95, p=None):
        if p is None:
            p = 0.5 / np.ones(len(X))
        if self.normalized_conformal:
            sigmas = self.de_y1.apply(X)
        else:
            sigmas = None
        return self.y1_estimator.predict_int(
            X, sigmas=sigmas, likelihood_ratios=1 / p, confidence=confidence
        )

    def evaluate(self, X, Y0, Y1, p=None, alpha=0.05, return_p_values=False):
        """
        Evaluates model
        """
        # Point prediction - compute RMSE incrementally
        y0_sq_err_sum = 0
        y1_sq_err_sum = 0
        ite_sq_err_sum = 0
        n = len(X)

        # Process in batches to reduce memory usage
        batch_size = 1000
        for i in range(0, n, batch_size):
            batch_slice = slice(i, min(i + batch_size, n))
            X_batch = X[batch_slice]
            Y0_batch = Y0[batch_slice]
            Y1_batch = Y1[batch_slice]
            p_batch = p[batch_slice] if p is not None else None

            y0_pred_batch = self.predict_y0(X_batch, p_batch)
            y1_pred_batch = self.predict_y1(X_batch, p_batch)

            y0_sq_err_sum += np.sum((y0_pred_batch - Y0_batch) ** 2)
            y1_sq_err_sum += np.sum((y1_pred_batch - Y1_batch) ** 2)
            ite_sq_err_sum += np.sum((y1_pred_batch - y0_pred_batch - (Y1_batch - Y0_batch)) ** 2)

            # Free memory explicitly
            del y0_pred_batch, y1_pred_batch

        rmse_y0 = np.sqrt(y0_sq_err_sum / n)
        rmse_y1 = np.sqrt(y1_sq_err_sum / n)
        rmse_ite = np.sqrt(ite_sq_err_sum / n)

        # Interval prediction
        # Process in batches for memory efficiency
        batch_size = 1000
        coverage_y0_sum = 0
        coverage_y1_sum = 0
        coverage_ite_sum = 0
        efficiency_y0_sum = 0
        efficiency_y1_sum = 0
        efficiency_ite_sum = 0

        for i in range(0, len(X), batch_size):
            batch_slice = slice(i, min(i + batch_size, len(X)))
            X_batch = X[batch_slice]
            Y0_batch = Y0[batch_slice]
            Y1_batch = Y1[batch_slice]
            p_batch = p[batch_slice] if p is not None else None

            # Get intervals for this batch
            y0_int = self.predict_int_y0(X_batch, p=p_batch, confidence=1 - alpha)
            y1_int = self.predict_int_y1(X_batch, p=p_batch, confidence=1 - alpha)
            ite_int = self.predict_int(X_batch, p=p_batch, confidence=1 - alpha)

            # Update running sums
            coverage_y0_sum += np.sum((Y0_batch >= y0_int[:, 0]) & (Y0_batch <= y0_int[:, 1]))
            coverage_y1_sum += np.sum((Y1_batch >= y1_int[:, 0]) & (Y1_batch <= y1_int[:, 1]))
            coverage_ite_sum += np.sum(
                ((Y1_batch - Y0_batch) >= ite_int[:, 0]) & ((Y1_batch - Y0_batch) <= ite_int[:, 1])
            )
            efficiency_y0_sum += np.sum(y0_int[:, 1] - y0_int[:, 0])
            efficiency_y1_sum += np.sum(y1_int[:, 1] - y1_int[:, 0])
            efficiency_ite_sum += np.sum(ite_int[:, 1] - ite_int[:, 0])

            # Free memory
            del y0_int, y1_int, ite_int

        coverage_y0 = coverage_y0_sum / n
        coverage_y1 = coverage_y1_sum / n
        coverage_ite = coverage_ite_sum / n
        efficiency_y0 = efficiency_y0_sum / n
        efficiency_y1 = efficiency_y1_sum / n
        efficiency_ite = efficiency_ite_sum / n
        # Distribution prediction
        batch_size = 1000
        crps_y0 = np.zeros(n)
        crps_y1 = np.zeros(n)
        crps_ite = np.zeros(n)
        ll_y0 = np.zeros(n)
        ll_y1 = np.zeros(n)
        ll_ite = np.zeros(n)

        for i in range(0, len(X), batch_size):
            batch_slice = slice(i, min(i + batch_size, len(X)))
            X_batch = X[batch_slice]
            Y0_batch = Y0[batch_slice]
            Y1_batch = Y1[batch_slice]
            p_batch = p[batch_slice] if p is not None else None
            y0_pred_cps = self.predict_cps_y0(X_batch, p_batch)
            y1_pred_cps = self.predict_cps_y1(X_batch, p_batch)
            ite_pred_cps = self.predict_cps(X_batch, p_batch)
            crps_y0[batch_slice] = crps_weighted(
                Y0_batch, y0_pred_cps[:, :, 0], weights=y0_pred_cps[:, :, 1], return_average=False
            )
            crps_y1[batch_slice] = crps_weighted(
                Y1_batch, y1_pred_cps[:, :, 0], weights=y1_pred_cps[:, :, 1], return_average=False
            )
            crps_ite[batch_slice] = crps_weighted(
                Y1_batch - Y0_batch,
                ite_pred_cps[:, :, 0],
                weights=ite_pred_cps[:, :, 1],
                return_average=False,
            )
            ll_y0[batch_slice] = loglikelihood(
                Y0_batch, y0_pred_cps[:, :, 0], weights=y0_pred_cps[:, :, 1], return_average=False
            )
            ll_y1[batch_slice] = loglikelihood(
                Y1_batch, y1_pred_cps[:, :, 0], weights=y1_pred_cps[:, :, 1], return_average=False
            )
            ll_ite[batch_slice] = loglikelihood(
                Y1_batch - Y0_batch,
                ite_pred_cps[:, :, 0],
                weights=ite_pred_cps[:, :, 1],
                return_average=False,
            )
            # Free memory
            del y0_pred_cps, y1_pred_cps, ite_pred_cps
        crps_y0 = np.mean(crps_y0)
        crps_y1 = np.mean(crps_y1)
        crps_ite = np.mean(crps_ite)
        ll_y0 = np.mean(ll_y0)
        ll_y1 = np.mean(ll_y1)
        ll_ite = np.mean(ll_ite)

        p_values_y0 = self.predict_p_value_y0(X, Y0, p)
        p_values_y1 = self.predict_p_value_y1(X, Y1, p)
        p_values_ite = self.predict_p_value(X, Y1 - Y0, p)

        dispersion_y0 = np.var(
            p_values_y0
        )  # dispersion of p-values, ideally 1 / 12 for uniform distribution
        dispersion_y1 = np.var(p_values_y1)
        dispersion_ite = np.var(p_values_ite)
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


class FCCN_countefactual:
    def __init__(self, fccn_model, treated=False):
        self.fccn_model = fccn_model
        self.treated = treated

    def fit(self, X, y):
        # model is already trained
        pass

    def predict(self, X, n_samples=1000):
        y0_samples, y1_samples = self.fccn_model.sample_y0_y1(X, n_samples)
        if self.treated:
            pred_cdf = np.sort(y1_samples, axis=1)
        else:
            pred_cdf = np.sort(y0_samples, axis=1)
        return pred_cdf


class CCT_Learner_FCCN(CCT_Learner):
    def __init__(
        self, input_size, hidden_size=25, alpha=5e-3, beta=1e-5, iters=20000, batch_size=128
    ):
        self.fccn_model = FCCN(input_size, hidden_size, alpha, beta)
        self.normalized_conformal = False
        self.fccn_iters = iters
        self.fccn_batch_size = batch_size

    def fit(self, X, y, W, p=None):
        self.y_min = np.min(y)
        self.y_max = np.max(y)
        if p is None:
            p = 0.5 / np.ones_like(W)
        (
            X_train,
            X_cal,
            y_train,
            y_cal,
            W_train,
            W_cal,
            p_train,
            p_cal,
        ) = train_test_split(X, y, W, p, test_size=0.25)
        self.fccn_model.train(
            X_train, y_train, W_train, iters=self.fccn_iters, batch_size=self.fccn_batch_size
        )
        self.y0_estimator = WrapProbabilisticRegressor(
            FCCN_countefactual(self.fccn_model, treated=False)
        )
        self.y1_estimator = WrapProbabilisticRegressor(
            FCCN_countefactual(self.fccn_model, treated=True)
        )
        self.y0_estimator.fit(X_train, y_train)
        self.y1_estimator.fit(X_train, y_train)
        self.y0_estimator.calibrate(
            X_cal[W_cal == 0], y_cal[W_cal == 0], likelihood_ratios=1 / (1 - p_cal[W_cal == 0])
        )
        self.y1_estimator.calibrate(
            X_cal[W_cal == 1], y_cal[W_cal == 1], likelihood_ratios=1 / p_cal[W_cal == 1]
        )


class NOFLITE_countefactual:
    def __init__(self, noflite_model, treated=False):
        self.noflite_model = noflite_model
        self.treated = treated

    def fit(self, X, y):
        # model is already trained
        pass

    def predict(self, X, n_samples=1000):
        y0_samples, y1_samples = self.noflite_model.sample_y0_y1(X, n_samples)
        if self.treated:
            pred_cdf = np.sort(y1_samples, axis=1)
        else:
            pred_cdf = np.sort(y0_samples, axis=1)
        return pred_cdf


class CCT_Learner_NOFLITE(CCT_Learner):
    def __init__(self, params):
        self.noflite_model = NOFLITE(params=params)
        self.normalized_conformal = False

    def fit(self, X, y, W, p=None):
        self.y_min = np.min(y)
        self.y_max = np.max(y)
        if p is None:
            p = 0.5 / np.ones_like(W)
        (
            X_train,
            X_cal,
            y_train,
            y_cal,
            W_train,
            W_cal,
            p_train,
            p_cal,
        ) = train_test_split(X, y, W, p, test_size=0.25)
        self.noflite_model.fit(X_train, y_train, W_train)
        self.y0_estimator = WrapProbabilisticRegressor(
            NOFLITE_countefactual(self.noflite_model, treated=False)
        )
        self.y1_estimator = WrapProbabilisticRegressor(
            NOFLITE_countefactual(self.noflite_model, treated=True)
        )
        self.y0_estimator.fit(X_train, y_train)
        self.y1_estimator.fit(X_train, y_train)
        self.y0_estimator.calibrate(
            X_cal[W_cal == 0], y_cal[W_cal == 0], likelihood_ratios=1 / (1 - p_cal[W_cal == 0])
        )
        self.y1_estimator.calibrate(
            X_cal[W_cal == 1], y_cal[W_cal == 1], likelihood_ratios=1 / p_cal[W_cal == 1]
        )


class CMC_T_Learner:
    def __init__(
        self,
        y0_estimator,
        y1_estimator,
        normalized_conformal=False,
        difficulty_estimator=None,
        MC_samples=100,
        pseudo_MC=True,
        max_min_y=True,
    ):
        self.y0_estimator = WrapRegressor(y0_estimator)
        self.y1_estimator = WrapRegressor(y1_estimator)
        self.normalized_conformal = normalized_conformal
        self.MC_samples = MC_samples
        self.pseudo_MC = pseudo_MC
        self.max_min_y = max_min_y
        if self.normalized_conformal:
            if difficulty_estimator is None:
                self.de_y0 = DifficultyEstimator()
                self.de_y1 = DifficultyEstimator()
                self.de_ite = DifficultyEstimator()
            else:
                self.de_y0 = deepcopy(difficulty_estimator)
                self.de_y1 = deepcopy(difficulty_estimator)
                self.de_ite = deepcopy(difficulty_estimator)

    def fit(self, X, y, W, p=None):
        # Split into calibration and training set for nuisance estimators
        # and for the final cate estimator
        if self.max_min_y:
            self.y_min = np.min(y)
            self.y_max = np.max(y)
        else:
            self.y_min = None
            self.y_max = None
        if p is None:
            p = 0.5 / np.ones_like(W)
        (
            X_train_nuisance,
            X_cal,
            y_train_nuisance,
            y_cal,
            W_train_nuisance,
            W_cal,
            p_train_nuisance,
            p_cal,
        ) = train_test_split(X, y, W, p, test_size=0.25)
        (
            X_prop_train_nuisance,
            X_cal_nuisance,
            y_prop_train_nuisance,
            y_cal_nuisance,
            W_prop_train_nuisance,
            W_cal_nuisance,
            p_prop_train_nuisance,
            p_cal_nuisance,
        ) = train_test_split(
            X_train_nuisance,
            y_train_nuisance,
            W_train_nuisance,
            p_train_nuisance,
            test_size=0.25,
        )

        self.y0_estimator.fit(
            X_prop_train_nuisance[W_prop_train_nuisance == 0],
            y_prop_train_nuisance[W_prop_train_nuisance == 0],
        )
        self.y1_estimator.fit(
            X_prop_train_nuisance[W_prop_train_nuisance == 1],
            y_prop_train_nuisance[W_prop_train_nuisance == 1],
        )
        # Fit difficulty estimators if normalized conformal
        if self.normalized_conformal:
            self.de_y0.fit(
                X_prop_train_nuisance[W_prop_train_nuisance == 0],
                y_prop_train_nuisance[W_prop_train_nuisance == 0],
            )
            self.de_y1.fit(
                X_prop_train_nuisance[W_prop_train_nuisance == 1],
                y_prop_train_nuisance[W_prop_train_nuisance == 1],
            )
            sigmas0 = self.de_y0.apply(X_cal_nuisance)[W_cal_nuisance == 0]
            sigmas1 = self.de_y1.apply(X_cal_nuisance)[W_cal_nuisance == 1]
        else:
            sigmas0 = None
            sigmas1 = None
        # Calibrate nuisance estimators
        if self.pseudo_MC:
            lr_cal_nuisance = np.zeros_like(p_cal_nuisance)
            lr_cal_nuisance[W_cal_nuisance == 0] = p_cal_nuisance[W_cal_nuisance == 0] / (
                1 - p_cal_nuisance[W_cal_nuisance == 0]
            )
            lr_cal_nuisance[W_cal_nuisance == 1] = (
                p_cal_nuisance[W_cal_nuisance == 1] / p_cal_nuisance[W_cal_nuisance == 1]
            )
        else:
            lr_cal_nuisance = np.zeros_like(p_cal_nuisance)
            lr_cal_nuisance[W_cal_nuisance == 0] = 1 / (1 - p_cal_nuisance[W_cal_nuisance == 0])
            lr_cal_nuisance[W_cal_nuisance == 1] = 1 / p_cal_nuisance[W_cal_nuisance == 1]
        self.y0_estimator.calibrate(
            X_cal_nuisance[W_cal_nuisance == 0],
            y_cal_nuisance[W_cal_nuisance == 0],
            sigmas=sigmas0,
            likelihood_ratios=lr_cal_nuisance[W_cal_nuisance == 0],
            cps=True,
        )
        self.y1_estimator.calibrate(
            X_cal_nuisance[W_cal_nuisance == 1],
            y_cal_nuisance[W_cal_nuisance == 1],
            sigmas=sigmas1,
            likelihood_ratios=lr_cal_nuisance[W_cal_nuisance == 1],
            cps=True,
        )
        if self.pseudo_MC:
            generator = self.generate_pseudo_MC_ite_samples
        else:
            generator = self.generate_MC_ite_samples
        ite_MC_cal = generator(X_cal, y_cal, W_cal, p_cal)
        ite_MC_train = generator(
            X_train_nuisance, y_train_nuisance, W_train_nuisance, p_train_nuisance
        )
        self.ite_estimator = WrapRegressor(self.IteEstimator(self.y0_estimator, self.y1_estimator))
        self.ite_estimator.fit(X_train_nuisance.repeat(self.MC_samples, axis=0), ite_MC_train)

        if self.normalized_conformal:
            self.de_ite.fit(X_train_nuisance.repeat(self.MC_samples, axis=0), ite_MC_train)
            sigmas_ite_cal = self.de_ite.apply(X_cal.repeat(self.MC_samples, axis=0))
        else:
            sigmas_ite_cal = None
        self.ite_estimator.calibrate(
            X_cal.repeat(self.MC_samples, axis=0), ite_MC_cal, sigmas=sigmas_ite_cal, cps=True
        )

    def predict(self, X, conformal_mean=False):
        return self.ite_estimator.predict(X)

    def predict_p_value(self, X, y):
        if self.normalized_conformal:
            sigmas = self.de_ite.apply(X)
        else:
            sigmas = None
        return self.ite_estimator.predict_cps(X, sigmas=sigmas, y=y, return_cpds=False)

    def predict_p_value_y0(self, X, y, p):
        if self.normalized_conformal:
            sigmas = self.de_y0.apply(X)
        else:
            sigmas = None
        if p is None:
            p = 0.5 / np.ones(len(X))
        return self.y0_estimator.predict_cps(
            X,
            y=y,
            sigmas=sigmas,
            likelihood_ratios=1 / (1 - p),
            return_cpds=False,
        )

    def predict_p_value_y1(self, X, y, p):
        if self.normalized_conformal:
            sigmas = self.de_y1.apply(X)
        else:
            sigmas = None
        if p is None:
            p = 0.5 / np.ones(len(X))
        return self.y1_estimator.predict_cps(
            X,
            y=y,
            sigmas=sigmas,
            likelihood_ratios=1 / p,
            return_cpds=False,
        )

    def predict_cps(self, X, return_cpds=True, lower_percentiles=None):
        if self.normalized_conformal:
            sigmas = self.de_ite.apply(X)
        else:
            sigmas = None
        return self.ite_estimator.predict_cps(
            X,
            sigmas=sigmas,
            return_cpds=return_cpds,
            lower_percentiles=lower_percentiles,
        )

    def predict_int(self, X, confidence=0.95):
        if self.normalized_conformal:
            sigmas = self.de_ite.apply(X)
        else:
            sigmas = None
        return self.ite_estimator.predict_int(X, sigmas=sigmas, confidence=confidence)

    def predict_y0(self, X):
        return self.y0_estimator.predict(X)

    def predict_y1(self, X):
        return self.y1_estimator.predict(X)

    def predict_cps_y0(self, X, p=None, return_cpds=True, lower_percentiles=None):
        if self.normalized_conformal:
            sigmas = self.de_y0.apply(X)
        else:
            sigmas = None
        if p is None:
            p = 0.5 / np.ones(len(X))
        return self.y0_estimator.predict_cps(
            X,
            sigmas=sigmas,
            likelihood_ratios=1 / (1 - p),
            return_cpds=return_cpds,
            lower_percentiles=lower_percentiles,
            y_min=self.y_min,
            y_max=self.y_max,
        )

    def predict_cps_y1(self, X, p=None, return_cpds=True, lower_percentiles=None):
        if self.normalized_conformal:
            sigmas = self.de_y1.apply(X)
        else:
            sigmas = None
        if p is None:
            p = 0.5 / np.ones(len(X))
        return self.y1_estimator.predict_cps(
            X,
            sigmas=sigmas,
            likelihood_ratios=1 / p,
            return_cpds=return_cpds,
            lower_percentiles=lower_percentiles,
            y_min=self.y_min,
            y_max=self.y_max,
        )

    def predict_int_y0(self, X, p=None, confidence=0.95):
        if self.normalized_conformal:
            sigmas = self.de_y0.apply(X)
        else:
            sigmas = None
        if p is None:
            p = 0.5 / np.ones(len(X))
        return self.y0_estimator.predict_int(
            X,
            sigmas=sigmas,
            likelihood_ratios=1 / (1 - p),
            confidence=confidence,
            y_min=self.y_min,
            y_max=self.y_max,
        )

    def predict_int_y1(self, X, p=None, confidence=0.95):
        if self.normalized_conformal:
            sigmas = self.de_y1.apply(X)
        else:
            sigmas = None
        if p is None:
            p = 0.5 / np.ones(len(X))
        return self.y1_estimator.predict_int(
            X,
            sigmas=sigmas,
            likelihood_ratios=1 / p,
            confidence=confidence,
            y_min=self.y_min,
            y_max=self.y_max,
        )

    def generate_pseudo_MC_ite_samples(self, X, y, W, p):
        """
        Generate Pseudo Monte Carlo sample of ITEs
        """
        ite_MC = np.zeros(len(X) * self.MC_samples)
        if self.normalized_conformal:
            sigmas0 = self.de_y0.apply(X)
            sigmas1 = self.de_y1.apply(X)

        for i in range(len(X)):
            if W[i] == 0:
                u1 = np.random.uniform(
                    low=0,
                    high=100,
                    size=self.MC_samples,
                )
                if self.normalized_conformal:
                    sigma1 = sigmas1[i].reshape(-1, 1)
                else:
                    sigma1 = None
                ite_MC[i * self.MC_samples : (i + 1) * self.MC_samples] = (
                    self.y1_estimator.predict_cps(
                        X[i].reshape(1, -1),
                        sigmas=sigma1,
                        likelihood_ratios=(1 - p[i]) / p[i],
                        lower_percentiles=u1,
                        y_min=self.y_min,
                        y_max=self.y_max,
                    )
                    - y[i].reshape(-1, 1)
                )
            else:
                u0 = np.random.uniform(
                    low=0,
                    high=100,
                    size=self.MC_samples,
                )
                if self.normalized_conformal:
                    sigma0 = sigmas0[i].reshape(-1, 1)
                else:
                    sigma0 = None
                ite_MC[i * self.MC_samples : (i + 1) * self.MC_samples] = y[i].reshape(
                    -1, 1
                ) - self.y0_estimator.predict_cps(
                    X[i].reshape(1, -1),
                    sigmas=sigma0,
                    likelihood_ratios=p[i] / (1 - p[i]),
                    lower_percentiles=u0,
                    y_min=self.y_min,
                    y_max=self.y_max,
                )
        return ite_MC

    def generate_MC_ite_samples(self, X, y, W, p):
        """
        Generate Monte Carlo sample of ITEs
        """
        ite_MC = np.zeros(len(X) * self.MC_samples)
        if self.normalized_conformal:
            sigmas0 = self.de_y0.apply(X)
            sigmas1 = self.de_y1.apply(X)

        for i in range(len(X)):
            u1 = np.random.uniform(
                low=0,
                high=100,
                size=self.MC_samples,
            )
            u0 = np.random.uniform(
                low=0,
                high=100,
                size=self.MC_samples,
            )
            if self.normalized_conformal:
                sigma0 = sigmas0[i].reshape(-1, 1)
                sigma1 = sigmas1[i].reshape(-1, 1)
            else:
                sigma0 = None
                sigma1 = None
            ite_MC[i * self.MC_samples : (i + 1) * self.MC_samples] = self.y1_estimator.predict_cps(
                X[i].reshape(1, -1),
                sigmas=sigma1,
                likelihood_ratios=1 / p[i],
                lower_percentiles=u1,
                y_min=self.y_min,
                y_max=self.y_max,
            ) - self.y0_estimator.predict_cps(
                X[i].reshape(1, -1),
                sigmas=sigma0,
                likelihood_ratios=1 / (1 - p[i]),
                lower_percentiles=u0,
                y_min=self.y_min,
                y_max=self.y_max,
            )
        return ite_MC

    class IteEstimator(BaseEstimator, RegressorMixin):
        def __init__(self, y0_estimator, y1_estimator):
            self.y0_estimator = y0_estimator
            self.y1_estimator = y1_estimator

        def fit(self, X, y):
            return self

        def predict(self, X):
            return self.y1_estimator.predict(X) - self.y0_estimator.predict(X)

    def evaluate(self, X, Y0, Y1, p=None, alpha=0.05, return_p_values=False):
        """
        Evaluates model
        """
        # Point prediction - compute RMSE incrementally
        y0_sq_err_sum = 0
        y1_sq_err_sum = 0
        ite_sq_err_sum = 0
        n = len(X)

        # Process in batches to reduce memory usage
        batch_size = 1000
        for i in range(0, n, batch_size):
            batch_slice = slice(i, min(i + batch_size, n))
            X_batch = X[batch_slice]
            Y0_batch = Y0[batch_slice]
            Y1_batch = Y1[batch_slice]
            p_batch = p[batch_slice] if p is not None else None

            y0_pred_batch = self.predict_y0(X_batch)
            y1_pred_batch = self.predict_y1(X_batch)

            y0_sq_err_sum += np.sum((y0_pred_batch - Y0_batch) ** 2)
            y1_sq_err_sum += np.sum((y1_pred_batch - Y1_batch) ** 2)
            ite_sq_err_sum += np.sum((y1_pred_batch - y0_pred_batch - (Y1_batch - Y0_batch)) ** 2)

            # Free memory explicitly
            del y0_pred_batch, y1_pred_batch

        rmse_y0 = np.sqrt(y0_sq_err_sum / n)
        rmse_y1 = np.sqrt(y1_sq_err_sum / n)
        rmse_ite = np.sqrt(ite_sq_err_sum / n)

        # Interval prediction
        # Process in batches for memory efficiency
        batch_size = 1000
        coverage_y0_sum = 0
        coverage_y1_sum = 0
        coverage_ite_sum = 0
        efficiency_y0_sum = 0
        efficiency_y1_sum = 0
        efficiency_ite_sum = 0

        for i in range(0, len(X), batch_size):
            batch_slice = slice(i, min(i + batch_size, len(X)))
            X_batch = X[batch_slice]
            Y0_batch = Y0[batch_slice]
            Y1_batch = Y1[batch_slice]
            p_batch = p[batch_slice] if p is not None else None

            # Get intervals for this batch
            y0_int = self.predict_int_y0(X_batch, p=p_batch, confidence=1 - alpha)
            y1_int = self.predict_int_y1(X_batch, p=p_batch, confidence=1 - alpha)
            ite_int = self.predict_int(X_batch, confidence=1 - alpha)

            # Update running sums
            coverage_y0_sum += np.sum((Y0_batch >= y0_int[:, 0]) & (Y0_batch <= y0_int[:, 1]))
            coverage_y1_sum += np.sum((Y1_batch >= y1_int[:, 0]) & (Y1_batch <= y1_int[:, 1]))
            coverage_ite_sum += np.sum(
                ((Y1_batch - Y0_batch) >= ite_int[:, 0]) & ((Y1_batch - Y0_batch) <= ite_int[:, 1])
            )
            efficiency_y0_sum += np.sum(y0_int[:, 1] - y0_int[:, 0])
            efficiency_y1_sum += np.sum(y1_int[:, 1] - y1_int[:, 0])
            efficiency_ite_sum += np.sum(ite_int[:, 1] - ite_int[:, 0])

            # Free memory
            del y0_int, y1_int, ite_int

        coverage_y0 = coverage_y0_sum / n
        coverage_y1 = coverage_y1_sum / n
        coverage_ite = coverage_ite_sum / n
        efficiency_y0 = efficiency_y0_sum / n
        efficiency_y1 = efficiency_y1_sum / n
        efficiency_ite = efficiency_ite_sum / n
        # Distribution prediction
        batch_size = 1000
        crps_y0 = np.zeros(n)
        crps_y1 = np.zeros(n)
        crps_ite = np.zeros(n)
        ll_y0 = np.zeros(n)
        ll_y1 = np.zeros(n)
        ll_ite = np.zeros(n)

        for i in range(0, len(X), batch_size):
            batch_slice = slice(i, min(i + batch_size, len(X)))
            X_batch = X[batch_slice]
            Y0_batch = Y0[batch_slice]
            Y1_batch = Y1[batch_slice]
            p_batch = p[batch_slice] if p is not None else None
            y0_pred_cps = self.predict_cps_y0(X_batch, p_batch)
            y1_pred_cps = self.predict_cps_y1(X_batch, p_batch)
            ite_pred_cps = self.predict_cps(X_batch)
            crps_y0[batch_slice] = crps_weighted(
                Y0_batch, y0_pred_cps[:, :, 0], weights=y0_pred_cps[:, :, 1], return_average=False
            )
            crps_y1[batch_slice] = crps_weighted(
                Y1_batch, y1_pred_cps[:, :, 0], weights=y1_pred_cps[:, :, 1], return_average=False
            )
            crps_ite[batch_slice] = crps_weighted(
                Y1_batch - Y0_batch,
                ite_pred_cps[:, :, 0],
                weights=ite_pred_cps[:, :, 1],
                return_average=False,
            )
            ll_y0[batch_slice] = loglikelihood(
                Y0_batch, y0_pred_cps[:, :, 0], weights=y0_pred_cps[:, :, 1], return_average=False
            )
            ll_y1[batch_slice] = loglikelihood(
                Y1_batch, y1_pred_cps[:, :, 0], weights=y1_pred_cps[:, :, 1], return_average=False
            )
            ll_ite[batch_slice] = loglikelihood(
                Y1_batch - Y0_batch,
                ite_pred_cps[:, :, 0],
                weights=ite_pred_cps[:, :, 1],
                return_average=False,
            )
            # Free memory
            del y0_pred_cps, y1_pred_cps, ite_pred_cps
        crps_y0 = np.mean(crps_y0)
        crps_y1 = np.mean(crps_y1)
        crps_ite = np.mean(crps_ite)
        ll_y0 = np.mean(ll_y0)
        ll_y1 = np.mean(ll_y1)
        ll_ite = np.mean(ll_ite)
        p_values_y0 = self.predict_p_value_y0(X, Y0, p)
        p_values_y1 = self.predict_p_value_y1(X, Y1, p)
        p_values_ite = self.predict_p_value(X, Y1 - Y0)

        dispersion_y0 = np.var(
            p_values_y0
        )  # dispersion of p-values, ideally 1 / 12 for uniform distribution
        dispersion_y1 = np.var(p_values_y1)
        dispersion_ite = np.var(p_values_ite)
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


class CMC_S_Learner:
    def __init__(
        self,
        y_estimator,
        normalized_conformal=False,
        difficulty_estimator=None,
        MC_samples=100,
        pseudo_MC=True,
        max_min_y=True,
    ):
        self.y_estimator = WrapRegressor(y_estimator)
        self.normalized_conformal = normalized_conformal
        self.MC_samples = MC_samples
        self.pseudo_MC = pseudo_MC
        self.max_min_y = max_min_y
        if self.normalized_conformal:
            if difficulty_estimator is None:
                self.de_y = DifficultyEstimator()
                self.de_ite = DifficultyEstimator()

            else:
                self.de_y = deepcopy(DifficultyEstimator())
                self.de_ite = deepcopy(difficulty_estimator)

    def fit(self, X, y, W, p=None):
        # Split into calibration and training set for nuisance estimators
        # and for the final cate estimator
        if self.max_min_y:
            self.y_min = np.min(y)
            self.y_max = np.max(y)
        else:
            self.y_min = None
            self.y_max = None
        if p is None:
            p = 0.5 / np.ones_like(W)
        (
            X_train_nuisance,
            X_cal,
            y_train_nuisance,
            y_cal,
            W_train_nuisance,
            W_cal,
            p_train_nuisance,
            p_cal,
        ) = train_test_split(X, y, W, p, test_size=0.25)
        (
            X_prop_train_nuisance,
            X_cal_nuisance,
            y_prop_train_nuisance,
            y_cal_nuisance,
            W_prop_train_nuisance,
            W_cal_nuisance,
            p_prop_train_nuisance,
            p_cal_nuisance,
        ) = train_test_split(
            X_train_nuisance, y_train_nuisance, W_train_nuisance, p_train_nuisance, test_size=0.25
        )

        X_prop_train_nuisance = self.add_W(X_prop_train_nuisance, W_prop_train_nuisance)
        X_cal_nuisance = self.add_W(X_cal_nuisance, W_cal_nuisance)

        self.W_bin_thresholds = [-np.inf, 0.1, np.inf]
        W_cal_nuisance_bins = binning(W_cal_nuisance, bins=self.W_bin_thresholds)

        self.y_estimator.fit(
            X_prop_train_nuisance,
            y_prop_train_nuisance,
        )
        # Fit difficulty estimators if normalized conformal
        if self.normalized_conformal:
            self.de_y.fit(X_prop_train_nuisance, y_prop_train_nuisance)
            sigmas = self.de_y.apply(X_cal_nuisance)
        else:
            sigmas = None
        # Calibrate nuisance estimators

        lr_cal_nuisance = np.zeros_like(p_cal_nuisance)
        if self.pseudo_MC:
            lr_cal_nuisance[W_cal_nuisance == 0] = p_cal_nuisance[W_cal_nuisance == 0] / (
                1 - p_cal_nuisance[W_cal_nuisance == 0]
            )
            lr_cal_nuisance[W_cal_nuisance == 1] = (
                p_cal_nuisance[W_cal_nuisance == 1] / p_cal_nuisance[W_cal_nuisance == 1]
            )
        else:
            lr_cal_nuisance[W_cal_nuisance == 0] = 1 / (1 - p_cal_nuisance[W_cal_nuisance == 0])
            lr_cal_nuisance[W_cal_nuisance == 1] = 1 / p_cal_nuisance[W_cal_nuisance == 1]
        self.y_estimator.calibrate(
            X_cal_nuisance,
            y_cal_nuisance,
            sigmas=sigmas,
            likelihood_ratios=lr_cal_nuisance,
            cps=True,
            bins=W_cal_nuisance_bins,
        )

        if self.pseudo_MC:
            generator = self.generate_pseudo_MC_ite_samples
        else:
            generator = self.generate_MC_ite_samples

        ite_MC_cal = generator(X_cal, y_cal, W_cal, p_cal)
        ite_MC_train = generator(
            X_train_nuisance, y_train_nuisance, W_train_nuisance, p_train_nuisance
        )

        self.ite_estimator = WrapRegressor(self.IteEstimator(self, self.y_estimator))
        self.ite_estimator.fit(X_train_nuisance.repeat(self.MC_samples, axis=0), ite_MC_train)

        if self.normalized_conformal:
            self.de_ite.fit(X_train_nuisance.repeat(self.MC_samples, axis=0), ite_MC_train)
            sigmas_ite_cal = self.de_ite.apply(X_cal.repeat(self.MC_samples, axis=0))
        else:
            sigmas_ite_cal = None

        self.ite_estimator.calibrate(
            X_cal.repeat(self.MC_samples, axis=0), ite_MC_cal, sigmas=sigmas_ite_cal, cps=True
        )

    def add_W(self, X, W):
        if isinstance(X, (pd.core.series.Series, pd.DataFrame)):
            X_W = deepcopy(X)
            X_W["W"] = W
            return X_W

        elif isinstance(X, np.ndarray):
            return np.hstack([X, W.reshape((-1, 1))])

        else:
            raise Exception("Unsupported type. Must be either Pandas Dataframe or Numpy array")

    def set_W(self, X, w):
        if isinstance(X, (pd.core.series.Series, pd.DataFrame)):
            X_W = deepcopy(X)
            X_W["W"] = w
            return X_W

        elif isinstance(X, np.ndarray):
            X_W = deepcopy(X)
            X_W[:, -1] = w
            return X_W

        else:
            raise Exception("Unsupported type. Must be either Pandas Dataframe or Numpy array")

    def predict(self, X, conformal_mean=False):
        return self.ite_estimator.predict(X)

    def predict_p_value(self, X, y):
        if self.normalized_conformal:
            sigmas = self.de_ite.apply(X)
        else:
            sigmas = None
        return self.ite_estimator.predict_cps(X, sigmas=sigmas, y=y, return_cpds=False)

    def predict_p_value_y0(self, X, y, p):
        X_0 = self.add_W(X, np.repeat(0.0, len(X)))
        X_0_W_bins = binning(np.repeat(0.0, len(X)), bins=self.W_bin_thresholds)
        if self.normalized_conformal:
            sigmas = self.de_y.apply(X_0)
        else:
            sigmas = None
        if p is None:
            p = 0.5 / np.ones(len(X))
        return self.y_estimator.predict_cps(
            X_0,
            y=y,
            sigmas=sigmas,
            likelihood_ratios=1 / (1 - p),
            return_cpds=False,
            bins=X_0_W_bins,
        )

    def predict_p_value_y1(self, X, y, p):
        X_1 = self.add_W(X, np.repeat(1.0, len(X)))
        X_1_W_bins = binning(np.repeat(1.0, len(X)), bins=self.W_bin_thresholds)
        if self.normalized_conformal:
            sigmas = self.de_y.apply(X_1)
        else:
            sigmas = None
        if p is None:
            p = 0.5 / np.ones(len(X))
        return self.y_estimator.predict_cps(
            X_1,
            y=y,
            sigmas=sigmas,
            likelihood_ratios=1 / p,
            return_cpds=False,
            bins=X_1_W_bins,
        )

    def predict_cps(self, X, return_cpds=True, lower_percentiles=None):
        if self.normalized_conformal:
            sigmas = self.de_ite.apply(X)
        else:
            sigmas = None
        return self.ite_estimator.predict_cps(
            X, sigmas=sigmas, return_cpds=return_cpds, lower_percentiles=lower_percentiles
        )

    def predict_int(self, X, confidence=0.95):
        if self.normalized_conformal:
            sigmas = self.de_ite.apply(X)
        else:
            sigmas = None
        return self.ite_estimator.predict_int(X, sigmas=sigmas, confidence=confidence)

    def predict_y0(self, X, p=None):
        X_0 = self.add_W(X, np.repeat(0.0, len(X)))
        if p is None:
            p = 0.5 / np.ones(len(X))
        return self.y_estimator.predict(X_0)

    def predict_y1(self, X, p=None):
        X_1 = self.add_W(X, np.repeat(1.0, len(X)))
        if p is None:
            p = 0.5 / np.ones(len(X))
        return self.y_estimator.predict(X_1)

    def predict_cps_y0(self, X, p=None, return_cpds=True, lower_percentiles=None):
        X_0 = self.add_W(X, np.repeat(0.0, len(X)))
        X_0_W_bins = binning(np.repeat(0.0, len(X)), bins=self.W_bin_thresholds)
        if self.normalized_conformal:
            sigmas = self.de_y.apply(X_0)
        else:
            sigmas = None
        if p is None:
            p = 0.5 / np.ones(len(X))
        return np.stack(
            self.y_estimator.predict_cps(
                X_0,
                sigmas=sigmas,
                likelihood_ratios=1 / (1 - p),
                return_cpds=return_cpds,
                lower_percentiles=lower_percentiles,
                bins=X_0_W_bins,
                y_min=self.y_min,
                y_max=self.y_max,
            ),
            axis=0,
        )

    def predict_cps_y1(self, X, p=None, return_cpds=True, lower_percentiles=None):
        X_1 = self.add_W(X, np.repeat(1.0, len(X)))
        X_1_W_bins = binning(np.repeat(1.0, len(X)), bins=self.W_bin_thresholds)

        if self.normalized_conformal:
            sigmas = self.de_y.apply(X_1)
        else:
            sigmas = None
        if p is None:
            p = 0.5 / np.ones(len(X))
        return np.stack(
            self.y_estimator.predict_cps(
                X_1,
                sigmas=sigmas,
                likelihood_ratios=1 / p,
                return_cpds=return_cpds,
                lower_percentiles=lower_percentiles,
                bins=X_1_W_bins,
                y_min=self.y_min,
                y_max=self.y_max,
            ),
            axis=0,
        )

    def predict_int_y0(self, X, p=None, confidence=0.95):
        X_0 = self.add_W(X, np.repeat(0.0, len(X)))
        X_0_W_bins = binning(np.repeat(0.0, len(X)), bins=self.W_bin_thresholds)

        if self.normalized_conformal:
            sigmas = self.de_y.apply(X_0)
        else:
            sigmas = None

        if p is None:
            p = 0.5 / np.ones(len(X))
        return self.y_estimator.predict_int(
            X_0,
            sigmas=sigmas,
            likelihood_ratios=1 / (1 - p),
            confidence=confidence,
            bins=X_0_W_bins,
            y_min=self.y_min,
            y_max=self.y_max,
        )

    def predict_int_y1(self, X, p=None, confidence=0.95):
        X_1 = self.add_W(X, np.repeat(1.0, len(X)))
        X_1_W_bins = binning(np.repeat(1.0, len(X)), bins=self.W_bin_thresholds)

        if self.normalized_conformal:
            sigmas = self.de_y.apply(X_1)
        else:
            sigmas = None

        if p is None:
            p = 0.5 / np.ones(len(X))
        return self.y_estimator.predict_int(
            X_1,
            sigmas=sigmas,
            likelihood_ratios=1 / p,
            confidence=confidence,
            bins=X_1_W_bins,
            y_min=self.y_min,
            y_max=self.y_max,
        )

    def generate_pseudo_MC_ite_samples(self, X, y, W, p):
        """
        Generate Pseudo Monte Carlo sample of ITEs
        """
        ite_MC = np.zeros(len(X) * self.MC_samples)

        X_0 = self.add_W(X, np.repeat(0.0, len(X)))
        X_1 = self.add_W(X, np.repeat(1.0, len(X)))

        if self.normalized_conformal:
            sigmas_X_0 = self.de_y.apply(X_0)
            sigmas_X_1 = self.de_y.apply(X_1)

        for i in range(len(X)):
            # TODO: check if the limtit with high and low is correct

            if W[i] == 0:
                if self.normalized_conformal:
                    sigma = sigmas_X_1[i].reshape(-1, 1)
                else:
                    sigma = None
                u1 = np.random.uniform(
                    low=0,
                    high=100,
                    size=self.MC_samples,
                )
                ite_MC[i * self.MC_samples : (i + 1) * self.MC_samples] = (
                    self.y_estimator.predict_cps(
                        X_1[i].reshape(1, -1),
                        sigmas=sigma,
                        likelihood_ratios=1 / p[i],
                        lower_percentiles=u1,
                        return_cpds=False,
                        bins=[1.0],
                        y_min=self.y_min,
                        y_max=self.y_max,
                    )
                    - y[i].reshape(-1, 1)
                )
            else:
                if self.normalized_conformal:
                    sigma = sigmas_X_0[i].reshape(-1, 1)
                else:
                    sigma = None
                u0 = np.random.uniform(
                    low=0,
                    high=100,
                    size=self.MC_samples,
                )
                ite_MC[i * self.MC_samples : (i + 1) * self.MC_samples] = y[i].reshape(
                    -1, 1
                ) - self.y_estimator.predict_cps(
                    X_0[i].reshape(1, -1),
                    sigmas=sigma,
                    likelihood_ratios=1 / (1 - p[i]),
                    lower_percentiles=u0,
                    bins=[0.0],
                    y_min=self.y_min,
                    y_max=self.y_max,
                )

        return ite_MC

    def generate_MC_ite_samples(self, X, y, W, p):
        """
        Generate Monte Carlo sample of ITEs
        """
        ite_MC = np.zeros(len(X) * self.MC_samples)

        X_0 = self.add_W(X, np.repeat(0.0, len(X)))
        X_1 = self.add_W(X, np.repeat(1.0, len(X)))

        if self.normalized_conformal:
            sigmas_X_0 = self.de_y.apply(X_0)
            sigmas_X_1 = self.de_y.apply(X_1)

        for i in range(len(X)):
            u1 = np.random.uniform(
                low=0,
                high=100,
                size=self.MC_samples,
            )
            u0 = np.random.uniform(
                low=0,
                high=100,
                size=self.MC_samples,
            )
            if self.normalized_conformal:
                sigma0 = sigmas_X_0[i].reshape(-1, 1)
                sigma1 = sigmas_X_1[i].reshape(-1, 1)
            else:
                sigma0 = None
                sigma1 = None

            ite_MC[i * self.MC_samples : (i + 1) * self.MC_samples] = self.y_estimator.predict_cps(
                X_1[i].reshape(1, -1),
                sigmas=sigma1,
                likelihood_ratios=1 / p[i],
                lower_percentiles=u1,
                bins=[1.0],
                y_min=self.y_min,
                y_max=self.y_max,
            ) - self.y_estimator.predict_cps(
                X_0[i].reshape(1, -1),
                sigmas=sigma0,
                likelihood_ratios=1 / (1 - p[i]),
                lower_percentiles=u0,
                bins=[0.0],
                y_min=self.y_min,
                y_max=self.y_max,
            )

        return ite_MC

    class IteEstimator(BaseEstimator, RegressorMixin):
        def __init__(self, outer_instance, y_estimator):
            self.y_estimator = y_estimator
            self.outer_instance = outer_instance

        def fit(self, X, y):
            return self

        def predict(self, X):
            X_0 = self.outer_instance.add_W(X, np.repeat(0, len(X)))
            X_1 = self.outer_instance.add_W(X, np.repeat(1, len(X)))

            return self.y_estimator.predict(X_1) - self.y_estimator.predict(X_0)

    def evaluate(self, X, Y0, Y1, p=None, alpha=0.05, return_p_values=False):
        """
        Evaluates model
        """
        # Point prediction - compute RMSE incrementally
        y0_sq_err_sum = 0
        y1_sq_err_sum = 0
        ite_sq_err_sum = 0
        n = len(X)

        # Process in batches to reduce memory usage
        batch_size = 1000
        for i in range(0, n, batch_size):
            batch_slice = slice(i, min(i + batch_size, n))
            X_batch = X[batch_slice]
            Y0_batch = Y0[batch_slice]
            Y1_batch = Y1[batch_slice]
            p_batch = p[batch_slice] if p is not None else None

            y0_pred_batch = self.predict_y0(X_batch)
            y1_pred_batch = self.predict_y1(X_batch)

            y0_sq_err_sum += np.sum((y0_pred_batch - Y0_batch) ** 2)
            y1_sq_err_sum += np.sum((y1_pred_batch - Y1_batch) ** 2)
            ite_sq_err_sum += np.sum((y1_pred_batch - y0_pred_batch - (Y1_batch - Y0_batch)) ** 2)

            # Free memory explicitly
            del y0_pred_batch, y1_pred_batch

        rmse_y0 = np.sqrt(y0_sq_err_sum / n)
        rmse_y1 = np.sqrt(y1_sq_err_sum / n)
        rmse_ite = np.sqrt(ite_sq_err_sum / n)

        # Interval prediction
        # Process in batches for memory efficiency
        batch_size = 1000
        coverage_y0_sum = 0
        coverage_y1_sum = 0
        coverage_ite_sum = 0
        efficiency_y0_sum = 0
        efficiency_y1_sum = 0
        efficiency_ite_sum = 0

        for i in range(0, len(X), batch_size):
            batch_slice = slice(i, min(i + batch_size, len(X)))
            X_batch = X[batch_slice]
            Y0_batch = Y0[batch_slice]
            Y1_batch = Y1[batch_slice]
            p_batch = p[batch_slice] if p is not None else None

            # Get intervals for this batch
            y0_int = self.predict_int_y0(X_batch, p=p_batch, confidence=1 - alpha)
            y1_int = self.predict_int_y1(X_batch, p=p_batch, confidence=1 - alpha)
            ite_int = self.predict_int(X_batch, confidence=1 - alpha)

            # Update running sums
            coverage_y0_sum += np.sum((Y0_batch >= y0_int[:, 0]) & (Y0_batch <= y0_int[:, 1]))
            coverage_y1_sum += np.sum((Y1_batch >= y1_int[:, 0]) & (Y1_batch <= y1_int[:, 1]))
            coverage_ite_sum += np.sum(
                ((Y1_batch - Y0_batch) >= ite_int[:, 0]) & ((Y1_batch - Y0_batch) <= ite_int[:, 1])
            )
            efficiency_y0_sum += np.sum(y0_int[:, 1] - y0_int[:, 0])
            efficiency_y1_sum += np.sum(y1_int[:, 1] - y1_int[:, 0])
            efficiency_ite_sum += np.sum(ite_int[:, 1] - ite_int[:, 0])

            # Free memory
            del y0_int, y1_int, ite_int

        coverage_y0 = coverage_y0_sum / n
        coverage_y1 = coverage_y1_sum / n
        coverage_ite = coverage_ite_sum / n
        efficiency_y0 = efficiency_y0_sum / n
        efficiency_y1 = efficiency_y1_sum / n
        efficiency_ite = efficiency_ite_sum / n
        # Distribution prediction
        batch_size = 1000
        crps_y0 = np.zeros(n)
        crps_y1 = np.zeros(n)
        crps_ite = np.zeros(n)
        ll_y0 = np.zeros(n)
        ll_y1 = np.zeros(n)
        ll_ite = np.zeros(n)

        for i in range(0, len(X), batch_size):
            batch_slice = slice(i, min(i + batch_size, len(X)))
            X_batch = X[batch_slice]
            Y0_batch = Y0[batch_slice]
            Y1_batch = Y1[batch_slice]
            p_batch = p[batch_slice] if p is not None else None
            y0_pred_cps = self.predict_cps_y0(X_batch, p_batch)
            y1_pred_cps = self.predict_cps_y1(X_batch, p_batch)
            ite_pred_cps = self.predict_cps(X_batch)
            crps_y0[batch_slice] = crps_weighted(
                Y0_batch, y0_pred_cps[:, :, 0], weights=y0_pred_cps[:, :, 1], return_average=False
            )
            crps_y1[batch_slice] = crps_weighted(
                Y1_batch, y1_pred_cps[:, :, 0], weights=y1_pred_cps[:, :, 1], return_average=False
            )
            crps_ite[batch_slice] = crps_weighted(
                Y1_batch - Y0_batch,
                ite_pred_cps[:, :, 0],
                weights=ite_pred_cps[:, :, 1],
                return_average=False,
            )
            ll_y0[batch_slice] = loglikelihood(
                Y0_batch, y0_pred_cps[:, :, 0], weights=y0_pred_cps[:, :, 1], return_average=False
            )
            ll_y1[batch_slice] = loglikelihood(
                Y1_batch, y1_pred_cps[:, :, 0], weights=y1_pred_cps[:, :, 1], return_average=False
            )
            ll_ite[batch_slice] = loglikelihood(
                Y1_batch - Y0_batch,
                ite_pred_cps[:, :, 0],
                weights=ite_pred_cps[:, :, 1],
                return_average=False,
            )
            # Free memory
            del y0_pred_cps, y1_pred_cps, ite_pred_cps
        crps_y0 = np.mean(crps_y0)
        crps_y1 = np.mean(crps_y1)
        crps_ite = np.mean(crps_ite)
        ll_y0 = np.mean(ll_y0)
        ll_y1 = np.mean(ll_y1)
        ll_ite = np.mean(ll_ite)

        p_values_y0 = self.predict_p_value_y0(X, Y0, p)
        p_values_y1 = self.predict_p_value_y1(X, Y1, p)
        p_values_ite = self.predict_p_value(X, Y1 - Y0)

        dispersion_y0 = np.var(
            p_values_y0
        )  # dispersion of p-values, ideally 1 / 12 for uniform distribution
        dispersion_y1 = np.var(p_values_y1)
        dispersion_ite = np.var(p_values_ite)
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


class CMC_X_Learner:
    def __init__(
        self,
        y0_estimator,
        y1_estimator,
        cate_estimator,
        normalized_conformal=False,
        difficulty_estimator=None,
        MC_samples=100,
        pseudo_MC=True,
        max_min_y=True,
    ):
        self.y0_estimator = WrapRegressor(y0_estimator)
        self.y1_estimator = WrapRegressor(y1_estimator)
        self.cate_estimator = WrapRegressor(cate_estimator)
        self.normalized_conformal = normalized_conformal
        self.MC_samples = MC_samples
        self.pseudo_MC = pseudo_MC
        self.max_min_y = max_min_y
        if self.normalized_conformal:
            if difficulty_estimator is None:
                self.de_y0 = DifficultyEstimator()
                self.de_y1 = DifficultyEstimator()
                self.de_ite = DifficultyEstimator()
            else:
                self.de_y0 = deepcopy(difficulty_estimator)
                self.de_y1 = deepcopy(difficulty_estimator)
                self.de_ite = deepcopy(difficulty_estimator)

    def fit(self, X, y, W, p=None):
        # Split into calibration and training set for nuisance estimators
        # and for the final cate estimator
        if self.max_min_y:
            self.y_min = np.min(y)
            self.y_max = np.max(y)
        else:
            self.y_min = None
            self.y_max = None
        if p is None:
            p = 0.5 / np.ones_like(W)
        (
            X_train_nuisance,
            X_cal,
            y_train_nuisance,
            y_cal,
            W_train_nuisance,
            W_cal,
            p_train_nuisance,
            p_cal,
        ) = train_test_split(X, y, W, p, test_size=0.25)
        (
            X_prop_train_nuisance,
            X_cal_nuisance,
            y_prop_train_nuisance,
            y_cal_nuisance,
            W_prop_train_nuisance,
            W_cal_nuisance,
            p_prop_train_nuisance,
            p_cal_nuisance,
        ) = train_test_split(
            X_train_nuisance, y_train_nuisance, W_train_nuisance, p_train_nuisance, test_size=0.25
        )

        self.y0_estimator.fit(
            X_prop_train_nuisance[W_prop_train_nuisance == 0],
            y_prop_train_nuisance[W_prop_train_nuisance == 0],
        )
        self.y1_estimator.fit(
            X_prop_train_nuisance[W_prop_train_nuisance == 1],
            y_prop_train_nuisance[W_prop_train_nuisance == 1],
        )
        # Fit difficulty estimators if normalized conformal
        if self.normalized_conformal:
            self.de_y0.fit(
                X_prop_train_nuisance[W_prop_train_nuisance == 0],
                y_prop_train_nuisance[W_prop_train_nuisance == 0],
            )
            self.de_y1.fit(
                X_prop_train_nuisance[W_prop_train_nuisance == 1],
                y_prop_train_nuisance[W_prop_train_nuisance == 1],
            )
            sigmas0 = self.de_y0.apply(X_cal_nuisance)[W_cal_nuisance == 0]
            sigmas1 = self.de_y1.apply(X_cal_nuisance)[W_cal_nuisance == 1]
        else:
            sigmas0 = None
            sigmas1 = None

        lr_cal_nuisance = np.zeros_like(p_cal_nuisance)
        if self.pseudo_MC:
            lr_cal_nuisance[W_cal_nuisance == 0] = p_cal_nuisance[W_cal_nuisance == 0] / (
                1 - p_cal_nuisance[W_cal_nuisance == 0]
            )
            lr_cal_nuisance[W_cal_nuisance == 1] = (
                p_cal_nuisance[W_cal_nuisance == 1] / p_cal_nuisance[W_cal_nuisance == 1]
            )
        else:
            lr_cal_nuisance[W_cal_nuisance == 0] = 1 / (1 - p_cal_nuisance[W_cal_nuisance == 0])
            lr_cal_nuisance[W_cal_nuisance == 1] = 1 / p_cal_nuisance[W_cal_nuisance == 1]
        # Calibrate nuisance estimators
        self.y0_estimator.calibrate(
            X_cal_nuisance[W_cal_nuisance == 0],
            y_cal_nuisance[W_cal_nuisance == 0],
            sigmas=sigmas0,
            likelihood_ratios=lr_cal_nuisance[W_cal_nuisance == 0],
            cps=True,
        )
        self.y1_estimator.calibrate(
            X_cal_nuisance[W_cal_nuisance == 1],
            y_cal_nuisance[W_cal_nuisance == 1],
            sigmas=sigmas1,
            likelihood_ratios=lr_cal_nuisance[W_cal_nuisance == 1],
            cps=True,
        )
        if self.pseudo_MC:
            generator = self.generate_pseudo_MC_ite_samples
        else:
            generator = self.generate_MC_ite_samples
        ite_MC_cal = generator(X_cal, y_cal, W_cal, p_cal)
        ite_MC_train = generator(
            X_train_nuisance, y_train_nuisance, W_train_nuisance, p_train_nuisance
        )

        self.ite_estimator = WrapRegressor(self.IteEstimator(self, self.cate_estimator))
        # self.ite_estimator.learner.fit_W(X_train_nuisance, y_train_nuisance, W_train_nuisance)
        self.ite_estimator.fit(X_train_nuisance.repeat(self.MC_samples, axis=0), ite_MC_train)

        if self.normalized_conformal:
            # self.de_ite.fit(X_train_nuisance.repeat(self.MC_samples, axis=0), ite_MC_train)
            self.de_ite.fit(X_train_nuisance.repeat(self.MC_samples, axis=0), ite_MC_train)
            sigmas_ite_cal = self.de_ite.apply(X_cal.repeat(self.MC_samples, axis=0))
        else:
            sigmas_ite_cal = None
        self.ite_estimator.calibrate(
            X_cal.repeat(self.MC_samples, axis=0), ite_MC_cal, sigmas=sigmas_ite_cal, cps=True
        )

    def predict(self, X, conformal_mean=False):
        return self.ite_estimator.predict(X)

    def predict_p_value(self, X, y):
        if self.normalized_conformal:
            sigmas = self.de_ite.apply(X)
        else:
            sigmas = None
        return self.ite_estimator.predict_cps(X, sigmas=sigmas, y=y, return_cpds=False)

    def predict_p_value_y0(self, X, y, p):
        if self.normalized_conformal:
            sigmas = self.de_y0.apply(X)
        else:
            sigmas = None
        if p is None:
            p = 0.5 / np.ones(len(X))
        return self.y0_estimator.predict_cps(
            X, sigmas=sigmas, likelihood_ratios=1 / (1 - p), y=y, return_cpds=False
        )

    def predict_p_value_y1(self, X, y, p):
        if self.normalized_conformal:
            sigmas = self.de_y1.apply(X)
        else:
            sigmas = None
        if p is None:
            p = 0.5 / np.ones(len(X))
        return self.y1_estimator.predict_cps(
            X, sigmas=sigmas, likelihood_ratios=1 / p, y=y, return_cpds=False
        )

    def predict_cps(self, X, return_cpds=True, lower_percentiles=None):
        if self.normalized_conformal:
            sigmas = self.de_ite.apply(X)
        else:
            sigmas = None
        return self.ite_estimator.predict_cps(
            X, sigmas=sigmas, return_cpds=return_cpds, lower_percentiles=lower_percentiles
        )

    def predict_int(self, X, confidence=0.95):
        if self.normalized_conformal:
            sigmas = self.de_ite.apply(X)
        else:
            sigmas = None
        return self.ite_estimator.predict_int(X, sigmas=sigmas, confidence=confidence)

    def predict_y0(self, X):
        return self.y0_estimator.predict(X)

    def predict_y1(self, X):
        return self.y1_estimator.predict(X)

    def predict_cps_y0(self, X, p=None, return_cpds=True, lower_percentiles=None):
        if self.normalized_conformal:
            sigmas = self.de_y0.apply(X)
        else:
            sigmas = None
        if p is None:
            p = 0.5 / np.ones(len(X))
        return self.y0_estimator.predict_cps(
            X,
            sigmas=sigmas,
            likelihood_ratios=1 / (1 - p),
            return_cpds=return_cpds,
            lower_percentiles=lower_percentiles,
            y_min=self.y_min,
            y_max=self.y_max,
        )

    def predict_cps_y1(self, X, p=None, return_cpds=True, lower_percentiles=None):
        if self.normalized_conformal:
            sigmas = self.de_y1.apply(X)
        else:
            sigmas = None
        if p is None:
            p = 0.5 / np.ones(len(X))
        return self.y1_estimator.predict_cps(
            X,
            sigmas=sigmas,
            likelihood_ratios=1 / p,
            return_cpds=return_cpds,
            lower_percentiles=lower_percentiles,
            y_min=self.y_min,
            y_max=self.y_max,
        )

    def predict_int_y0(self, X, p=None, confidence=0.95):
        if self.normalized_conformal:
            sigmas = self.de_y0.apply(X)
        else:
            sigmas = None
        if p is None:
            p = 0.5 / np.ones(len(X))
        return self.y0_estimator.predict_int(
            X,
            sigmas=sigmas,
            likelihood_ratios=1 / (1 - p),
            confidence=confidence,
            y_min=self.y_min,
            y_max=self.y_max,
        )

    def predict_int_y1(self, X, p=None, confidence=0.95):
        if self.normalized_conformal:
            sigmas = self.de_y1.apply(X)
        else:
            sigmas = None
        if p is None:
            p = 0.5 / np.ones(len(X))
        return self.y1_estimator.predict_int(
            X,
            sigmas=sigmas,
            likelihood_ratios=1 / p,
            confidence=confidence,
            y_min=self.y_min,
            y_max=self.y_max,
        )

    def generate_pseudo_MC_ite_samples(self, X, y, W, p):
        """
        Generate Pseudo Monte Carlo sample of ITEs
        """
        ite_MC = np.zeros(len(X) * self.MC_samples)
        if self.normalized_conformal:
            sigmas0 = self.de_y0.apply(X)
            sigmas1 = self.de_y1.apply(X)

        for i in range(len(X)):
            # TODO: check if the limtit with high and low is correct
            if W[i] == 0:
                u1 = np.random.uniform(
                    low=0,
                    high=100,
                    size=self.MC_samples,
                )
                if self.normalized_conformal:
                    sigma1 = sigmas1[i].reshape(-1, 1)
                else:
                    sigma1 = None
                ite_MC[i * self.MC_samples : (i + 1) * self.MC_samples] = (
                    self.y1_estimator.predict_cps(
                        X[i].reshape(1, -1),
                        sigmas=sigma1,
                        likelihood_ratios=1 / p[i],
                        lower_percentiles=u1,
                        y_min=self.y_min,
                        y_max=self.y_max,
                    )
                    - y[i].reshape(-1, 1)
                )
            else:
                u0 = np.random.uniform(
                    low=0,
                    high=100,
                    size=self.MC_samples,
                )
                if self.normalized_conformal:
                    sigma0 = sigmas0[i].reshape(-1, 1)
                else:
                    sigma0 = None
                ite_MC[i * self.MC_samples : (i + 1) * self.MC_samples] = y[i].reshape(
                    -1, 1
                ) - self.y0_estimator.predict_cps(
                    X[i].reshape(1, -1),
                    sigmas=sigma0,
                    likelihood_ratios=1 / (1 - p[i]),
                    lower_percentiles=u0,
                    y_min=self.y_min,
                    y_max=self.y_max,
                )
        return ite_MC

    def generate_MC_ite_samples(self, X, y, W, p):
        """
        Generate Monte Carlo sample of ITEs
        """
        ite_MC = np.zeros(len(X) * self.MC_samples)
        if self.normalized_conformal:
            sigmas0 = self.de_y0.apply(X)
            sigmas1 = self.de_y1.apply(X)

        for i in range(len(X)):
            u1 = np.random.uniform(
                low=0,
                high=100,
                size=self.MC_samples,
            )
            u0 = np.random.uniform(
                low=0,
                high=100,
                size=self.MC_samples,
            )
            if self.normalized_conformal:
                sigma0 = sigmas0[i].reshape(-1, 1)
                sigma1 = sigmas1[i].reshape(-1, 1)
            else:
                sigma0 = None
                sigma1 = None
            ite_MC[i * self.MC_samples : (i + 1) * self.MC_samples] = self.y1_estimator.predict_cps(
                X[i].reshape(1, -1),
                sigmas=sigma1,
                likelihood_ratios=1 / p[i],
                lower_percentiles=u1,
                y_min=self.y_min,
                y_max=self.y_max,
            ) - self.y0_estimator.predict_cps(
                X[i].reshape(1, -1),
                sigmas=sigma0,
                likelihood_ratios=1 / (1 - p[i]),
                lower_percentiles=u0,
                y_min=self.y_min,
                y_max=self.y_max,
            )
        return ite_MC

    class IteEstimator(BaseEstimator, RegressorMixin):
        def __init__(self, outer_instance, cate_estimator):
            self.outer_instance = outer_instance
            self.cate_estimator = cate_estimator

        def fit_W(self, X, y, W):
            d0_hat = self.outer_instance.y1_estimator.predict(X[W == 0]) - y[W == 0]
            d1_hat = y[W == 1] - self.outer_instance.y0_estimator.predict(X[W == 1])

            # same generator to get the same shuffle for both datasets
            np_random_generator = np.random.default_rng(seed=42)
            X_shuffle = np.vstack([X[W == 0], X[W == 1]])
            # outputs None but shuffles the array internally. shuffles by default on axis=0
            np_random_generator.shuffle(X_shuffle)

            np_random_generator = np.random.default_rng(seed=42)
            d_hat_shuffle = np.append(d0_hat, d1_hat)
            np_random_generator.shuffle(d_hat_shuffle)

            self.cate_estimator.fit(X_shuffle, d_hat_shuffle)
            return self

        # for crepes library compatibility
        def fit(self, X, y):
            self.cate_estimator.fit(X, y)
            return self

        def predict(self, X):
            return self.cate_estimator.predict(X)

    def evaluate(self, X, Y0, Y1, p=None, alpha=0.05, return_p_values=False):
        """
        Evaluates model
        """
        # Point prediction - compute RMSE incrementally
        y0_sq_err_sum = 0
        y1_sq_err_sum = 0
        ite_sq_err_sum = 0
        n = len(X)

        # Process in batches to reduce memory usage
        batch_size = 1000
        for i in range(0, n, batch_size):
            batch_slice = slice(i, min(i + batch_size, n))
            X_batch = X[batch_slice]
            Y0_batch = Y0[batch_slice]
            Y1_batch = Y1[batch_slice]
            p_batch = p[batch_slice] if p is not None else None

            y0_pred_batch = self.predict_y0(X_batch)
            y1_pred_batch = self.predict_y1(X_batch)

            y0_sq_err_sum += np.sum((y0_pred_batch - Y0_batch) ** 2)
            y1_sq_err_sum += np.sum((y1_pred_batch - Y1_batch) ** 2)
            ite_sq_err_sum += np.sum((y1_pred_batch - y0_pred_batch - (Y1_batch - Y0_batch)) ** 2)

            # Free memory explicitly
            del y0_pred_batch, y1_pred_batch

        rmse_y0 = np.sqrt(y0_sq_err_sum / n)
        rmse_y1 = np.sqrt(y1_sq_err_sum / n)
        rmse_ite = np.sqrt(ite_sq_err_sum / n)

        # Interval prediction
        # Process in batches for memory efficiency
        batch_size = 1000
        coverage_y0_sum = 0
        coverage_y1_sum = 0
        coverage_ite_sum = 0
        efficiency_y0_sum = 0
        efficiency_y1_sum = 0
        efficiency_ite_sum = 0

        for i in range(0, len(X), batch_size):
            batch_slice = slice(i, min(i + batch_size, len(X)))
            X_batch = X[batch_slice]
            Y0_batch = Y0[batch_slice]
            Y1_batch = Y1[batch_slice]
            p_batch = p[batch_slice] if p is not None else None

            # Get intervals for this batch
            y0_int = self.predict_int_y0(X_batch, p=p_batch, confidence=1 - alpha)
            y1_int = self.predict_int_y1(X_batch, p=p_batch, confidence=1 - alpha)
            ite_int = self.predict_int(X_batch, confidence=1 - alpha)

            # Update running sums
            coverage_y0_sum += np.sum((Y0_batch >= y0_int[:, 0]) & (Y0_batch <= y0_int[:, 1]))
            coverage_y1_sum += np.sum((Y1_batch >= y1_int[:, 0]) & (Y1_batch <= y1_int[:, 1]))
            coverage_ite_sum += np.sum(
                ((Y1_batch - Y0_batch) >= ite_int[:, 0]) & ((Y1_batch - Y0_batch) <= ite_int[:, 1])
            )
            efficiency_y0_sum += np.sum(y0_int[:, 1] - y0_int[:, 0])
            efficiency_y1_sum += np.sum(y1_int[:, 1] - y1_int[:, 0])
            efficiency_ite_sum += np.sum(ite_int[:, 1] - ite_int[:, 0])

            # Free memory
            del y0_int, y1_int, ite_int

        coverage_y0 = coverage_y0_sum / n
        coverage_y1 = coverage_y1_sum / n
        coverage_ite = coverage_ite_sum / n
        efficiency_y0 = efficiency_y0_sum / n
        efficiency_y1 = efficiency_y1_sum / n
        efficiency_ite = efficiency_ite_sum / n
        # Distribution prediction
        batch_size = 1000
        crps_y0 = np.zeros(n)
        crps_y1 = np.zeros(n)
        crps_ite = np.zeros(n)
        ll_y0 = np.zeros(n)
        ll_y1 = np.zeros(n)
        ll_ite = np.zeros(n)

        for i in range(0, len(X), batch_size):
            batch_slice = slice(i, min(i + batch_size, len(X)))
            X_batch = X[batch_slice]
            Y0_batch = Y0[batch_slice]
            Y1_batch = Y1[batch_slice]
            p_batch = p[batch_slice] if p is not None else None
            y0_pred_cps = self.predict_cps_y0(X_batch, p_batch)
            y1_pred_cps = self.predict_cps_y1(X_batch, p_batch)
            ite_pred_cps = self.predict_cps(X_batch)
            crps_y0[batch_slice] = crps_weighted(
                Y0_batch, y0_pred_cps[:, :, 0], weights=y0_pred_cps[:, :, 1], return_average=False
            )
            crps_y1[batch_slice] = crps_weighted(
                Y1_batch, y1_pred_cps[:, :, 0], weights=y1_pred_cps[:, :, 1], return_average=False
            )
            crps_ite[batch_slice] = crps_weighted(
                Y1_batch - Y0_batch,
                ite_pred_cps[:, :, 0],
                weights=ite_pred_cps[:, :, 1],
                return_average=False,
            )
            ll_y0[batch_slice] = loglikelihood(
                Y0_batch, y0_pred_cps[:, :, 0], weights=y0_pred_cps[:, :, 1], return_average=False
            )
            ll_y1[batch_slice] = loglikelihood(
                Y1_batch, y1_pred_cps[:, :, 0], weights=y1_pred_cps[:, :, 1], return_average=False
            )
            ll_ite[batch_slice] = loglikelihood(
                Y1_batch - Y0_batch,
                ite_pred_cps[:, :, 0],
                weights=ite_pred_cps[:, :, 1],
                return_average=False,
            )
            # Free memory
            del y0_pred_cps, y1_pred_cps, ite_pred_cps
        crps_y0 = np.mean(crps_y0)
        crps_y1 = np.mean(crps_y1)
        crps_ite = np.mean(crps_ite)
        ll_y0 = np.mean(ll_y0)
        ll_y1 = np.mean(ll_y1)
        ll_ite = np.mean(ll_ite)

        p_values_y0 = self.predict_p_value_y0(X, Y0, p)
        p_values_y1 = self.predict_p_value_y1(X, Y1, p)
        p_values_ite = self.predict_p_value(X, Y1 - Y0)

        dispersion_y0 = np.var(
            p_values_y0
        )  # dispersion of p-values, ideally 1 / 12 for uniform distribution
        dispersion_y1 = np.var(p_values_y1)
        dispersion_ite = np.var(p_values_ite)
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
