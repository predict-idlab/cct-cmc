from copy import deepcopy

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from crepes import WrapRegressor  # type: ignore
from crepes.extras import DifficultyEstimator, binning  # type: ignore
from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore


class Conformal_MC_T_Learner:
    def __init__(
        self,
        y0_estimator,
        y1_estimator,
        adaptive_conformal=False,
        difficulty_estimator=None,
        MC_samples=100,
        pseudo_MC=True,
    ):
        self.y0_estimator = WrapRegressor(y0_estimator)
        self.y1_estimator = WrapRegressor(y1_estimator)
        self.adaptive_conformal = adaptive_conformal
        self.MC_samples = MC_samples
        self.pseudo_MC = pseudo_MC
        if self.adaptive_conformal:
            if difficulty_estimator is None:
                self.de_y0 = DifficultyEstimator()
                self.de_y1 = DifficultyEstimator()
                self.de_ite = DifficultyEstimator()
            else:
                self.de_y0 = deepcopy(difficulty_estimator)
                self.de_y1 = deepcopy(difficulty_estimator)
                self.de_ite = deepcopy(difficulty_estimator)

    def fit(self, X, y, W):
        # Split into calibration and training set for nuisance estimators
        # and for the final cate estimator
        (
            X_train_nuisance,
            X_cal,
            y_train_nuisance,
            y_cal,
            W_train_nuisance,
            W_cal,
        ) = train_test_split(X, y, W, test_size=0.25)
        (
            X_prop_train_nuisance,
            X_cal_nuisance,
            y_prop_train_nuisance,
            y_cal_nuisance,
            W_prop_train_nuisance,
            W_cal_nuisance,
        ) = train_test_split(X_train_nuisance, y_train_nuisance, W_train_nuisance, test_size=0.25)

        self.y0_estimator.fit(
            X_prop_train_nuisance[W_prop_train_nuisance == 0],
            y_prop_train_nuisance[W_prop_train_nuisance == 0],
        )
        self.y1_estimator.fit(
            X_prop_train_nuisance[W_prop_train_nuisance == 1],
            y_prop_train_nuisance[W_prop_train_nuisance == 1],
        )
        # Fit difficulty estimators if adaptive conformal
        if self.adaptive_conformal:
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
        self.y0_estimator.calibrate(
            X_cal_nuisance[W_cal_nuisance == 0],
            y_cal_nuisance[W_cal_nuisance == 0],
            sigmas=sigmas0,
            cps=True,
        )
        self.y1_estimator.calibrate(
            X_cal_nuisance[W_cal_nuisance == 1],
            y_cal_nuisance[W_cal_nuisance == 1],
            sigmas=sigmas1,
            cps=True,
        )
        if self.pseudo_MC:
            generator = self.generate_pseudo_MC_ite_samples
        else:
            generator = self.generate_MC_ite_samples
        ite_MC_cal = generator(X_cal, y_cal, W_cal)
        ite_MC_train = generator(X_train_nuisance, y_train_nuisance, W_train_nuisance)
        self.ite_estimator = WrapRegressor(self.IteEstimator(self.y0_estimator, self.y1_estimator))
        self.ite_estimator.fit(X_train_nuisance.repeat(self.MC_samples, axis=0), ite_MC_train)

        if self.adaptive_conformal:
            self.de_ite.fit(X_train_nuisance.repeat(self.MC_samples, axis=0), ite_MC_train)
            sigmas_ite_cal = self.de_ite.apply(X_cal.repeat(self.MC_samples, axis=0))
        else:
            sigmas_ite_cal = None
        self.ite_estimator.calibrate(
            X_cal.repeat(self.MC_samples, axis=0), ite_MC_cal, sigmas=sigmas_ite_cal, cps=True
        )

    def predict(self, X, conformal_mean=False):
        if conformal_mean:
            return self.ite_estimator.predict(X) + np.mean(self.ite_estimator.cps.alphas)
        return self.ite_estimator.predict(X)

    def predict_cps(self, X, return_cpds=True, lower_percentiles=None):
        if self.adaptive_conformal:
            sigmas = self.de_ite.apply(X)
        else:
            sigmas = None
        return self.ite_estimator.predict_cps(
            X, sigmas=sigmas, return_cpds=return_cpds, lower_percentiles=lower_percentiles
        )

    def predict_int(self, X, confidence=0.95):
        if self.adaptive_conformal:
            sigmas = self.de_ite.apply(X)
        else:
            sigmas = None
        return self.ite_estimator.predict_int(X, sigmas=sigmas, confidence=confidence)

    def predict_y0(self, X):
        return self.y0_estimator.predict(X)

    def predict_y1(self, X):
        return self.y1_estimator.predict(X)

    def predict_cps_y0(self, X, return_cpds=True, lower_percentiles=None):
        if self.adaptive_conformal:
            sigmas = self.de_y0.apply(X)
        else:
            sigmas = None
        return self.y0_estimator.predict_cps(
            X, sigmas=sigmas, return_cpds=return_cpds, lower_percentiles=lower_percentiles
        )

    def predict_cps_y1(self, X, return_cpds=True, lower_percentiles=None):
        if self.adaptive_conformal:
            sigmas = self.de_y1.apply(X)
        else:
            sigmas = None
        return self.y1_estimator.predict_cps(
            X, sigmas=sigmas, return_cpds=return_cpds, lower_percentiles=lower_percentiles
        )

    def predict_int_y0(self, X, confidence=0.95):
        if self.adaptive_conformal:
            sigmas = self.de_y0.apply(X)
        else:
            sigmas = None
        return self.y0_estimator.predict_int(X, sigmas=sigmas, confidence=confidence)

    def predict_int_y1(self, X, confidence=0.95):
        if self.adaptive_conformal:
            sigmas = self.de_y1.apply(X)
        else:
            sigmas = None
        return self.y1_estimator.predict_int(X, sigmas=sigmas, confidence=confidence)

    def generate_pseudo_MC_ite_samples(self, X, y, W):
        """
        Generate Pseudo Monte Carlo sample of ITEs
        """
        ite_MC = np.zeros(len(X) * self.MC_samples)
        if self.adaptive_conformal:
            sigmas0 = self.de_y0.apply(X)
            sigmas1 = self.de_y1.apply(X)

        m0 = len(self.y0_estimator.cps.alphas)
        m1 = len(self.y1_estimator.cps.alphas)

        for i in range(len(X)):
            # TODO: check if the limtit with high and low is correct
            if W[i] == 0:
                u1 = np.random.uniform(
                    low=(1 / (m1 + 1)) * 100,
                    high=(m1 / (m1 + 1)) * 100,
                    size=self.MC_samples,
                )
                if self.adaptive_conformal:
                    sigma1 = sigmas1[i].reshape(-1, 1)
                else:
                    sigma1 = None
                ite_MC[
                    i * self.MC_samples : (i + 1) * self.MC_samples
                ] = self.y1_estimator.predict_cps(
                    X[i].reshape(1, -1), sigmas=sigma1, lower_percentiles=u1
                ) - y[
                    i
                ].reshape(
                    -1, 1
                )
            else:
                u0 = np.random.uniform(
                    low=(1 / (m0 + 1)) * 100,
                    high=(m0 / (m0 + 1)) * 100,
                    size=self.MC_samples,
                )
                if self.adaptive_conformal:
                    sigma0 = sigmas0[i].reshape(-1, 1)
                else:
                    sigma0 = None
                ite_MC[i * self.MC_samples : (i + 1) * self.MC_samples] = y[i].reshape(
                    -1, 1
                ) - self.y0_estimator.predict_cps(
                    X[i].reshape(1, -1), sigmas=sigma0, lower_percentiles=u0
                )
        return ite_MC

    def generate_MC_ite_samples(self, X, y, W):
        """
        Generate Monte Carlo sample of ITEs
        """
        ite_MC = np.zeros(len(X) * self.MC_samples)
        if self.adaptive_conformal:
            sigmas0 = self.de_y0.apply(X)
            sigmas1 = self.de_y1.apply(X)

        m0 = len(self.y0_estimator.cps.alphas)
        m1 = len(self.y1_estimator.cps.alphas)

        for i in range(len(X)):
            u1 = np.random.uniform(
                low=(1 / (m1 + 1)) * 100,
                high=(m1 / (m1 + 1)) * 100,
                size=self.MC_samples,
            )
            u0 = np.random.uniform(
                low=(1 / (m0 + 1)) * 100,
                high=(m0 / (m0 + 1)) * 100,
                size=self.MC_samples,
            )
            if self.adaptive_conformal:
                sigma0 = sigmas0[i].reshape(-1, 1)
                sigma1 = sigmas1[i].reshape(-1, 1)
            else:
                sigma0 = None
                sigma1 = None
            ite_MC[i * self.MC_samples : (i + 1) * self.MC_samples] = self.y1_estimator.predict_cps(
                X[i].reshape(1, -1), sigmas=sigma1, lower_percentiles=u1
            ) - self.y0_estimator.predict_cps(
                X[i].reshape(1, -1), sigmas=sigma0, lower_percentiles=u0
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


class Conformal_MC_S_Learner:
    def __init__(
        self,
        y_estimator,
        adaptive_conformal=False,
        difficulty_estimator=None,
        MC_samples=100,
        pseudo_MC=True,
    ):
        self.y_estimator = WrapRegressor(y_estimator)
        self.adaptive_conformal = adaptive_conformal
        self.MC_samples = MC_samples
        self.pseudo_MC = pseudo_MC
        if self.adaptive_conformal:
            if difficulty_estimator is None:
                self.de_y = DifficultyEstimator()
                self.de_ite = DifficultyEstimator()

            else:
                self.de_y = deepcopy(DifficultyEstimator())
                self.de_ite = deepcopy(difficulty_estimator)

    def fit(self, X, y, W):
        # Split into calibration and training set for nuisance estimators
        # and for the final cate estimator
        (
            X_train_nuisance,
            X_cal,
            y_train_nuisance,
            y_cal,
            W_train_nuisance,
            W_cal,
        ) = train_test_split(X, y, W, test_size=0.25)
        (
            X_prop_train_nuisance,
            X_cal_nuisance,
            y_prop_train_nuisance,
            y_cal_nuisance,
            W_prop_train_nuisance,
            W_cal_nuisance,
        ) = train_test_split(X_train_nuisance, y_train_nuisance, W_train_nuisance, test_size=0.25)

        # X_train_nuisance = self.add_W(X_train_nuisance, W_train_nuisance)
        # X_cal = self.add_W(X_cal, W_cal)
        X_prop_train_nuisance = self.add_W(X_prop_train_nuisance, W_prop_train_nuisance)
        X_cal_nuisance = self.add_W(X_cal_nuisance, W_cal_nuisance)

        self.W_bin_thresholds = [-np.inf, 0.1, np.inf]
        W_cal_nuisance_bins = binning(W_cal_nuisance, bins=self.W_bin_thresholds)

        self.y_estimator.fit(
            X_prop_train_nuisance,
            y_prop_train_nuisance,
        )
        # Fit difficulty estimators if adaptive conformal
        if self.adaptive_conformal:
            self.de_y.fit(X_prop_train_nuisance, y_prop_train_nuisance)
            sigmas = self.de_y.apply(X_cal_nuisance)
        else:
            sigmas = None
        # Calibrate nuisance estimators

        self.y_estimator.calibrate(
            X_cal_nuisance, y_cal_nuisance, sigmas=sigmas, cps=True, bins=W_cal_nuisance_bins
        )

        if self.pseudo_MC:
            generator = self.generate_pseudo_MC_ite_samples
        else:
            generator = self.generate_MC_ite_samples

        ite_MC_cal = generator(X_cal, y_cal, W_cal)
        ite_MC_train = generator(X_train_nuisance, y_train_nuisance, W_train_nuisance)

        self.ite_estimator = WrapRegressor(self.IteEstimator(self, self.y_estimator))
        self.ite_estimator.fit(X_train_nuisance.repeat(self.MC_samples, axis=0), ite_MC_train)

        if self.adaptive_conformal:
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
        if conformal_mean:
            return self.ite_estimator.predict(X) + np.mean(self.ite_estimator.cps.alphas)
        return self.ite_estimator.predict(X)

    def predict_cps(self, X, return_cpds=True, lower_percentiles=None):
        if self.adaptive_conformal:
            sigmas = self.de_ite.apply(X)
        else:
            sigmas = None
        return self.ite_estimator.predict_cps(
            X, sigmas=sigmas, return_cpds=return_cpds, lower_percentiles=lower_percentiles
        )

    def predict_int(self, X, confidence=0.95):
        if self.adaptive_conformal:
            sigmas = self.de_ite.apply(X)
        else:
            sigmas = None
        return self.ite_estimator.predict_int(X, sigmas=sigmas, confidence=confidence)

    def predict_y0(self, X):
        X_0 = self.add_W(X, np.repeat(0.0, len(X)))

        return self.y_estimator.predict(X_0)

    def predict_y1(self, X):
        X_1 = self.add_W(X, np.repeat(1.0, len(X)))

        return self.y_estimator.predict(X_1)

    def predict_cps_y0(self, X, return_cpds=True, lower_percentiles=None):
        X_0 = self.add_W(X, np.repeat(0.0, len(X)))
        X_0_W_bins = binning(np.repeat(0.0, len(X)), bins=self.W_bin_thresholds)

        if self.adaptive_conformal:
            sigmas = self.de_y.apply(X_0)
        else:
            sigmas = None
        return self.y_estimator.predict_cps(
            X_0,
            sigmas=sigmas,
            return_cpds=return_cpds,
            lower_percentiles=lower_percentiles,
            bins=X_0_W_bins,
        )

    def predict_cps_y1(self, X, return_cpds=True, lower_percentiles=None):
        X_1 = self.add_W(X, np.repeat(1.0, len(X)))
        X_1_W_bins = binning(np.repeat(1.0, len(X)), bins=self.W_bin_thresholds)

        if self.adaptive_conformal:
            sigmas = self.de_y.apply(X_1)
        else:
            sigmas = None
        return self.y_estimator.predict_cps(
            X_1,
            sigmas=sigmas,
            return_cpds=return_cpds,
            lower_percentiles=lower_percentiles,
            bins=X_1_W_bins,
        )

    def predict_int_y0(self, X, confidence=0.95):
        X_0 = self.add_W(X, np.repeat(0.0, len(X)))
        X_0_W_bins = binning(np.repeat(0.0, len(X)), bins=self.W_bin_thresholds)

        if self.adaptive_conformal:
            sigmas = self.de_y.apply(X_0)
        else:
            sigmas = None
        return self.y_estimator.predict_int(
            X_0, sigmas=sigmas, confidence=confidence, bins=X_0_W_bins
        )

    def predict_int_y1(self, X, confidence=0.95):
        X_1 = self.add_W(X, np.repeat(1.0, len(X)))
        X_1_W_bins = binning(np.repeat(1.0, len(X)), bins=self.W_bin_thresholds)

        if self.adaptive_conformal:
            sigmas = self.de_y.apply(X_1)
        else:
            sigmas = None
        return self.y_estimator.predict_int(
            X_1, sigmas=sigmas, confidence=confidence, bins=X_1_W_bins
        )

    def generate_pseudo_MC_ite_samples(self, X, y, W):
        """
        Generate Pseudo Monte Carlo sample of ITEs
        """
        ite_MC = np.zeros(len(X) * self.MC_samples)

        X_0 = self.add_W(X, np.repeat(0.0, len(X)))
        X_1 = self.add_W(X, np.repeat(1.0, len(X)))

        if self.adaptive_conformal:
            sigmas_X_0 = self.de_y.apply(X_0)
            sigmas_X_1 = self.de_y.apply(X_1)

        m0 = len(self.y_estimator.cps.alphas[1][0])
        m1 = len(self.y_estimator.cps.alphas[1][1])

        for i in range(len(X)):
            # TODO: check if the limtit with high and low is correct

            if W[i] == 0:
                if self.adaptive_conformal:
                    sigma = sigmas_X_1[i].reshape(-1, 1)
                else:
                    sigma = None
                u1 = np.random.uniform(
                    low=(1 / (m1 + 1)) * 100,
                    high=(m1 / (m1 + 1)) * 100,
                    size=self.MC_samples,
                )
                ite_MC[
                    i * self.MC_samples : (i + 1) * self.MC_samples
                ] = self.y_estimator.predict_cps(
                    X_1[i].reshape(1, -1),
                    sigmas=sigma,
                    lower_percentiles=u1,
                    return_cpds=False,
                    bins=[1.0],
                ) - y[
                    i
                ].reshape(
                    -1, 1
                )
            else:
                if self.adaptive_conformal:
                    sigma = sigmas_X_0[i].reshape(-1, 1)
                else:
                    sigma = None
                u0 = np.random.uniform(
                    low=(1 / (m0 + 1)) * 100,
                    high=(m0 / (m0 + 1)) * 100,
                    size=self.MC_samples,
                )
                ite_MC[i * self.MC_samples : (i + 1) * self.MC_samples] = y[i].reshape(
                    -1, 1
                ) - self.y_estimator.predict_cps(
                    X_0[i].reshape(1, -1), sigmas=sigma, lower_percentiles=u0, bins=[0.0]
                )

        return ite_MC

    def generate_MC_ite_samples(self, X, y, W):
        """
        Generate Monte Carlo sample of ITEs
        """
        ite_MC = np.zeros(len(X) * self.MC_samples)

        X_0 = self.add_W(X, np.repeat(0.0, len(X)))
        X_1 = self.add_W(X, np.repeat(1.0, len(X)))

        if self.adaptive_conformal:
            sigmas_X_0 = self.de_y.apply(X_0)
            sigmas_X_1 = self.de_y.apply(X_1)

        m0 = len(self.y_estimator.cps.alphas[1][0])
        m1 = len(self.y_estimator.cps.alphas[1][1])

        for i in range(len(X)):
            u1 = np.random.uniform(
                low=(1 / (m1 + 1)) * 100,
                high=(m1 / (m1 + 1)) * 100,
                size=self.MC_samples,
            )
            u0 = np.random.uniform(
                low=(1 / (m0 + 1)) * 100,
                high=(m0 / (m0 + 1)) * 100,
                size=self.MC_samples,
            )
            if self.adaptive_conformal:
                sigma0 = sigmas_X_0[i].reshape(-1, 1)
                sigma1 = sigmas_X_1[i].reshape(-1, 1)
            else:
                sigma0 = None
                sigma1 = None

            ite_MC[i * self.MC_samples : (i + 1) * self.MC_samples] = self.y_estimator.predict_cps(
                X_1[i].reshape(1, -1), sigmas=sigma1, lower_percentiles=u1, bins=[1.0]
            ) - self.y_estimator.predict_cps(
                X_0[i].reshape(1, -1), sigmas=sigma0, lower_percentiles=u0, bins=[0.0]
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


class Conformal_MC_X_Learner:
    def __init__(
        self,
        y0_estimator,
        y1_estimator,
        cate_estimator,
        adaptive_conformal=False,
        difficulty_estimator=None,
        MC_samples=100,
        pseudo_MC=True,
    ):
        self.y0_estimator = WrapRegressor(y0_estimator)
        self.y1_estimator = WrapRegressor(y1_estimator)
        self.cate_estimator = WrapRegressor(cate_estimator)
        self.adaptive_conformal = adaptive_conformal
        self.MC_samples = MC_samples
        self.pseudo_MC = pseudo_MC
        if self.adaptive_conformal:
            if difficulty_estimator is None:
                self.de_y0 = DifficultyEstimator()
                self.de_y1 = DifficultyEstimator()
                self.de_ite = DifficultyEstimator()
            else:
                self.de_y0 = deepcopy(difficulty_estimator)
                self.de_y1 = deepcopy(difficulty_estimator)
                self.de_ite = deepcopy(difficulty_estimator)

    def fit(self, X, y, W):
        # Split into calibration and training set for nuisance estimators
        # and for the final cate estimator
        (
            X_train_nuisance,
            X_cal,
            y_train_nuisance,
            y_cal,
            W_train_nuisance,
            W_cal,
        ) = train_test_split(X, y, W, test_size=0.25)
        (
            X_prop_train_nuisance,
            X_cal_nuisance,
            y_prop_train_nuisance,
            y_cal_nuisance,
            W_prop_train_nuisance,
            W_cal_nuisance,
        ) = train_test_split(X_train_nuisance, y_train_nuisance, W_train_nuisance, test_size=0.25)

        self.y0_estimator.fit(
            X_prop_train_nuisance[W_prop_train_nuisance == 0],
            y_prop_train_nuisance[W_prop_train_nuisance == 0],
        )
        self.y1_estimator.fit(
            X_prop_train_nuisance[W_prop_train_nuisance == 1],
            y_prop_train_nuisance[W_prop_train_nuisance == 1],
        )
        # Fit difficulty estimators if adaptive conformal
        if self.adaptive_conformal:
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
        self.y0_estimator.calibrate(
            X_cal_nuisance[W_cal_nuisance == 0],
            y_cal_nuisance[W_cal_nuisance == 0],
            sigmas=sigmas0,
            cps=True,
        )
        self.y1_estimator.calibrate(
            X_cal_nuisance[W_cal_nuisance == 1],
            y_cal_nuisance[W_cal_nuisance == 1],
            sigmas=sigmas1,
            cps=True,
        )
        if self.pseudo_MC:
            generator = self.generate_pseudo_MC_ite_samples
        else:
            generator = self.generate_MC_ite_samples
        ite_MC_cal = generator(X_cal, y_cal, W_cal)
        ite_MC_train = generator(X_train_nuisance, y_train_nuisance, W_train_nuisance)

        self.ite_estimator = WrapRegressor(self.IteEstimator(self, self.cate_estimator))
        # self.ite_estimator.learner.fit_W(X_train_nuisance, y_train_nuisance, W_train_nuisance)
        self.ite_estimator.fit(X_train_nuisance.repeat(self.MC_samples, axis=0), ite_MC_train)

        if self.adaptive_conformal:
            # self.de_ite.fit(X_train_nuisance.repeat(self.MC_samples, axis=0), ite_MC_train)
            self.de_ite.fit(X_train_nuisance.repeat(self.MC_samples, axis=0), ite_MC_train)
            sigmas_ite_cal = self.de_ite.apply(X_cal.repeat(self.MC_samples, axis=0))
        else:
            sigmas_ite_cal = None
        self.ite_estimator.calibrate(
            X_cal.repeat(self.MC_samples, axis=0), ite_MC_cal, sigmas=sigmas_ite_cal, cps=True
        )

    def predict(self, X, conformal_mean=False):
        if conformal_mean:
            return self.ite_estimator.predict(X) + np.mean(self.ite_estimator.cps.alphas)
        return self.ite_estimator.predict(X)

    def predict_cps(self, X, return_cpds=True, lower_percentiles=None):
        if self.adaptive_conformal:
            sigmas = self.de_ite.apply(X)
        else:
            sigmas = None
        return self.ite_estimator.predict_cps(
            X, sigmas=sigmas, return_cpds=return_cpds, lower_percentiles=lower_percentiles
        )

    def predict_int(self, X, confidence=0.95):
        if self.adaptive_conformal:
            sigmas = self.de_ite.apply(X)
        else:
            sigmas = None
        return self.ite_estimator.predict_int(X, sigmas=sigmas, confidence=confidence)

    def predict_y0(self, X):
        return self.y0_estimator.predict(X)

    def predict_y1(self, X):
        return self.y1_estimator.predict(X)

    def predict_cps_y0(self, X, return_cpds=True, lower_percentiles=None):
        if self.adaptive_conformal:
            sigmas = self.de_y0.apply(X)
        else:
            sigmas = None
        return self.y0_estimator.predict_cps(
            X, sigmas=sigmas, return_cpds=return_cpds, lower_percentiles=lower_percentiles
        )

    def predict_cps_y1(self, X, return_cpds=True, lower_percentiles=None):
        if self.adaptive_conformal:
            sigmas = self.de_y1.apply(X)
        else:
            sigmas = None
        return self.y1_estimator.predict_cps(
            X, sigmas=sigmas, return_cpds=return_cpds, lower_percentiles=lower_percentiles
        )

    def predict_int_y0(self, X, confidence=0.95):
        if self.adaptive_conformal:
            sigmas = self.de_y0.apply(X)
        else:
            sigmas = None
        return self.y0_estimator.predict_int(X, sigmas=sigmas, confidence=confidence)

    def predict_int_y1(self, X, confidence=0.95):
        if self.adaptive_conformal:
            sigmas = self.de_y1.apply(X)
        else:
            sigmas = None
        return self.y1_estimator.predict_int(X, sigmas=sigmas, confidence=confidence)

    def generate_pseudo_MC_ite_samples(self, X, y, W):
        """
        Generate Pseudo Monte Carlo sample of ITEs
        """
        ite_MC = np.zeros(len(X) * self.MC_samples)
        if self.adaptive_conformal:
            sigmas0 = self.de_y0.apply(X)
            sigmas1 = self.de_y1.apply(X)

        m0 = len(self.y0_estimator.cps.alphas)
        m1 = len(self.y1_estimator.cps.alphas)

        for i in range(len(X)):
            # TODO: check if the limtit with high and low is correct
            if W[i] == 0:
                u1 = np.random.uniform(
                    low=(1 / (m1 + 1)) * 100,
                    high=(m1 / (m1 + 1)) * 100,
                    size=self.MC_samples,
                )
                if self.adaptive_conformal:
                    sigma1 = sigmas1[i].reshape(-1, 1)
                else:
                    sigma1 = None
                ite_MC[
                    i * self.MC_samples : (i + 1) * self.MC_samples
                ] = self.y1_estimator.predict_cps(
                    X[i].reshape(1, -1), sigmas=sigma1, lower_percentiles=u1
                ) - y[
                    i
                ].reshape(
                    -1, 1
                )
            else:
                u0 = np.random.uniform(
                    low=(1 / (m0 + 1)) * 100,
                    high=(m0 / (m0 + 1)) * 100,
                    size=self.MC_samples,
                )
                if self.adaptive_conformal:
                    sigma0 = sigmas0[i].reshape(-1, 1)
                else:
                    sigma0 = None
                ite_MC[i * self.MC_samples : (i + 1) * self.MC_samples] = y[i].reshape(
                    -1, 1
                ) - self.y0_estimator.predict_cps(
                    X[i].reshape(1, -1), sigmas=sigma0, lower_percentiles=u0
                )
        return ite_MC

    def generate_MC_ite_samples(self, X, y, W):
        """
        Generate Monte Carlo sample of ITEs
        """
        ite_MC = np.zeros(len(X) * self.MC_samples)
        if self.adaptive_conformal:
            sigmas0 = self.de_y0.apply(X)
            sigmas1 = self.de_y1.apply(X)

        m0 = len(self.y0_estimator.cps.alphas)
        m1 = len(self.y1_estimator.cps.alphas)

        for i in range(len(X)):
            u1 = np.random.uniform(
                low=(1 / (m1 + 1)) * 100,
                high=(m1 / (m1 + 1)) * 100,
                size=self.MC_samples,
            )
            u0 = np.random.uniform(
                low=(1 / (m0 + 1)) * 100,
                high=(m0 / (m0 + 1)) * 100,
                size=self.MC_samples,
            )
            if self.adaptive_conformal:
                sigma0 = sigmas0[i].reshape(-1, 1)
                sigma1 = sigmas1[i].reshape(-1, 1)
            else:
                sigma0 = None
                sigma1 = None
            ite_MC[i * self.MC_samples : (i + 1) * self.MC_samples] = self.y1_estimator.predict_cps(
                X[i].reshape(1, -1), sigmas=sigma1, lower_percentiles=u1
            ) - self.y0_estimator.predict_cps(
                X[i].reshape(1, -1), sigmas=sigma0, lower_percentiles=u0
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
