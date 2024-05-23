import time
import warnings
from copy import deepcopy

import numpy as np  # type: ignore
from crepes_weighted import ConformalPredictor, WrapRegressor  # type: ignore
from crepes_weighted.extras import DifficultyEstimator  # type: ignore
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split  # type: ignore


class NaiveWCP:
    def __init__(
        self,
        y0_estimator,
        y1_estimator,
        normalized_conformal=False,
        difficulty_estimator=None,
    ):
        self.y0_estimator = WrapRegressor(y0_estimator)
        self.y1_estimator = WrapRegressor(y1_estimator)
        self.normalized_conformal = normalized_conformal
        self.difficulty_estimator = difficulty_estimator
        if self.normalized_conformal:
            if difficulty_estimator is None:
                self.de_y0 = DifficultyEstimator()
                self.de_y1 = DifficultyEstimator()
            else:
                self.de_y0 = deepcopy(difficulty_estimator)
                self.de_y1 = deepcopy(difficulty_estimator)

    def fit(self, X, y, W, ps):
        # Split into calibration and training set for nuisance estimators
        # and for the final cate estimator
        (
            X_train_nuisance,
            X_cal,
            y_train_nuisance,
            y_cal,
            W_train_nuisance,
            W_cal,
            ps_train_nuisance,
            ps_cal,
        ) = train_test_split(X, y, W, ps, test_size=0.5)
        self.y0_estimator.fit(
            X_train_nuisance[W_train_nuisance == 0],
            y_train_nuisance[W_train_nuisance == 0],
        )
        self.y1_estimator.fit(
            X_train_nuisance[W_train_nuisance == 1],
            y_train_nuisance[W_train_nuisance == 1],
        )
        # Fit difficulty estimators if normalized conformal
        if self.normalized_conformal:
            self.de_y0.fit(
                X_train_nuisance[W_train_nuisance == 0], y_train_nuisance[W_train_nuisance == 0]
            )
            self.de_y1.fit(
                X_train_nuisance[W_train_nuisance == 1], y_train_nuisance[W_train_nuisance == 1]
            )
            sigmas0 = self.de_y0.apply(X_cal)[W_cal == 0]
            sigmas1 = self.de_y1.apply(X_cal)[W_cal == 1]
        else:
            sigmas0 = None
            sigmas1 = None
        # Calibrate nuisance estimators
        w0 = 1 / (1 - ps_cal)
        w1 = 1 / ps_cal
        self.y0_estimator.calibrate(
            X_cal[W_cal == 0],
            y_cal[W_cal == 0],
            likelihood_ratios=w0,
            sigmas=sigmas0,
        )
        self.y1_estimator.calibrate(
            X_cal[W_cal == 1],
            y_cal[W_cal == 1],
            likelihood_ratios=w1,
            sigmas=sigmas1,
        )

    def predict(self, X):
        y0 = self.y0_estimator.predict(X)
        y1 = self.y1_estimator.predict(X)
        return y1 - y0

    def predict_int(self, X, ps=None, confidence=0.95):
        adj_confidence = 1 - (1 - confidence) / 2
        if ps is None:
            w0 = None
            w1 = None
        else:
            w0 = 1 / (1 - ps)
            w1 = 1 / ps
        if self.normalized_conformal:
            sigmas0 = self.de_y0.apply(X)
            sigmas1 = self.de_y1.apply(X)
        else:
            sigmas0 = None
            sigmas1 = None
        y0 = self.y0_estimator.predict_int(
            X,
            sigmas=sigmas0,
            likelihood_ratios=w0,
            confidence=adj_confidence,
        )
        y1 = self.y1_estimator.predict_int(
            X,
            sigmas=sigmas1,
            likelihood_ratios=w1,
            confidence=adj_confidence,
        )
        ite = np.stack((y1[:, 0] - y0[:, 1], y1[:, 1] - y0[:, 0]), axis=1)
        return ite


class NestedWCP:
    def __init__(
        self,
        y0_estimator,
        y1_estimator,
        ite_estimator=None,
        normalized_conformal=False,
        difficulty_estimator=None,
        exact=False,
    ):
        self.y0_estimator = WrapRegressor(y0_estimator)
        self.y1_estimator = WrapRegressor(y1_estimator)
        if exact:
            if ite_estimator is None:
                ite_estimator = ConformalInferenceInterval(RandomForestRegressor())
            self.ite_estimator = ConformalInferenceInterval(ite_estimator)
        else:
            if ite_estimator is None:
                self.ite_estimator_l = GradientBoostingRegressor(loss="quantile", alpha=0.4)
                self.ite_estimator_u = GradientBoostingRegressor(loss="quantile", alpha=0.6)
            else:
                warnings.warn("Watch out ite_estimator probably needs to be quantile estimator ...")
                self.ite_estimator_l = deepcopy(ite_estimator)
                self.ite_estimator_u = deepcopy(ite_estimator)
        self.normalized_conformal = normalized_conformal
        if self.normalized_conformal:
            if difficulty_estimator is None:
                self.de_y0 = DifficultyEstimator()
                self.de_y1 = DifficultyEstimator()
            else:
                self.de_y0 = deepcopy(difficulty_estimator)
                self.de_y1 = deepcopy(difficulty_estimator)
        self.exact = exact
        self.calibrated = False

    def fit(self, X, y, W, ps, confidence=0.95):
        self.confidence = confidence
        (
            X_fold1,
            X_fold2,
            y_fold1,
            y_fold2,
            W_fold1,
            W_fold2,
            ps_fold1,
            ps_fold2,
        ) = train_test_split(X, y, W, ps, test_size=0.25)
        (
            X_fold1_train,
            X_fold1_cal,
            y_fold1_train,
            y_fold1_cal,
            W_fold1_train,
            W_fold1_cal,
            ps_fold1_train,
            ps_fold1_cal,
        ) = train_test_split(X_fold1, y_fold1, W_fold1, ps_fold1, test_size=0.9)
        self.y0_estimator.fit(
            X_fold1_train[W_fold1_train == 0],
            y_fold1_train[W_fold1_train == 0],
        )
        self.y1_estimator.fit(
            X_fold1_train[W_fold1_train == 1],
            y_fold1_train[W_fold1_train == 1],
        )
        # Fit difficulty estimators if normalized conformal
        if self.normalized_conformal:
            self.de_y0.fit(X_fold1_train[W_fold1_train == 0], y_fold1_train[W_fold1_train == 0])
            self.de_y1.fit(X_fold1_train[W_fold1_train == 1], y_fold1_train[W_fold1_train == 1])
            sigmas0 = self.de_y0.apply(X_fold1_cal)[W_fold1_cal == 0]
            sigmas1 = self.de_y1.apply(X_fold1_cal)[W_fold1_cal == 1]
        else:
            sigmas0 = None
            sigmas1 = None
        w0_fold1_cal = ps_fold1_cal[W_fold1_cal == 0] / (1 - ps_fold1_cal[W_fold1_cal == 0])
        w1_fold1_cal = (1 - ps_fold1_cal[W_fold1_cal == 1]) / ps_fold1_cal[W_fold1_cal == 1]
        # Calibrate nuisance estimators
        self.y0_estimator.calibrate(
            X_fold1_cal[W_fold1_cal == 0],
            y_fold1_cal[W_fold1_cal == 0],
            likelihood_ratios=w0_fold1_cal,
            sigmas=sigmas0,
        )
        self.y1_estimator.calibrate(
            X_fold1_cal[W_fold1_cal == 1],
            y_fold1_cal[W_fold1_cal == 1],
            likelihood_ratios=w1_fold1_cal,
            sigmas=sigmas1,
        )

        # Counterfactual inference on the calibration set
        w0_fold2 = np.ones(len(ps_fold2[W_fold2 == 1]))
        w1_fold2 = np.ones(len(ps_fold2[W_fold2 == 0]))
        if self.exact:
            confidence = 1 - (1 - confidence) / 2
        if self.normalized_conformal:
            sigmas0 = self.de_y0.apply(X_fold2[W_fold2 == 1])
            sigmas1 = self.de_y1.apply(X_fold2[W_fold2 == 0])
        else:
            sigmas0 = None
            sigmas1 = None
        y0_fold2 = self.y0_estimator.predict_int(
            X_fold2[W_fold2 == 1],
            sigmas0,
            likelihood_ratios=w0_fold2,
            confidence=confidence,
            y_min=np.min(y[W == 0]),
            y_max=np.max(y[W == 0]),
        )
        y1_fold2 = self.y1_estimator.predict_int(
            X_fold2[W_fold2 == 0],
            sigmas1,
            likelihood_ratios=w1_fold2,
            confidence=confidence,
            y_min=np.min(y[W == 1]),
            y_max=np.max(y[W == 1]),
        )

        ite_fold2 = np.zeros((len(y_fold2), 2))
        ite_fold2[W_fold2 == 0] = y1_fold2 - y_fold2[W_fold2 == 0].reshape(-1, 1)
        ite_fold2[W_fold2 == 1] = np.flip(y_fold2[W_fold2 == 1].reshape(-1, 1) - y0_fold2, axis=1)
        # Fit ITE estimator
        if self.exact:
            self.ite_estimator.fit(X_fold2, ite_fold2)
        else:
            self.ite_estimator_l.fit(X_fold2, ite_fold2[:, 0])
            self.ite_estimator_u.fit(X_fold2, ite_fold2[:, 1])
        self.calibrated = True

    def predict(self, X):
        y0 = self.y0_estimator.predict(X)
        y1 = self.y1_estimator.predict(X)
        return y1 - y0

    def predict_int(self, X, ps=None):
        assert self.calibrated, "Model must be calibrated before prediction"
        if self.exact:
            return self.ite_estimator.predict(X, gamma=(1 - self.confidence) / 2)
        return np.stack((self.ite_estimator_l.predict(X), self.ite_estimator_u.predict(X)), axis=1)


class ConformalInferenceInterval:
    def __init__(
        self,
        learner,
    ) -> None:
        self.learner_l = deepcopy(learner)
        self.learner_u = deepcopy(learner)

    def fit(self, X, y):
        # Split into calibration and training set for nuisance estimators
        # and for the final cate estimator
        (
            X_train,
            X_cal,
            y_train,
            y_cal,
        ) = train_test_split(X, y, test_size=0.25)
        self.learner_l.fit(
            X_train,
            y_train[:, 0],
        )
        self.learner_u.fit(
            X_train,
            y_train[:, 1],
        )

        y_cal_hat_l = self.learner_l.predict(X_cal)
        y_cal_hat_u = self.learner_u.predict(X_cal)

        self.gammas = np.sort(np.maximum(y_cal_hat_l - y_cal[:, 0], y_cal[:, 1] - y_cal_hat_u))[
            ::-1
        ]

    def predict(self, X, gamma=0.025):
        gamma_index = int(gamma * (len(self.gammas) + 1)) - 1
        gamma = self.gammas[gamma_index]

        y_hat_l = self.learner_l.predict(X)
        y_hat_u = self.learner_u.predict(X)

        return np.stack((y_hat_l - gamma, y_hat_u + gamma), axis=1)
