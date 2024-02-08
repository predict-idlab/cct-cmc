import numpy as np  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

from src.conformal_metalearners.drlearner import conformalMetalearner


class CM_learner:
    def __init__(
        self, n_folds=5, alpha=0.1, base_learner="GBM", quantile_regression=True, metalearner="DR"
    ):
        self.model = conformalMetalearner(
            n_folds=n_folds,
            alpha=alpha,
            base_learner=base_learner,
            quantile_regression=quantile_regression,
            metalearner=metalearner,
        )

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
        ) = train_test_split(X, y, W, ps, test_size=0.25)
        self.model.fit(X_train_nuisance, W_train_nuisance, y_train_nuisance, ps_train_nuisance)
        self.model.conformalize(self.model.alpha, X_cal, W_cal, y_cal, ps_cal)

    def predict(self, X):
        cate, _, _ = self.model.predict(X)
        return cate

    def predict_int(self, X):
        _, cate_l, cate_u = self.model.predict(X)
        return np.stack((cate_l, cate_u), axis=-1)
