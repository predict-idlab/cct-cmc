import time
import warnings
from copy import deepcopy

import numpy as np  # type: ignore
from crepes import ConformalPredictor  # type: ignore
from crepes.extras import DifficultyEstimator  # type: ignore
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split  # type: ignore


class WeightedConformalRegressor(ConformalPredictor):
    """
    A conformal regressor transforms point predictions (regression
    values) into prediction intervals, for a certain confidence level.
    """

    def __repr__(self):
        if self.fitted:
            return (
                f"WeigthedConformalRegressor(fitted={self.fitted}, "
                f"normalized={self.normalized}, "
                f"mondrian={self.mondrian})"
            )
        else:
            return f"WeigthedConformalRegressor(fitted={self.fitted})"

    def fit(self, residuals, sigmas=None, conformal_weight=None, bins=None):
        """
        Fit conformal regressor.

        Parameters
        ----------
        residuals : array-like of shape (n_values,)
            true values - predicted values
        sigmas: array-like of shape (n_values,), default=None
            difficulty estimates
        bins : array-like of shape (n_values,), default=None
            Mondrian categories

        Returns
        -------
        self : object
            Fitted ConformalRegressor.

        Examples
        --------
        Assuming that ``y_cal`` and ``y_hat_cal`` are vectors with true
        and predicted targets for some calibration set, then a standard
        conformal regressor can be formed from the residuals:

        .. code-block:: python

           residuals_cal = y_cal - y_hat_cal

           from crepes import ConformalRegressor

           cr_std = ConformalRegressor()

           cr_std.fit(residuals_cal)

        Assuming that ``sigmas_cal`` is a vector with difficulty estimates,
        then a normalized conformal regressor can be fitted in the following
        way:

        .. code-block:: python

           cr_norm = ConformalRegressor()
           cr_norm.fit(residuals_cal, sigmas=sigmas_cal)

        Assuming that ``bins_cals`` is a vector with Mondrian categories
        (bin labels), then a Mondrian conformal regressor can be fitted in the
        following way:

        .. code-block:: python

           cr_mond = ConformalRegressor()
           cr_mond.fit(residuals_cal, bins=bins_cal)

        A normalized Mondrian conformal regressor can be fitted in the
        following way:

        .. code-block:: python

           cr_norm_mond = ConformalRegressor()
           cr_norm_mond.fit(residuals_cal, sigmas=sigmas_cal,
                            bins=bins_cal)
        """
        tic = time.time()
        abs_residuals = np.abs(residuals)
        self.conformal_weight = conformal_weight
        if bins is None:
            self.mondrian = False
            if sigmas is None:
                self.normalized = False
                sort_idx = np.argsort(abs_residuals)[::-1]
                self.alphas = abs_residuals[sort_idx]
            else:
                self.normalized = True
                sort_idx = np.argsort(abs_residuals / sigmas)[::-1]
                self.alphas = (abs_residuals / sigmas)[sort_idx]
            if conformal_weight is not None:
                self.conformal_weight = self.conformal_weight[sort_idx]
        else:
            self.mondrian = True
            bin_values = np.unique(bins)
            if sigmas is None:
                self.normalized = False
                self.alphas = (
                    bin_values,
                    [np.sort(abs_residuals[bins == b])[::-1] for b in bin_values],
                )
            else:
                self.normalized = True
                self.alphas = (
                    bin_values,
                    [
                        np.sort(abs_residuals[bins == b] / sigmas[bins == b])[::-1]
                        for b in bin_values
                    ],
                )
        self.fitted = True
        toc = time.time()
        self.time_fit = toc - tic
        return self

    def predict(
        self,
        y_hat,
        sigmas=None,
        conformal_weight=None,
        bins=None,
        confidence=0.95,
        acount_for_small_bins=False,
        y_min=-np.inf,
        y_max=np.inf,
    ):
        """
        Predict using conformal regressor.

        Parameters
        ----------
        y_hat : array-like of shape (n_values,)
            predicted values
        sigmas : array-like of shape (n_values,), default=None
            difficulty estimates
        bins : array-like of shape (n_values,), default=None
            Mondrian categories
        confidence : float in range (0,1), default=0.95
            confidence level
        y_min : float or int, default=-numpy.inf
            minimum value to include in prediction intervals
        y_max : float or int, default=numpy.inf
            maximum value to include in prediction intervals

        Returns
        -------
        intervals : ndarray of shape (n_values, 2)
            prediction intervals

        Examples
        --------
        Assuming that ``y_hat_test`` is a vector with predicted targets for a
        test set and ``cr_std`` a fitted standard conformal regressor, then
        prediction intervals at the 99% confidence level can be obtained by:

        .. code-block:: python

           intervals = cr_std.predict(y_hat_test, confidence=0.99)

        Assuming that ``sigmas_test`` is a vector with difficulty estimates for
        the test set and ``cr_norm`` a fitted normalized conformal regressor,
        then prediction intervals at the default (95%) confidence level can be
        obtained by:

        .. code-block:: python

           intervals = cr_norm.predict(y_hat_test, sigmas=sigmas_test)

        Assuming that ``bins_test`` is a vector with Mondrian categories (bin
        labels) for the test set and ``cr_mond`` a fitted Mondrian conformal
        regressor, then the following provides prediction intervals at the
        default confidence level, where the intervals are lower-bounded by 0:

        .. code-block:: python

           intervals = cr_mond.predict(y_hat_test, bins=bins_test,
                                       y_min=0)

        Note
        ----
        In case the specified confidence level is too high in relation to the
        size of the calibration set, a warning will be issued and the output
        intervals will be of maximum size.
        """
        tic = time.time()
        intervals = np.zeros((len(y_hat), 2))
        if not self.mondrian:
            if self.conformal_weight is None:
                alpha_index = int((1 - confidence) * (len(self.alphas) + 1)) - 1
            else:
                conformal_weights = self.conformal_weight.reshape(1, -1).repeat(
                    len(conformal_weight), axis=0
                ) / (self.conformal_weight.sum() + conformal_weight.reshape(-1, 1))
                conformal_weight = conformal_weight.reshape(-1, 1) / (
                    self.conformal_weight.sum() + conformal_weight.reshape(-1, 1)
                )
                alpha_index = (
                    (np.cumsum(conformal_weights, axis=1) + conformal_weight) > (1 - confidence)
                ).argmax(axis=1) - 1
            alpha = self.alphas[alpha_index[alpha_index >= 0]]
            if self.normalized:
                intervals[alpha_index >= 0, 0] = (
                    y_hat[alpha_index >= 0] - alpha * sigmas[alpha_index >= 0]
                )
                intervals[alpha_index >= 0, 1] = (
                    y_hat[alpha_index >= 0] + alpha * sigmas[alpha_index >= 0]
                )
            else:
                intervals[alpha_index >= 0, 0] = y_hat[alpha_index >= 0] - alpha
                intervals[alpha_index >= 0, 1] = y_hat[alpha_index >= 0] + alpha

            if any(alpha_index < 0):
                warnings.warn(
                    "the no. of calibration examples is too small"
                    "for the chosen confidence level; the "
                    "intervals will be of maximum size"
                )
                if acount_for_small_bins:
                    intervals[alpha_index < 0, 0] = y_min
                    intervals[alpha_index < 0, 1] = y_max
                else:
                    intervals[alpha_index < 0, 0] = y_hat[alpha_index < 0] - self.alphas[0]
                    intervals[alpha_index < 0, 1] = y_hat[alpha_index < 0] + self.alphas[0]
        else:
            bin_values, bin_alphas = self.alphas
            bin_indexes = [np.argwhere(bins == b).T[0] for b in bin_values]
            alpha_indexes = np.array(
                [
                    int((1 - confidence) * (len(bin_alphas[b]) + 1)) - 1
                    for b in range(len(bin_values))
                ]
            )
            too_small_bins = np.argwhere(alpha_indexes < 0)
            if len(too_small_bins) > 0:
                if len(too_small_bins[:, 0]) < 11:
                    bins_to_show = " ".join([str(bin_values[i]) for i in too_small_bins[:, 0]])
                else:
                    bins_to_show = " ".join(
                        [str(bin_values[i]) for i in too_small_bins[:10, 0]] + ["..."]
                    )
                warnings.warn(
                    "the no. of calibration examples is too "
                    "small for the chosen confidence level "
                    f"in the following bins: {bins_to_show}; "
                    "the corresponding intervals will be of "
                    "maximum size"
                )
            bin_alpha = np.array(
                [
                    bin_alphas[b][alpha_indexes[b]] if alpha_indexes[b] >= 0 else np.inf
                    for b in range(len(bin_values))
                ]
            )
            if self.normalized:
                for b in range(len(bin_values)):
                    intervals[bin_indexes[b], 0] = (
                        y_hat[bin_indexes[b]] - bin_alpha[b] * sigmas[bin_indexes[b]]
                    )
                    intervals[bin_indexes[b], 1] = (
                        y_hat[bin_indexes[b]] + bin_alpha[b] * sigmas[bin_indexes[b]]
                    )
            else:
                for b in range(len(bin_values)):
                    intervals[bin_indexes[b], 0] = y_hat[bin_indexes[b]] - bin_alpha[b]
                    intervals[bin_indexes[b], 1] = y_hat[bin_indexes[b]] + bin_alpha[b]
        if y_min > -np.inf:
            intervals[intervals < y_min] = y_min
        if y_max < np.inf:
            intervals[intervals > y_max] = y_max
        toc = time.time()
        self.time_predict = toc - tic
        return intervals

    def evaluate(
        self,
        y_hat,
        y,
        sigmas=None,
        conformal_weight=None,
        bins=None,
        confidence=0.95,
        y_min=-np.inf,
        y_max=np.inf,
        metrics=None,
    ):
        """
        Evaluate conformal regressor.

        Parameters
        ----------
        y_hat : array-like of shape (n_values,)
            predicted values
        y : array-like of shape (n_values,)
            correct target values
        sigmas : array-like of shape (n_values,), default=None
            difficulty estimates
        bins : array-like of shape (n_values,), default=None
            Mondrian categories
        confidence : float in range (0,1), default=0.95
            confidence level
        y_min : float or int, default=-numpy.inf
            minimum value to include in prediction intervals
        y_max : float or int, default=numpy.inf
            maximum value to include in prediction intervals
        metrics : a string or a list of strings,
                  default=list of all metrics, i.e.,
                  ["error", "eff_mean", "eff_med", "time_fit", "time_evaluate"]

        Returns
        -------
        results : dictionary with a key for each selected metric
            estimated performance using the metrics

        Examples
        --------
        Assuming that ``y_hat_test`` and ``y_test`` are vectors with predicted
        and true targets for a test set, ``sigmas_test`` and ``bins_test`` are
        vectors with difficulty estimates and Mondrian categories (bin labels)
        for the test set, and ``cr_norm_mond`` is a fitted normalized Mondrian
        conformal regressor, then the latter can be evaluated at the default
        confidence level with respect to error and mean efficiency (interval
        size) by:

        .. code-block:: python

           results = cr_norm_mond.evaluate(y_hat_test, y_test,
                                           sigmas=sigmas_test, bins=bins_test,
                                           metrics=["error", "eff_mean"])
        """
        tic = time.time()
        if metrics is None:
            metrics = ["error", "eff_mean", "eff_med", "time_fit", "time_evaluate"]
        test_results = {}
        intervals = self.predict(y_hat, sigmas, conformal_weight, bins, confidence, y_min, y_max)
        if "error" in metrics:
            test_results["error"] = 1 - np.mean(
                np.logical_and(intervals[:, 0] <= y, y <= intervals[:, 1])
            )
        if "eff_mean" in metrics:
            test_results["eff_mean"] = np.mean(intervals[:, 1] - intervals[:, 0])
        if "eff_med" in metrics:
            test_results["eff_med"] = np.median(intervals[:, 1] - intervals[:, 0])
        if "time_fit" in metrics:
            test_results["time_fit"] = self.time_fit
        toc = time.time()
        self.time_evaluate = toc - tic
        if "time_evaluate" in metrics:
            test_results["time_evaluate"] = self.time_evaluate
        return test_results


class WeightedWrapRegressor:
    """
    A learner wrapped with a :class:`.ConformalRegressor`
    or :class:`.ConformalPredictiveSystem`.
    """

    def __init__(self, learner):
        self.wcr = None
        self.calibrated = False
        self.learner = learner

    def __repr__(self):
        if self.calibrated:
            return (
                f"WeightedWrapRegressor(learner={self.learner}, "
                f"calibrated={self.calibrated}, "
                f"predictor={self.wcr})"
            )
        else:
            return f"WrapRegressor(learner={self.learner}, calibrated={self.calibrated})"

    def fit(self, X, y, **kwargs):
        """
        Fit learner.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features),
           set of objects
        y : array-like of shape (n_samples,),
            target values
        kwargs : optional arguments
           any additional arguments are forwarded to the
           ``fit`` method of the ``learner`` object

        Returns
        -------
        None

        Examples
        --------
        Assuming ``X_train`` and ``y_train`` to be an array and vector
        with training objects and labels, respectively, a random
        forest may be wrapped and fitted by:

        .. code-block:: python

           from sklearn.ensemble import RandomForestRegressor
           from crepes import WrapRegressor

           rf = WrapRegressor(RandomForestRegressor())
           rf.fit(X_train, y_train)

        Note
        ----
        The learner, which can be accessed by ``rf.learner``, may be fitted
        before as well as after being wrapped.

        Note
        ----
        All arguments, including any additional keyword arguments, to
        :meth:`.fit` are forwarded to the ``fit`` method of the learner.
        """
        self.learner.fit(X, y, **kwargs)

    def predict(self, X):
        """
        Predict with learner.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features),
           set of objects

        Returns
        -------
        y : array-like of shape (n_samples,),
            values predicted using the ``predict``
            method of the ``learner`` object.

        Examples
        --------
        Assuming ``w`` is a :class:`.WrapRegressor` object for which the wrapped
        learner ``w.learner`` has been fitted, (point) predictions of the
        learner can be obtained for a set of test objects ``X_test`` by:

        .. code-block:: python

           y_hat = w.predict(X_test)

        The above is equivalent to:

        .. code-block:: python

           y_hat = w.learner.predict(X_test)
        """
        return self.learner.predict(X)

    def calibrate(self, X, y, sigmas=None, conformal_weight=None, bins=None, oob=False):
        """
        Fit a :class:`.ConformalRegressor` or
        :class:`.ConformalPredictiveSystem` using learner.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features),
           set of objects
        y : array-like of shape (n_samples,),
            target values
        sigmas: array-like of shape (n_samples,), default=None
            difficulty estimates
        bins : array-like of shape (n_samples,), default=None
            Mondrian categories
        oob : bool, default=False
           use out-of-bag estimation
        cps : bool, default=False
            if cps=False, the method fits a :class:`.ConformalRegressor`
            and otherwise, a :class:`.ConformalPredictiveSystem`

        Returns
        -------
        self : object
            The :class:`.WrapRegressor` object is updated with a fitted
            :class:`.ConformalRegressor` or :class:`.ConformalPredictiveSystem`

        Examples
        --------
        Assuming ``X_cal`` and ``y_cal`` to be an array and vector,
        respectively, with objects and labels for the calibration set,
        and ``w`` is a :class:`.WrapRegressor` object for which the learner
        has been fitted, a standard conformal regressor is formed by:

        .. code-block:: python

           w.calibrate(X_cal, y_cal)

        Assuming that ``sigmas_cal`` is a vector with difficulty estimates,
        a normalized conformal regressor is obtained by:

        .. code-block:: python

           w.calibrate(X_cal, y_cal, sigmas=sigmas_cal)

        Assuming that ``bins_cals`` is a vector with Mondrian categories (bin
        labels), a Mondrian conformal regressor is obtained by:

        .. code-block:: python

           w.calibrate(X_cal, y_cal, bins=bins_cal)

        A normalized Mondrian conformal regressor is generated in the
        following way:

        .. code-block:: python

           w.calibrate(X_cal, y_cal, sigmas=sigmas_cal, bins=bins_cal)

        By providing the option ``oob=True``, the conformal regressor
        will be calibrating using out-of-bag predictions, allowing
        the full set of training objects (``X_train``) and labels (``y_train``)
        to be used, e.g.,

        .. code-block:: python

           w.calibrate(X_train, y_train, oob=True)

        By providing the option ``cps=True``, each of the above calls will instead
        generate a :class:`.ConformalPredictiveSystem`, e.g.,

        .. code-block:: python

           w.calibrate(X_cal, y_cal, sigmas=sigmas_cal, cps=True)

        Note
        ----
        Enabling out-of-bag calibration, i.e., setting ``oob=True``, requires
        that the wrapped learner has an attribute ``oob_prediction_``, which
        e.g., is the case for a ``sklearn.ensemble.RandomForestRegressor``, if
        enabled when created, e.g., ``RandomForestRegressor(oob_score=True)``

        Note
        ----
        The use of out-of-bag calibration, as enabled by ``oob=True``,
        does not come with the theoretical validity guarantees of the regular
        (inductive) conformal regressors and predictive systems, due to that
        calibration and test instances are not handled in exactly the same way.
        """
        if oob:
            residuals = y - self.learner.oob_prediction_
        else:
            residuals = y - self.predict(X)
        self.wcr = WeightedConformalRegressor()
        self.wcr.fit(residuals, sigmas=sigmas, conformal_weight=conformal_weight, bins=bins)
        self.calibrated = True
        return self

    def predict_int(
        self,
        X,
        sigmas=None,
        conformal_weight=None,
        bins=None,
        confidence=0.95,
        acount_for_small_bins=False,
        y_min=-np.inf,
        y_max=np.inf,
    ):
        """
        Predict interval using fitted :class:`.ConformalRegressor` or
        :class:`.ConformalPredictiveSystem`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features),
           set of objects
        sigmas : array-like of shape (n_samples,), default=None
            difficulty estimates
        bins : array-like of shape (n_samples,), default=None
            Mondrian categories
        confidence : float in range (0,1), default=0.95
            confidence level
        y_min : float or int, default=-numpy.inf
            minimum value to include in prediction intervals
        y_max : float or int, default=numpy.inf
            maximum value to include in prediction intervals

        Returns
        -------
        intervals : ndarray of shape (n_samples, 2)
            prediction intervals

        Examples
        --------
        Assuming that ``X_test`` is a set of test objects and ``w`` is a
        :class:`.WrapRegressor` object that has been calibrated, i.e.,
        :meth:`.calibrate` has been applied, prediction intervals at the
        99% confidence level can be obtained by:

        .. code-block:: python

           intervals = w.predict_int(X_test, confidence=0.99)

        Assuming that ``sigmas_test`` is a vector with difficulty estimates for
        the test set and ``w`` is a :class:`.WrapRegressor` object that has been
        calibrated with both residuals and difficulty estimates, prediction
        intervals at the default (95%) confidence level can be obtained by:

        .. code-block:: python

           intervals = w.predict_int(X_test, sigmas=sigmas_test)

        Assuming that ``bins_test`` is a vector with Mondrian categories (bin
        labels) for the test set and ``w`` is a :class:`.WrapRegressor` object
        that has been calibrated with both residuals and bins, the following
        provides prediction intervals at the default confidence level, where the
        intervals are lower-bounded by 0:

        .. code-block:: python

           intervals = w.predict_int(X_test, bins=bins_test, y_min=0)

        Note
        ----
        In case the specified confidence level is too high in relation to the
        size of the calibration set, a warning will be issued and the output
        intervals will be of maximum size.

        Note
        ----
        Note that ``sigmas`` and ``bins`` will be ignored by
        :meth:`.predict_int`, if the :class:`.WrapRegressor` object has been
        calibrated without specifying any such values.

        Note
        ----
        Note that an error will be reported if ``sigmas`` and ``bins`` are not
        provided to :meth:`.predict_int`, if the :class:`.WrapRegressor` object
        has been calibrated with such values.
        """
        if not self.calibrated:
            raise RuntimeError(("predict_int requires that calibrate has been" "called first"))
        else:
            y_hat = self.learner.predict(X)
            return self.wcr.predict(
                y_hat,
                sigmas=sigmas,
                conformal_weight=conformal_weight,
                bins=bins,
                confidence=confidence,
                acount_for_small_bins=acount_for_small_bins,
                y_min=y_min,
                y_max=y_max,
            )

    def evaluate(
        self,
        X,
        y,
        sigmas=None,
        conformal_weight=None,
        bins=None,
        confidence=0.95,
        y_min=-np.inf,
        y_max=np.inf,
        metrics=None,
    ):
        """
        Evaluate :class:`.ConformalRegressor` or
        :class:`.ConformalPredictiveSystem`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
           set of objects
        y : array-like of shape (n_samples,)
            correct target values
        sigmas : array-like of shape (n_samples,), default=None,
            difficulty estimates
        bins : array-like of shape (n_samples,), default=None,
            Mondrian categories
        confidence : float in range (0,1), default=0.95
            confidence level
        y_min : float or int, default=-numpy.inf
            minimum value to include in prediction intervals
        y_max : float or int, default=numpy.inf
            maximum value to include in prediction intervals
        metrics : a string or a list of strings, default=list of all
            metrics; for a learner wrapped with a conformal regressor
            these are "error", "eff_mean","eff_med", "time_fit", and
            "time_evaluate", while if wrapped with a conformal predictive
            system, the metrics also include "CRPS".

        Returns
        -------
        results : dictionary with a key for each selected metric
            estimated performance using the metrics

        Examples
        --------
        Assuming that ``X_test`` is a set of test objects, ``y_test`` is a
        vector with true targets, ``sigmas_test`` and ``bins_test`` are
        vectors with difficulty estimates and Mondrian categories (bin labels)
        for the test set, and ``w`` is a calibrated :class:`.WrapRegressor`
        object, then the latter can be evaluated at the 90% confidence level
        with respect to error, mean and median efficiency (interval size) by:

        .. code-block:: python

           results = w.evaluate(X_test, y_test, sigmas=sigmas_test,
                                bins=bins_test, confidence=0.9,
                                metrics=["error", "eff_mean", "eff_med"])

        Note
        ----
        If included in the list of metrics, "CRPS" (continuous-ranked
        probability score) will be ignored if the :class:`.WrapRegressor` object
        has been calibrated with the (default) option ``cps=False``, i.e., the
        learner is wrapped with a :class:`.ConformalRegressor`.

        Note
        ----
        The use of the metric ``CRPS`` may consume a lot of memory, as a matrix
        is generated for which the number of elements is the product of the
        number of calibration and test objects, unless a Mondrian approach is
        employed; for the latter, this number is reduced by increasing the number
        of bins.

        Note
        ----
        The reported result for ``time_fit`` only considers fitting the
        conformal regressor or predictive system; not for fitting the
        learner.
        """
        if not self.calibrated:
            raise RuntimeError(("evaluate requires that calibrate has been" "called first"))
        else:
            y_hat = self.learner.predict(X)
            return self.wcr.evaluate(
                y_hat,
                y,
                sigmas=sigmas,
                conformal_weight=conformal_weight,
                bins=bins,
                confidence=confidence,
                y_min=y_min,
                y_max=y_max,
            )


class NaiveWCP:
    def __init__(
        self,
        y0_estimator,
        y1_estimator,
        adaptive_conformal=False,
        difficulty_estimator=None,
    ):
        self.y0_estimator = WeightedWrapRegressor(y0_estimator)
        self.y1_estimator = WeightedWrapRegressor(y1_estimator)
        self.adaptive_conformal = adaptive_conformal
        self.difficulty_estimator = difficulty_estimator
        if self.adaptive_conformal:
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
        # Fit difficulty estimators if adaptive conformal
        if self.adaptive_conformal:
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
            conformal_weight=w0,
            sigmas=sigmas0,
        )
        self.y1_estimator.calibrate(
            X_cal[W_cal == 1],
            y_cal[W_cal == 1],
            conformal_weight=w1,
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
        if self.adaptive_conformal:
            sigmas0 = self.de_y0.apply(X)
            sigmas1 = self.de_y1.apply(X)
        else:
            sigmas0 = None
            sigmas1 = None
        y0 = self.y0_estimator.predict_int(
            X, sigmas=sigmas0, conformal_weight=w0, confidence=adj_confidence
        )
        y1 = self.y1_estimator.predict_int(
            X, sigmas=sigmas1, conformal_weight=w1, confidence=adj_confidence
        )
        ite = np.stack((y1[:, 0] - y0[:, 1], y1[:, 1] - y0[:, 0]), axis=1)
        return ite


class NestedWCP:
    def __init__(
        self,
        y0_estimator,
        y1_estimator,
        ite_estimator=None,
        adaptive_conformal=False,
        difficulty_estimator=None,
        exact=False,
    ):
        self.y0_estimator = WeightedWrapRegressor(y0_estimator)
        self.y1_estimator = WeightedWrapRegressor(y1_estimator)
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
        self.adaptive_conformal = adaptive_conformal
        if self.adaptive_conformal:
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
        # Fit difficulty estimators if adaptive conformal
        if self.adaptive_conformal:
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
            conformal_weight=w0_fold1_cal,
            sigmas=sigmas0,
        )
        self.y1_estimator.calibrate(
            X_fold1_cal[W_fold1_cal == 1],
            y_fold1_cal[W_fold1_cal == 1],
            conformal_weight=w1_fold1_cal,
            sigmas=sigmas1,
        )

        # Counterfactual inference on the calibration set
        w0_fold2 = np.ones(len(ps_fold2[W_fold2 == 1]))
        w1_fold2 = np.ones(len(ps_fold2[W_fold2 == 0]))
        if self.exact:
            confidence = 1 - (1 - confidence) / 2
        if self.adaptive_conformal:
            sigmas0 = self.de_y0.apply(X_fold2[W_fold2 == 1])
            sigmas1 = self.de_y1.apply(X_fold2[W_fold2 == 0])
        else:
            sigmas0 = None
            sigmas1 = None
        y0_fold2 = self.y0_estimator.predict_int(
            X_fold2[W_fold2 == 1],
            sigmas0,
            w0_fold2,
            confidence=confidence,
            acount_for_small_bins=False,
        )
        y1_fold2 = self.y1_estimator.predict_int(
            X_fold2[W_fold2 == 0],
            sigmas1,
            w1_fold2,
            confidence=confidence,
            acount_for_small_bins=False,
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
