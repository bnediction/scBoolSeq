""" Binarization suite following scikit-learn's transformer API"""

import typing
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import distance

import sklearn
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    _SetOutputMixin,
    OneToOneFeatureMixin,
)

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn import utils as skutils
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import KNeighborsClassifier

from ._core_utils import (
    validated_sklearn_transform,
    with_pandas_output,
    slice_dispatcher,
)
from ._exceptions import NotASubsetOfExpectedColumnsError
from ._scboolseq_core import compute_criteria
from ._types import _sklearnArrayOrFrameCheck, _ArrayOrFrame, _ArrayOrSeries
from .simulation import (
    biased_simulation_from_binary_state,
    _boolean_trajectory_moments,
    _MOMENTS,
)


__all__ = [
    "NullBinarizer",
    "GaussianMixtureBinarizer",
    "SymmetricBinarizer",
    "ZeroInflatedBinarizer",
    "QuantileBinarizer",
    "scBoolSeqBinarizer",
]


class _BaseBinarizer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Abstract Base Class for Binarizer"""

    def _validate_data(
        self,
        X: _sklearnArrayOrFrameCheck = "no_validation",
        y: _sklearnArrayOrFrameCheck = "no_validation",
        reset: bool = True,
        validate_separately: bool = False,
        **check_params,
    ):
        """Perform additional checks when calling BaseEstimator._validate_data"""
        _require_df = getattr(self, "require_df", False)
        if _require_df:
            if not isinstance(X, pd.DataFrame):
                _i = self.__class__.__name__
                raise TypeError(
                    f"{_i}() expected a pandas.DataFrame, got {type(X)} instead."
                )
            _cols = X.columns
            _idx = X.index

            _validated = X
            # pylint: disable=attribute-defined-outside-init
            if reset:
                self.column_names_ = X.columns
            else:
                if not X.columns.isin(self.column_names_).all():
                    raise NotASubsetOfExpectedColumnsError(
                        "New data contains features missing at fit time"
                    )
                if len(X.columns) < len(self.column_names_):
                    _validated = pd.DataFrame(
                        data=0.0, columns=self.column_names_, index=X.index
                    )
                    _validated[X.columns] = X

        else:
            # perhaps it is better to take this out of the else
            # and deal with column subsetting in an outer part.
            _validated = super()._validate_data(
                X=X,
                y=y,
                reset=reset,
                validate_separately=validate_separately,
                **check_params,
            )

        _y_null = True
        if isinstance(_validated, tuple):
            _y_null = False
            X, y = _validated
        else:
            X = _validated

        if _require_df and not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(data=X, columns=_cols, index=_idx)

        return X if _y_null else (X, y)

    def transform(self, X):
        """placeholder"""

    def predict(self, X):
        """Wrapper for self.transform"""
        return self.transform(X)

    def binarize(self, X):
        """Wrapper for self.transform"""
        return self.transform(X)


class NullBinarizer(_BaseBinarizer):
    """Does not binarize. It always returns np.nan
    Created for a consistent interface."""

    def __init__(self, require_df: bool = False):
        self.require_df = require_df

    def fit(self, X, y=None):
        """Validate data and set dummy attribute
        to pass sklearn tests"""
        X = self._validate_data(X)
        # pylint: disable=attribute-defined-outside-init
        self.is_fitted_ = True
        # pylint: enable=attribute-defined-outside-init
        return self

    @validated_sklearn_transform
    def transform(self, X):
        """Return undetermined values"""
        return np.full(X.shape, np.nan)


class GaussianMixtureBinarizer(_BaseBinarizer):
    """Binarize values according to a 2-component
    Gaussian Mixture Model.

    This binarizer supposes each feature has a bimodal distribution.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
    String describing the type of covariance parameters to use.
    Must be one of:

    - 'full': each component has its own general covariance matrix.
    - 'tied': all components share the same general covariance matrix.
    - 'diag': each component has its own diagonal covariance matrix.
    - 'spherical': each component has its own single variance.
    """

    valid_covariance_types = frozenset(("full", "tied", "diag", "spherical"))

    def __init__(
        self,
        confidence: float = 0.95,
        covariance_type: str = "full",
        random_state=None,
        require_df: bool = False,
    ):
        self.confidence = confidence
        if covariance_type not in self.valid_covariance_types:
            raise ValueError(
                f"Invalid covariance_type, pick one of {self.valid_covariance_types}"
            )
        self.covariance_type: str = covariance_type
        self.random_state = random_state
        self.require_df: bool = require_df

    def fit(self, X, y=None):
        """Compute feature-wise Gaussian Mixture for binarization.
        :param: y represents a boolean matrix (same dimensions as X)
                  which indicates the entries to consider for fitting
                  the estimator. Default behaviour (when y = None)
                  is considering all entries"""
        X = self._validate_data(X)
        if y is None:
            y = np.full(X.shape, 1, dtype=bool)

        # pylint: disable=attribute-defined-outside-init
        _gmms = [  # Create a separate Gaussian Mixture for each feature
            GaussianMixture(
                n_components=2,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
            )
            for i in range(X.shape[1])
        ]

        ## Old version (which raises a FutureWarning)
        ## FutureWarning: 'DataFrame.swapaxes' is deprecated and will be removed in a future version.
        ## Please use 'DataFrame.transpose' instead.
        ## Apparently, there is no fix for this (links worked as of Fri 17 May 14:29:54 CEST 2024)
        ## https://github.com/numpy/numpy/issues/24889
        ## https://stackoverflow.com/questions/77576750/futurewarning-dataframe-swapaxes-is-deprecated-and-will-be-removed-in-a-futur
        # self.gaussian_mixtures_ = list(
        #    map(  # Save the trained Gaussian Mixtures
        #        lambda _gmm_, _feature_, _mask_: _gmm_.fit(
        #            _feature_[_mask_]  # Subsetting a df yields the correct dimensions
        #            if isinstance(_feature_, pd.DataFrame)
        #            else _feature_[_mask_].reshape(-1, 1)
        #        ),
        #        _gmms,
        #        np.array_split(X, X.shape[1], axis=1),
        #        np.array_split(y, y.shape[1], axis=1)
        #    )
        # )

        def _gmm_helper(gmm, feature, mask):
            """Train a GaussianMixtureModel (gmm) on a single column."""
            if isinstance(feature, pd.DataFrame):
                return gmm.fit(feature[mask])
            return gmm.fit(feature[mask].reshape(-1, 1))

        self.gaussian_mixtures_ = [
            _gmm_helper(
                _gmms[i],
                slice_dispatcher(X)[:, np.array([i])],
                slice_dispatcher(y)[:, np.array([i])],
            )
            for i in range(len(_gmms))
        ]

        self.means_ = np.concatenate(
            [_gmm.means_.T for _gmm in self.gaussian_mixtures_], axis=0
        )
        # `self.m1_gt_m2_` will be used to reorder probas when needed,
        # so that m_0 < m_1 for all Gaussian Mixtures (needed for unified binarization op.)
        self.m1_gt_m2_ = self.means_[:, 0] > self.means_[:, 1]
        # pylint: enable=attribute-defined-outside-init
        return self

    @validated_sklearn_transform
    def transform(self, X):
        """Binarize X according to the probability of
        belonging to each one of the two modes"""
        y = np.full(X.shape, np.nan)

        # _raw_probas = list(
        #    map(
        #        lambda _gmm_, _feature_: _gmm_.predict_proba(
        #            _feature_.compressed().reshape(-1, 1)
        #            if isinstance(_feature_, np.ma.MaskedArray)
        #            else _feature_
        #        ),
        #        self.gaussian_mixtures_,
        #        np.array_split(X, X.shape[1], axis=1),
        #    )
        # )
        def _maybe_masked_array_proba(gmm, feature):
            """Fit the GaussianMixtureModel `gmm on the given data array `feature`.
            Compress the array if it is masked."""
            if isinstance(feature, np.ma.MaskedArray):
                return gmm.predict_proba(feature.compressed().reshape(-1, 1))
            return gmm.predict_proba(feature)

        _raw_probas = [
            _maybe_masked_array_proba(
                self.gaussian_mixtures_[i],
                slice_dispatcher(X)[:, np.array([i])],
            )
            for i in range(len(self.gaussian_mixtures_))
        ]
        _probas = list(
            map(
                lambda _reversed_, _probas_: (
                    _probas_[:, [1, 0]] if _reversed_ else _probas_
                ),
                self.m1_gt_m2_,
                _raw_probas,
            )
        )
        for i, _proba in enumerate(_probas):
            _false_mask = _proba[:, 0] >= self.confidence
            _true_mask = _proba[:, 1] >= self.confidence
            y[_false_mask, i] = 0
            y[_true_mask, i] = 1

        return y


class SymmetricBinarizer(_BaseBinarizer):
    """Binarize values greater than a user specified threshold.
    The input's distribution is assumed to be symmetric and centered
    around 0."""

    def __init__(self, threshold: float = 0.25, require_df: bool = False):
        self.threshold: float = threshold
        self.require_df: bool = require_df

    def fit(self, X, y=None):
        """Included for API compatibility only. This method has no effect."""
        X = self._validate_data(X)
        return self

    @validated_sklearn_transform
    def transform(self, X):
        """Binarize X by checking if values are smaller or larger than X"""
        y = np.full(X.shape, np.nan)
        _true_mask = np.apply_along_axis(lambda x: x > self.threshold, 1, X)
        _false_mask = np.apply_along_axis(lambda x: x < -self.threshold, 1, X)
        y[_true_mask] = 1
        y[_false_mask] = 0

        return y


class ZeroInflatedBinarizer(_BaseBinarizer):
    """Zero or not binarizer"""

    # pylint: disable=no-member
    def __init__(
        self,
        zeroes_are: float = np.nan,
        threshold: float = np.finfo(float).resolution,
        require_df: bool = False,
    ):
        self.zeroes_are: float = zeroes_are
        self.threshold: float = threshold
        self.require_df: bool = require_df

    # pylint: enable=no-member

    def fit(self, X, y=None):
        """Included for API compatibility only. This method has no effect."""
        X = self._validate_data(X)
        return self

    @validated_sklearn_transform
    def transform(self, X):
        """Zero-or-not binarization."""
        y = np.full(X.shape, self.zeroes_are)
        _true_mask = np.apply_along_axis(lambda x: x > self.threshold, 1, X)
        y[_true_mask] = 1

        return y


class QuantileBinarizer(_BaseBinarizer):
    """Binarize using Tukey Fences for Outlier assignment
    true > upper_quantile + alpha * IQR
    false < lower_quantile - alpha * IQR
    """

    def __init__(
        self,
        margin_quantile: float = 0.05,
        alpha: float = 0.0,
        require_df: bool = False,
    ):
        self.margin_quantile: float = margin_quantile
        self.alpha: float = alpha
        self.require_df: bool = require_df

    def fit(self, X, y=None):
        """Compute feature-wise IQR and margin quantiles
        :param: y is included for sklean API compatibility only."""
        X = self._validate_data(X)
        # pylint: disable=attribute-defined-outside-init
        _margins = np.array([self.margin_quantile, 1.0 - self.margin_quantile])
        self.quantiles_ = np.quantile(X, axis=0, q=_margins)
        self.lower_quantile_ = self.quantiles_[0, :]
        self.upper_quantile_ = self.quantiles_[1, :]
        # self.iqr_ = self.upper_quantile_ - self.lower_quantile_
        _iqr_ = np.quantile(X, axis=0, q=np.array([0.25, 0.75]))
        self.iqr_ = _iqr_[1, :] - _iqr_[0, :]
        self.true_thresh_ = self.upper_quantile_ + self.alpha * self.iqr_
        self.false_thresh_ = self.lower_quantile_ - self.alpha * self.iqr_
        # pylint: enable=attribute-defined-outside-init

        return self

    @validated_sklearn_transform
    def transform(self, X):
        """Binarize X according to the Tukey Fences defined by the
        train data (self.fit(train_data)) and the provided
        margin quantile and alpha multiplier (constructor parameters)."""
        y = np.full(X.shape, np.nan)
        _true_mask = np.apply_along_axis(lambda x: x > self.true_thresh_, 1, X)
        _false_mask = np.apply_along_axis(lambda x: x < self.false_thresh_, 1, X)
        y[_true_mask] = 1
        y[_false_mask] = 0
        return y


class scBoolSeqBinarizer(_BaseBinarizer):
    """Pure Python implementation of scBoolSeq's binarization scheme."""

    valid_covariance_types = GaussianMixtureBinarizer.valid_covariance_types
    zero_inflated_binarizers = {
        "zero_or_not": ZeroInflatedBinarizer,
        "quantile": QuantileBinarizer,
    }

    def __init__(
        self,
        confidence: float = 0.95,
        covariance_type: str = "full",
        margin_quantile: float = 0.05,
        alpha: float = 0.0,
        zeroinf_binarizer: str = "zero_or_not",
        zeroes_are: float = np.nan,
        dor_threshold: float = 0.99,
        amplitude_threshold: float = 10.0,
        half_life_parameter: str = "MeanNZ",
        warm_start: bool = True,
        require_df: bool = True,
    ):
        if covariance_type not in self.valid_covariance_types:
            raise ValueError(
                f"Covariance type should be one of {self.valid_covariance_types}"
            )
        self.confidence: float = confidence
        self.covariance_type: str = covariance_type
        self.margin_quantile: float = margin_quantile
        self.alpha: float = alpha
        self.zeroinf_binarizer: str = zeroinf_binarizer
        self.zeroes_are: float = zeroes_are
        self.dor_threshold: float = dor_threshold
        self.amplitude_threshold: float = amplitude_threshold
        self.half_life_parameter: str = half_life_parameter
        self.warm_start: bool = warm_start
        self.require_df: bool = require_df

    def set_params(self, **params):
        """Call BaseEstimator.set_params() and
        handle the propagation to nested binarizers."""
        # Call BaseEstimator set_params() to handle all standard checks
        _ = super().set_params(**params)

        # Changing the dor threshold requires re-computing criteria
        # (more or less) genes could become 'Discarded'
        if "dor_threshold" in params:
            if hasattr(self, "criteria_"):
                del self.criteria_
            if hasattr(self, "simulation_criteria_"):
                del self.simulation_criteria_

        # Similarly there may be a better way to handle the embedded estimators but
        # this method should work
        if hasattr(self, "estimators_"):
            # Check overlap in the changed parameter(s) and propagate it
            for _estimator in self.estimators_.values():
                if any(param in _estimator.get_params() for param in params):
                    del self.estimators_
                    break

        if hasattr(self, "simulation_estimators_"):
            # Check overlap in the changed parameter(s) and propagate it
            for _estimator in self.simulation_estimators_.values():
                if any(param in _estimator.get_params() for param in params):
                    del self.simulation_estimators_
                    break

        return self

    @with_pandas_output
    def fit(self, X, y=None, simulation=True):
        """Compute feature-wise criteria to classify genes'
        distributions into 4 types:
            * Discarded
            * Zero-Inflated
            * Unimodal
            * Bimodal
        """
        # `_should_reset` helps preventing the estimator instance falling into a
        # invalid/inconsistent state in which the set of transformers
        # and the gene entries of the criteria_ frame are not exactly the same.
        _should_reset = not (self.warm_start and hasattr(self, "column_names_"))
        X = self._validate_data(X, reset=_should_reset)

        # pylint: disable=attribute-defined-outside-init
        if not (self.warm_start and hasattr(self, "criteria_")):
            self.criteria_ = compute_criteria(
                X,
                thresholds=dict(
                    DropOutRate=self.dor_threshold,
                    AmplitudeDenominator=self.amplitude_threshold,
                ),
                simulation_criteria=False,
            )

        if simulation:
            if not (self.warm_start and hasattr(self, "simulation_criteria_")):
                self.simulation_criteria_ = compute_criteria(
                    X,
                    thresholds=dict(DropOutRate=self.dor_threshold),
                    simulation_criteria=True,
                )

        # These nested access and construction are only possible because
        # scBoolSeq's parameters are a superset of all the binarizers
        _zeroinf_bin = self.zero_inflated_binarizers[self.zeroinf_binarizer]
        _estimators = {
            "ZeroInf": _zeroinf_bin(  # Dynamically construct the desired zeroinf binarizer
                **{
                    key: self.get_params()[key]
                    for key in _zeroinf_bin().get_params()
                    if key
                    in self.get_params()  # ZeroInfBinarizer has a threshold param
                    # which is unnecessary to include here.
                }
            ),
            "Bimodal": GaussianMixtureBinarizer(
                confidence=self.confidence,
                covariance_type=self.covariance_type,
                require_df=self.require_df,
            ),
            "Unimodal": QuantileBinarizer(
                margin_quantile=self.margin_quantile,
                alpha=self.alpha,
                require_df=self.require_df,
            ),
            "Discarded": NullBinarizer(require_df=self.require_df),
        }
        self.estimators_ = skutils.Bunch()
        for category, frame in self.criteria_.groupby("Category"):
            self.estimators_.update(
                {category: _estimators[category].fit(X[frame.index])}
            )

        if simulation:
            _sim_estimators = {
                "Bimodal": GaussianMixtureBinarizer(
                    confidence=self.confidence,
                    covariance_type=self.covariance_type,
                )
            }
            self.simulation_estimators_ = skutils.Bunch()
            # _sim_bimodal_genes = self.simulation_criteria_.query("Category == 'Bimodal'").index
            for category, frame in self.simulation_criteria_.groupby("Category"):
                # This check is basically category == 'Bimodal'
                if category in _sim_estimators:
                    self.simulation_estimators_.update(
                        {
                            category: _sim_estimators[category].fit(
                                X=X[frame.index],
                                # For sampling/simulation estimators, we only consider non-zero entries.
                                y=~np.isclose(X[frame.index], 0.0),
                            )
                        }
                    )

            if "Bimodal" in self.simulation_estimators_:
                _bimodal_means = np.concatenate(
                    [
                        gmm.means_.reshape(1, 2)
                        for gmm in self.simulation_estimators_.Bimodal.gaussian_mixtures_
                    ],
                    axis=0,
                )
                # Covariances can be 1 or 2
                # the following _raw_vars_ and _vars ensure that they
                # will have the correct shape in order to be saved as two
                # variances, even if it is one covariance, twice.
                _raw_vars_ = [
                    gmm.covariances_
                    for gmm in self.simulation_estimators_.Bimodal.gaussian_mixtures_
                ]
                _vars = (
                    [np.repeat(_var, 2) for _var in _raw_vars_]
                    if self.covariance_type == "tied"
                    else _raw_vars_
                )
                _bimodal_variances = np.concatenate(
                    [_var.reshape(1, 2) for _var in _vars],
                    axis=0,
                )
                _bimodal_weights = np.concatenate(
                    [
                        gmm.weights_.reshape(1, 2)
                        for gmm in self.simulation_estimators_.Bimodal.gaussian_mixtures_
                    ],
                    axis=0,
                )
                _reversed = _bimodal_means[:, 0] > _bimodal_means[:, 1]
                # Correct the order in each array
                _bimodal_means[_reversed, :] = _bimodal_means[:, [1, 0]][_reversed, :]
                _bimodal_variances[_reversed, :] = _bimodal_variances[:, [1, 0]][
                    _reversed, :
                ]
                _bimodal_weights[_reversed, :] = _bimodal_weights[:, [1, 0]][
                    _reversed, :
                ]

                self.simulation_criteria_.loc[
                    self.simulation_criteria_.query("Category == 'Bimodal'").index,
                    ["Gaussian_Mean_1", "Gaussian_Mean_2"],
                ] = _bimodal_means
                self.simulation_criteria_.loc[
                    self.simulation_criteria_.query("Category == 'Bimodal'").index,
                    ["Gaussian_Variance_1", "Gaussian_Variance_2"],
                ] = _bimodal_variances
                self.simulation_criteria_.loc[
                    self.simulation_criteria_.query("Category == 'Bimodal'").index,
                    ["Gaussian_Proportion_1", "Gaussian_Proportion_2"],
                ] = _bimodal_weights

            self.simulation_criteria_.loc[:, "Beta_1"] = (
                np.log(2.0) / self.criteria_[self.half_life_parameter]
            )

            # Classifier for biased_sampling from boolean states
            # TODO: set this as an hyperparameter of the class
            self.boolean_category_classifier_ = KNeighborsClassifier(
                n_neighbors=15, weights="distance"
            )
            # This attribute is needed in order to prevent assigning Boolean Genes
            # to genes in the reference which cannot be sampled.
            self.kept_genes_: pd.Index = self.criteria_.query(
                "Category != 'Discarded'"
            ).index

            if len(self.kept_genes_) > 0:
                self.scaled_moments_: pd.Dataframe = self.criteria_.loc[
                    self.kept_genes_, _MOMENTS
                ].copy(deep=True)
                self.scaled_moments_ /= self.scaled_moments_.abs().max()

                self.boolean_category_classifier_.fit(
                    self.scaled_moments_, self.criteria_.loc[self.kept_genes_, "Category"]
                )
                self.scaled_moments_.loc[self.kept_genes_, "Category"] = self.criteria_.loc[
                    self.kept_genes_, "Category"
                ]

        return self

    @with_pandas_output
    @validated_sklearn_transform
    def transform(self, X: _ArrayOrFrame) -> _ArrayOrFrame:
        """_summary_

        Args:
            X (_ArrayOrFrame): _description_

        Returns:
            _ArrayOrFrame: _description_
        """
        check_is_fitted(
            self,
            attributes=("criteria_", "estimators_"),
            msg="scBoolSeq partially fitted (or not at all). Please call '.fit()'",
        )
        _frames = []
        for category, frame in self.criteria_.groupby("Category"):
            _frames.append(self.estimators_[category].transform(X[frame.index]))

        _joint = pd.concat(_frames, axis=1)

        return _joint

    @with_pandas_output
    @validated_sklearn_transform
    def inverse_transform(
        self,
        X: _ArrayOrFrame,
        n_jobs=None,
        random_state=None,
        dropout_mode: str = "exponential",
    ) -> _ArrayOrFrame:
        """_summary_

        Args:
            X (_ArrayOrFrame): _description_
            n_jobs (_type_, optional): _description_. Defaults to None.
            random_state (_type_, optional): _description_. Defaults to None.
            dropout_mode (str, optional): _description_. Defaults to "exponential".

        Returns:
            _type_: _description_
        """
        check_is_fitted(
            self,
            attributes=(
                "simulation_criteria_",
                "simulation_estimators_",
            ),
            msg="Call '.fit(simulation=True)' to enable the inverse transform",
        )
        return biased_simulation_from_binary_state(
            X,
            self.simulation_criteria_,
            n_threads=n_jobs,
            seed=random_state,
            dropout_mode=dropout_mode,
        )

    def synthetic_rna_from_boolean_states(self, *args, **kwargs) -> _ArrayOrFrame:
        """Wrapper for self.sample_counts()"""
        return self.sample_counts(*args, **kwargs)

    def sample_counts(
        self,
        X: _ArrayOrFrame,
        n_samples_per_state: int = 1,
        n_jobs: typing.Optional[int] = None,
        random_state=None,
        dropout_mode: str = "exponential",
        dropout_rates: typing.Optional[_ArrayOrSeries] = None,
    ) -> _ArrayOrFrame:
        """_summary_

        Args:
            X (_ArrayOrFrame): _description_
            n_jobs (_type_, optional): _description_. Defaults to None.
            random_state (_type_, optional): _description_. Defaults to None.
            dropout_mode (str, optional): _description_. Defaults to "exponential".

        Returns:
            _type_: _description_
        """
        check_is_fitted(
            self,
            attributes=(
                "simulation_criteria_",
                "simulation_estimators_",
                "boolean_category_classifier_",
            ),
            msg="Call '.fit(simulation=True)' to enable the inverse transform",
        )

        # 0# dropout_rates verification
        dropout_replacement: typing.Optional[pd.Series] = None
        if dropout_rates is not None:
            # If the user provides a Series, they know which dropout rate(s) they want to
            # override. They should all be contained within the Boolean frame.
            if isinstance(dropout_rates, pd.Series):
                if isinstance(X, pd.DataFrame):
                    if not dropout_rates.index.isin(X.columns).all():
                        raise ValueError(
                            "Provided dropout rates for genes"
                            f"not observed in X {dropout_rates.index.difference(X.columns)}"
                        )
                    else:
                        dropout_replacement = dropout_rates
                else:
                    # Passing a pd.Series of dropout_rates for a np.ndarray of samples is not supported.
                    raise TypeError(
                        "Params `dropout_rates` and `X` should be pandas.Series and pandas.DataFrame"
                    )
            elif isinstance(dropout_rates, np.ndarray):
                if dropout_rates.shape[0] != X.shape[1]:
                    raise ValueError(
                        f"Shape mismatch: Provided {dropout_rates.shape[0]} dropout rates"
                        f" for {X.shape[1]} genes."
                        f"Param `dropout_rates` should match exactly the number of genes"
                        "in the Boolean states `X` param."
                    )
                # If the sizes are correct, the sorted dropout matching will be done after
                # matching Boolean distributions to the reference.

        # 1# Boolean distribution classification

        boolean_trace_criteria = _boolean_trajectory_moments(X)[_MOMENTS]
        _category_hat = self.boolean_category_classifier_.predict(
            boolean_trace_criteria
        )
        boolean_trace_criteria.loc[:, "Category"] = pd.Series(
            _category_hat, index=X.columns
        )

        # 2# Compute by-category euclidean distances in scaled moment space
        by_category_distances = {}
        for category, frame in boolean_trace_criteria.groupby("Category"):
            _ref = self.scaled_moments_.query(f"Category == '{category}'")
            by_category_distances.update(
                {
                    category: pd.DataFrame(
                        distance.cdist(frame[_MOMENTS], _ref[_MOMENTS]),
                        columns=_ref.index,
                        index=frame.index,
                    )
                }
            )

        # 3# Match Boolean genes to those of the reference
        ##  by-category, sorted by descending variance, pick the closest match
        ##  and then exclude it from the candidate list (bijective matching).
        taken = pd.Index([], dtype="object")
        best_match = pd.Series()
        # for category in by_cat_distances:
        for category, frame in boolean_trace_criteria.groupby("Category"):
            # for gene, _dists in by_cat_distances[category].iterrows():
            _dists = by_category_distances[category]
            for gene in frame.sort_values(by="Variance", ascending=False).index:
                _ibm = _dists.loc[gene, _dists.columns.difference(taken)].idxmin()
                best_match[gene] = _ibm
                taken = taken.append(pd.Index([_ibm]))

        sample_criteria: pd.DataFrame = (
            self.simulation_criteria_.loc[best_match, :].copy(deep=True)
            # .copy() is needed because overwriting the dropout is a destructive operation
            .set_index(best_match.index)
        )

        # 4# Update dropout rates (if provided by the user).
        if dropout_rates is not None:
            if not dropout_replacement:  # Do sorted matching of the dropout rates.
                sorted_previous_dor: pd.Series = sample_criteria[
                    "DropOutRate"
                ].sort_values()
                _sorted_new_dor: np.ndarray = np.sort(dropout_rates)
                dropout_replacement: pd.Series = pd.Series(
                    _sorted_new_dor, index=sorted_previous_dor.index
                )
            # Save a copy of the reference dropout
            sample_criteria.loc[:, "ReferenceDropOutRate"] = sample_criteria[
                "DropOutRate"
            ]
            # Overwrite it with the new sorted one
            sample_criteria.loc[dropout_replacement.index, "DropOutRate"] = (
                dropout_replacement
            )

        # 5# Perform biased sampling from parametric distributions and dropout simulation
        return biased_simulation_from_binary_state(
            X,
            sample_criteria,
            n_threads=n_jobs,
            n_samples=n_samples_per_state,
            seed=random_state,
            dropout_mode=dropout_mode,
        )
