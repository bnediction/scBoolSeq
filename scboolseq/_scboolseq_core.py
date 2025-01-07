""" """

import typing
import copy
import diptest
import numpy as np
import pandas as pd
import scipy.stats as ss
import statsmodels.api as sm
import warnings
from sklearn.mixture import GaussianMixture

from ._types import _ArrayOrFrame, _OptionalDict

_DEFAULT_THRESHOLDS = {
    "DropOutRate": 0.95,
    "BI": 1.5,
    "Dip": 0.05,
    "Kurtosis": 1,
    "AmplitudeDenominator": 10,
}


def compute_bimodality_index(data) -> float:
    """."""
    # data = data.values if isinstance(data)
    gmm = GaussianMixture(n_components=2, covariance_type="tied")
    gmm.fit(data.reshape(-1, 1))
    means = gmm.means_.flatten()

    std = np.sqrt(gmm.covariances_[0])
    delta = np.abs(means[0] - means[1]) / std

    p = gmm.predict(data.reshape(-1, 1)).mean()
    bim_index_array = delta * np.sqrt(p * (1 - p))
    return bim_index_array[0]


def compute_density_peak(data):
    """Using statsmodels.nonparametric.KDEUnivariate with default parameters"""
    BW_TYPE = 'normal_reference' # This should be the default bandwidth type.
    try:
        bw = sm.nonparametric.bandwidths.select_bandwidth(data, bw=BW_TYPE, kernel=None)
    except RuntimeError:
        warnings.warn(
            "KDE bandwidth computation failed (bandwidth is 0.0); Falling back to NaN.",
            RuntimeWarning,
            stacklevel=2
        )
        return np.nan

    kernel = sm.nonparametric.KDEUnivariate(data)  # data is 1-D vector
    # kernel.fit(bw="silverman") # Tried for debugging, error commes from zero vectors.
    kernel.fit(bw=BW_TYPE)
    return kernel.support[np.argmax(kernel.density)]


def classify_from_criteria(
    criteria: pd.DataFrame,
    thresholds: _OptionalDict = None,
    simulation_criteria: bool = False,
    f_bimodal_index: typing.Callable = compute_bimodality_index,
    data: typing.Optional[_ArrayOrFrame] = None,
):
    """
    Based on the method described in
    "Personalization of logical models with multi-omics data allows clinical stratification of patients"
    Frontiers in Physiology
    Beal et al, 2019.
    """
    _thresholds = copy.deepcopy(_DEFAULT_THRESHOLDS)
    if thresholds is not None:
        _thresholds.update(thresholds)

    if data is not None:
        if isinstance(data, pd.DataFrame):
            data = data.values
    # criteria = criteria.copy(deep=True)
    median_amplitude_thresh = (
        criteria["Amplitude"].median() / _thresholds["AmplitudeDenominator"]
    )

    # New category column
    category = pd.Series(name="Category", index=criteria.index, data=pd.NA)

    # Discarded:
    _ampli = criteria["Amplitude"] < median_amplitude_thresh
    _dor = criteria["DropOutRate"] > _thresholds["DropOutRate"]
    category[criteria.where(_dor | _ampli).dropna().index] = "Discarded"

    # Bimodal:
    _unclassified1 = category.isna()
    # _bi = criteria["BI"] > _thresholds["BI"]
    criteria.loc[:, "BI"] = 0.0
    # ^ 0.0 is not the actual value but used to avoid problems
    # when performing comparisons
    _dip = criteria["Dip"] < _thresholds["Dip"]
    _kurt = criteria["Kurtosis"] < _thresholds["Kurtosis"]
    # possible_bimodal = data[data.columns[_dip & _kurt]]
    _n_poss_bim = (_dip & _kurt).sum()
    if _n_poss_bim:
        possible_bimodal = data[:, _dip & _kurt]

        criteria.loc[_dip & _kurt, "BI"] = np.apply_along_axis(
            f_bimodal_index, 0, possible_bimodal
        )
    _bi = criteria["BI"] > _thresholds["BI"]

    category[criteria.where(_unclassified1 & _bi & _dip & _kurt).dropna().index] = (
        "Bimodal"
    )

    # Zero-Inflated
    if not simulation_criteria:
        _unclassified2 = category.isna()
        _zeroinf = criteria["DenPeak"] < median_amplitude_thresh
        category[criteria.where(_unclassified2 & _zeroinf).dropna().index] = "ZeroInf"

    # Unimodal
    _unclassified3 = category.isna()
    category[_unclassified3] = "Unimodal"

    return criteria.join(category)


def _compute_criteria(
    data: _ArrayOrFrame,
    thresholds: _OptionalDict = None,
    simulation_criteria=False,
    skip_slow=False,
):
    """Compute statistical criteria and call classify_from_criteria()
    in order to classify the distributions.
    Thresholds default to the module constant _DEFAULT_THRESHOLDS.
    If specified, values provided in thresholds will update (replace)
    the defaults."""
    _thresholds = copy.deepcopy(_DEFAULT_THRESHOLDS)
    if thresholds is not None:
        _thresholds.update(thresholds)

    _is_df = False
    if isinstance(data, pd.DataFrame):
        _is_df = True
        _cols = data.columns
        data = data.values

    masked_data = np.ma.masked_where(np.isclose(data, 0.0), data)
    # These quantities are the same for binarization and simulation
    _criteria_dict = {
        "Mean": data.mean(axis=0),
        "MeanNZ": masked_data.mean(axis=0),  # .compressed(),
        "Median": np.median(data, axis=0),
        "MedianNZ": np.ma.median(masked_data, axis=0),  # .compressed(),
        "GeometricMean": ss.gmean(masked_data, axis=0),
        "HarmonicMean": ss.hmean(masked_data, axis=0),
        "Variance": data.var(axis=0),
        "VarianceNZ": masked_data.var(axis=0),  # .compressed(),
        "DropOutRate": (data < 10 * np.finfo(float).eps).mean(axis=0),
    }

    # Change the function definitions to match the np array type
    # pylint: disable=unnecessary-lambda-assignment,unnecessary-lambda
    if simulation_criteria:
        data = masked_data
        # extract the DipTest p-value [0] is the statistic
        _f_dip = lambda _x: diptest.diptest(_x.compressed())[1]
        _f_bim = lambda _x: compute_bimodality_index(_x.compressed())
        _f_den_peak = lambda _x: compute_density_peak(_x.compressed())
    else:
        _f_dip = lambda _x: diptest.diptest(_x)[1]
        _f_bim = lambda _x: compute_bimodality_index(_x)
        _f_den_peak = lambda _x: compute_density_peak(_x)
    # pylint: enable=unnecessary-lambda-assignment,unnecessary-lambda

    _other_criteria = {
        "Amplitude": data.max(axis=0) - data.min(axis=0),
        "Dip": np.apply_along_axis(_f_dip, 0, data),
        "Kurtosis": ss.kurtosis(data, axis=0),
        "Skewness": ss.skew(data, axis=0),
        "DenPeak": np.apply_along_axis(_f_den_peak, 0, data),
    }
    # if not skip_slow:
    #    # Add a Boolean Mask to prevent this from being done
    #    # on all genes. Only those which fullfill the other
    #    # two bimodality criteria should be analysed.
    #    _other_criteria.update(
    #        {
    #            "BI": np.apply_along_axis(_f_bim, 0, data),
    #        }
    #    )

    _criteria_dict.update(_other_criteria)
    _criteria: pd.DataFrame = pd.DataFrame(_criteria_dict)

    if _is_df:
        _criteria.index = _cols

    # return classify_from_criteria(_criteria, _thresholds, simulation_criteria)
    return _criteria, _f_bim, masked_data


def compute_criteria(
    data: _ArrayOrFrame,
    thresholds: _OptionalDict = None,
    simulation_criteria=False,
    skip_slow=False,
):
    """Compute statistical criteria and call classify_from_criteria()
    in order to classify the distributions.
    Thresholds default to the module constant _DEFAULT_THRESHOLDS.
    If specified, values provided in thresholds will update (replace)
    the defaults."""
    _criteria, _f_bim, _masked_data = _compute_criteria(
        data,
        thresholds=thresholds,
        simulation_criteria=simulation_criteria,
        skip_slow=skip_slow,
    )
    return (
        _criteria
        if skip_slow
        else classify_from_criteria(
            _criteria,
            thresholds,
            simulation_criteria,
            _f_bim,
            (_masked_data if simulation_criteria else data),
        )
    )
