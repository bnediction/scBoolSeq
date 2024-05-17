"""
    Module to simulate gene expression count data
    from learnt criteria.
"""

import warnings
import functools
import multiprocessing as mp

from typing import Tuple, Callable, List, Optional, Union, Iterable, Dict, Iterator

import scipy.stats as ss
import scipy.optimize as opt
import numpy as np
import pandas as pd

# import sklearn
# from sklearn import preprocessing as preproc
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

from ._core_utils import slice_dispatcher

# Typing aliases
RandomWalkGenerator = Union[Iterator[Dict[str, int]], List[Dict[str, int]]]
SampleCountSpec = Union[int, range, Iterable[int]]
_RandType = Union[np.random.Generator, int]

# TODO: remove this rng
# module-wide numpy random number generator instance
_GLOBAL_RNG = np.random.default_rng()

# Constants
_DEFAULT_BOOL_DIST_CLASSIFIER = KNeighborsClassifier(n_neighbors=15, weights="distance")
_PROBA_CLIP_BOUNDARY: float = 0.95
_DEFAULT_HALF_LIFE_CALCULATION: str = "arithmetic"
# ^ Numerically stable, no other method is consistently or significantly better.
# It also has a natural, explainable biological significance
_EXP_EPSILON: float = 1.0
# ^ Shorthand for _EXPONENTIAL_EPSILON
# It was set to 1.0 and not a real machine epsilon such as `np.finfo(float).eps`
# because the machine epsilon can lead to numerical instability when calculating
# the harmonic and geometric means.
_EXP_HALF_LIFE_CALCULATION_MODES = {
    "arithmetic": np.mean,
    "harmonic": ss.hmean,
    "geometric": ss.gmean,
    "median": np.median,
    "midpoint": lambda x: (np.max(x) + np.min(x)) / 2.0,
}
_VALID_DROPOUT_MODES = ("uniform", "exponential", "sigmoid")
_DEFAULT_DROPOUT_MODE = "exponential"
_MOMENTS = moments = ["Mean", "Variance", "Skewness", "Kurtosis"]


def set_module_rng_seed(seed: int) -> np.random.Generator:
    """Fix the module-wide random number generator
    as well as the R seed.

    parameters
    ----------
            seed : any seed accepted by numpy.random.default_rng
                   see `help(numpy.random.default_rng)`
                   for more details.

    returns
    -------
            The package-wide global random number generator
            (result of calling `numpy.random.default_rng(seed)`)

    Caveats:
        TL;DR
            If the seed is not an integer, it will not be set on
            the embedded R instance.

        Seed has a type hint 'int' because this is the recommended
        type. This function attempts to set the seed both on the
        python (numpy) and R sides. To our knowledge, there is no
        explicit conversion rule provided by rpy2 for other types
        of seeds accepted by numpy.random.default_rng().
        If a numpy object is provided to seed the global generator
        (such as SeedSequence, BitGenerator, Generator), nothing
        will happen on the R side and a warning message will be
        raised printed.
    """
    # declare global to override the generator
    # that has been imported on different sections
    # pylint: disable=global-statement
    global _GLOBAL_RNG
    # pylint: enable=global-statement
    _GLOBAL_RNG = np.random.default_rng(seed)

    return _GLOBAL_RNG


def __sim_zero_inf(
    _lambda: float,
    size: int,
    ignore_deprecation: bool = False,
    rng: Optional[_RandType] = None,
) -> np.ndarray:
    """DEPRECATED . Simulate zero-inflated genes
    This function samples from an exponential distribution with parameter
    lambda
    """
    rng = rng or _GLOBAL_RNG
    rng = (
        np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    )
    if not ignore_deprecation:
        __err_message = ["Error: ", "ZeroInf genes cannot be directly sampled"]
        raise DeprecationWarning("".join(__err_message))
    return rng.exponential(scale=1 / _lambda, size=size)


def _sim_unimodal(
    mean: float, std_dev: float, size: int, rng: Optional[_RandType] = None
) -> np.ndarray:
    """Simulate the expression of unimodal genes using
    a normal (gaussian) distribution."""
    rng = rng or _GLOBAL_RNG
    rng = (
        np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    )
    return rng.normal(loc=mean, scale=std_dev, size=size)


def _sim_bimodal(
    mean1: float,
    mean2: float,
    std_dev_1: float,
    std_dev_2: float,
    weights: Tuple[float],
    size: int,
    rng: Optional[_RandType] = None,
):
    """Simulate bimodal genes. The modelling is achieved using a gaussian mixture.

    Parametres
    ----------

    Weights should be a tuple (or eventually a list) containing
    """
    rng = rng or _GLOBAL_RNG
    rng = (
        np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    )
    # Parameters of the mixture components
    # if isinstance(std_dev, (list, tuple, np.ndarray)):
    #    std_dev_1, std_dev_2 = std_dev
    # else:
    #    std_dev_1 = std_dev_2 = std_dev
    norm_params = np.array([[mean1, std_dev_1], [mean2, std_dev_2]])
    # A stream of indices from which to choose the component
    mixture_idx = rng.choice(len(weights), size=size, replace=True, p=weights)
    # y is the mixture sample
    return np.fromiter(
        (ss.norm.rvs(*(norm_params[i]), random_state=rng) for i in mixture_idx),
        dtype=np.float64,
    )


def _dropout_mask(
    dropout_rate: float, size: int, rng: Optional[_RandType] = None
) -> np.ndarray:
    """Dropout mask to obtain the same dropout_rate as originally estimated"""
    rng = rng or _GLOBAL_RNG
    rng = (
        np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    )
    return rng.choice(
        (0, 1), size=size, replace=True, p=(dropout_rate, 1.0 - dropout_rate)
    )


def _estimate_exponential_parameter(
    data: Union[np.ndarray, pd.Series],
    mode: str = _DEFAULT_HALF_LIFE_CALCULATION,
) -> float:
    """Estimate \\beta_1 for the exponential decay model
    of drop-out probabilities.

    Probabilistic model:

    $$
        p = \\beta_0 * e^{\\beta_1 * (x - x_0)}
    $$

    Note: \\beta_0 can be easily calculated as follows:

        \\beta_0 = 2.0 * criteria["DropOutRate"]

    This is the actual computation performed in
    `scboolseq.scBoolSeq.simulation_fit`

    returns:
    -------
        \\beta_1
    """
    positive_entries = data > 0.0
    positive_data = data[positive_entries]

    _min_positive_value = np.min(positive_data)
    x = positive_data - _min_positive_value + _EXP_EPSILON
    _mu = _EXP_HALF_LIFE_CALCULATION_MODES[mode](x)

    # b_0 is now calculated in scboolseq.scBoolSeq.simulation_fit
    # b_0 = 2.0 * dropout_rate
    b_1 = np.log(2.0) / _mu

    return b_1


def _exponential_normalization_constant(data, lambda_):
    """Give a normalization constant to match the desired DOR"""
    # The integral was calculated analytically
    upper, lower = data.min(), data.max()
    I = lambda x: np.exp(-lambda_ * x) * (lambda_ * x + 1)
    return I(upper) - I(lower)


def _generate_objective_function_exponential(data, tau):
    """tau is the desired dropout rate"""

    def objective_function(params):
        _beta, _lambda = params
        objective = data.shape[0] * tau
        probas = _beta * np.exp(-_lambda * data)
        probas[probas >= 0.95] = 0.90
        return np.power(probas.sum() - objective, 2)

    # def objective_function(params):
    #    _lambda = params
    #    objective = data.shape[0] * tau
    #    probas = np.exp(-_lambda * data)
    #    # probas[probas >= 0.95] = 0.95
    #    return np.power(probas.sum() - objective, 2)

    return objective_function


# """Simulate dropout using an exponential decay probabilistic strategy
# """Simulate dropout using an exponential decay probabilistic strategy"""


def get_exponential_dropout_probabilities(
    data: Union[np.ndarray, pd.Series],
    b_0: float,
    b_1: float,
    rng: Optional[_RandType] = None,
) -> Union[np.ndarray, pd.Series]:
    """Simulate dropout using an exponential decay probabilistic strategy.

    Args:
        data (Union[np.ndarray, pd.Series]): Expression values subject to dropout
        b_0 (float): reference dropout rate
        b_1 (float): lambda rate parameter
        rng (Optional[_RandType], optional): Integer or np.random.default_rng() instance. Defaults to None.

    Returns:
        Union[np.ndarray, pd.Series]: _description_
    """
    rng = rng or _GLOBAL_RNG
    rng = (
        np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    )

    decayed = data.copy()
    positive_entries = data > 0.0
    positive_data = data[positive_entries]

    _min_positive_value = np.min(positive_data)
    x = positive_data

    raw_p = np.exp(-b_1 * x)
    b_0_opt = (x.shape[0] * b_0) / raw_p.sum()  # analytical optimum

    dropout_probabilities = b_0_opt * raw_p
    correction_mask = dropout_probabilities >= _PROBA_CLIP_BOUNDARY
    ## Clipped version:
    dropout_probabilities[correction_mask] = _PROBA_CLIP_BOUNDARY

    return dropout_probabilities


def exponential_decay_dropout(
    data: Union[np.ndarray, pd.Series],
    b_0: float,
    b_1: float,
    rng: Optional[_RandType] = None,
) -> Union[np.ndarray, pd.Series]:
    """Simulate dropout using an exponential decay probabilistic strategy.
    b_1 is

    Args:
        data (Union[np.ndarray, pd.Series]): _description_
        b_0 (float): _description_
        b_1 (float): _description_
        rng (Optional[_RandType], optional): _description_. Defaults to None.

    Returns:
        Union[np.ndarray, pd.Series]: _description_
    """
    rng = rng or _GLOBAL_RNG
    rng = (
        np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    )

    decayed = data.copy()
    positive_entries = data > 0.0
    positive_data = data[positive_entries]

    _min_positive_value = np.min(positive_data)
    x = positive_data

    raw_p = np.exp(-b_1 * x)
    b_0_opt = (x.shape[0] * b_0) / raw_p.sum()  # analytical optimum

    dropout_probabilities = b_0_opt * raw_p
    correction_mask = dropout_probabilities >= _PROBA_CLIP_BOUNDARY
    ## Clipped version:
    dropout_probabilities[correction_mask] = _PROBA_CLIP_BOUNDARY
    ## Stochastic version
    # dropout_probabilities[correction_mask] = rng.uniform(
    #    0.95, 0.99, size=correction_mask.sum()
    # )
    dropout_events = rng.binomial(1, p=dropout_probabilities)
    dropout_mask = 1.0 - dropout_events
    decayed[positive_entries] *= dropout_mask

    return decayed


def _generate_objective_function_sigmoid(data, tau, center: Callable = np.mean):
    """tau is the desired dropout rate"""

    def objective_function(params):
        lambda_ = params
        objective = data.shape[0] * tau
        probas = 1 - 1 / (1 + np.exp(-lambda_ * data + center(data)))
        return np.power(probas.sum() - objective, 2)

    return objective_function


def _sigmoid_dropout_probabilities(data, lambda_: float, center: Callable = np.mean):
    """determine sigmoid dropout probabilities"""
    p = 1 - 1 / (1 + np.exp(-lambda_ * data + center(data)))
    return p


def sigmoid_decay_dropout(
    data: Union[np.ndarray, pd.Series],
    correction_dor: float,
    lambda_: float,
    rng: Optional[_RandType] = None,
):
    """Simulate dropout events
    Probabilities are determined by an inverse sigmoid function
    """
    # setup rng
    rng = rng or _GLOBAL_RNG
    rng = (
        np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    )

    # select the positive entries for simulating dropout
    decayed = data.copy()
    positive_entries = data > 0.0
    positive_data = data[positive_entries]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Use scipy to to optimise the shape of the sigmoid
        obj_func = _generate_objective_function_sigmoid(positive_data, correction_dor)
        constraints = ({"type": "ineq", "fun": lambda x: np.array([x[0]])},)
        bounds = ((0.0001, 100),)
        x0 = np.array([lambda_])  # our initial guess is the estimated half-life param
        # np.log(2.0) / np.mean(data)
        opt_res = opt.minimize(obj_func, x0=x0, constraints=constraints, bounds=bounds)
        dropout_probabilities = _sigmoid_dropout_probabilities(
            positive_data, lambda_=opt_res["x"]
        )

    dropout_events = rng.binomial(1, p=dropout_probabilities)
    dropout_mask = 1.0 - dropout_events
    decayed[positive_entries] *= dropout_mask

    return decayed


def sample_gene(
    criterion: pd.Series,
    n_samples: int,
    enforce_dropout_rate: bool = True,
    rng: Optional[_RandType] = None,
    dropout_mode: str = _DEFAULT_DROPOUT_MODE,
) -> pd.Series:
    """Simulate the expression of a gene, using the information provided by
    the criteria dataframe of a profile_binr.scBoolSeq class.

    Parametres
    ----------

    criterion : an entry (row) of the criteria dataframe of a scBoolSeq class,
    trained on the dataset which you want to sample.

    n_samples : number of samples to generate

    enforce_dropout_rate : should random entries of the gene be set to zero
    in order to preserve the dropout rate estimated whilst computing the criteria
    for the original expression dataset ?
    """
    if dropout_mode not in _VALID_DROPOUT_MODES:
        raise ValueError(
            f"Invalid dropout mode '{dropout_mode}'. Valid modes are: {_VALID_DROPOUT_MODES}"
        )
    rng = rng or _GLOBAL_RNG
    rng = (
        np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    )
    _data: np.array

    if criterion["Category"] == "Discarded":
        _data = np.full(n_samples, np.nan)
        return pd.Series(data=_data, name=criterion.name)

    if criterion["Category"] == "Unimodal":
        _data = _sim_unimodal(
            mean=criterion["MeanNZ"],
            std_dev=np.sqrt(criterion["VarianceNZ"]),
            size=n_samples,
            rng=rng,
        )
    elif criterion["Category"] == "Bimodal":
        _data = _sim_bimodal(
            mean1=criterion["Gaussian_Mean_1"],
            mean2=criterion["Gaussian_Mean_2"],
            std_dev_1=np.sqrt(criterion["Gaussian_Variance_1"]),
            std_dev_2=np.sqrt(criterion["Gaussian_Variance_2"]),
            weights=(
                criterion["Gaussian_Proportion_1"],
                criterion["Gaussian_Proportion_2"],
            ),
            size=n_samples,
            rng=rng,
        )
    elif criterion["Category"] == "ZeroInf":
        __err_message = ["Error: ", "ZeroInf genes cannot be directly sampled"]
        raise DeprecationWarning("".join(__err_message))
    else:
        raise ValueError(f"Unknown category `{criterion['Category']}`, aborting")

    _negative_data = _data < 0.0
    _data[_negative_data] = 0.0
    natural_dor = sum(_negative_data) / len(_data)
    if enforce_dropout_rate and natural_dor < criterion["DropOutRate"]:
        correction_dor = criterion["DropOutRate"] - natural_dor
        if dropout_mode == "uniform":
            _data *= _dropout_mask(dropout_rate=correction_dor, size=n_samples, rng=rng)
        elif dropout_mode == "exponential":
            _data = exponential_decay_dropout(
                data=_data,
                b_0=correction_dor,
                b_1=criterion["Beta_1"],
                rng=rng,
            )
        elif dropout_mode == "exponential_probas":
            _data = get_exponential_dropout_probabilities(
                data=_data,
                b_0=correction_dor,
                b_1=criterion["Beta_1"],
                rng=rng,
            )
        elif dropout_mode == "sigmoid":
            _data = sigmoid_decay_dropout(_data, correction_dor, criterion["Beta_1"])

    return pd.Series(data=_data, name=criterion.name)


def _sample_sequential(
    criteria: pd.DataFrame,
    n_samples: int,
    enforce_dropout_rate: bool = True,
    rng: Optional[_RandType] = None,
    dropout_mode: str = _DEFAULT_DROPOUT_MODE,
) -> pd.DataFrame:
    """Simulate samples from a criteria dataframe, sequentially"""
    # rng = rng or _GLOBAL_RNG # bad idea in most cases: This is not thread safe
    # or more precisely this will generate identical streams
    rng = (
        np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    )
    return (
        criteria.apply(
            lambda y: sample_gene(
                y,
                n_samples=n_samples,
                enforce_dropout_rate=enforce_dropout_rate,
                rng=rng,
                dropout_mode=dropout_mode,
            ),
            axis=1,
        ).T.dropna(how="all", axis=1),
    )[0]


def sample_from_criteria(
    criteria: pd.DataFrame,
    n_samples: int,
    enforce_dropout_rate: bool = True,
    n_threads: int = mp.cpu_count(),
    dropout_mode: str = _DEFAULT_DROPOUT_MODE,
) -> pd.DataFrame:
    """
    Create a new expression dataframe from a criteria dataframe using multithreading

    WARNING : this function has no seeding mechanism and will use
    the module's numpy.random.Generator which is not thread-safe.
    """
    _partial_simulation_function: Callable = functools.partial(
        _sample_sequential,
        n_samples=n_samples,
        enforce_dropout_rate=enforce_dropout_rate,
        dropout_mode=dropout_mode,
    )

    if criteria.shape[0] < n_threads:
        n_threads = criteria.shape[0]

    # _df_splitted_ls: List[pd.DataFrame] = np.array_split(criteria, n_threads)
    _df_splitted_ls: List[pd.DataFrame] = [
        criteria.iloc[i, :] for _, i in KFold(n_threads, shuffle=False).split(criteria)
    ]
    with mp.Pool(n_threads) as pool:
        ret_list = pool.map(_partial_simulation_function, _df_splitted_ls)

    return pd.concat(ret_list, axis=1)


def random_nan_binarizer(
    binary_df: pd.DataFrame,
    probs: Tuple[float] = (0.5, 0.5),
    rng: Optional[_RandType] = None,
):
    """Assuming df is a binary matrix produced by calling
    profile_binr.scBoolSeq.binarise() this function should
    randomly resolve all NaN entries to either 1 or 0

    first probability is the probability of resolving
    to one and the second is the probability of resolving
    to zero.
    probs = (p(nan->1), p(nan->0))

    This function operates on a deep copy of the df argument.
    """
    rng = rng or _GLOBAL_RNG
    rng = (
        np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    )
    binary_df = binary_df.copy(deep=True)
    _na_mask = binary_df.isna()
    _size = _na_mask.shape[0] * _na_mask.shape[1]
    binary_df[_na_mask] = rng.choice((1.0, 0.0), size=_size, p=probs).reshape(
        _na_mask.shape
    )

    return binary_df


def random_nan_binariser(*args, **kwargs):
    """alias for scboolseq.simulation.random_nan_binarizer"""
    return random_nan_binarizer(*args, **kwargs)


def simulate_unimodal_distribution(
    value: float,
    mean: float,
    std_dev: float,
    size: int,
    rng: Optional[_RandType] = None,
) -> pd.Series:
    """
    A wrapper for scipy.stats.halfnorm, used to perform
    biased sampling from a Boolean state.

    params:
    ------
            * value   : 0 or 1.
                * 0   : The left half of the normal distribution
                * 1   : The right half of the normal distribution
            * mean    : Mean of the normal distribution whose half should be sampled
            * std_dev : The standard deviation of the normal distribution ... ^
            * size    : The number of random variates to be sampled from the
                        half normal distribution.
    """
    rng = rng or _GLOBAL_RNG
    rng = (
        np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    )
    # random variate right hand side
    _rv_rhs = ss.halfnorm.rvs(loc=mean, scale=std_dev, size=size, random_state=rng)
    # random variate left hand side
    _rv_lhs = mean - (_rv_rhs - mean)
    return _rv_rhs if np.isclose(value, 1.0) else _rv_lhs


def simulate_bimodal_gene(
    binary_gene: pd.Series,
    criterion: pd.Series,
    rng: Optional[_RandType] = None,
    dropout_mode: str = _DEFAULT_DROPOUT_MODE,
) -> pd.Series:
    """
    Simulate bimodal gene expression, by sampling the appropriate
    modality of the Gaussian Mixture used to represent the bimodal
    distribution of the gene.

    params:
    ------
        * binary_gene : A fully determined (only 0 or 1 as entries) Boolean pandas series.
        * criterion : A slice of a criteria DataFrame, containing the criterion
                      needed in order to simulate the given gene.
        * rng : A random number genereator instance, an integer or None.

    returns:
    -------
        * A pandas series, resulting of performing the biased sampling over
          the Boolean states.
    """
    rng = rng or _GLOBAL_RNG
    rng = (
        np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    )
    if binary_gene.name != criterion.name:
        raise ValueError(
            f"Gene mismatch, data: `{binary_gene.name}`, criterion: `{criterion.name}` "
        )
    simulated_normalised_expression = pd.Series(
        np.nan, index=binary_gene.index, name=criterion.name, dtype=float
    )
    one_mask = binary_gene > 0
    zero_mask = one_mask.apply(lambda x: not x)

    simulated_from_ones = ss.norm.rvs(
        loc=criterion["Gaussian_Mean_2"],
        scale=np.sqrt(criterion["Gaussian_Variance_1"]),
        size=sum(one_mask),  # change for one_mask.sum() ?
        random_state=rng,
    )
    simulated_from_zeros = ss.norm.rvs(
        loc=criterion["Gaussian_Mean_1"],
        scale=np.sqrt(criterion["Gaussian_Variance_2"]),
        size=sum(zero_mask),  # change for one_mask.sum() ?
        random_state=rng,
    )
    simulated_negative = simulated_from_zeros < 0.0
    simulated_from_zeros[simulated_negative] = 0.0

    simulated_normalised_expression[one_mask] = simulated_from_ones
    simulated_normalised_expression[zero_mask] = simulated_from_zeros

    return simulated_normalised_expression


def simulate_unimodal_gene(
    binary_gene: pd.Series,
    criterion: pd.Series,
    rng: Optional[_RandType] = None,
    dropout_mode: str = _DEFAULT_DROPOUT_MODE,
) -> pd.Series:
    """
    Simulate unimodal gene expression from Boolean states by sampling
    the appropriate side of a gaussian (under or above the mean).

    params:
    ------
        * binary_gene : A fully determined (only 0 or 1 as entries) Boolean pandas series.
        * criterion : A slice of a criteria DataFrame, containing the criterion
                      needed in order to simulate the given gene.
        * rng : A random number genereator instance, an integer or None.

    returns:
    -------
        * A pandas series, resulting of performing the biased sampling over
          the Boolean states.
    """
    rng = rng or _GLOBAL_RNG
    rng = (
        np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    )
    if binary_gene.name != criterion.name:
        raise ValueError(
            f"Gene mismatch, data: `{binary_gene.name}`, criterion: `{criterion.name}` "
        )
    simulated_normalised_expression = pd.Series(
        0.0, index=binary_gene.index, name=criterion.name, dtype=float
    )
    one_mask = binary_gene > 0
    zero_mask = one_mask.apply(lambda x: not x)

    simulated_from_ones = simulate_unimodal_distribution(
        1.0,
        mean=criterion["Mean"],
        std_dev=np.sqrt(criterion["Variance"]),
        size=sum(one_mask),
        rng=rng,
    )

    simulated_from_zeros = simulate_unimodal_distribution(
        0.0,
        mean=criterion["Mean"],
        std_dev=np.sqrt(criterion["Variance"]),
        size=sum(zero_mask),
        rng=rng,
    )

    simulated_negative = simulated_from_zeros < 0.0
    simulated_from_zeros[simulated_negative] = 0.0
    # natural_dor = sum(simulated_negative) / len(binary_gene)

    simulated_normalised_expression[zero_mask] = simulated_from_zeros
    simulated_normalised_expression[one_mask] = simulated_from_ones

    return simulated_normalised_expression


# maybeTODO: refactor, factorize the dropout step to have it here instead of within the cases.


def simulate_gene(
    binary_gene: pd.Series,
    criterion: pd.Series,
    rng: Optional[_RandType] = None,
    dropout_mode: str = _DEFAULT_DROPOUT_MODE,
) -> pd.Series:
    """Simulate the expression of a gene, using the information provided by
    the criteria dataframe of a profile_binr.scBoolSeq class.

    Parameters
    ----------

    criterion : an entry (row) of the criteria dataframe of a scBoolSeq class,
    trained on the dataset which you want to simulate.

    n_samples : number of samples to generate

    enforce_dropout_rate : should random entries of the gene be set to zero
    in order to preserve the dropout rate estimated whilst computing the criteria
    for the original expression dataset ?
    """
    rng = rng or _GLOBAL_RNG
    rng = (
        np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    )

    sampled_data: pd.Series
    _category = criterion["Category"]
    if _category == "Discarded":
        sampled_data = pd.Series(
            np.nan, index=binary_gene.index, name=criterion.name, dtype=float
        )
    elif criterion["Category"] == "Unimodal":
        sampled_data = simulate_unimodal_gene(binary_gene, criterion, rng=rng)
    elif criterion["Category"] == "Bimodal":
        sampled_data = simulate_bimodal_gene(binary_gene, criterion, rng=rng)
    else:
        raise ValueError(f"Unknown category `{criterion['Category']}`, aborting")

    sample_dor = np.isclose(sampled_data, 0.0).mean()
    if dropout_mode and (sample_dor < criterion["DropOutRate"]):
        _correction_dor = criterion["DropOutRate"] - sample_dor

        if dropout_mode == "uniform":
            sampled_data *= _dropout_mask(
                dropout_rate=_correction_dor,
                size=sampled_data.shape[0],
                rng=rng,
            )
        else:
            sampled_data = exponential_decay_dropout(
                data=sampled_data,
                b_0=_correction_dor,
                b_1=criterion["Beta_1"],
                rng=rng,
            )

    return sampled_data


def _simulate_subset(
    binary_df: pd.DataFrame,
    simulation_criteria: pd.DataFrame,
    rng: Optional[_RandType] = None,
    dropout_mode: str = _DEFAULT_DROPOUT_MODE,
) -> pd.Series:
    """helper function, wrapper for apply"""
    rng = rng or _GLOBAL_RNG
    rng = (
        np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    )
    return binary_df.apply(
        lambda x: simulate_gene(
            x, simulation_criteria.loc[x.name, :], rng=rng, dropout_mode=dropout_mode
        )
    )


def biased_simulation_from_binary_state(
    binary_df: pd.DataFrame,
    simulation_criteria: pd.DataFrame,
    n_threads: Optional[int] = None,
    n_samples: Optional[int] = None,
    seed: Optional[int] = None,
    dropout_mode: str = _DEFAULT_DROPOUT_MODE,
) -> pd.DataFrame:
    """n_threads defaults to multiprocessing.cpu_count()"""

    n_threads = min(abs(n_threads), mp.cpu_count()) if n_threads else mp.cpu_count()
    # verify binarised genes are contained in the simulation criteria index
    if not all(x in simulation_criteria.index for x in binary_df.columns):
        raise ValueError(
            "'binary_df' contains at least one gene for which there is no simulation criterion."
        )
    if n_samples is not None:
        binary_df = pd.concat(n_samples * [binary_df], ignore_index=False)
    # match the order
    simulation_criteria = simulation_criteria.loc[binary_df.columns, :]

    # Generate independent random number generators,
    # with good quality seeds according to :
    # https://numpy.org/doc/stable/reference/random/parallel.html
    _seed_sequence_generator = np.random.SeedSequence(seed)
    child_seeds = _seed_sequence_generator.spawn(n_threads)
    streams = [np.random.default_rng(s) for s in child_seeds]
    modes = len(child_seeds) * [dropout_mode]

    if n_threads > 1:
        n_threads = min(n_threads, simulation_criteria.shape[0])
        # _bws := by worker slices
        _bws = [
            _ws for _, _ws in KFold(n_threads, shuffle=False).split(simulation_criteria)
        ]
        # _criteria_ls = np.array_split(simulation_criteria, n_threads)
        # _binary_ls = np.array_split(binary_df, n_threads, axis=1)
        _criteria_ls = [simulation_criteria.iloc[_ws, :] for _ws in _bws]
        _binary_ls = [binary_df.iloc[:, _ws] for _ws in _bws]
        with mp.Pool(n_threads) as pool:
            ret_list = pool.starmap(
                _simulate_subset, zip(_binary_ls, _criteria_ls, streams, modes)
            )
        return pd.concat(ret_list, axis=1)
    else:
        return _simulate_subset(
            binary_df=binary_df,
            simulation_criteria=simulation_criteria,
            rng=streams[0],
            dropout_mode=dropout_mode,
        )


def _boolean_trajectory_moments(
    trajectory: pd.DataFrame,
) -> pd.DataFrame:
    """Compute scaled (Max abs) moments for a Boolean trajectory"""
    # Here we know that some genes' Boolean values may be highly homogeneous (if not identical
    # over the whole Boolean trace). So this is expected and there is no need to worry the user
    # because the precision loss will be manually compensated.
    # ```
    # RuntimeWarning: Precision loss occurred in moment calculation due to catastrophic cancellation.
    # This occurs when the data are nearly identical. Results may be unreliable.
    # "Skewness": ss.skew(trajectory),
    # "Kurtosis": ss.kurtosis(trajectory),
    # ```
    with warnings.catch_warnings(action="ignore"):
        _boolean_moments: pd.DataFrame = pd.DataFrame(
            {
                "Mean": trajectory.mean(),
                "Variance": trajectory.var(),
                "DropOutRate": (trajectory == 0).mean(),
                "Skewness": ss.skew(trajectory),
                "Kurtosis": ss.kurtosis(trajectory),
            }
        )
    # Here the maxabs scaling is necessary in order to have comparable moments.
    _boolean_moments /= _boolean_moments.abs().max()
    _has_nans = _boolean_moments.isna().sum(axis=1) > 0
    _has_zero_variance = np.isclose(_boolean_moments.Variance, 0.0)
    _all_one = np.isclose(_boolean_moments.Mean, 1.0)
    _all_zero = np.isclose(_boolean_moments.Mean, 0.0)
    assert (
        _has_nans.values == _has_zero_variance
    ).all(), "Check that input is binary (1 and 0 only, with a numerical dtype)"
    # Those with zero variance having all ones are assigned the min
    # skewness because their mass is completely shifted to the right
    _boolean_moments.loc[(_has_zero_variance & _all_one), "Skewness"] = -1.0
    # The same reasoning is applied for mass shifted to the left
    _boolean_moments.loc[(_has_zero_variance & _all_zero), "Skewness"] = 1.0
    # These zero-variance distributions are the most leptokurtic
    _boolean_moments.loc[(_has_zero_variance & (_all_one | _all_zero)), "Kurtosis"] = (
        1.0
    )

    return _boolean_moments


# def make_boolean_classification_pipeline(
#    preprocessing_steps: List[sklearn.base.OneToOneFeatureMixin] = [
#        ("scaler", preproc.MaxAbsScaler())
#    ],
#    classifier: sklearn.base.ClassifierMixin = _DEFAULT_BOOL_DIST_CLASSIFIER,
#    output_type: Optional["str"] = None,
# ):
#    """ """
#    pipeline = Pipeline(
#        [
#            *preprocessing_steps,
#            ("classifier", classifier),
#        ]
#    )
#    if output_type is not None:
#        pipeline.set_output(transform=output_type)
#    return pipeline
