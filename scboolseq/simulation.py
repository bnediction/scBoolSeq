"""
    Module to simulate gene expression count data
    from learnt criteria.
"""
import warnings
import functools
import multiprocessing as mp

from typing import Tuple, Callable, List, Optional, Union, Iterable, Dict, Iterator

import scipy.stats as ss
import numpy as np
import pandas as pd

# Typing aliases
RandomWalkGenerator = Union[Iterator[Dict[str, int]], List[Dict[str, int]]]
SampleCountSpec = Union[int, range, Iterable[int]]
_RandType = Union[np.random.Generator, int]

# module-wide numpy random number generator instance
_GLOBAL_RNG = np.random.default_rng()

# Constants
_DEFAULT_HALF_LIFE_CALCULATION = "arithmetic"
# ^ Numerically stable, no other method is consistently or significantly better.
# It also has a natural, explainable biological significance
_EXP_EPSILON = 1.0
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
_VALID_DROUPOUT_MODES = ("uniform", "exponential")
_DEFAULT_DROPOUT_MODE = "exponential"


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
    global _GLOBAL_RNG
    _GLOBAL_RNG = np.random.default_rng(seed)
    if isinstance(seed, int):
        # import is performed within the function,
        # to prevent an import error due to circular import
        from .core import scBoolSeq

        _sc_bool_seq = scBoolSeq(pd.DataFrame())
        _sc_bool_seq.r(f"set.seed({seed})")
    else:
        _no_seed_warning_message_ls = [
            f"seed of type {type(seed)} can not be passed to the embedded R instance",
            "the seed was not set on the R side.",
        ]
        warnings.warn(" ".join(_no_seed_warning_message_ls))

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
    std_dev: float,
    weights: Tuple[float],
    size: int,
    rng: Optional[_RandType] = None,
):
    """Simulate bimodal genes. The modelling is achieved using a gaussian mixture.
    The variance is assumed to be tied i.e. both components of the mixture have the same
    variance.

    Parametres
    ----------

    Weights should be a tuple (or eventually a list) containing
    """
    rng = rng or _GLOBAL_RNG
    rng = (
        np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    )
    # Parameters of the mixture components
    norm_params = np.array([[mean1, std_dev], [mean2, std_dev]])
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


def exponential_decay_dropout(
    data: Union[np.ndarray, pd.Series],
    b_0: float,
    b_1: float,
    rng: Optional[_RandType] = None,
):
    """Simulate dropout using an exponential decay probabilistic strategy"""
    rng = rng or _GLOBAL_RNG
    rng = (
        np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    )

    decayed = data.copy()
    positive_entries = data > 0.0
    positive_data = data[positive_entries]

    # TEST THE NECESSITY OF THE _EXP_EPSILON IN THIS CONTEXT
    _min_positive_value = np.min(positive_data)
    x = positive_data - _min_positive_value  # + _EXP_EPSILON

    dropout_probabilities = b_0 * np.exp(-b_1 * x)  # (x - _EXP_EPSILON))
    dropout_probabilities[dropout_probabilities >= 1.0] = 0.95
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
    if dropout_mode not in _VALID_DROUPOUT_MODES:
        raise ValueError(
            f"Invalid dropout mode '{dropout_mode}'. Valid modes are: {_VALID_DROUPOUT_MODES}"
        )
    rng = rng or _GLOBAL_RNG
    rng = (
        np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    )
    _data: np.array

    if criterion["Category"] == "Discarded":
        _data = np.full(n_samples, np.nan)
    elif criterion["Category"] == "Unimodal":
        _data = _sim_unimodal(
            mean=criterion["mean"],
            std_dev=np.sqrt(criterion["variance"]),
            size=n_samples,
            rng=rng,
        )
        _negative_data = _data < 0.0
        _data[_negative_data] = 0.0
        natural_dor = sum(_negative_data) / len(_data)
        if enforce_dropout_rate and natural_dor < criterion["DropOutRate"]:
            correction_dor = criterion["DropOutRate"] - natural_dor
            if dropout_mode == "uniform":
                _data *= _dropout_mask(
                    dropout_rate=correction_dor, size=n_samples, rng=rng
                )
            else:
                _data = exponential_decay_dropout(
                    data=_data,
                    b_0=criterion["beta_0"],
                    b_1=criterion["beta_1"],
                    rng=rng,
                )
    elif criterion["Category"] == "Bimodal":
        _data = _sim_bimodal(
            mean1=criterion["gaussian_mean1"],
            mean2=criterion["gaussian_mean2"],
            std_dev=np.sqrt(criterion["gaussian_variance"]),
            weights=(criterion["gaussian_prob1"], criterion["gaussian_prob2"]),
            size=n_samples,
            rng=rng,
        )
        _negative_data = _data < 0.0
        _data[_negative_data] = 0.0
        natural_dor = sum(_negative_data) / len(_data)
        if enforce_dropout_rate and natural_dor < criterion["DropOutRate"]:
            correction_dor = criterion["DropOutRate"] - natural_dor
            if dropout_mode == "uniform":
                _data *= _dropout_mask(
                    dropout_rate=correction_dor, size=n_samples, rng=rng
                )
            else:
                _data = exponential_decay_dropout(
                    data=_data,
                    b_0=criterion["beta_0"],
                    b_1=criterion["beta_1"],
                    rng=rng,
                )
    elif criterion["Category"] == "ZeroInf":
        __err_message = ["Error: ", "ZeroInf genes cannot be directly sampled"]
        raise DeprecationWarning("".join(__err_message))
        # _data = __sim_zero_inf(_lambda=criterion["lambda"], size=n_samples, rng=rng)
    else:
        raise ValueError(f"Unknown category `{criterion['Category']}`, aborting")

    return pd.Series(data=_data, name=criterion.name)


def _sample_sequential(
    criteria: pd.DataFrame,
    n_samples: int,
    enforce_dropout_rate: bool = True,
    rng: Optional[_RandType] = None,
    dropout_mode: str = _DEFAULT_DROPOUT_MODE,
) -> pd.DataFrame:
    """Simulate samples from a criteria dataframe, sequentially"""
    rng = rng or _GLOBAL_RNG
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

    _df_splitted_ls: List[pd.DataFrame] = np.array_split(criteria, n_threads)
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
        raise ValueError("Criterion and gene mismatch")
    simulated_normalised_expression = pd.Series(
        np.nan, index=binary_gene.index, name=criterion.name, dtype=float
    )
    one_mask = binary_gene > 0
    zero_mask = one_mask.apply(lambda x: not x)

    simulated_from_ones = ss.norm.rvs(
        loc=criterion["gaussian_mean2"],
        scale=np.sqrt(criterion["gaussian_variance"]),
        size=sum(one_mask),  # change for one_mask.sum() ?
        random_state=rng,
    )
    simulated_from_zeros = ss.norm.rvs(
        loc=criterion["gaussian_mean1"],
        scale=np.sqrt(criterion["gaussian_variance"]),
        size=sum(zero_mask),  # change for one_mask.sum() ?
        random_state=rng,
    )
    # First approach to simulating the DropOutRate,
    # put all negative simulated values to zero
    simulated_negative = simulated_from_zeros < 0.0
    simulated_from_zeros[simulated_negative] = 0.0
    natural_dor = sum(simulated_negative) / len(binary_gene)

    simulated_normalised_expression[one_mask] = simulated_from_ones
    simulated_normalised_expression[zero_mask] = simulated_from_zeros

    if natural_dor < criterion["DropOutRate"]:
        # check how many values do we need to put to zero
        _correction_dor = criterion["DropOutRate"] - natural_dor

        if dropout_mode == "uniform":
            simulated_normalised_expression *= _dropout_mask(
                dropout_rate=_correction_dor,
                size=len(simulated_normalised_expression),
                rng=rng,
            )
        else:
            simulated_normalised_expression = exponential_decay_dropout(
                data=simulated_normalised_expression,
                b_0=criterion["beta_0"],
                b_1=criterion["beta_1"],
                rng=rng,
            )

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
        raise ValueError("Criterion and gene mismatch")
    simulated_normalised_expression = pd.Series(
        0.0, index=binary_gene.index, name=criterion.name, dtype=float
    )
    one_mask = binary_gene > 0
    zero_mask = one_mask.apply(lambda x: not x)

    simulated_from_ones = simulate_unimodal_distribution(
        1.0,
        mean=criterion["mean"],
        std_dev=np.sqrt(criterion["variance"]),
        size=sum(one_mask),
        rng=rng,
    )

    simulated_from_zeros = simulate_unimodal_distribution(
        0.0,
        mean=criterion["mean"],
        std_dev=np.sqrt(criterion["variance"]),
        size=sum(zero_mask),
        rng=rng,
    )

    simulated_negative = simulated_from_zeros < 0.0
    simulated_from_zeros[simulated_negative] = 0.0
    natural_dor = sum(simulated_negative) / len(binary_gene)

    simulated_normalised_expression[zero_mask] = simulated_from_zeros
    simulated_normalised_expression[one_mask] = simulated_from_ones

    if natural_dor < criterion["DropOutRate"]:
        _correction_dor = criterion["DropOutRate"] - natural_dor

        if dropout_mode == "uniform":
            simulated_normalised_expression *= _dropout_mask(
                dropout_rate=_correction_dor,
                size=len(simulated_normalised_expression),
                rng=rng,
            )
        else:
            simulated_normalised_expression = exponential_decay_dropout(
                data=simulated_normalised_expression,
                b_0=criterion["beta_1"],
                b_1=criterion["beta_1"],
                rng=rng,
            )

    return simulated_normalised_expression


def simulate_gene(
    binary_gene: pd.Series,
    criterion: pd.Series,
    rng: Optional[_RandType] = None,
    dropout_mode: str = _DEFAULT_DROPOUT_MODE,
) -> pd.Series:
    """Simulate the expression of a gene, using the information provided by
    the criteria dataframe of a profile_binr.scBoolSeq class.

    Parametres
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

    if criterion["Category"] == "Discarded":
        return pd.Series(
            np.nan, index=binary_gene.index, name=criterion.name, dtype=float
        )
    elif criterion["Category"] == "Unimodal":
        return simulate_unimodal_gene(
            binary_gene, criterion, rng=rng, dropout_mode=dropout_mode
        )
    elif criterion["Category"] == "Bimodal":
        return simulate_bimodal_gene(
            binary_gene, criterion, rng=rng, dropout_mode=dropout_mode
        )
    else:
        raise ValueError(f"Unknown category `{criterion['Category']}`, aborting")


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

    # Generate independent random number generators, with good quality seeds
    # according to :
    # https://numpy.org/doc/stable/reference/random/parallel.html
    _seed_sequence_generator = np.random.SeedSequence(seed)
    child_seeds = _seed_sequence_generator.spawn(n_threads)
    streams = [np.random.default_rng(s) for s in child_seeds]
    modes = len(child_seeds) * [dropout_mode]

    if n_threads > 1:
        _criteria_ls = np.array_split(simulation_criteria, n_threads)
        _binary_ls = np.array_split(binary_df, n_threads, axis=1)
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


def __simulate_from_boolean_trajectory(
    boolean_trajectory_df: pd.DataFrame,
    criteria_df: pd.DataFrame,
    n_samples_per_state: SampleCountSpec = 1,
    n_threads: Optional[int] = None,
    rng_seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    WARNING:  do not use this function, as it need to be debugged.

    Generate `n_samples_per_state`, for each one of the
    states found in `boolean_trajectory_df`.
    The biased simulation from the binary state is performed
    according to `criteria_df`.

    Parameter `n_samples_per_state` is of type SampleCountSpec,
    defined as :
    >>> SampleCountSpec = Union[int, range, Iterable[int]]
    The behaviour changes depending on the type.
    * If an int is given, all states of the boolean trajectory will
      be sampled `n_samples_per_state` times.
    * If a range is given, a random number of samples (within the range)
      will be created for each observation.
    * If a list is given


    If specified, parameter `rng_seed` allows 100% reproductible results,
    which means that given the same set of parameters with the given seed
    will always produce the same pseudo random values.

    The aim of this function is generating synthetic data to be
    used as an input to STREAM, in order to evaluate the performance
    of the PROFILE-based simulation method we have developped.

    Returns
    -------
        A tuple : (simulated_expression_dataframe, metadata)

    """
    n_threads = min(abs(n_threads), mp.cpu_count()) if n_threads else mp.cpu_count()
    # for all runs to obtain the same results the seeds of each run should be fixed
    _rng = np.random.default_rng(rng_seed)
    _simulation_seeds = _rng.integers(
        123, rng_seed, size=len(boolean_trajectory_df.index)
    )
    # Generate independent random number generators,
    # with good quality seeds according to :
    # https://numpy.org/doc/stable/reference/random/parallel.html
    print("Spawning seeds...", flush=True, end="\t")
    _seed_sequence_generator = np.random.SeedSequence(rng_seed)
    child_seeds = _seed_sequence_generator.spawn(n_threads)
    streams = [np.random.default_rng(s) for s in child_seeds]
    print("Done", flush=True)

    _n_states = len(boolean_trajectory_df.index)

    # multiple dispatch for n_samples_per_state
    if isinstance(n_samples_per_state, int):
        sample_sizes = [n_samples_per_state] * _n_states
    elif isinstance(n_samples_per_state, range):
        sample_sizes = _rng.integers(
            n_samples_per_state.start,
            n_samples_per_state.stop,
            size=len(boolean_trajectory_df.index),
        )
    elif isinstance(n_samples_per_state, Iterable):
        sample_sizes = list(n_samples_per_state)
        # check we have enough sample sizes for each one of the observed states
        if not len(sample_sizes) == _n_states:
            raise ValueError(
                " ".join(
                    [
                        "`n_samples_per_state` should contain",
                        f"exactly {_n_states} entries, received {len(sample_sizes)}",
                    ]
                )
            )
    else:
        raise TypeError(
            f"Invalid type `{type(n_samples_per_state)}` for parameter n_samples_per_state"
        )

    # generate synthetic samples
    # synthetic_samples = []
    # for size, rnd_walk_step, _rng_seed in zip(
    #    sample_sizes, boolean_trajectory_df.iterrows(), _simulation_seeds
    # ):
    #    _idx, binary_state = rnd_walk_step

    #    synthetic_samples.append(
    #        biased_simulation_from_binary_state(
    #            binary_state.to_frame().T,
    #            criteria_df,
    #            n_samples=size,
    #            n_threads=n_threads,
    #            seed=_rng_seed,
    #        )
    #        .reset_index()
    #        .rename(columns={"index": "kind"})
    #    )
    # match the simulation order :

    print("Align and split boolean frame and criteria...", flush=True, end="\t")
    criteria_df = criteria_df.loc[boolean_trajectory_df.columns, :]
    _criteria_ls = np.array_split(criteria_df, n_threads)
    _binary_ls_no_sample_size = np.array_split(boolean_trajectory_df, n_threads, axis=1)
    _binary_ls = map(
        lambda frame, size: pd.concat(size * [frame], ignore_index=False),
        _binary_ls_no_sample_size,
        sample_sizes,
    )
    print("Done", flush=True)
    print("Enter multiprocessing pool.", flush=True)
    with mp.Pool(n_threads) as pool:
        raw_synthetic_samples = pool.starmap(
            _simulate_subset, zip(_binary_ls, _criteria_ls, streams)
        )
    print(f"Terminate {n_threads} workers.", flush=True)

    # synthetic_samples = [
    #    sample.reset_index().rename(columns={boolean_trajectory_df.index.name: "kind"})
    #    for sample in raw_synthetic_samples
    # ]
    print("Concatenating all experiments into a single frame...", end="\t", flush=True)
    synthetic_single_cell_experiment = pd.concat(
        (
            sample.reset_index().rename(
                columns={boolean_trajectory_df.index.name: "kind"}
            )
            for sample in raw_synthetic_samples  # merge all samples into a single frame
        ),
        axis="rows",
        ignore_index=True,
    )
    print("Done", flush=True)

    print("Transforming index...", sep="\t", flush=True)
    # create an informative, artificial, and unique index
    synthetic_single_cell_experiment = synthetic_single_cell_experiment.set_index(
        map(
            lambda idx, enum: f"{idx}_{enum}",
            synthetic_single_cell_experiment.index,
            list(range(1, len(synthetic_single_cell_experiment.index) + 1)),
        )
    )
    # synthetic_single_cell_experiment = synthetic_single_cell_experiment.set_index(
    #    synthetic_single_cell_experiment.kind
    #    + "_"
    #    + synthetic_single_cell_experiment.index.to_frame()
    #    .reset_index()
    #    .index.map(lambda x: f"obs{str(x)}")
    # )
    print("Done")

    print("Create colour map...", sep="\t", flush=True)
    # Create a colour map for different cell types
    _RGB_values = list("0123456789ABCDEF")
    color_map = {
        i: "#" + "".join([_rng.choice(_RGB_values) for j in range(6)])
        for i in boolean_trajectory_df.index.unique().to_list()
    }
    print("Done", flush=True)

    # Create a metadata frame
    print("Create metadata frame...", sep="\t", flush=True)
    # cell_colours = (
    #    synthetic_single_cell_experiment.kind.apply(lambda x: color_map[x])
    #    .to_frame()
    #    .rename(columns={"kind": "label_color"})
    # )
    # metadata = pd.concat(
    #    [synthetic_single_cell_experiment.kind, cell_colours], axis="columns"
    # )
    # metadata = metadata.rename(columns={"kind": "label"})
    metadata = (
        pd.DataFrame(  # index arg should not be necessary, both series have the same ?
            {
                "label": synthetic_single_cell_experiment.kind,
                "label_color": synthetic_single_cell_experiment.kind.apply(
                    lambda x: color_map[x]
                ),
            }
        )
    )
    print("Done", flush=True)
    # drop the number of activated genes from the synthetic expression frame
    synthetic_single_cell_experiment = synthetic_single_cell_experiment[
        synthetic_single_cell_experiment.columns[1:]
    ]

    return synthetic_single_cell_experiment, metadata
