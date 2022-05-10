"""
    Tools for assessing the results of scBoolSeq's simulation data.
"""

import typing
import collections
from functools import reduce
import numpy as np
import pandas as pd


from ..simulation import _RandType

ConfInt = collections.namedtuple("ConfInt", ["lower", "upper"])


def summarise_by_criteria(data: pd.DataFrame, criteria: pd.DataFrame) -> pd.DataFrame:
    """Summarise by criteria to get the needed mean vs std plots"""
    return pd.DataFrame(
        {"Mean": data.mean(), "Variance": data.var(), "Category": criteria.Category}
    )


def compare_profiles(
    criteria: pd.DataFrame,
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    label_a: str = "frame_a",
    label_b: str = "frame_b",
):
    """Compare the profile of two samples sharing a single criteria DataFrame"""
    summary_a = summarise_by_criteria(frame_a, criteria)
    summary_b = summarise_by_criteria(frame_b, criteria)
    summary_a["Data"] = label_a
    summary_b["Data"] = label_b
    merged = pd.concat([summary_a, summary_b], axis="rows")
    merged = merged.reset_index()

    return merged


def jaccard_index_with_nans(df1: pd.DataFrame, df2: pd.DataFrame):
    """Compute the Jaccard similarity index between two indetically
    labeled DataFrames. The entry frames are supposed to contain entries
    belonging to the following set := {0.0, 1.0, np.nan}.
    It's intended to compare frames resulting from the binarization of
    synthetic datasets as follows:

    # bash:
    curl -fOL https://github.com/pinellolab/STREAM/raw/master/stream/tests/datasets/Nestorowa_2016/data_Nestorowa.tsv.gz
    >>> import pandas as pd
    >>> from scboolseq import scBoolSeq
    >>> from scboolseq.simulation import random_nan_binariser
    >>> RNG_SEED = 1234
    >>> # we use the transpose method (pd.read_csv(...).T) because scBoolSeq supposes columns represent genes
    >>> # and rows observations and this dataset is oriented conversely.
    >>> data = pd.read_csv("data_Nestorowa.tsv.gz", index_col=0, sep="\t").T
    >>> scbool = scBoolSeq(data=data, r_seed=RNG_SEED).fit().simulation_fit()
    >>> partial_bin = scbool.binarize(data)
    >>> fully_binarized = random_nan_binariser(partial_bin, rng=RNG_SEED)
    >>> # Now we binarize and simulate two datasets and use the function to compare them
    >>> bin1 = scbool.binarize(scbool.simulate(fully_binarized), seed=545)
    >>> bin2 = scbool.binarize(scbool.simulate(fully_binarized), seed=434)
    >>> jaccard_index_with_nans(bin1, bin2)
    >>> # the result
    """
    _columns_equal = bool(
        (df1.columns == df2.columns).cumprod()[-1]
    )  # if they are equal, last value will be 1
    _index_equal = bool((df1.index == df2.index).cumprod()[-1])  # ibidem
    if not (_columns_equal and _index_equal):
        raise ValueError("DataFrames are not identically labeled, aborting comparison.")
    vec1 = df1.fillna(-1.0).values.flatten()
    vec2 = df2.fillna(-1.0).values.flatten()
    intersection = (vec1 == vec2).sum()
    return intersection / (len(vec1) + len(vec2) - intersection)


def bootstrap(
    data: typing.Union[pd.Series, np.array],
    statistic: typing.Callable = np.mean,
    bootstrap_reps: int = 500,
    alpha=0.05,
    rng: typing.Optional[_RandType] = None,
) -> ConfInt:
    """Compute a standard bootstrap (reverse percentile) confidence
    interval of level (1-`alpha`) for the given `statistic`,
    using `bootstrap_reps` bootstrap realisations."""

    # check if the vector contains at least one NaN
    if reduce(lambda f, g: f or g, np.isnan(data)):
        raise ValueError("parameter 'data' contains NaN entries")

    rng = rng or np.random.default_rng()
    rng = (
        np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    )

    _theta: float = statistic(data)
    _boot_samples: np.array = np.array(
        [
            statistic(rng.choice(data, replace=True, size=len(data)))
            for i in range(bootstrap_reps)
        ]
    )
    _boot_dist: np.array = _boot_samples - _theta
    _b = np.quantile(_boot_dist, q=(1 - alpha / 2, alpha / 2))
    return ConfInt(*(_theta - _b[0], _theta - _b[1]))
