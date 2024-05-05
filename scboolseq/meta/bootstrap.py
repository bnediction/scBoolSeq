"""
"""

import typing
import collections
from functools import reduce

import pandas as pd
import numpy as np
import scipy.stats as stats


class ConfInt(collections.namedtuple("ConfInt", ["lower", "upper"])):
    def __contains__(self, value):
        return self.lower <= value <= self.upper

    def __gt__(self, value):
        return self.lower > value and self.upper > value

    def __lt__(self, value):
        return self.lower < value and self.upper < value

    def __ge__(self, value):
        return self.lower >= value and self.upper >= value

    def __le__(self, value):
        return self.lower <= value and self.upper <= value


def bootstrap(
    x: typing.Union[pd.Series, np.array],
    statistic: typing.Callable = np.mean,
    B: int = 500,
    alpha=0.05,
    p=None,
    random_state: typing.Optional[typing.Union[int, np.random.Generator]] = None,
) -> ConfInt:
    """Compute a standard bootstrap (reverse percentile) confidence
    interval of level (1-`alpha`) for the given `statistic`,
    using `B` bootstrap realisations."""
    random_state = np.random.default_rng(random_state)

    # check if the vector contains NaNs,
    if reduce(lambda f, g: f or g, np.isnan(x)):
        raise ValueError("parameter 'x' contains NaN entries")

    _theta: float = statistic(x)
    _boot_samples: np.array = np.array(
        [
            statistic(random_state.choice(x, replace=True, size=len(x), p=p))
            for i in range(B)
        ]
    )
    _boot_dist: np.array = _boot_samples - _theta
    _b = np.quantile(_boot_dist, q=(1 - alpha / 2, alpha / 2))
    return ConfInt(*(_theta - _b[0], _theta - _b[1]))


def aggregate_bootstrap_frame(b_ci_frame: pd.DataFrame, name: str) -> pd.Series:
    """
    b_ci_frame stands for Bootstrap Confidence Interval DataFrame.
    """
    positive_mask = (b_ci_frame > 0).all(axis=0)
    negative_mask = (b_ci_frame < 0).all(axis=0)
    aggregated = pd.Series(np.nan, index=b_ci_frame.columns, name=name)
    aggregated[positive_mask] = 1.0
    aggregated[negative_mask] = 0.0
    return aggregated
