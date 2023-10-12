""" Custom types """

# STDLIB
import typing
from pathlib import Path

# DATA
import numpy as np
import pandas as pd

# ESTIMATORS
from sklearn.base import BaseEstimator, TransformerMixin

# Filesystem
_PathLike = typing.Union[Path, str]
# Python data types
_OptionalDict = typing.Optional[typing.Dict[str, typing.Any]]
_OptionalCallable = typing.Optional[typing.Callable]
# Numerical data types
_ArrayOrFrame = typing.Union[np.ndarray, pd.DataFrame]
_ArrayOrSeries = typing.Union[np.ndarray, pd.Series]
_sklearnArrayOrFrameCheck = typing.Union[str, _ArrayOrFrame]

_sklearnEstOrTrans = typing.Union[BaseEstimator, TransformerMixin]
