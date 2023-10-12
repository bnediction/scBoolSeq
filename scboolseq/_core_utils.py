"""Core utils used to reduce boilerplate 
within the scBoolSeq codebase.
These are not intended to be called directly."""

# standard library
import typing
import functools
from pathlib import Path

# third party
import pandas as pd
from sklearn.utils.validation import check_is_fitted

# local imports
from ._types import _ArrayOrFrame, _sklearnEstOrTrans, _OptionalCallable


def first_arg_is_path(func: typing.Callable):
    """Decorator: Cast the first argument to pathlib.Path"""

    @functools.wraps(func)
    def path_is_casted(*args, **kwargs):
        if not isinstance(args[0], Path):
            _casted_args = (Path(args[0]), *args[1:])
            return func(*_casted_args, **kwargs)
        else:
            return func(*args, **kwargs)

    return path_is_casted


def validated_sklearn_transform(transform_method: typing.Callable):
    """Perform all checks needed to comply with sklearn's API.
    Conceals some extra checks for DataFrames.
    Use this decorator to drastically reduce boilerplate in the
    definition of custom transformers."""

    # pylint: disable=invalid-name
    @functools.wraps(transform_method)
    def _validated_(
        transformer_instance: _sklearnEstOrTrans, X: _ArrayOrFrame, *args, **kwargs
    ):
        """Do the boring part of verifications"""
        # Fundamental:
        check_is_fitted(transformer_instance)
        # pylint: disable=protected-access
        _return_df = False
        if isinstance(X, pd.DataFrame):
            _return_df = True
            _in_cols = X.columns
            _in_idx = X.index
        X = transformer_instance._validate_data(X, reset=False)

        y = transform_method(transformer_instance, X, *args, **kwargs)
        if _return_df:
            colnames = _in_cols
            if hasattr(transformer_instance, "column_names_"):
                colnames = transformer_instance.column_names_
            y = pd.DataFrame(data=y, columns=colnames, index=_in_idx)
            y = y[_in_cols]

        return y

    return _validated_
