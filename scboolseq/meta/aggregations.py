""" """

import typing
from typing import Union, TypeAlias, Dict

import numpy as np
import pandas as pd

_Frame: TypeAlias = pd.DataFrame
_GroupBy: TypeAlias = pd.core.groupby.DataFrameGroupBy
_GroupDict: TypeAlias = Dict[typing.Any, pd.Index]
_Groups: TypeAlias = Union[_GroupBy, _GroupDict]


def boolean_to_arithmetic(df: _Frame) -> _Frame:
    """ Perform (by group) the mapping 
               /
               | -1  if x == 0
        f(x) = | 0   if x is np.nan
               | 1   if x == 1
               \ 
    """
    return df.replace(0, -1).fillna(0)


def arithmetic_to_boolean(df: _Frame) -> _Frame:
    """ Perform (by group) the mapping 
               /
               | -1  if x == 0
        f(x) = | 0   if x is np.nan
               | 1   if x == 1
               \ 
    """
    return df.replace(0, np.nan).replace(-1, 0)


# This class could be much smaller
# by delegating the calls to pandas
# creating the dedicated class attributes
# but this seems an overkill for the moment.
class CellAggregator:
    """Aggregate single cells with different strategies.
    This supposes that cells have been binarised using scBoolSeq.

    Note that not all strategies are created equal.
    _meta_strategies: Require subsequent binarisation
    _binary_strategies: Are already resolved (in the [0, np.nan, 1] extension of the Boolean domain)
    """

    _all_strategies = (
        "sum",
        "mean",
        "median",
        "mode",
        "min",
        "max",
    )
    _meta_strategies = (
        "sum",
        "mean",
    )
    _binary_strategies = (
        "mode",
        "median",
        "min",
        "max",
    )

    def __init__(self, binary_df: _Frame, groups: _Groups, copy: bool = True):
        if copy:
            binary_df = binary_df.copy(deep=True)
        self.binary_df: _Frame = binary_df
        if isinstance(groups, _GroupBy):
            groups = groups.groups
        self.groups: _GroupDict = groups

    @property
    def _arithmetic_map(self):
        """ Perform (by group) the mapping 
               /
               | -1  if x == 0
        f(x) = | 0   if x is np.nan
               | 1   if x == 1
               \ 
        This returns a dictionnary.
        """
        return {
            group_name: self.binary_df.loc[group_index, :].replace(0, -1).fillna(0)
            for group_name, group_index in self.groups.items()
        }

    @property
    def sum(self):
        """by-group sum (after self._arithmetic_map)"""
        return pd.DataFrame.from_dict(
            {k: v.sum() for k, v in self._arithmetic_map.items()}
        ).T

    @property
    def mean(self):
        """by-group mean (after self._arithmetic_map)"""
        return pd.DataFrame.from_dict(
            {k: v.mean() for k, v in self._arithmetic_map.items()}
        ).T

    @property
    def median(self):
        """by-group median (after self._arithmetic_map)"""
        return pd.DataFrame.from_dict(
            {k: v.median() for k, v in self._arithmetic_map.items()}
        ).T.pipe(arithmetic_to_boolean)

    @property
    def min(self):
        """by-group min (after self._arithmetic_map)"""
        return pd.DataFrame.from_dict(
            {k: v.min() for k, v in self._arithmetic_map.items()}
        ).T.pipe(arithmetic_to_boolean)

    @property
    def max(self):
        """by-group max (after self._arithmetic_map)"""
        return pd.DataFrame.from_dict(
            {k: v.max() for k, v in self._arithmetic_map.items()}
        ).T.pipe(arithmetic_to_boolean)

    @property
    def mode(self):
        """by-group (first) mode"""
        return pd.DataFrame.from_dict(
            {
                group_name: self.binary_df.loc[group_index, :].mode().loc[0, :]
                for group_name, group_index in self.groups.items()
            }
        ).T
