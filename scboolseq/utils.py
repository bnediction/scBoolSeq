""" Utility functions. """

import pickle
import typing
from time import time

# import typing
import pandas as pd

from sklearn.utils import Bunch

from ._types import _PathLike, _OptionalDict
from ._core_utils import first_arg_is_path


@first_arg_is_path
def parse_data_directory(
    data_dir: _PathLike,
    glob_pattern: str = "*.?sv",
    _globals: _OptionalDict = None,
    **csv_kwargs,
):
    """Parse a whole directory of csv/tsv files to pandas data frames
    by default, index column is supposed to be the fist (index_col=0).
    This can be overridden using **csv_kw"""
    if "index_col" not in csv_kwargs:
        csv_kwargs["index_col"] = 0
    data_frames = Bunch(
        **{
            file.name.replace(file.suffix, "").replace(" ", "_"): pd.read_csv(
                file.resolve(),
                sep=(
                    "\t" if "t" in file.suffix.lower() else ","
                ),  # This adaptation using the file name might be fragile
                **csv_kwargs,
            )
            for file in data_dir.glob(glob_pattern)
        }
    )

    if _globals:
        for frame in data_frames:
            _globals[frame] = data_frames[frame]

    return data_frames


@first_arg_is_path
def parse_pickles(
    data_dir: _PathLike,
    _globals: _OptionalDict = None,
    suffix: typing.Optional[str] = None,
):
    """Parse a whole directory of pkl files"""

    suffix = suffix or "pkl"
    _pickles = {}
    for file in data_dir.glob(f"*{suffix.replace('.', '')}"):
        with open(file, "rb") as _pk_file:
            _pickles.update({file.name.replace(file.suffix, ""): pickle.load(_pk_file)})

    if _globals:
        for name, pkl in _pickles.items():
            _globals[name] = pkl

    return _pickles


def select_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """Select columns whose data type is different from
    "object".

    Args:
        df (pd.DataFrame): with numeric and non-numeric data.

    Returns:
        pd.DataFrame: Only numeric columns
    """
    return frame[frame.columns[frame.dtypes != "O"]]


class Timer(object):
    """A simple timer class used to measure execution time,
    without all the problems related to using timeit."""

    def __init__(self, description):
        self.description = description
        self.start: float
        self.end: float

    def __enter__(self):
        self.start = time()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.end = time()
        _formater = lambda t: f"{t // 60**2}:{(t % 60**2) // 60}:{t % 60}"
        print(f"{self.description}: {round(self.end - self.start, 5)}", flush=True)
