"""
Tools to perform the normalisation of a count matrix
of RNA-seq reads.

Credits :
" BNediction ;  pinellolab/STREAM "
"""

__all__ = ["normalize", "normalise", "log_transform", "log_normalize", "log_normalise"]

import numpy as np
import pandas as pd

from typing import Optional


def normalize(raw_counts: pd.DataFrame, method: Optional[str] = None) -> pd.DataFrame:
    """
    Normalize count matrix.

    Parameters
    ----------
    raw_counts_df: pandas.DataFrame
        A raw count matrix containing the number of
        reads per gene (row), per sample (column).

    method: str, optional (default: "RPM")
        Choose from : {
            "RPM" or "CPM": Reads (counts) per million mapped reads (library-size correct)
                    * RPM does not consider the transcript length normalization.
                    * suitable for sequencing protocols where reads are generated irrespective of gene length

            "RPKM", "TPM" : to be implemented
        }

    Returns
    -------
    a pandas.DataFrame, containing normalized read counts
    """
    method = method or "RPM"
    method = method.upper()

    if method not in ["RPM", "CPM", "RPKM", "TPM"]:
        raise ValueError(f"unrecognized method {method}")
    if method == "RPM" or method == "CPM":
        return raw_counts / raw_counts.sum() * 1e6
    else:
        raise NotImplementedError(
            "\n".join(
                [
                    "method {method} has not been implemented.",
                    "Feel free to open a pull request at :",
                    "https://github.com/bnediction/scBoolSeq",
                ]
            )
        )


def normalise(*args, **kwargs) -> pd.DataFrame:
    """alias for normalize. See help(normalize)"""
    return normalize(*args, **kwargs)


def log_transform(
    data: pd.DataFrame, base: Optional[int] = None, constant: Optional[int] = None
) -> pd.DataFrame:
    """
    log transform expression data

    Parameters
    ----------
    data: pd.DataFrame
        Annotated data matrix.
    base: int, optional (default: 2)
        The base used to calculate logarithm
    constant: int, optional (default: 1)
        The number to be added to all entries, to
        prevent indetermination of the transformation
        when a read is equal to zero.

    Returns
    -------
    a pandas.DataFrame, containing log-transformed counts

    np.log(data + constante) / np.log(base)
    i.e.
    np.log(data + 1) / np.log(2)
    for default parameters
    """
    base = base or 2
    constant = constant or 1
    return np.log(data + constant) / np.log(base)


def log_normalize(
    data: pd.DataFrame,
    method: Optional[str] = None,
    base: Optional[int] = None,
    constant: Optional[int] = None,
) -> pd.DataFrame:
    """
    Chain the call to normalize and log_transform

    Parameters
    ----------
    data: pandas.DataFrame
        raw expression counts

    method: String
        optional
    """
    _normalized = normalize(data, method=method)
    return log_transform(_normalized, base=base, constant=constant)


def log_normalise(*args, **kwargs) -> pd.DataFrame:
    """alias for log_normalize. See help(log_normalize)"""
    return log_normalize(*args, **kwargs)
