"""
    Discretize (binarize) bulk and single-cell
    RNA-Seq data, using a set of criteria.
"""

from typing import Optional
import numpy as np
import pandas as pd

def _binarize_discarded(gene: pd.Series, criteria: pd.DataFrame):
    """ """
    return pd.Series(np.nan, index=gene.index)

def _binarize_bimodal(gene: pd.Series, criteria: pd.DataFrame, alpha: float):
    """ """
    _binary_gene = pd.Series(np.nan, index=gene.index)
    _criterion = criteria.loc[gene.name, :]
    bim_thresh_up = _criterion["bim_thresh_up"]
    bim_thresh_down = _criterion["bim_thresh_down"]
    _binary_gene[gene >= bim_thresh_up] = 1.0
    _binary_gene[gene <= bim_thresh_down] = 0.0
    return _binary_gene

def _binarize_unimodal_and_zeroinf(gene: pd.Series, criteria: pd.DataFrame, alpha: float):
    """ """
    _binary_gene = pd.Series(np.nan, index=gene.index)
    _criterion = criteria.loc[gene.name, :]
    unim_thresh_up = (
        _criterion["unimodal_high_quantile"] + alpha * _criterion["IQR"]
    )
    unim_thresh_down = (
        _criterion["unimodal_low_quantile"] - alpha * _criterion["IQR"]
    )
    _binary_gene[gene > unim_thresh_up] = 1.0
    _binary_gene[gene < unim_thresh_down] = 0.0
    return _binary_gene

_binarization_function_by_category = {
    "ZeroInf": _binarize_unimodal_and_zeroinf,
    "Unimodal": _binarize_unimodal_and_zeroinf,
    "Bimodal": _binarize_bimodal,
    "Discarded": _binarize_discarded,
}

def _binarize_gene(gene: pd.Series, criteria: pd.DataFrame, alpha: float) -> pd.Series:
    """ """
    return _binarization_function_by_category[criteria.loc[gene.name, "Category"]](
        gene, criteria, alpha
    )

def binarize(criteria: pd.DataFrame, alpha: float, data: pd.DataFrame) -> pd.DataFrame:
    """ binarize `data` according to `criteria`."""
    return data.apply(_binarize_gene, args=[criteria, alpha])
