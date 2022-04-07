"""
    Discretize (binarize) bulk and single-cell
    RNA-Seq data, using a set of criteria.
"""

import numpy as np
import pandas as pd


def _binarize_discarded(gene: pd.Series, *args):
    """Helper function for the binarization of discarded genes.
    Not intended to be called directly. The inclusion of the variadic
    argument *args was introduced to avoid type errors when calling it
    from scboolseq.core.scBoolSeq"""
    return pd.Series(np.nan, index=gene.index)


def _binarize_bimodal(gene: pd.Series, criteria: pd.DataFrame, *args):
    """Helper function for the binarization of bimodal genes.
    Not intended to be called directly. The inclusion of the variadic
    argument *args was introduced to avoid type errors when calling it
    from scboolseq.core.scBoolSeq"""
    _binary_gene = pd.Series(np.nan, index=gene.index)
    _criterion = criteria.loc[gene.name, :]
    bim_thresh_up = _criterion["bim_thresh_up"]
    bim_thresh_down = _criterion["bim_thresh_down"]
    _binary_gene[gene >= bim_thresh_up] = 1.0
    _binary_gene[gene <= bim_thresh_down] = 0.0
    return _binary_gene


def _binarize_unimodal_and_zeroinf(
    gene: pd.Series, criteria: pd.DataFrame, alpha: float
):
    """Helper function for the binarization of unimodal and zero-inflated genes.
    Not intended to be called directly."""
    _binary_gene = pd.Series(np.nan, index=gene.index)
    _criterion = criteria.loc[gene.name, :]
    unim_thresh_up = _criterion["unimodal_high_quantile"] + alpha * _criterion["IQR"]
    unim_thresh_down = _criterion["unimodal_low_quantile"] - alpha * _criterion["IQR"]
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
    """Helper function for the binarization of a single gene.
    Not intended to be called directly. It is internally used by
    scboolseq.binarization._binarize and scboolseq.binarization.binarize"""
    return _binarization_function_by_category[criteria.loc[gene.name, "Category"]](
        gene, criteria, alpha
    )


def _binarize(criteria: pd.DataFrame, alpha: float, data: pd.DataFrame) -> pd.DataFrame:
    """binarize `data` according to `criteria`,
    using `alpha` multiplier for the IQR, for Unimodal
    and ZeroInf genes."""
    return data.apply(_binarize_gene, args=[criteria, alpha])


def binarize(data: pd.DataFrame, criteria: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """binarize `data` according to `criteria`,
    using `alpha` multiplier for the IQR, for Unimodal and ZeroInf genes."""
    return data.apply(_binarize_gene, args=[criteria, alpha])


def binarise(*args, **kwargs) -> pd.DataFrame:
    """alias for binarize. See scBoolSeq.binarization.binarize"""
    return binarize(*args, **kwargs)
