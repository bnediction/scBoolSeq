""" Size-factor estimation for synthetic scRNA-seq data from Boolean Traces.

This is an experimental module which complements the generation of
scRNA-seq using scBoolSeq (see below for reference). 

The idea is to compute cell-wise criteria and to build a regression
model which can correctly estimate size factors (used for 
normalisation in standard scRNA-seq analysis pipelines) in order to
obtain "raw counts" from the simulated log-transformed scRNA-seq data.


=========
Reference
=========
scBoolSeq: Linking scRNA-Seq Statistics and Boolean Dynamics

Credits: Institut Curie, BNeDiction

 Gustavo Magaña López
 Laurence Calzone
 Andrei Zinovyev
 Loïc Paulevé

https://doi.org/10.1101/2023.10.23.563518 
https://www.biorxiv.org/content/10.1101/2023.10.23.563518v1
"""

## Definining the estimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import (
    clone,
    BaseEstimator,
    _SetOutputMixin,
    RegressorMixin,
)


## Basic numerical
import sklearn
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import model_selection as sk_modsel
from sklearn import preprocessing as sk_preproc

## Core statistical inference
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import (
    Poisson,
    GeneralizedPoisson,  # Counts
    NegativeBinomial,
    NegativeBinomialP,  # idem. var > mu
    # ZeroInflatedPoisson, ZeroINflatedNegativeBinomialP,
)

## Local imports (scboolseq)
import scboolseq.binarization as scbin
from scboolseq._scboolseq_core import compute_criteria

__all__ = [
    "scipy_negative_binomial_mle",
    "statsmodels_negative_binomial_mle",
    "SizeFactorRegressor",
]


def scipy_negative_binomial_mle(data):
    """Compute the parameters needed to define a Negative Binomial
    distribution using scipy (scipy.stats.nbinom)
    ================================================================
    @param `data` : A numpy.ndarray, pandas.Series or alike
                    (should define the methods .mean() and .var())
    ================================================================
    Returns : (n, p) : A tuple containing the parameters of the
                       Negative Binomial Distribution, as defined
                       by scipy.
    ================================================================
    Example usage (in the context of single-cell transcriptomics):
    >>> data = adata.obs["total_counts"] # per-cell counts (lib size)
    >>> n, p = scipy_negative_binomial_mle(data)
    >>> print(n, p) # inspect the parameters
    >>> mle_nbinom = scipy.stats.nbinom(n, p) # create a distribution
    >>> new_samples = mle_nbinom.rvs(1000) # get 1000 realisations
    """
    mu = data.mean()
    s2 = data.var()
    p = mu / s2
    n = np.power(mu, 2) / (s2 - mu)
    return n, p


def statsmodels_negative_binomial_mle(data):
    """Wrapper for `sm.NegativeBinomial(data, np.ones_like(data)).fit()`"""
    return sm.NegativeBinomial(data, np.ones_like(data)).fit()


class SizeFactorRegressor(RegressorMixin, BaseEstimator):
    """ """

    def __init__(
        self,
        formula="total_counts ~ HarmonicMean",
        preproc_transformer=sk_preproc.StandardScaler(),
        clone_preproc=False,
    ):
        self.formula = formula
        self.preproc_transformer = clone(preproc_transformer)  # needed ?
        self.clone_preproc = clone_preproc

    def fit(self, X, y=None, **kwargs):
        """kwargs are passed to statsmodels.formula.api.glm().fit(**kwargs)"""
        if y is None:
            raise ValueError("requires y to be passed, but the target y is None")
        # omit this ?
        # X, y = check_X_y(X, y)
        __em = ["fit", "transform"]
        if trans := self.preproc_transformer:
            if not all(hasattr(trans, _method) for _method in __em):
                raise AttributeError(
                    f"Invalid {type(self.preproc_transformer)=}."
                    "A valid transformer must implement at the following methods: "
                    f"{__em}"
                )
            X = trans.fit(X).transform(X)

        DATA = X.join(y)
        self.glm_spec_ = smf.glm(
            formula=self.formula,
            data=DATA,
            family=sm.families.NegativeBinomial(),
        )
        self.glm_fitted_ = self.glm_spec_.fit(**kwargs)

        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        if trans := self.preproc_transformer:
            if self.clone_preproc:
                X = clone(trans).fit_transform(X)
            else:
                X = trans.transform(X)

        return (
            self.glm_fitted_.predict(X).rename(self.glm_spec_.endog_names).astype(int)
        )

    def score(self, X, y):
        check_is_fitted(self, "is_fitted_")
        y_hat = self.predict(X)
        y_hat = y_hat.rename(f"predicted_{y_hat.name}")
        error_frame = y_hat.to_frame().join(y)
        error_frame.loc[:, "error"] = error_frame[y_hat.name] - error_frame[y.name]
        error_frame.loc[:, "rel_error"] = (
            np.abs(error_frame["error"]) / error_frame[y.name]
        )
        return error_frame

    def de_normalize(self, X, target_sum):
        """return an estimate of the de-normalized X frame
        here we assume that X has genes as columns and cells as rows."""
        _cell_wise_criteria = compute_criteria(X.T, skip_slow=True)
        _new_size_factors = self.predict(_cell_wise_criteria)
        return (
            (np.expm1(X) / target_sum)
            .T.apply(lambda x: x * _new_size_factors[x.name])
            .astype(int)
            .T
        )
