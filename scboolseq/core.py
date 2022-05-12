"""
    scBoolSeq: scRNA-Seq data binarization and synthetic generation from Boolean dynamics.

    author: "Gustavo Magaña López"
    credits: "BNeDiction; Institut Curie"
"""

__all__ = ["scBoolSeq"]

from typing import NoReturn, Any, Optional, Union, Tuple
from pathlib import Path
import logging

import random
import string
import math

import multiprocessing
from functools import partial

# rpy2 conversion
import rpy2.robjects as r_objs
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import globalenv as GLOBALENV
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from rpy2.rinterface_lib.embedded import RNotReadyError

# data management
import numpy as np
import pandas as pd

# local imports
from .simulation import (
    biased_simulation_from_binary_state,
    _estimate_exponential_parameter,
    _DEFAULT_HALF_LIFE_CALCULATION,
    _DEFAULT_DROPOUT_MODE,
)
from .utils.stream_helpers import simulate_from_boolean_trajectory, SampleCountSpec
from .binarization import _binarize as binarization_binarize

# R source code locations :
__SCBOOLSEQ_DIR__ = Path(__file__).resolve().parent.joinpath("_R")
__SCBOOLSEQ_SRC__ = __SCBOOLSEQ_DIR__.joinpath("PROFILE_source.R").resolve()
__SCBOOLSEQ_BOOTSTRAP__ = __SCBOOLSEQ_DIR__.joinpath("install_deps.R").resolve()

rpy2_logger.setLevel(logging.ERROR)  # will display errors, but not warnings


class scBoolSeq(object):
    """
    scBoolSeq: scRNA-Seq data binarization and synthetic generation
               from Boolean dynamics.

    Objects of this class should be instantiated by passing
    none, one, or more of the follwing pandas.DataFrame :

        * data: SAMPLES AS ROWS and GENES AS COLUMNS.

        * criteria: GENES AS ROWS, STATISTICAL PARAMS AS COLUMNS.

        * simulation_criteria: idem as criteria.

        * optionally, an integer `r_seed` to fix the seed
          of the embedded R instance's random number generator.

    Here is an example of the intended usage of this class.

    import the data :
    >>> exp_data = pd.read_csv("example_gene_expression.csv")

    NOTE :
        The dataframe should contain the GENES as COLUMNS
        and SAMPLES as ROWS. The individual (sample) identifier
        should be the index.

    create a scBoolSeq instance
    >>> scbsq = scBoolSeq(exp_data, r_seed=1234)
    # compute "criteria" used to determine the proper binarization
    # rule for each gene in the dataset
    >>> scbsq.fit()
    >>> scbsq.binarize(scbsq.data)
    >>> scbsq.binarize(new_data) # binarize new observations
                                   # of the same genes (or a subset).
    """

    @staticmethod
    def _r_bool(value: bool) -> str:
        """Return a string representation of a Python boolean
        which is valid in R syntax. To be used when generating
        string literals which can be directly passed to self.r
        (an R instance invoked by rpy2 at initialization)."""
        if not isinstance(value, bool):
            raise TypeError(f"R boolean representation of {type(value)} undefined")
        return "T" if value else "F"

    @staticmethod
    def _r_none() -> str:
        """Return "NULL", R's equivalent of Python's None
        This is necessary as rpy2 provides no conversion rule for Python's None."""
        return "NULL"

    @staticmethod
    def _random_string(length: int = 10) -> str:
        """Return a random string og length `n` containing ascii lowercase letters and digits."""
        return "".join(
            random.choice(string.ascii_lowercase + string.digits) for i in range(length)
        )

    @staticmethod
    def _build_r_error_hint(error_str: str) -> str:
        """Build an auxiliary error string, associated to common causes
        of RRuntimeErrors."""
        _err_ls = (
            "",
            f"{str(error_str)}\n",
            "There was an error while calling compute_criteria() in R",
            "If the error is cryptic (as R errors generally are),"
            "some likely causes might be:",
            "\t * insufficient RAM",
            "\t * the descriptor files might have been deleted or corrupted",
            "\t * the data.frame contains non-numerical entries",
            "\t * the data.frame is empty",
        )
        return "\n".join(_err_ls)

    @staticmethod
    def _check_df_contains_no_nan(
        _df: pd.DataFrame, _parameter_name: str = "data"
    ) -> NoReturn:
        """Helper method for checking the validity of a given DataFrame.
        rpy2 provides no conversion rule for the multiple types of NaN that
        exist on R and Python. Passing a DataFrame with missing entries to the
        R backend may result in undefined behaviour."""
        _na_count = _df.isna().sum().sum()
        if _na_count:
            raise ValueError(
                " ".join(
                    [
                        f"Parameter `{_parameter_name}` has {_na_count} NaN entries",
                        "this will cause undefined behaviour when computing criteria",
                        "or simulating expression data. "
                        "Please verify that all entries of your dataframe ",
                        "are valid numerical entries. ",
                    ]
                )
            )

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        criteria: Optional[pd.DataFrame] = None,
        simulation_criteria: Optional[pd.DataFrame] = None,
        r_seed: Optional[int] = None,
    ):
        # self._addr will be used to keep track of R objects related to the instance :
        self._addr: str = str(hex(id(self)))
        self.r = r_objs.r
        self.r_globalenv = GLOBALENV
        self._valid_categories = ("ZeroInf", "Bimodal", "Discarded", "Unimodal")
        self.data: Optional[pd.DataFrame] = data
        self.criteria: Optional[pd.DataFrame] = criteria
        self.zero_inf_criteria: Optional[pd.DataFrame] = None
        self.simulation_criteria: Optional[pd.DataFrame] = simulation_criteria
        self._zero_inf_idx: pd.core.indexes.base.Index

        # try loading all packages and functions, installing them upon failure
        try:
            with open(__SCBOOLSEQ_SRC__, "r", encoding="utf-8") as _scbsq_source:
                self.r("".join(_scbsq_source.readlines()))
        except RRuntimeError:
            print("\nERROR : one or more R dependencies are not installed")
            print("Trying to automatically satisfy missing dependencies\n")
            try:
                # install dependencies :
                with open(__SCBOOLSEQ_BOOTSTRAP__, "r", encoding="utf-8") as _scbs_boot:
                    self.r("".join(_scbs_boot.readlines()))
                print("\n Missing dependencies successfully installed \n")
                # re-import the R source as functions were not saved because
                # of the previous RRuntimeError
                with open(__SCBOOLSEQ_SRC__, "r", encoding="utf-8") as _scbs_src:
                    self.r("".join(_scbs_src.readlines()))
            except RRuntimeError as _rer:
                print("Bootstrapping the installation of R dependencies failed:")
                raise _rer from None

        # if provided, set R rng seed
        if r_seed is not None:
            if not isinstance(r_seed, int):
                raise TypeError(
                    f"param `r_seed` must be of type int, not {type(r_seed)}"
                )
            self.r(f"set.seed({r_seed})")

    def __repr__(self):
        _attrs = (
            f"has_data={self._has_data}",
            f"can_binarize={self._is_trained}",
            f"can_simulate={self._can_simulate}",
        )
        return f"scBoolSeq({', '.join(_attrs)})"

    def r_ls(self):
        """Return a list containing all the names in the main R environment."""
        return list(self.r("ls()"))

    @property
    def _has_data(self) -> bool:
        """Boolean indicating if the instance possesses a reference 'data' DataFrame"""
        return hasattr(self, "data") and isinstance(self.data, pd.DataFrame)

    @property
    def _is_trained(self) -> bool:
        """Boolean indicating if the instance can be used to binarize expression."""
        return hasattr(self, "criteria") and isinstance(self.criteria, pd.DataFrame)

    @property
    def _can_simulate(self) -> bool:
        """Boolean indicating if the instance can be used to simulate expression."""
        return hasattr(self, "simulation_criteria") and isinstance(
            self.simulation_criteria, pd.DataFrame
        )

    @property
    def _data_in_r(self) -> bool:
        """Determine if the data to fit the criteria is present in the R environment."""
        return f"META_RNA_{self._addr}" in self.r_ls()

    def r_instantiate_data(
        self, data: Union[pd.DataFrame, Any], identifier: str
    ) -> NoReturn:
        """
        Instantiate a DataFrame within the R embedded process,
        bind it to the provided identifier on the main environment.

        Parameters:
                  data: Any data, although the original converter is intended for pandas DataFrames
            identifier: A string, the desired variable name for the object in the R env.
        """
        try:
            with localconverter(r_objs.default_converter + pandas2ri.converter):
                self.r_globalenv[identifier] = r_objs.conversion.py2rpy(data)
        except RRuntimeError as _rer:
            print(
                "An error ocurred while instantiating the data within the R session",
                "This is likely due to the type of the objects contained in the DataFrame",
                "(rpy2 may not have implemented the needed conversion rule).",
                "",
                sep="\n",
            )
            raise RRuntimeError(str(_rer)) from None
            # change this to display R's original error ?

    def fit(
        self,
        n_threads: int = multiprocessing.cpu_count(),
        dor_threshold: float = 0.95,
        unimodal_margin_quantile: float = 0.25,
        mask_zero_entries: bool = False,
    ) -> "scBoolSeq":
        """
        Compute the criteria needed to decide which binarization rule
        will be applied to each gene. This is performed by calling
        the corresponding R function `compute_criteria()` via rpy2.

        Arguments:
        ---------

            * n_threads: The number of parallel threads (processes) to be used.

            * dor_threshold: The DropOut Rate (percentage of zero entries) after which
                             genes should be discarded. For example `dor_threshold = 0.5`
                             will discard genes having a DropOut Rate > 50%. This means
                             that binarisation and synthetic generation for these genes
                             will not occur.

            * unimodal_margin_quantile:
                    Binarisation of "Unimodal" and "ZeroInflated" genes is quantile-based.
                    This parameter is needed to compute the binarisation thresholds:

                    threshold_true = quantile(gene, 1 - unimodal_margin_quantile) + \\alpha * IQR
                    threshold_false = quantile(gene, unimodal_margin_quantile) - \\alpha * IQR

            * mask_zero_entries: Wether zero entries should be ignored at criteria estimation.
                         Setting it to True is discouraged. This parameter is really needed
                         to calculate the simulation criteria, but it was included in this method
                         to have a uniform API.



        Returns:
        -------
            self (a reference to the object itself). This is done
            so that the criteria can be directly accessed, or other
            methods can be called. Examples :
            >>> scbsq = scBoolSeq(normalised_expr_data_frame)
            >>> criteria = scbsq.fit().criteria
            Or, for instance:
            >>> binarized = scbsq.fit().binarize()

        "Side effects" : a pandas.DataFrame containing the criteria and the label
        for each gene will be stored within the class.

        It is accessible via :
        >>> self.criteria

        IMPORTANT : This function calls R code which uses descriptors and memory
        mappings which are temporarily stored in files containing the
        current datetime and a small hash to ensure the unicity of names.

        Do not manipulate or erase files which look like this:
            SCBOOLSEQ_backing_file_DFNPO6252Q_Wed Mar 30 16:53:00 2022.bin
            SCBOOLSEQ_backing_file_DFNPO6252Q_Wed Mar 30 16:53:00 2022.desc

        They will be automatically removed one the criteria is calculated
        for all genes in the dataset.
        """
        if unimodal_margin_quantile > 0.5:
            raise ValueError(
                " ".join(
                    [
                        "unimodal_margin_quantile should be smaller than 0.5,"
                        "otherwise the binarisation rule for unimodal and zero-inflated genes",
                        "will be undefined (lower_bound > upper_bound)",
                    ]
                )
            )

        # the data must be instantiated before performing the R call :
        if self._has_data:
            if not self._is_trained and not self._data_in_r:
                self._check_df_contains_no_nan(self.data)
                self.r_instantiate_data(self.data, f"META_RNA_{self._addr}")
        else:
            raise AttributeError(
                "\n".join(
                    (
                        "Cannot compute criteria without a reference 'data' DataFrame,",
                        "please assign a valid RNA-Seq DataFrame to `self.data`",
                    )
                )
            )

        # call compute_criteria only once
        if not self._is_trained:
            params = [
                f"exp_dataset = META_RNA_{self._addr}",
                f"n_threads = {n_threads}",
                f"dor_threshold = {dor_threshold}",
                f"mask_zero_entries = {self._r_bool(mask_zero_entries)}",
                f"unimodal_margin_quantile = {unimodal_margin_quantile}",
            ]
            try:
                with localconverter(r_objs.default_converter + pandas2ri.converter):
                    self.criteria = r_objs.conversion.rpy2py(
                        self.r(
                            f"criteria_{self._addr} <- compute_criteria({', '.join(params)})"
                        )
                    )
                self.criteria["dor_threshold"] = dor_threshold
            except RRuntimeError as _rer:
                raise RRuntimeError(self._build_r_error_hint(_rer)) from None

        return self

    def simulation_fit(
        self,
        n_threads: int = multiprocessing.cpu_count(),
        dor_threshold: float = 0.95,
        unimodal_margin_quantile: float = 0.25,
        half_life_method: str = _DEFAULT_HALF_LIFE_CALCULATION,
        mask_zero_entries: bool = True,
    ) -> "scBoolSeq":
        """Basically the same as scBoolSeq_instance.fit(), but masking zero
        entries in order to better approximate the parameters of the distribution
        and independently simulate drop-out events.

        Parameters to perform this drop-out simulations are calculated
        using scboolseq.simulation._estimate_exponential_parameter().
        See this function's docstring for further information.

        Possible values for parameter `half_life_method` are:
            * arithmetic \\
            * harmonic     | <---- pythagorean means
            * geometric   /
            * median
            * midpoint (max(x) + min(x)) / 2.0
        """
        if not self._is_trained:
            raise AttributeError(
                "\n".join(
                    [
                        "Cannot compute simulation fit without a valid `criteria` DataFrame",
                        "Call self.fit() before calling this method.",
                    ]
                )
            )
        if not self._has_data:
            raise AttributeError(
                "\n".join(
                    [
                        "Cannot compute simulation fit because the instance has no valid `data`",
                        "please assign a valid RNA-Seq DataFrame to `self.data`",
                    ]
                )
            )
        _unimodal_margin_quantile = self.criteria.unimodal_margin_quantile.unique()[0]
        if not math.isclose(
            # Getting the first of all unique values is safe
            # unless the criteria dataframe has been tampered with
            _unimodal_margin_quantile,
            unimodal_margin_quantile,
        ):
            raise ValueError(
                " ".join(
                    [
                        "Specified unimodal margin quantile differs from the binarization's",
                        f" {unimodal_margin_quantile} != {_unimodal_margin_quantile}",
                        "The discrepancy between these two will cause inconsistent results",
                    ]
                )
            )
        _dor_threshold = self.criteria.dor_threshold.unique()[0]
        if not math.isclose(_dor_threshold, dor_threshold):
            raise ValueError(
                " ".join(
                    [
                        "Specified DropOutRate threshold differs from the binarization's",
                        f"{dor_threshold} != {_dor_threshold}",
                        "The discrepancy between these two will cause inconsistent results",
                    ]
                )
            )

        self._zero_inf_idx = self.criteria[self.criteria.Category == "ZeroInf"].index
        params = [
            f"exp_dataset = META_RNA_{self._addr}",
            f"n_threads = {n_threads}",
            f"dor_threshold = {dor_threshold}",
            f"mask_zero_entries = {self._r_bool(mask_zero_entries)}",
            f"unimodal_margin_quantile = {unimodal_margin_quantile}",
        ]
        with localconverter(r_objs.default_converter + pandas2ri.converter):
            self.simulation_criteria = r_objs.conversion.rpy2py(
                self.r(
                    f"simulation_criteria_{self._addr} <- compute_criteria({', '.join(params)})"
                )
            )

        self.simulation_criteria.loc[:, "beta_0"] = (
            2.0 * self.simulation_criteria["DropOutRate"].values
        )

        # self.simulation_criteria.loc[:, "beta_1"] = self.data.apply(
        #   _estimate_exponential_parameter
        # )
        # ^ np.apply_along axis is preferred since it is, in general, at
        # least 10 times faster than pandas.apply
        _beta_1_estimator = partial(
            _estimate_exponential_parameter, mode=half_life_method
        )
        self.simulation_criteria.loc[:, "beta_1"] = np.apply_along_axis(
            _beta_1_estimator, 0, self.data.values
        )
        # force ZeroInf genes to be simulated as Unimodal
        self.simulation_criteria.loc[
            self.simulation_criteria.Category == "ZeroInf", "Category"
        ] = "Unimodal"

        return self

    def _binarize_or_normalize(
        self,
        action: str,
        data: pd.DataFrame,
        gene: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Not intended to be called directly.
        Handle calls to self.r_binarize and self.r_normalize by dynamically constructing the R call.
        """

        assert action in [
            "binarize",
            "normalize",
        ], f"No method defined for action {action}"

        if not self._is_trained:
            raise AttributeError(
                f"Cannot {action} without the criteria DataFrame. Call self.fit() first."
            )

        _df_name: str = ""
        _rm_df: bool = False
        if data is not None:
            _df_name = f"bin_{self._addr}_{self._random_string()}"
            self.r_instantiate_data(data, _df_name)
            _rm_df = True
        else:
            _df_name = f"META_RNA_{self._addr}"
            data = self.data

        params = [
            f"exp_dataset = {_df_name}",
            f"ref_dataset = META_RNA_{self._addr}",
            f"ref_criteria = criteria_{self._addr}",
        ]

        if gene is not None:
            params.append(f"gene = {gene}")

        try:
            with localconverter(r_objs.default_converter + pandas2ri.converter):
                _df: pd.DataFrame = r_objs.conversion.rpy2py(
                    self.r(
                        f"""
                            {action}_exp({', '.join(params)}) %>%
                                apply(2, function(x) tidyr::replace_na(x, replace = -1)) %>%
                                    as.data.frame()
                    """
                    )
                )
                return _df.replace(-1.0, np.nan)
        except RRuntimeError as _rer:
            _err_help_ls = [
                "",
                f"{_rer}\n",
                "Please verify that your genes (columns) are a subset",
                "of the genes used to compute the criteria.\n",
                "Difference between the dataframes' column sets",
                "(genes contained in data for which there is no established criteria):",
                f"{set(data.columns).difference(set(self.data.columns))}",
            ]
            raise RRuntimeError("\n".join(_err_help_ls)) from None
        finally:
            if _rm_df:
                _ = self.r(f"rm({_df_name})")

    def binarize(
        self,
        data: pd.DataFrame = None,
        alpha: float = 1.0,
        n_threads: int = multiprocessing.cpu_count(),
        include_discarded: bool = False,
    ):
        """Binarize given RNA-Seq `data`, according to self.criteria"""
        if not self._is_trained:
            raise AttributeError(
                "Cannot binarize without the criteria DataFrame. Call self.fit() first."
            )

        data = data.copy(deep=True)
        self._check_df_contains_no_nan(_df=data, _parameter_name="data")
        # self._alpha = alpha

        # verify binarised genes are contained in the simulation criteria index
        if not all(gene in self.criteria.index for gene in data.columns):
            raise ValueError(
                "'data' contains genes for which there is no simulation criterion."
            )

        # Redundant integrity check, verify that the user did not tamper with
        # the criteria dataframe:
        if not all(
            category in self._valid_categories for category in self.criteria.Category
        ):
            raise ValueError(
                "\n".join(
                    [
                        "Corrupted criteria DataFrame,",
                        f"The set of categories : {self.criteria.Category.unique()}"
                        "is not a subset of",
                        f"the set of valid categories : {self._valid_categories}",
                    ]
                )
            )

        _binary_ls = np.array_split(data, n_threads, axis=1)
        _partial_binarize = partial(binarization_binarize, self.criteria, alpha)
        with multiprocessing.Pool(n_threads) as pool:
            ret_list = pool.map(_partial_binarize, _binary_ls)

        result = pd.concat(ret_list, axis=1)
        result = (
            result
            if include_discarded
            else result.loc[
                :, self.criteria[self.criteria.Category != "Discarded"].index
            ]
        )

        return result

    def r_binarise(self, *args, **kwargs) -> pd.DataFrame:
        """alias for self.r_binarize. See help(scBoolSeq.r_binarize)"""
        return self.r_normalize(*args, **kwargs)

    def r_binarize(
        self, data: pd.DataFrame, gene: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Binarize expression data.

        NOTICE: Legacy function. This version has no option to parametrize the
                binarization threshold. Furthermore, given that the binarization
                is done sequentially on the R backend, it will be much slower
                than the parallel Python implementation.

        Parameters:
            data: a (optional) pandas.DataFrame containing genes as columns and rows as measurements
                  Defaults to self.data (the data used to compute the criteria).

            gene: an (optional) string determining the gene name to binarize. It must be contained
                  the dataframe to binarize (and the previously computed criteria).
        """
        return self._binarize_or_normalize("binarize", data, gene)

    def binarise(self, *args, **kwargs) -> pd.DataFrame:
        """alias for self.binarize. See help(scBoolSeq.binarize)"""
        return self.binarize(*args, **kwargs)

    def r_normalise(self, *args, **kwargs) -> pd.DataFrame:
        """alias for self.normalize. See help(scBoolSeq.r_normalize)"""
        return self.r_normalize(*args, **kwargs)

    def r_normalize(
        self, data: pd.DataFrame, gene: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Normalize expression data.

        Parameters:
            data: a (optional) pandas.DataFrame containing genes as columns and rows as measurements
                  Defaults to self.data (the data used to compute the criteria).

            gene: an (optional) string determining the gene name to normalize. It must be contained
                  the dataframe to binarize (and the previously computed criteria).

        NOTICE: This function passes `data` to the R backend, which may be slow depending on
        the dataframe's size.
        """
        return self._binarize_or_normalize("normalize", data, gene)

    def simulate(
        self,
        binary_df,
        n_threads: Optional[int] = None,
        n_samples: Optional[int] = None,
        seed: Optional[int] = None,
        dropout_mode: str = _DEFAULT_DROPOUT_MODE,
    ) -> pd.DataFrame:
        """
        Perform biased simulation based on a fully determined binary DataFrame.

        wrapper for profile_binr.simulation.biased_simulation_from_binary_state
        """
        if not self._can_simulate:
            raise AttributeError("Call .simulation_fit() first")
        self._check_df_contains_no_nan(_df=binary_df, _parameter_name="binary_df")
        n_threads = (
            min(abs(n_threads), multiprocessing.cpu_count())
            if n_threads
            else multiprocessing.cpu_count()
        )
        return biased_simulation_from_binary_state(
            binary_df,
            self.simulation_criteria,
            n_threads=n_threads,
            n_samples=n_samples,
            seed=seed,
            dropout_mode=dropout_mode,
        )

    # TODO: remove this method
    def simulate_with_metadata(
        self,
        binary_df: pd.DataFrame,
        n_samples_per_state: SampleCountSpec = 1,
        rng_seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simulate a single cell experiment from a boolean trace,
        specifying how many samples (cells) should be produced for each
        binary state. This method returns both a synthetic RNA-Seq DataFrame
        and a metadata DataFrame which can be used to apply trajectory
        reconstruction methods.

        wrapper for scBoolSeq.utils.stream_helpers.simulate_from_boolean_trajectory
        """
        if not self._can_simulate:
            raise AttributeError("Call .simulation_fit() first")
        self._check_df_contains_no_nan(_df=binary_df, _parameter_name="binary_df")

        return simulate_from_boolean_trajectory(
            boolean_trajectory_df=binary_df,
            criteria_df=self.simulation_criteria,
            n_samples_per_state=n_samples_per_state,
            rng_seed=rng_seed,
        )

    def clear_r_envir(self):
        """Remove an all R objects that have been created by the scBoolSeq
        instance."""
        _instance_objs = [f"'{obj}'" for obj in self.r_ls() if self._addr in obj]
        return self.r(f"rm(list = c({', '.join(_instance_objs)}))")

    def __del__(self):
        """This method is intended to reduce the memory footprint once the object
        is destroyed by eliminating all the frames it instanciated on the R-side
        (such as the criteria and simulation_criteria frames).
        WARNING 1 : not intended to be called directly.
        WARNING 2 : There is no guarantee that calling `del object` will effectively
        call this method. This means that the R-side objects won't be erased until
        Python's garbage collector decides to clean up this object.
        """
        # The try/except section is needed because the embedded R instance
        # can be destroyed before the object upon the Python interpreter's exit.
        try:
            _ = self.clear_r_envir()
        except RNotReadyError as _rnrerr:
            pass
