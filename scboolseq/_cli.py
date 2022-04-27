"""Command line parser for scBoolSeq"""

import multiprocessing as mp
from pathlib import Path
from functools import reduce
import argparse
import sys
import datetime as dt
import toml

import pandas as pd

# local relative import
from .core import scBoolSeq

_YIELD_TIMESTAMP = (
    lambda: str(dt.datetime.now())
    .split(".", maxsplit=1)[0]
    .replace(" ", "_")
    .replace(":", "h", 1)
    .replace(":", "m", 1)
)


class scBoolSeqRunner(object):
    """Runner used to call scboolseq.core.scBoolSeq using
    the arguments parsed by scboolseq._cli.scBoolSeqCLIParser"""

    # Class constant used to validate inputs
    valid_actions = ("binarize", "synthesize")
    # Valid simulation categories
    # (used to verify that users specified simulation criteria
    # and not binarization criteria when calling `synthesize`)
    simulation_categories_set = {
        "Bimodal",
        "Discarded",
        "Unimodal",
    }
    suffix_separator_dict = {"csv": ",", "tsv": "\t"}

    def validate_file_suffixes(self, path, param_name=None):
        """Validate the parameter's suffix.
        Return the list of suffixes of path if validation passes.
        Raise ValueError upon failed validation."""
        _suffixes = [suffix.replace(".", "").lower() for suffix in path.suffixes]
        if not any(_s in _suffixes for _s in self.suffix_separator_dict):
            _msg1 = "Unknown file extension, please provide a csv or tsv file."
            _msg1 += f" (param {param_name})" if param_name is not None else ""
            raise ValueError(_msg1)
        if all(_s in _suffixes for _s in self.suffix_separator_dict):
            _msg2 = "Ambiguous file name, cannot determine file type. Aborting."
            _msg2 += f" (param {param_name})" if param_name is not None else ""
            raise ValueError(_msg2)

        return _suffixes

    def f_frame_or_none(self, str_path_or_none, param_name=None):
        """Helper function, return a DataFrame if the path parameter has been parsed."""
        if str_path_or_none is None:
            return str_path_or_none
        else:
            _csv_kw = {}
            _path = Path(str_path_or_none)
            _suffixes = self.validate_file_suffixes(_path, param_name=param_name)
            return pd.read_csv(
                str_path_or_none,
                index_col=0,
                sep=self.suffix_separator_dict[_suffixes[0]],
            )

    @staticmethod
    def f_out_file(path, suffix: str = "binarized"):
        """Default name for output file"""
        return path.parent.joinpath(
            path.name.replace(path.suffix, f"_{suffix}{path.suffixes[0]}")
        ).resolve()

    def __init__(self, action: str, timestamp: str = ""):
        if action not in self.valid_actions:
            raise ValueError(
                f"Unknown action `{action}`. Available options are {self.valid_actions}"
            )
        self.action = action
        self.timestamp = timestamp

    def __repr__(self):
        return f"scBoolSeqRunner(action={self.action}, timestamp={self.timestamp})"

    def binarize(self, args):
        """binarize expression data"""

        in_file = Path(args["in_file"]).resolve()
        _in_frame = self.f_frame_or_none(in_file)
        in_frame = _in_frame.T if args.get("genes_are_rows", False) else _in_frame
        out_file = (
            Path(args.get("output")).resolve()
            if args.get("output")
            else self.f_out_file(in_file)
        )

        scbs = scBoolSeq(r_seed=args.get("rng_seed"))
        _data = self.f_frame_or_none(args.get("reference"))
        scbs.data = (
            _data.T
            if _data is not None and args.get("genes_are_rows", False)
            else _data
        )
        scbs.criteria = self.f_frame_or_none(args.get("criteria"))

        if not scbs._is_trained:
            if not scbs._has_data:
                scbs.data = in_frame
            scbs.fit(
                n_threads=args.get("n_threads") or mp.cpu_count(),
                unimodal_margin_quantile=args.get("margin_quantile", 0.25),
                dor_threshold=args.get("dor_threshold", 0.95),
            )
        else:
            print(
                f"Ignoring parameter --margin_quantile={args.get('margin_quantile', 0.25)}, "
                "as --criteria has been specified",
                sep=" ",
            )
        if args["dump_criteria"]:
            _name = reduce(
                lambda xx, yy: xx.replace(yy, ""), [in_file.name, *in_file.suffixes]
            )
            scbs.criteria.to_csv(
                f"scBoolSeq_criteria_{_name}_{self.timestamp}{in_file.suffixes[0]}",
                sep=self.suffix_separator_dict[self.validate_file_suffixes(in_file)[0]],
            )

        _binarized = scbs.binarize(
            in_frame,
            alpha=args.get("alpha", 1.0),
            include_discarded=not args.get("exclude_discarded"),
            n_threads=args.get("n_threads") or mp.cpu_count(),
        )
        binarized = _binarized.T if args.get("genes_are_rows") else _binarized
        _suffix = out_file.suffix.lower().replace(".", "")
        _out_separator = self.suffix_separator_dict["csv"]
        if _suffix not in self.suffix_separator_dict:
            print("WARNING: Unknown output file extension, defaulting to csv")
        else:
            _out_separator = self.suffix_separator_dict[
                out_file.suffix.lower().replace(".", "")
            ]
        binarized.to_csv(
            out_file,
            sep=_out_separator,
        )

    def synthesize(self, args):
        """synthesize RNA-Seq data from boolean dynamics"""

        in_file = Path(args["in_file"]).resolve()
        _in_frame = self.f_frame_or_none(in_file)
        in_frame = _in_frame.T if args.get("genes_are_rows", False) else _in_frame
        out_file = (
            Path(args.get("output")).resolve()
            if args.get("output")
            else self.f_out_file(in_file, suffix="synthetic")
        )

        scbs = scBoolSeq(r_seed=args.get("rng_seed"))
        _data = self.f_frame_or_none(args.get("reference"))
        scbs.data = (
            _data.T
            if _data is not None and args.get("genes_are_rows", False)
            else _data
        )
        scbs.simulation_criteria = self.f_frame_or_none(args.get("simulation_criteria"))
        scbs._check_df_contains_no_nan(in_frame, "in_file")

        if scbs.simulation_criteria is not None:
            _sim_categories = set(scbs.simulation_criteria["Category"].unique())
            if not _sim_categories.issubset(self.simulation_categories_set):
                _extra_categories = _sim_categories.difference(
                    self.simulation_categories_set
                )
                raise ValueError(
                    f"Unknown categories found in simulation_criteria: {_extra_categories}"
                )

        if not scbs._can_simulate:
            scbs.fit(
                n_threads=args.get("n_threads") or mp.cpu_count(),
                dor_threshold=args.get("dor_threshold", 0.95),
            ).simulation_fit(
                n_threads=args.get("n_threads") or mp.cpu_count(),
            )

        if args["dump_criteria"]:
            _name = reduce(
                lambda xx, yy: xx.replace(yy, ""), [in_file.name, *in_file.suffixes]
            )
            scbs.simulation_criteria.to_csv(
                f"scBoolSeq_simulation_criteria_{_name}_{self.timestamp}{in_file.suffixes[0]}",
                sep=self.suffix_separator_dict[self.validate_file_suffixes(in_file)[0]],
            )

        _synthetic = scbs.simulate(
            binary_df=in_frame,
            n_threads=args.get("n_threads") or mp.cpu_count(),
            n_samples=args.get("n_samples", 1),
            seed=args.get("rng_seed"),
        )
        synthetic = _synthetic.T if args.get("genes_are_rows", False) else _synthetic
        _suffix = out_file.suffix.lower().replace(".", "")
        _out_separator = self.suffix_separator_dict["csv"]
        if _suffix not in self.suffix_separator_dict:
            print("WARNING: Unknown output file extension, defaulting to csv")
        else:
            _out_separator = self.suffix_separator_dict[
                out_file.suffix.lower().replace(".", "")
            ]
        synthetic.to_csv(
            out_file,
            sep=_out_separator,
        )

    def __call__(self, args):
        """Binarize or synthesize via scBoolSeq"""
        getattr(self, self.action)(args)


class scBoolSeqCLIParser(object):
    """Semantic command line parser, serving as an interface for
    the embedded tool scBoolSeq."""

    def __init__(self):
        self.main_parser = argparse.ArgumentParser(
            description="""
                scBoolSeq: bulk and single-cell RNA-Seq data binarization and synthetic 
                generation from Boolean dynamics.""",
            usage="""scBoolSeq <command> [<args>]

Available commands:
\t* binarize\t Binarize a RNA-Seq dataset.
\t* synthesize\t Simulate a RNA-Seq experiment from Boolean dynamics.
\t* from_file\t Repeat a binarization or synthethic generation experiment, based on a config file.

NOTE on TSV/CSV file specs:
* If '.csv', the file is assumed to use the standard separator for columns ','.
* The index (gene or sample identifiers) is assumed to be the first column.
* The scBoolSeq is designed with consistency in mind. 
  The `output` (binarized or synthetic expression frame) will have the same disposition 
  (genes x observations | observations x genes) as the `input`. 
  If a `reference` is specified, its disposition must match the `input`'s.
""",
        )
        self.main_parser.add_argument("command", help="Subcommand to run")
        # Used to prevent users from binarizing observation instead of gene expression
        self._disposition_mandatory_options = ("genes_are_columns", "genes_are_rows")
        # parse_args defaults to [1:] for args
        # exclude the rest of the args too, or validation will fail
        args = self.main_parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")
            self.main_parser.print_help()
            sys.exit(1)
        # dispatch the specified command
        getattr(self, args.command)()

    def _verify_mandatory_mutually_exclusive_options(
        self, parsed_args_dict, options, parser
    ):
        """Check that one of the two mandatory mutually exclusive
        options has been specified. Exit with an informative help message."""
        if not any(parsed_args_dict.get(_key) for _key in options):
            parser.print_help()
            print(
                f"Please specify either --{options[0]}",
                f"or --{options[1]}",
                sep=" ",
            )
            sys.exit(1)

    def binarize(self):
        """Binarize log2(CPM + 1) expression data."""
        parser = argparse.ArgumentParser(
            description="""
            Distribution-based binarization of RNA-Seq data.
                * Bimodal Genes: Binarization based on the probability of membership
                to each one of the two modalities. 
                * Unimodal and Zero-Inflated genes: Binarization based on parametric outlier
                assignment. By default, Tukey Fences [Q1 - alpha * IQR, Q3 + alpha * IQR],
                with (alpha == 1); The margin quantile (q25 yields Q1 and Q3) and alpha parameters
                can be specified as detailed below.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "in_file",
            type=lambda x: Path(x).resolve().as_posix(),
            help="""
            A csv/tsv file containing normalized expression data.""",
        )
        _genes_columns_or_rows = parser.add_mutually_exclusive_group(required=False)
        _genes_columns_or_rows.add_argument(
            "--genes-are-columns",
            action="store_true",
            help="""Are genes stored as columns and observations as rows?""",
        )
        _genes_columns_or_rows.add_argument(
            "--genes-are-rows",
            action="store_true",
            help="""Are genes stored as rows and observations as columns?""",
        )
        _ref_data_or_criteria = parser.add_mutually_exclusive_group(required=False)
        _ref_data_or_criteria.add_argument(
            "--reference",
            type=lambda x: Path(x).resolve().as_posix(),
            help="""
            A "reference" csv/tsv file containing a column for each gene and a line for each observation (cell/sample).
            The "reference" will be used to compute the criteria needed for binarization.
            Expression data must be normalized before using this tool.""",
        )
        _ref_data_or_criteria.add_argument(
            "--criteria",
            type=lambda x: Path(x).resolve().as_posix(),
            help="""
            A "criteria" csv/tsv file, previously computed using this tool,
            having the default columns (or any extra criteria added by the user, which will
            be ignored) and a row for at least each gene contained in `in_file` (criteria for
            extra genes will simply be ignored).
            The criteria DataFrame will be used to binarize `in_file`.""",
        )
        parser.add_argument(
            "--alpha",
            type=float,
            default=1.0,
            help="""Multiplier for IQR.""",
        )
        parser.add_argument(
            "--margin-quantile",
            type=float,
            default=0.25,
            help="""Margin quantile for parametric Tukey fences.""",
        )
        parser.add_argument(
            "--dor-threshold",
            type=float,
            default=0.95,
            help="""DropOutRate (DOR) threshold. All genes having a DOR
            greater or equal to the specified value will be classified
            as discarded (no binarization or simulation will be applied).""",
        )
        parser.add_argument(
            "--n-threads",
            type=int,
            default=mp.cpu_count(),
            help="""The number of parallel processes to be used.""",
        )
        parser.add_argument(
            "--rng-seed",
            type=int,
            help="""An integer which will be used to seed both R's
            and Python's (numpy) random number generators.""",
        )
        parser.add_argument(
            "--exclude-discarded",
            action="store_true",
            help="""Should discarded genes be excluded from the output file?""",
        )
        parser.add_argument(
            "--output",
            type=lambda x: Path(x).resolve().as_posix(),
            help="""The name (can be a full path) of the file in which results should
            be stored. Defaults to `in_file`_binarized.csv/tsv""",
        )
        parser.add_argument(
            "--dump-criteria",
            action="store_true",
            help="""Should the computed criteria be saved to a csv file
            to be reutilized afterwards?""",
        )
        parser.add_argument(
            "--dump-config",
            action="store_true",
            help="""
            Should the specified CLI arguments be dumped to a toml configuration
            file?""",
        )
        # ignore the command and the subcommand, parse all options :
        args = dict(vars(parser.parse_args(sys.argv[2:])))
        self._verify_mandatory_mutually_exclusive_options(
            args, self._disposition_mandatory_options, parser
        )
        if args.get("genes_are_columns") is not None:
            args["genes_are_rows"] = not args.pop("genes_are_columns")
        args.update({"action": "binarize"})
        _timestamp = _YIELD_TIMESTAMP()

        if args["dump_config"]:
            _config_dest = f"scBoolSeq_experiment_config_{_timestamp}.toml"
            if Path(_config_dest).resolve().exists():
                # avoid overwritting existing config files
                raise FileExistsError("Config file already exists, aborting")
            with open(_config_dest, "w", encoding="utf-8") as _c_f:
                _ = args.pop("dump_config")
                toml.dump(args, _c_f)

        binarizer = scBoolSeqRunner(action=args.pop("action"), timestamp=_timestamp)
        binarizer(args)

    def synthesize(self):
        """Create a synthetic RNA-Seq dataset from a reference
        expression dataset (or simulation_criteria) and a frame containing
        a trace of Boolean states."""
        parser = argparse.ArgumentParser(
            description="""
            Synthesis of RNA-Seq data via biased sampling from Boolean states.
            Using a set of "simulation_criteria" and or a "reference" RNA-Seq dataset.
            Genes will be sampled from a Gaussian of a Gaussian mixture with two modalities.
            The DropOut rates will be comparable to those of reference dataset (or simulation_criteria).
            """,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "in_file",
            type=lambda x: Path(x).resolve().as_posix(),
            help="""
            A csv/tsv file containing FULLY DETERMINED Boolean states.
            These should be represented as zeros (false) and ones (true).
            """,
        )
        _genes_columns_or_rows = parser.add_mutually_exclusive_group(required=False)
        _genes_columns_or_rows.add_argument(
            "--genes-are-columns",
            action="store_true",
            help="""Are genes stored as columns and observations as rows?""",
        )
        _genes_columns_or_rows.add_argument(
            "--genes-are-rows",
            action="store_true",
            help="""Are genes stored as rows and observations as columns?""",
        )
        _ref_data_or_criteria = parser.add_mutually_exclusive_group(required=False)
        _reference_mandatory_options = ("reference", "simulation_criteria")
        _ref_data_or_criteria.add_argument(
            "--reference",
            type=lambda x: Path(x).resolve().as_posix(),
            help="""
            A "reference" csv/tsv file containing a column for each gene and a line for each observation (cell/sample).
            The "reference" will be used to compute the `simulation_criteria` needed in order to perform biased sampling
            and simulation.
            Expression data must be normalized before using this tool.""",
        )
        _ref_data_or_criteria.add_argument(
            "--simulation-criteria",
            type=lambda x: Path(x).resolve().as_posix(),
            help="""
            A "simulation_criteria" csv/tsv file, previously computed using this tool,
            having the default columns (or any extra criteria added by the user, which will
            be ignored) and a row for at least each gene contained in `in_file` (criteria for
            extra genes will simply be ignored).
            The criteria DataFrame will be used to generate a synthetic RNA-Seq dataset
            from the binary frame `in_file`.""",
        )
        parser.add_argument(
            "--dor-threshold",
            type=float,
            default=0.95,
            help="""DropOutRate (DOR) threshold. All genes having a DOR
            greater or equal to the specified value will be classified
            as discarded (no binarization or simulation will be applied).""",
        )
        parser.add_argument(
            "--n-samples",
            type=int,
            default=1,
            help="""Number of samples to be simulated per binary state""",
        )
        parser.add_argument(
            "--n-threads",
            type=int,
            default=mp.cpu_count(),
            help="""The number of parallel processes to be used.""",
        )
        parser.add_argument(
            "--output",
            type=lambda x: Path(x).resolve().as_posix(),
            help="""The name (can be a full path) of the file in which results should
            be stored. Defaults to `in_file`_synthetic.csv""",
        )
        parser.add_argument(
            "--rng-seed",
            type=int,
            help="""An integer which will be used to seed both R's
            and Python's (numpy) random number generators.""",
        )
        parser.add_argument(
            "--dump-criteria",
            action="store_true",
            help="""Should the computed simulation criteria be saved to a csv file
            to be reutilized afterwards?""",
        )
        parser.add_argument(
            "--dump-config",
            action="store_true",
            help="""
            Should the specified CLI arguments be dumped to a toml configuration
            file?""",
        )
        args = dict(vars(parser.parse_args(sys.argv[2:])))
        self._verify_mandatory_mutually_exclusive_options(
            args, _reference_mandatory_options, parser
        )
        self._verify_mandatory_mutually_exclusive_options(
            args, self._disposition_mandatory_options, parser
        )
        if args.get("genes_are_columns") is not None:
            args["genes_are_rows"] = not args.pop("genes_are_columns")
        args.update({"action": "synthesize"})
        _timestamp = _YIELD_TIMESTAMP()

        if args["dump_config"]:
            _config_dest = f"scBoolSeq_experiment_config_{_timestamp}.toml"
            if Path(_config_dest).resolve().exists():
                # avoid overwritting existing config files
                raise FileExistsError("Config file already exists, aborting")
            with open(_config_dest, "w", encoding="utf-8") as _c_f:
                _ = args.pop("dump_config")
                toml.dump(args, _c_f)

        simulator = scBoolSeqRunner(action=args.pop("action"), timestamp=_timestamp)
        simulator(args)

    def from_file(self):
        """Read the configuration params from a TOML config file"""
        parser = argparse.ArgumentParser(
            description="Read a toml configuration file to parse arguments for simulations"
        )
        parser.add_argument(
            "config_file",
            type=lambda p: Path(p).resolve(),
            help="""
            A toml configuration file. See https://toml.io/en/

            For the parameters' keys and values, run:
            `$ scBoolSeq [binarize | synthesize] -h`""",
        )
        parser.add_argument(
            "--dump-config",
            action="store_true",
            help="""
            Should the parameters defined in the specified config file be dumped to
            a new toml configuration file?""",
        )
        args = dict(vars(parser.parse_args(sys.argv[2:])))
        with open(args["config_file"], "r", encoding="utf-8") as config:
            params = toml.load(config)

        if "action" not in params.keys():
            parser.print_help()
            print(
                f"\nConfig file {args['config_file']} has no specified action, aborting."
            )
            sys.exit(1)

        if params["action"] not in scBoolSeqRunner.valid_actions:
            parser.print_help()
            print(
                f"\nConfig file {args['config_file']} requested an unknown action: ",
                f"`{params['action']}`.",
                f"Valid actions are: {scBoolSeqRunner.valid_actions}",
                sep="\n",
            )
            sys.exit(1)

        _timestamp = _YIELD_TIMESTAMP
        if args["dump_config"]:
            _config_dest = f"scBoolSeq_experiment_config_{_timestamp}.toml"
            if Path(_config_dest).resolve().exists():
                # avoid overwritting existing config files
                raise FileExistsError("Config file already exists, aborting")
            with open(_config_dest, "w", encoding="utf-8") as _c_f:
                _ = args.pop("dump_config")
                toml.dump(params, _c_f)

        binarizer_or_simulator = scBoolSeqRunner(
            action=params.pop("action"), timestamp=_timestamp
        )
        binarizer_or_simulator(params)
