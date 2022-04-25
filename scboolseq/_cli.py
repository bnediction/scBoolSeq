"""Command line parser for scBoolSeq"""

import multiprocessing as mp
from pathlib import Path
import argparse
import sys
import datetime as dt
import toml

import pandas as pd

# local relative import
from .core import scBoolSeq


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

    @staticmethod
    def f_frame_or_none(str_path_or_none):
        """Helper function, return a DataFrame if the path parameter has been parsed."""
        return (
            pd.read_csv(str_path_or_none, index_col=0)
            if str_path_or_none is not None
            else str_path_or_none
        )

    @staticmethod
    def f_out_file(path, suffix: str = "binarized"):
        """Default name for output file"""
        return path.parent.joinpath(
            path.name.replace(path.suffix, f"_{suffix}{path.suffix}")
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
        in_frame = self.f_frame_or_none(in_file)
        out_file = args.get("output") or self.f_out_file(in_file)

        scbs = scBoolSeq(r_seed=args.get("rng_seed"))
        scbs.data = self.f_frame_or_none(args.get("reference"))
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
            _name = in_file.name.replace(in_file.suffix, "")
            scbs.criteria.to_csv(f"scBoolSeq_criteria_{_name}_{self.timestamp}.csv")

        print(f"exclude_discarded={args.get('exclude_discarded')}")
        binarized = scbs.binarize(
            in_frame,
            alpha=args.get("alpha", 1.0),
            include_discarded=not args.get("exclude_discarded"),
            n_threads=args.get("n_threads") or mp.cpu_count(),
        )
        binarized.to_csv(out_file)

    def synthesize(self, args):
        """synthesize RNA-Seq data from boolean dynamics"""

        in_file = Path(args["in_file"]).resolve()
        in_frame = self.f_frame_or_none(in_file)
        out_file = args.get("output") or self.f_out_file(in_file, suffix="synthetic")

        scbs = scBoolSeq(r_seed=args.get("rng_seed"))
        scbs.data = self.f_frame_or_none(args.get("reference"))
        scbs.simulation_criteria = self.f_frame_or_none(args.get("simulation_criteria"))

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
            _name = in_file.name.replace(in_file.suffix, "")
            scbs.criteria.to_csv(
                f"scBoolSeq_simulation_criteria_{_name}_{self.timestamp}.csv"
            )

        synthetic = scbs.simulate(
            binary_df=in_frame,
            n_threads=args.get("n_threads") or mp.cpu_count(),
            n_samples=args.get("n_samples", 1),
            seed=args.get("rng_seed"),
        )
        synthetic.to_csv(out_file)

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
            usage="""scboolseq <command> [<args>]

Available commands:
\t* binarize\t  Binarize a RNA-Seq dataset.
\t* synthesize\t Simulate a RNA-Seq experiment from Boolean dynamics.
\t* from_file\t Repeat a binarization or synthethic generation experiment, based on a config file.

NOTE on CSV file specs:
* Assumed to use the standard separator for columns ','.
* The index is assumed to be the first column.
""",
        )
        self.main_parser.add_argument("command", help="Subcommand to run")
        # parse_args defaults to [1:] for args
        # exclude the rest of the args too, or validation will fail
        args = self.main_parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")
            self.main_parser.print_help()
            sys.exit(1)
        # dispatch the specified command
        getattr(self, args.command)()

    def binarize(self):
        """Binarize log2(CPM + 1) expression data."""
        parser = argparse.ArgumentParser(
            description="""
            Distribution-based binarization of RNA-Seq data.
                * Bimodal Genes: Binarization based on the probability of membership
                to each one of the two modalities. 
                * Unimodal and Zero-Inflated genes: Binarization based on parametric outlier
                assignment. By default, Tukey Fences [Q1 - alpha * IQR, Q3 + alpha * IQR],
                with (alpha == 1); The margin quantile (q25 yields Q1 and Q3) and alpha parameter
                can be optionally specified via the command line interface.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "in_file",
            type=lambda x: Path(x).resolve().as_posix(),
            help="""
            A csv file containing a column for each gene and a line for each observation (cell/sample).
            Expression data must be normalized before using this tool.""",
        )
        _ref_data_or_criteria = parser.add_mutually_exclusive_group(required=False)
        _ref_data_or_criteria.add_argument(
            "--reference",
            type=lambda x: Path(x).resolve().as_posix(),
            help="""
            A "reference" csv file containing a column for each gene and a line for each observation (cell/sample).
            The "reference" will be used to compute the criteria needed for binarization.
            Expression data must be normalized before using this tool.""",
        )
        _ref_data_or_criteria.add_argument(
            "--criteria",
            type=lambda x: Path(x).resolve().as_posix(),
            help="""
            A "criteria" csv file, previously computed using this tool,
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
            "--margin_quantile",
            type=float,
            default=0.25,
            help="""Margin quantile for parametric Tukey fences.""",
        )
        parser.add_argument(
            "--dor_threshold",
            type=float,
            default=0.95,
            help="""DropOutRate (DOR) threshold. All genes having a DOR
            greater or equal to the specified value will be classified
            as discarded (no binarization or simulation will be applied).""",
        )
        parser.add_argument(
            "--n_threads",
            type=int,
            default=mp.cpu_count(),
            help="""The number of parallel processes to be used.""",
        )
        parser.add_argument(
            "--rng_seed",
            type=int,
            help="""An integer which will be used to seed both R's
            and Python's (numpy) random number generators.""",
        )
        parser.add_argument(
            "--exclude_discarded",
            action="store_true",
            help="""Should discarded genes be excluded from the output file?""",
        )
        parser.add_argument(
            "--output",
            type=lambda x: Path(x).resolve().as_posix(),
            help="""The name (can be a full path) of the file in which results should
            be stored. Defaults to `in_file`_binarized.csv""",
        )
        parser.add_argument(
            "--dump_criteria",
            action="store_true",
            help="""Should the computed criteria be saved to a csv file
            to be reutilized afterwards?""",
        )
        parser.add_argument(
            "--dump_config",
            action="store_true",
            help="""
            Should the specified CLI arguments be dumped to a toml configuration
            file?""",
        )
        # ignore the command and the subcommand, parse all options :
        args = dict(vars(parser.parse_args(sys.argv[2:])))
        args.update({"action": "binarize"})
        _timestamp = str(dt.datetime.now()).split(".", maxsplit=1)[0].replace(" ", "_")

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
            A csv file containing a column for each gene and a line for each observation 
            (an observation here is defined as a FULLY DETERMINED Boolean state). Preferrably,
            Boolean states should be represented as zeros and ones.
            """,
        )
        _ref_data_or_criteria = parser.add_mutually_exclusive_group(required=False)
        _reference_mandatory_options = ("reference", "simulation_criteria")
        _ref_data_or_criteria.add_argument(
            "--reference",
            type=lambda x: Path(x).resolve().as_posix(),
            help="""
            A "reference" csv file containing a column for each gene and a line for each observation (cell/sample).
            The "reference" will be used to compute the `simulation_criteria` needed in order to perform biased sampling
            and simulation.
            Expression data must be normalized before using this tool.""",
        )
        _ref_data_or_criteria.add_argument(
            "--simulation_criteria",
            type=lambda x: Path(x).resolve().as_posix(),
            help="""
            A "simulation_criteria" csv file, previously computed using this tool,
            having the default columns (or any extra criteria added by the user, which will
            be ignored) and a row for at least each gene contained in `in_file` (criteria for
            extra genes will simply be ignored).
            The criteria DataFrame will be used to generate a synthetic RNA-Seq dataset
            from the binary frame `in_file`.""",
        )
        parser.add_argument(
            "--dor_threshold",
            type=float,
            default=0.95,
            help="""DropOutRate (DOR) threshold. All genes having a DOR
            greater or equal to the specified value will be classified
            as discarded (no binarization or simulation will be applied).""",
        )
        parser.add_argument(
            "--n_samples",
            type=int,
            default=1,
            help="""Number of samples to be simulated per binary state""",
        )
        parser.add_argument(
            "--n_threads",
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
            "--rng_seed",
            type=int,
            help="""An integer which will be used to seed both R's
            and Python's (numpy) random number generators.""",
        )
        parser.add_argument(
            "--dump_criteria",
            action="store_true",
            help="""Should the computed simulation criteria be saved to a csv file
            to be reutilized afterwards?""",
        )
        parser.add_argument(
            "--dump_config",
            action="store_true",
            help="""
            Should the specified CLI arguments be dumped to a toml configuration
            file?""",
        )
        args = dict(vars(parser.parse_args(sys.argv[2:])))
        if not any(args[_key] for _key in _reference_mandatory_options):
            parser.print_help()
            print(
                "\nCannot simulate without any reference.",
                f"Please specify either --{_reference_mandatory_options[0]}",
                f"or --{_reference_mandatory_options[1]}",
                sep=" ",
            )
            sys.exit(1)
        args.update({"action": "synthesize"})
        _timestamp = str(dt.datetime.now()).split(".", maxsplit=1)[0].replace(" ", "_")

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
        # NOT prefixing the argument with -- means it's not optional
        parser.add_argument(
            "config_file",
            type=lambda p: Path(p).resolve(),
            help="""
            A toml configuration file. See https://toml.io/en/

            For the parameters' keys and values, run:
            `$ scBoolSeq [binarize | synthesize] -h`""",
        )
        parser.add_argument(
            "--dump_config",
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

        _timestamp = str(dt.datetime.now()).split(".", maxsplit=1)[0].replace(" ", "_")
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
