""" Command line parser for IRM """

from pathlib import Path
import argparse
import sys
import datetime as dt
import toml

from .core import scBoolSeq


class scBoolSeqRunner(object):
    """Runner used to call scboolseq.core.scBoolSeq using
    the arguments parsed by scboolseq._cli.scBoolSeqCLIParser"""

    def __init__(self, action: str):
        self._valid_actions = ("binarize", "synthesize")
        if action not in self._valid_actions:
            raise ValueError(
                f"Unknown action `{action}`. Available options are {self._valid_actions}"
            )
        self.action = action

    def binarize(self, args):
        """binarize expression data"""
        print("binarizing")
        for arg, val in args.items():
            print(f"arg({arg}) = {val}")

    def synthesize(self, args):
        """synthesize RNA-Seq data from boolean dynamics"""
        print("synthesizing")
        for arg, val in args.items():
            print(f"arg({arg}) = {val}")

    def __call__(self, args):
        f"""{self.action} via scBoolSeq"""
        getattr(self, self.action)(args)


class scBoolSeqCLIParser(object):
    """Semantic command line parser, serving as an interface for
    the embedded tool scBoolSeq.
    """

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="""
                scBoolSeq: bulk and single-cell RNA-Seq data binarization and synthetic 
                generation from Boolean dynamics
            """,
            usage="""scboolseq <command> [<args>]

Available commands:
\t* binarize\t  Binarize a RNA-Seq dataset.
\t* synthesize\t Simulate a RNA-Seq experiment from Boolean dynamics.
\t* from_file\t Repeat a binarization or synthethic generation experiment, based on a config file.
""",
        )
        parser.add_argument("command", help="Subcommand to run")
        # parse_args defaults to [1:] for args
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")
            parser.print_help()
            sys.exit(1)
        # use dispatch pattern to invoke method with same name
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
                with alpha <- 1; The margin quantile (q25 yields Q1 and Q3) and alpha parameter
                can be optionally specified via the command line interface.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "in_file",
            type=lambda x: Path(x).resolve(),
            help="""
            A csv file containing a column for each gene and a line for each observation (cell/sample).
            Expression data must be normalized before using this tool.""",
        )
        _ref_data_or_criteria = parser.add_mutually_exclusive_group(required=False)
        _ref_data_or_criteria.add_argument(
            "--reference",
            type=lambda x: Path(x).resolve(),
            help="""
            A "reference" csv file containing a column for each gene and a line for each observation (cell/sample).
            The "reference" will be used to compute the criteria needed for binarization.
            Expression data must be normalized before using this tool.""",
        )
        _ref_data_or_criteria.add_argument(
            "--criteria",
            type=lambda x: Path(x).resolve(),
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
            help="""Multiplier for IQR (float: %(default)f)""",
        )
        parser.add_argument(
            "--margin_quantile",
            type=float,
            default=0.25,
            help="""Margin quantile for parametric Tukey fences (float: %(default)f)""",
        )
        parser.add_argument(
            "--dump_config",
            type=bool,
            default=False,
            help="""
            Should the specified parameters be dumped to a toml configuration
            file? (bool: %(default)d)""",
        )
        # ignore the command and the subcommand, parse all options :
        args = dict(vars(parser.parse_args(sys.argv[2:])))
        args.update({"action": "binarize"})

        if args["dump_config"]:
            _timestamp = (
                str(dt.datetime.now()).split(".", maxsplit=1)[0].replace(" ", "_")
            )
            _config_dest = f"scBoolSeq_experiment_config_{_timestamp}.toml"
            if Path(_config_dest).resolve().exists():
                # avoid overwritting existing config files
                raise FileExistsError("Config file already exists, aborting")
            with open(_config_dest, "w", encoding="utf-8") as _c_f:
                toml.dump(args, _c_f)

        binarizer = scBoolSeqRunner(action=args.pop("action"))
        binarizer(args)

    def from_file(self):
        """Read the configuration params from a config parser."""
        raise NotImplementedError("not yet.")
