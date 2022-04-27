
# scBoolSeq

scRNA-Seq data binarisation and synthetic generation from Boolean dynamics.

## Installation

### Conda

```
conda install -c conda-forge -c colomoto scboolseq
```

### Pip

You need `R` installed.

```
pip install scboolseq
```

TODO: how to install R dependencies

## Usage

### Command line

scBoolSeq provides a rich CLI allowing programmatic access to its main functionalities, namely the `binarization` of RNA-Seq data and the 
generation of synthetic RNA-Seq data `synthesis` reflecting activation states from Boolean Network simulations. Once correctly instaled, 
the tool's and subcommand's help explain all the possible parameters. Bellow you have an overview 

#### Main CLI

```bash
$ scBoolSeq -h
usage: scBoolSeq <command> [<args>]

Available commands:
	* binarize	 Binarize a RNA-Seq dataset.
	* synthesize	 Simulate a RNA-Seq experiment from Boolean dynamics.
	* from_file	 Repeat a binarization or synthethic generation experiment, based on a config file.

NOTE on TSV/CSV file specs:
* If '.csv', the file is assumed to use the standard separator for columns ','.
* The index is assumed to be the first column.

scBoolSeq: bulk and single-cell RNA-Seq data binarization and synthetic generation from Boolean dynamics.

positional arguments:
  command     Subcommand to run

optional arguments:
  -h, --help  show this help message and exit
```

#### Binarization

Minimal example of binarization, specifying some optional parameters.

```bash
curl -fOL https://github.com/pinellolab/STREAM/raw/master/stream/tests/datasets/Nestorowa_2016/data_Nestorowa.tsv.gz

ls
# data_Nestorowa.tsv.gz
time scBoolSeq binarize data_Nestorowa.tsv.gz --transpose_in --output Nestorowa_binarized.csv --n_threads 8 --dump_config --dump_criteria
# ________________________________________________________
# Executed in   34.49 secs   fish           external 
#   usr time   30.04 secs  1211.00 micros   30.04 secs 
#   sys time    3.90 secs  171.00 micros    3.89 secs 

ls
# data_Nestorowa.tsv.gz    scBoolSeq_criteria_data_Nestorowa_2022-04-26_12h46m48.tsv
# Nestorowa_binarized.csv  scBoolSeq_experiment_config_2022-04-26_12h46m48.toml

```

### Python API

TODO
