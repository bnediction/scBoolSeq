
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

##### Synthetic generation from Boolean states

**todo** modify this after API modifications.

```bash
cat minimal_boolean_example.csv 
# the output is not commented out so that it can be copied
# and perhaps be read with `x = pandas.read_clipboard(sep=',')`
cell_id,Kdm3a,Coro2b,8430408G22Rik,Clec9a,Phf6
HSPC_025,1.0,1.0,1.0,1.0,0.0
HSPC_031,1.0,1.0,0.0,0.0,0.0
HSPC_037,0.0,1.0,0.0,0.0,0.0
LT-HSC_001,0.0,1.0,1.0,1.0,1.0
HSPC_001,0.0,1.0,0.0,1.0,1.0
HSPC_008,1.0,0.0,1.0,0.0,1.0
HSPC_014,0.0,1.0,1.0,1.0,0.0
HSPC_020,0.0,0.0,1.0,0.0,1.0
HSPC_026,0.0,1.0,1.0,1.0,1.0
HSPC_038,0.0,0.0,0.0,1.0,1.0
LT-HSC_002,0.0,0.0,0.0,0.0,0.0
HSPC_002,0.0,0.0,0.0,0.0,1.0
HSPC_009,1.0,0.0,1.0,0.0,0.0
HSPC_015,0.0,1.0,0.0,0.0,1.0
HSPC_021,1.0,1.0,1.0,0.0,0.0


scBoolSeq synthesize minimal_boolean_example.csv --reference data_Nestorowa.tsv.gz\
     --transpose_ref --n_samples 40 --output new_synthetic.tsv --n_threads 12
```

### Python API

```python
import pandas as pd
from scboolseq import scBoolSeq
from scboolseq.simulation import random_nan_binariser


# read in the normalized expression data
nestorowa = pd.read_csv("data_Nestorowa.tsv.gz", index_col=0, sep="\t")
nestorowa.iloc[1:5, 1:5] 
#                HSPC_031  HSPC_037  LT-HSC_001  HSPC_001
# Kdm3a          6.877725  0.000000    0.000000  0.000000
# Coro2b         0.000000  6.913384    8.178374  9.475577
# 8430408G22Rik  0.000000  0.000000    0.000000  0.000000
# Clec9a         0.000000  0.000000    0.000000  0.000000
#
# NOTE : here, genes are rows and observations are columns

# scBoolSeq expects genes to be columns, thus we transpose the DataFrame.
scbool_nest = scBoolSeq(data=nestorowa.T, r_seed=1234)
scbool_nest
# scBoolSeq(has_data=True, can_simulate=False, can_simulate=False)
scbool_nest.fit() # compute binarization criteria
# scBoolSeq(has_data=True, can_binarize=True, can_simulate=False)

scbool_nestorowa.simulation_fit() # compute simulation criteria
# scBoolSeq(has_data=True, can_binarize=True, can_simulate=True)

binarized = scbool_nestorowa.binarize(nestorowa.T)
binarized.iloc[1:5, 1:5] 
#             Kdm3a  Coro2b  8430408G22Rik  Phf6
# HSPC_031      1.0     NaN            NaN   0.0
# HSPC_037      0.0     1.0            NaN   0.0
# LT-HSC_001    0.0     1.0            NaN   1.0
# HSPC_001      0.0     1.0            NaN   1.0

fully_bin = binarized.iloc[1:5, 1:5].pipe(random_nan_binariser) # randomly (equiprobably) binarize undetermined values
fully_bin 
#             Kdm3a  Coro2b  8430408G22Rik  Phf6
# HSPC_031      1.0     0.0            1.0   0.0
# HSPC_037      0.0     1.0            1.0   0.0
# LT-HSC_001    0.0     1.0            0.0   1.0
# HSPC_001      0.0     1.0            1.0   1.0

# create a synthetic frame, with two samples per boolean state,
# fixing the rng's seed for reproductibility
# specyfing the number of threads to use
scbool_nestorowa.simulate(fully_bin, n_threads=4, seed=1234, n_samples=2) 
#               Kdm3a    Coro2b  8430408G22Rik      Phf6
# HSPC_031    7.328819  0.000000       8.087928  0.923352
# HSPC_037    1.003712  6.843611       7.003577  0.000000
# LT-HSC_001  0.000000  0.000000       0.000000  5.174053
# HSPC_001    1.672793  0.000000       0.000000  4.481709
# HSPC_031    8.536391  1.060373       0.000000  3.267464
# HSPC_037    1.055816  5.479887       0.000000  3.836276
# LT-HSC_001  0.000000  0.000000       0.000000  8.131221
# HSPC_001    2.451340  0.000000       0.000000  9.969012

```
