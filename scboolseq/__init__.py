"""
The PROFILE methodology for the binarisation and normalisation of RNA-seq data.
"""

from .core import scBoolSeq
from .utils.normalization import log_transform, normalize, log_normalize

__version__ = "0.1.0"
__author__ = "Gustavo Magaña López"
__credits__ = "BNeDiction; Institut Curie"
