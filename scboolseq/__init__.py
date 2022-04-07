"""
scBoolSeq:
scRNA-Seq data binarisation and synthetic generation from Boolean dynamics.

author: "Gustavo Magaña López"
credits: "BNediction ; Institut Curie"
"""

from .core import scBoolSeq
from .utils.normalization import log_transform, normalize, log_normalize

__version__ = "0.1.0"
__author__ = "Gustavo Magaña López"
__credits__ = "BNeDiction; Institut Curie"
