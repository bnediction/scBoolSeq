"""
scBoolSeq: Linking scRNA-Seq with Boolean Dynamics

author: "Gustavo Magaña López"
credits: "BNediction ; Institut Curie"
"""

__version__ = "9999"
__author__ = "Gustavo Magaña López"
__credits__ = "BNeDiction; Institut Curie"

# Packages
from . import binarization
from . import meta

# Main Class
from .binarization import scBoolSeqBinarizer as scBoolSeq
