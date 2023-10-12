"""
scBoolSeq: Linking scRNA-Seq with Boolean Dynamics

author: "Gustavo Magaña López"
credits: "BNediction ; Institut Curie"
"""

__version__ = "0.2.0"
__author__ = "Gustavo Magaña López"
__credits__ = "BNeDiction; Institut Curie"

# Packages
from . import binarization

# Main Class
from .binarization import scBoolSeqBinarizer as scBoolSeq
