"""
scBoolSeq-meta: Define meta-observations from binarised data. 
author: "Gustavo Magaña López"
credits: "BNediction ; Institut Curie"
"""

__version__ = "0.2.0"
__author__ = "Gustavo Magaña López"
__credits__ = "BNeDiction; Institut Curie"

# Packages
from . import bootstrap

# classes and functions
from .aggregations import CellAggregator
from .metrics import meta_marker_counter
