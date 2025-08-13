#!/usr/bin/env python
# -*- coding: utf-8

from setuptools import find_packages
from setuptools import setup

NAME = "scBoolSeq"
VERSION = "9999"

setup(
    name=NAME,
    version=VERSION,
    author="Gustavo Magaña López",
    author_email="gustavo.magana@labri.fr",
    url="https://github.com/bnediction/scBoolSeq",
    packages=find_packages(exclude=("tests",)),
    license="BSD-3",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords='',
    description="scBoolSeq: Linking scRNA-Seq Statistics and Boolean Dynamics.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "pandas",
        "diptest",
        "scikit-learn>=1.6.0",# should include numpy and scipy
        "statsmodels"
    ],
)

