#!/usr/bin/env python
# -*- coding: utf-8

from setuptools import find_packages
from setuptools import setup

NAME = "scBoolSeq"
VERSION = "0.1.0"

setup(
    name=NAME,
    version=VERSION,
    author="Gustavo Magaña López",
    author_email="gustavo.magana-lopez@u-psud.fr",
    url="https://github.com/bnediction/scBoolSeq",
    packages=find_packages(exclude=("tests",)),
    license="BSD-3",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords='',
    description="scRNA-Seq data binarisation and synthetic generation from Boolean dynamics.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "pandas",
        "matplotlib",
        "toml",
        "plotnine",
        "rpy2",
    ],
    package_data={'profile_binr': ['_R/*.R']}
)

