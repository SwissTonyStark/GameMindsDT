#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages
import setuptools

def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    init = open(os.path.join("dt_mine_rl", "__init__.py"), "r").read().split()
    return init[init.index("__version__") + 1][1:-1]

setup(
    name="DT Mine RL Project", 
    version=get_version(),
    description="DT Mine RL Project",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    author="GameMindsDT-team",
    author_email="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Intended Audience :: Academic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    keywords="learning from human feedback reinforcement learning minecraft pytorch with a decision transformer model",
    python_requires='>=3.8',
)
