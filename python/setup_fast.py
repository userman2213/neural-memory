#!/usr/bin/env python3
"""
Build Cython extension for Neural Memory hot paths.

Usage:
    python setup_fast.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "fast_ops",
        sources=["fast_ops.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-march=native"],
    ),
]

setup(
    name="neural-memory-fast-ops",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
)
