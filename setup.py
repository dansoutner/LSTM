#!/usr/bin/env python

"""
setup.py  to build mandelbot code with cython
"""
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy # to get includes


setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("LSTM", ["LSTM.pyx"], )],
    include_dirs = [numpy.get_include(),],
)
