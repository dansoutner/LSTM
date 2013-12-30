#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
setup.py  to build LSTM lib code with cython
"""
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy        # to get includes


arpalm_pyx = Extension("ArpaLM",
					["ArpaLM.py"],)

lda_pyx = Extension("lda",
					["lda.py"],)

lstm_pyx = Extension("LSTM",
					["LSTM.pyx"],
					#extra_compile_args=['-fopenmp', '-march=native', '-Ofast', '-O3', '-flto', '-fwhole-program', '-funroll-loops' ],
					extra_compile_args=['-march=native', '-Ofast', '-O3', '-flto', '-fwhole-program', '-funroll-loops' ],
					#extra_link_args=['-fopenmp']
					)

setup(
	cmdclass = {'build_ext': build_ext},
	#ext_modules = [arpalm_pyx, lda_pyx, lstm_pyx],
	ext_modules = [lstm_pyx],
	include_dirs = [numpy.get_include()],
)

