# -*- coding:utf-8 -*-

"""
setup.py  to build LSTM lib code with cython
"""
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy        # to get includes


fast_pyx = Extension("fast",
					["fast.pyx"],
					extra_compile_args=['-fopenmp', '-march=native', '-Ofast', '-O3', '-flto', '-fwhole-program', ],
					#extra_compile_args=['-march=native', '-Ofast', '-O3', '-flto', '-fwhole-program', ],
					extra_link_args=['-fopenmp']
					)

setup(
	cmdclass = {'build_ext': build_ext},
	ext_modules = [fast_pyx],
	include_dirs = [numpy.get_include(),],
)

