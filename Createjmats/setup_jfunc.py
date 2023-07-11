from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import sys

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

setup(
	name = "Jfunc_cython_v6",
	ext_modules = cythonize(
		Extension(
			"Jfunc_cython_v6",
			sources = ["Jfunc_cython_v6.pyx"],
            #include_dirs = ["/opt/homebrew/include/"],
			#include_dirs= [numpy.get_include(), "/opt/homebrew/include/", "/opt/homebrew/Cellar/"],
			include_dirs = [numpy.get_include(), "/opt/homebrew/include/", "/usr/local/"],
            libraries = ['gmp', 'mpfr', 'mpc'],
		),
		annotate = True),
	install_requires = ["numpy"],
	zip_safe = False
)

# python setup.py build_ext --inplace
