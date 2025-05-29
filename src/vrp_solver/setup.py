from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

class get_pybind_include:
    """Helper class to determine the pybind11 include path"""
    def __str__(self):
        import pybind11
        return pybind11.get_include()

ext_modules = [
    Extension(
        'improved_genetic_vrp_final',
        ['improved_genetic_vrp_final.cpp'],
        include_dirs=[
            get_pybind_include(),
        ],
        language='c++',
        extra_compile_args=['-std=c++17'],
    ),
]

setup(
    name='improved_genetic_vrp_final',
    version='0.1',
    author='TuNombre',
    author_email='tucorreo@example.com',
    description='VRP Solver avanzado con GA y mejoras multi-hilo',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)
