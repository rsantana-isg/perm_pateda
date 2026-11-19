from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "perm_pateda._cython.distances",
        ["src/perm_pateda/_cython/distances.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "perm_pateda._cython.consensus",
        ["src/perm_pateda/_cython/consensus.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "perm_pateda._cython.learning.mallows",
        ["src/perm_pateda/_cython/learning/mallows.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "perm_pateda._cython.sampling.mallows",
        ["src/perm_pateda/_cython/sampling/mallows.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "perm_pateda._cython.algorithms.permutation",
        ["src/perm_pateda/_cython/algorithms/permutation.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
    "perm_pateda._cython.learning.plackett_luce",
    ["src/perm_pateda/_cython/learning/plackett_luce.pyx"],
    include_dirs=[np.get_include()],
    ),
    Extension(
    "perm_pateda._cython.sampling.plackett_luce",
    ["src/perm_pateda/_cython/sampling/plackett_luce.pyx"],
    include_dirs=[np.get_include()],
    ),

    Extension(
        "perm_pateda._cython.learning.lehmer",
        ["src/perm_pateda/_cython/learning/lehmer.pyx"],
        include_dirs=[np.get_include()],
    ),

    Extension(
        "perm_pateda._cython.sampling.lehmer",
        ["src/perm_pateda/_cython/sampling/lehmer.pyx"],
        include_dirs=[np.get_include()],
        ),
    Extension(
    "perm_pateda._cython.representations.lehmer",
    ["src/perm_pateda/_cython/representations/lehmer.pyx"],
    include_dirs=[np.get_include()],
    ),
]

setup(
    name="perm_pateda",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
        },
    ),
)