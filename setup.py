from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "dGLMNET_cython",
        ["dGLMNET_cython.pyx"],
        extra_compile_args=['/openmp'],
    )
]

setup(
    ext_modules=cythonize(ext_modules),
)