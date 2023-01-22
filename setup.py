"""Setup script for jax_newton_raphson."""

from setuptools import setup

setup(name='jax_newton_raphson',
      author="Cambridge Yang",
      author_email="camyang@csail.mit.edu",
      description="Newton-Raphson minimizer in JAX",
      url="https://github.com/thisiscam/jax_newton_raphson",
      version='0.1.0',
      py_modules=['jax_newton_raphson'],
      install_requires=[
          'jax',
          'chex',
      ])
