from setuptools import setup, find_packages

setup(name="gapstat",
      version = '0.0.1',
      author = 'John M. Maloney',
      package_dir = {'':'src'},
      py_modules = ['gapstat'],
      install_requires = ['scikit-learn']
)