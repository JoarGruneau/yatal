from setuptools import find_packages, setup

setup(name='yatal',
      author='Joar Gruneau',
      description='Yet Another technical Analysis Library',
      license='MIT',
      version='0.1.0',
      packages=find_packages(),
      long_description=open('README.txt').read(),
      setup_requires=['numpy>=1.16.2', 'numba>=0.41.0'])
