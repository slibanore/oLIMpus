#!/usr/bin/env python

from setuptools import setup, find_packages

# SarahLibanore: install oLIMpus and zeus21
setup(
    name='oLIMpus',
          version='2.0',
          description='oLIMpus: cross-correlating lines with Zeus21.',
          url='https://github.com/slibanore/Zeus21',
          author='Sarah Libanore',
          author_email='libanore@bgu.ac.il',
          license='MIT',
          packages=find_packages(),
          long_description=open('README.md').read(),
          install_requires=[
           "mcfit",
           "classy",
           "numexpr",
           "astropy",
           "pyfftw",
           "powerbox",
           "tqdm",
           "matplotlib",
            "zeus21 @ git+https://github.com/ZeusCosmo/Zeus21.git@zeus21_hack",
       ],
)
