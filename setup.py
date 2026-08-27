#!/usr/bin/env python
"""oLIMpus installation.

The version is NOT written here. It comes from oLIMpus/_version.py, which reads the
MAJOR from the VERSION file and derives the MINOR from the git history; see that module.
We load it by path rather than importing the package, because importing oLIMpus pulls in
zeus21, CLASS and mcfit, which are not yet installed when setup.py runs.
"""

from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).resolve().parent

_ns = {"__file__": str(HERE / "oLIMpus" / "_version.py")}
exec((HERE / "oLIMpus" / "_version.py").read_text(), _ns)
VERSION = _ns["get_version"]()

# freeze it so an sdist/wheel built from this tree carries the number git gave us
_ns["freeze"](VERSION)

setup(
    name='oLIMpus',
    version=VERSION,
    description='oLIMpus: an effective model for line intensity mapping auto- and '
                'cross-power spectra in cosmic dawn and reionization.',
    url='https://github.com/slibanore/oLIMpus',
    author='Sarah Libanore',
    author_email='libanore@bgu.ac.il',
    license='MIT',
    packages=find_packages(include=['oLIMpus', 'oLIMpus.*']),
    package_data={'oLIMpus': ['*.jpeg', '*.png', '_static_version.txt']},
    long_description=(HERE / 'README.md').read_text(),
    long_description_content_type='text/markdown',
    python_requires='>=3.9',
    install_requires=[
        # zeus21 is a hard dependency and oLIMpus v2 needs the zeus21_hack branch,
        # which is not on PyPI. A PEP 508 direct reference is honoured by
        # `pip install .` and `pip install -e .`; it is only forbidden on PyPI uploads.
        "zeus21 @ git+https://github.com/ZeusCosmo/Zeus21@zeus21_hack",
        "numpy",
        "scipy",
        "mcfit",
        "classy",
        "numexpr",
        "astropy",
        "pyfftw",
        "powerbox",
        "tqdm",
        "matplotlib",
    ],
    extras_require={
        'docs': ["sphinx", "myst_parser"],
        'notebooks': ["jupyter", "ipykernel"],
    },
)
