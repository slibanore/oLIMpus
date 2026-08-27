#!/bin/bash
set -e

# Creates a conda environment named 'oLIMpus' with python 3.10 and installs the code.
# CLASS must already be installed in that environment (see the README).

conda create --name oLIMpus python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate oLIMpus

conda install -y cython ipykernel pygments pexpect

# installs oLIMpus and, through install_requires, Zeus21 from the zeus21_hack branch
pip install .

python -m ipykernel install --user --name oLIMpus --display-name "oLIMpus"

echo "Conda environment 'oLIMpus' is ready. Version: $(python -c 'import oLIMpus' 2>/dev/null | head -1)"
