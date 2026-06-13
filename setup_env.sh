#!/bin/bash
conda deactivate
conda env remove --name oLIMpus --all -y
conda create --name oLIMpus python=3.10 -y

source $(conda info --base)/etc/profile.d/conda.sh
conda activate oLIMpus

conda install -y cython ipykernel pygments pexpect
pip install --upgrade .

echo "Conda environment 'oLIMpus' is set up and packages are installed."