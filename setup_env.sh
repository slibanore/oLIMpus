#!/bin/bash

# Create a new Conda environment named 'myenv' with Python 3.10
conda deactivate
conda env remove --name oLIMpus --all -y
conda create --name oLIMpus python=3.10

# Activate the environment
conda activate oLIMpus

# Install required Python packages
conda install cython 
conda install ipykernel 
conda install pygments 
conda install pexpect 
pip install .

echo "Conda environment 'oLIMpus' is set up and packages are installed."