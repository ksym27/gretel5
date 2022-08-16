#!/bin/bash

if [[ $# -eq 0 ]] ; then
    echo 'Please provide environment name'
    exit 1
fi

ENV_NAME=$1

# basic env
# basic env
conda create --yes -n $ENV_NAME python=3.7 \
    geopandas \
    jupyterlab \
    matplotlib \
    nbconvert \
    networkx \
    notebook \
    numpy \y
    osmnx \
    pandas \
    pylint \
    rope \
    scikit-learn \
    scipy \
    seaborn \
    tabulate \
    tensorboardx \
    termcolor \
    tqdm \
    yapf \
    -c conda-forge

source activate $ENV_NAME


pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install torch-geometric
pip install pygsp



source deactivate

echo "Created environment $ENV_NAME"
