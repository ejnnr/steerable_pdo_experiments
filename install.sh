#!/bin/bash

conda env create -f environment.yml
conda activate steerable_pdo_experiments
pip install git+git://github.com/treverhines/RBF.git
pip install torch-scatter torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip install git+git://github.com/ejnnr/steerable_pdos.git@experiments
pip install -e .