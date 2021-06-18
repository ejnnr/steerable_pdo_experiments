#!/bin/bash

conda env create -f environment.yml
conda activate steerable_pdo_experiments
pip install git+git://github.com/ejnnr/RBF.git
pip install git+git://github.com/ejnnr/steerable_pdos.git
pip install -e .