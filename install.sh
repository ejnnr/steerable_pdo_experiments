#!/bin/bash

set -e

poetry install
# There are some issues installing RBF with poetry
# caused by the fact that it uses cython during installation.
# We circumvent that by installing it manually afterward.
poetry run pip install git+https://github.com/ejnnr/RBF
# Finally, install the version of the e2cnn library needed
# for these experiments
poetry run pip install git+https://github.com/ejnnr/steerable_pdos@pdo_econv