#!/bin/bash

# Define paths
PACKAGE_NAME="irrevolutions"
WHEEL_DIR="wheelhouse"
DIST_DIR="dist"

# Clean previous builds
rm -rf $WHEEL_DIR $DIST_DIR
mkdir $WHEEL_DIR

# Install required tools
pip install setuptools wheel cibuildwheel

# Build the package
python setup.py bdist_wheel
# cibuildwheel --platform linux

# Download dependencies
# pip download -r requirements.txt --dest=$WHEEL_DIR
pip download -r requirements.txt --platform manylinux2014_x86_64 --dest wheelhouse --no-deps
#  --python-version 3.9
# Transfer to the cluster
scp -r $WHEEL_DIR $DIST_DIR/*.whl leonbala@irene-fr.ccc.cea.fr:/ccc/work/cont003/gen14282/leonbala/wheels