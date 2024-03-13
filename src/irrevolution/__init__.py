# library include

import matplotlib.pyplot as plt

import dolfinx.io
import numpy as np
import yaml
import json
import sys
import os
from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.plot
from dolfinx import log
import ufl
from dolfinx.io import XDMFFile

import logging
logging.basicConfig(level=logging.INFO)
logging.critical(
    f"DOLFINx version: {dolfinx.__version__} based on GIT commit: {dolfinx.git_commit_hash} of https://github.com/FEniCS/dolfinx/")

from petsc4py import PETSc
comm = MPI.COMM_WORLD

from dolfinx.fem import (
    Constant,
    Function,
    FunctionSpace,
    assemble_scalar,
    dirichletbc,
    form,
    locate_dofs_geometrical,
    set_bc,
)
