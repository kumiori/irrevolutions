#!/usr/bin/env python3
import json
import logging
import os
import sys
from pathlib import Path

import dolfinx
import dolfinx.mesh
import dolfinx.plot
import numpy as np
import pandas as pd
import petsc4py
import pyvista
import ufl
import yaml
from dolfinx.fem import (
    Constant,
    Function,
    assemble_scalar,
    dirichletbc,
    form,
    locate_dofs_geometrical,
    set_bc,
)
from dolfinx.common import list_timings
from dolfinx.fem.petsc import assemble_vector, set_bc
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc

from irrevolutions.algorithms.am import HybridSolver
from irrevolutions.algorithms.so import BifurcationSolver, StabilitySolver
from irrevolutions.solvers import SNESSolver
from irrevolutions.solvers.function import vec_to_functions
from irrevolutions.utils import (
    ColorPrint,
    _logger,
    _write_history_data,
    history_data,
    norm_H1,
    norm_L2,
)
from irrevolutions.utils.plots import (
    plot_AMit_load,
    plot_energies,
    plot_force_displacement,
)
from  irrevolutions.test.test_1d import _AlternateMinimisation1D as am1d


petsc4py.init(sys.argv)
comm = MPI.COMM_WORLD

# Mesh on node model_rank and then distribute
model_rank = 0

def run_computation(parameters, storage=None):

    return


def load_parameters(file_path, ndofs, model="at1"):
    """
    Load parameters from a YAML file.

    Args:
        file_path (str): Path to the YAML parameter file.

    Returns:
        dict: Loaded parameters.
    """
    import hashlib

    with open(file_path) as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    parameters["model"]["model_dimension"] = 1
    parameters["model"]["model_type"] = "1D"
    # parameters["model"]["mu"] = 1
    parameters["model"]["w1"] = 1

    parameters["geometry"]["geom_type"] = "discrete-damageable"
    # Get mesh parameters

    if model == "at2":
        parameters["loading"]["min"] = 0.9
        parameters["loading"]["max"] = 0.9
        parameters["loading"]["steps"] = 1

    elif model == "at1":
        parameters["loading"]["min"] = 0.0
        parameters["loading"]["max"] = 1.5
        parameters["loading"]["steps"] = 20

    parameters["geometry"]["geom_type"] = "traction-bar"
    parameters["geometry"]["mesh_size_factor"] = 4
    parameters["geometry"]["N"] = ndofs

    parameters["stability"]["cone"]["cone_max_it"] = 400000
    parameters["stability"]["cone"]["cone_atol"] = 1e-6
    parameters["stability"]["cone"]["cone_rtol"] = 1e-6
    parameters["stability"]["cone"]["scaling"] = 1e-2

    parameters["model"]["w1"] = 1
    parameters["model"]["ell"] = 0.2
    parameters["model"]["k_res"] = 0.0

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    return parameters, signature


if __name__ == "__main__":
    # Set the logging level
    logging.basicConfig(level=logging.INFO)

    # Load parameters
    parameters, signature = load_parameters(
        os.path.join(os.path.dirname(__file__), "parameters.yaml"), 100, "at1")

    # Run computation
    run_computation(parameters)