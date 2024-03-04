import logging
import json
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import sys
import os

import dolfinx
import dolfinx.plot
from dolfinx import log
import ufl

from dolfinx.fem import (
    Constant,
    Function,
    FunctionSpace,
    locate_dofs_geometrical,
    assemble_scalar,
    dirichletbc,
    form,
    set_bc,
)
from dolfinx.fem.petsc import assemble_vector
from dolfinx.mesh import CellType
from dolfinx.io import XDMFFile, gmshio
from dolfinx.common import Timer, list_timings, TimingType

from mpi4py import MPI
import petsc4py
from petsc4py import PETSc

sys.path.append("../")
from models import DamageElasticityModel as Brittle
from algorithms.am import AlternateMinimisation, HybridSolver
from algorithms.so_merged import BifurcationSolver, StabilitySolver
from solvers import SNESSolver
from meshes.primitives import mesh_bar_gmshapi
from utils import ColorPrint
from utils.plots import plot_energies
from utils import norm_H1, norm_L2


# Define the Computation class
class Computation:
    def __init__(self, parameters_file):
        self.parameters_file = parameters_file
        self.parameters = self.load_parameters(parameters_file)
        self.signature = self.generate_signature(self.parameters)
        self.output_directory = f"output/{self.signature}"
        self.mesh = None
        self.V_u = None
        self.V_alpha = None
        self.u = None
        self.u_ = None
        self.zero_u = None
        self.alpha = None
        self.zero_alpha = None
        self.alphadot = None
        self.alpha_lb = None
        self.alpha_ub = None
        self.bcs_u = []
        self.bcs_alpha = []
        self.model = None
        self.solver = None
        self.hybrid = None
        self.bifurcation = None
        self.cone = None
        self.history_data = {
            "load": [],
            "elastic_energy": [],
            "fracture_energy": [],
            "total_energy": [],
            "solver_data": [],
            "cone_data": [],
            "cone-eig": [],
            "eigs": [],
            "uniqueness": [],
            "inertia": [],
            "F": [],
            "alphadot_norm": [],
            "rate_12_norm": [],
            "unscaled_rate_12_norm": [],
            "cone-stable": [],
        }

    def load_parameters(self, parameters_file):
        with open(parameters_file) as f:
            parameters = yaml.load(f, Loader=yaml.FullLoader)
        return parameters

    def generate_signature(self, parameters):
        signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()
        return signature

    def create_mesh_and_files(self):
        # Code to create mesh and output directories
        pass

    def create_function_spaces(self):
        # Code to create function spaces
        pass

    def create_functions(self):
        # Code to create functions
        pass

    def create_boundary_conditions(self):
        # Code to create boundary conditions
        pass

    def setup_energy_terms(self):
        # Code to setup energy terms
        pass

    def create_solvers(self):
        # Code to create solvers
        pass

    def run_load_steps(self):
        # Code to run load steps
        pass

    def save_results(self):
        # Code to save results
        pass

    def run_computation(self):
        # Wrapper to run the entire computation
        pass


# Example usage
parameters_file = "../test/parameters.yml"
comp = Computation(parameters_file)
comp.run_computation()
