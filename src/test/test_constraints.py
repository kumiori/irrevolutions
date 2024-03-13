# mixin for testing constraints on the 2d pizza example
# just to draw from the class

import os
import sys
sys.path.append("../")
import solvers.restriction as restriction
from algorithms.so import BifurcationSolver
import test_binarydataio as bio
from test_extend import test_extend_vector
from test_cone_project import _cone_project_restricted
from test_spa import load_minimal_constraints
from utils import _logger
import dolfinx
import ufl
import numpy as np
from dolfinx.io import XDMFFile
import random

from petsc4py import PETSc
from mpi4py import MPI
import pickle 
import logging

_logger.setLevel(logging.CRITICAL)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class StabilitySolverTester(BifurcationSolver):
    def __init__(self, errors, Ar, xk, constraints, F):

        self.V_u
        self.V_alpha

    def run_constraints_test(self):
        return self.setup_constraints(self.alpha_old)

