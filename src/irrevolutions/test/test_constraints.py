# mixin for testing constraints on the 2d pizza example
# just to draw from the class

import sys
sys.path.append("../")
from algorithms.so import BifurcationSolver
from irrevolutions.utils import _logger

from mpi4py import MPI
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

