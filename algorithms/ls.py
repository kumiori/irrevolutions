import logging
from pydoc import cli
from time import clock_settime

import dolfinx
from dolfinx.fem import (
    Constant,
    Function,
    FunctionSpace,
    dirichletbc,
    form,
    assemble_scalar,
    locate_dofs_geometrical,
)
from petsc4py import PETSc
from slepc4py import SLEPc
from dolfinx.cpp.log import log, LogLevel
import ufl
import numpy as np
from pathlib import Path

import mpi4py
import numpy as np
from ufl import Measure


comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# create a class naled linesearch

class LineSearch(object):
    def __init__(self, energy, state):
        super(LineSearch, self).__init__()
        # self.u_0 = dolfin.Vector(state['u'].vector())
        # self.alpha_0 = dolfin.Vector(state['alpha'].vector())

        self.energy = energy
        self.state = state
        # initial state

        self.V_u = state["u"].function_space
        self.V_alpha = state["alpha"].function_space
        self.mesh = self.V_u.mesh

        self.u0 = Function(state['u'].function_space)
        self.alpha0 = Function(state['alpha'].function_space)


    def search(self, state, perturbation, m=3, mode=0):
        dx = Measure("dx", domain=self.mesh) #-> volume measure

        __import__('pdb').set_trace()

        en_0 = assemble_scalar(form(self.energy))

        # get admissible interval

        # discretise interval for polynomial interpolation at order m

        # compute energy at discretised points

        # compute polynomial coefficients

        # compute minimum of polynomial

        # return arg-minimum of 1d energy-perturbations