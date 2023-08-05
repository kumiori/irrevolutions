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

    
    def admissible_interval(self, state, perturbation, alpha_lb, bifurcation):
        alpha = state["alpha"]
        # beta = perturbation["beta"]
        beta = bifurcation[1]

        one = max(1., max(alpha.vector[:]))
        upperbound = one
        lowerbound = alpha_lb


        # positive
        mask = np.int32(np.where(beta.vector[:]>0)[0])

        hp2 = (one-alpha.vector[mask])/beta.vector[mask]  if len(mask)>0 else [np.inf]
        hp1 = (alpha_lb.vector[mask]-alpha.vector[mask])/beta.vector[mask]  if len(mask)>0 else [-np.inf]
        hp = (max(hp1), min(hp2))

        # negative
        mask = np.int32(np.where(beta.vector[:]<0)[0])

        hn2 = (one-alpha.vector[mask])/beta.vector[mask] if len(mask)>0 else [-np.inf]
        hn1 = (alpha_lb.vector[mask]-alpha.vector[mask])/beta.vector[mask]  if len(mask)>0 else [np.inf]
        hn = (max(hn2), min(hn1))

        hmax = np.array(np.min([hp[1], hn[1]]))
        hmin = np.array(np.max([hp[0], hn[0]]))

        hmax_glob = np.array(0.0,'d')
        hmin_glob = np.array(0.0,'d')

        comm.Allreduce(hmax, hmax_glob, op=mpi4py.MPI.MIN)
        comm.Allreduce(hmin, hmin_glob, op=mpi4py.MPI.MAX)

        hmax = float(hmax_glob)
        hmin = float(hmin_glob)


        if hmin>0:
            log(LogLevel.INFO, 'Line search troubles: found hmin>0')
            return (0., 0.)
        if hmax==0 and hmin==0:
            log(LogLevel.INFO, 'Line search failed: found zero step size')
            # import pdb; pdb.set_trace()
            return (0., 0.)
        if hmax < hmin:
            log(LogLevel.INFO, 'Line search failed: optimal h* not admissible')
            return (0., 0.)
            # get next perturbation mode

        assert hmax > hmin, 'hmax > hmin'
        __import__('pdb').set_trace()
        
        return (hmin, hmax)
