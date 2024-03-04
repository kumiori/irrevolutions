import logging
from pydoc import cli
from time import clock_settime
from utils import norm_H1, norm_L2

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
import random


comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# create a class naled linesearch


class LineSearch(object):
    def __init__(self, energy, state, linesearch_parameters={}):
        super(LineSearch, self).__init__()
        # self.u_0 = dolfin.Vector(state['u'].vector())
        # self.alpha_0 = dolfin.Vector(state['alpha'].vector())

        self.energy = energy
        self.state = state
        # initial state
        self.parameters = linesearch_parameters

        self.V_u = state["u"].function_space
        self.V_alpha = state["alpha"].function_space
        self.mesh = self.V_u.mesh

        self.u0 = Function(state["u"].function_space)
        self.alpha0 = Function(state["alpha"].function_space)

    def search(self, state, perturbation, interval, m=2, method="min"):
        # m = self.parameters["order"]

        v = perturbation["v"]
        beta = perturbation["beta"]

        u_0 = Function(state["u"].function_space)
        alpha_0 = Function(state["alpha"].function_space)

        state["u"].vector.copy(u_0.vector)
        state["alpha"].vector.copy(alpha_0.vector)

        en_0 = assemble_scalar(form(self.energy))

        # get admissible interval
        hmin, hmax = interval

        # discretise interval for polynomial interpolation at order m
        htest = np.linspace(hmin, hmax, np.int32(m + 1))
        energies_1d = []
        perturbation_norms = []

        # compute energy at discretised points

        for h in htest:
            with state["u"].vector.localForm() as u_local, state[
                "alpha"
            ].vector.localForm() as alpha_local, alpha_0.vector.localForm() as alpha0_local, u_0.vector.localForm() as u0_local, v.vector.localForm() as v_local, beta.vector.localForm() as beta_local:
                u_local.array[:] = u0_local.array[:] + h * v_local.array[:]
                alpha_local.array[:] = alpha0_local.array[:] + h * beta_local.array[:]

            state["u"].vector.ghostUpdate(
                addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
            )
            state["alpha"].vector.ghostUpdate(
                addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
            )

            yh_norm = np.sum([norm_H1(func) for func in state.values()])
            perturbation_norms.append(yh_norm)

            en_h = assemble_scalar(form(self.energy))
            energies_1d.append(en_h - en_0)

        # restore state
        u_0.vector.copy(state["u"].vector)
        alpha_0.vector.copy(state["alpha"].vector)

        # compute polynomial coefficients

        z = np.polyfit(htest, energies_1d, m)
        p = np.poly1d(z)

        # compute minimum of polynomial
        if m == 2:
            logging.info("Line search using quadratic interpolation")
            h_opt = -z[1] / (2 * z[0])
        else:
            logging.info(
                "Line search using polynomial interpolation (order {})".format(m)
            )
            h = np.linspace(0, hmax, 30)
            h_opt = h[np.argmin(p(h))]

        if method == "random":
            h_opt = random.uniform(hmin, hmax)

        return h_opt, energies_1d, p, z

    def perturb(self, state, perturbation, h):
        v = perturbation["v"]
        beta = perturbation["beta"]

        z0_norm = np.sum([norm_H1(func) for func in state.values()])

        with state["u"].vector.localForm() as u_local, state[
            "alpha"
        ].vector.localForm() as alpha_local, v.vector.localForm() as v_local, beta.vector.localForm() as beta_local:
            u_local.array[:] = u_local.array[:] + h * v_local.array[:]
            alpha_local.array[:] = alpha_local.array[:] + h * beta_local.array[:]

        state["u"].vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
        )
        state["alpha"].vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
        )

        zh_norm = np.sum([norm_H1(func) for func in state.values()])

        logging.critical(f"Initial state norm: {z0_norm}")
        logging.critical(f"Perturbation norm: {zh_norm}")

        return state

    def admissible_interval(self, state, perturbation, alpha_lb, bifurcation):
        """Computes the admissible interval for the line search, based on
        the solution to the rate problem"""

        alpha = state["alpha"]
        # beta = perturbation["beta"]
        beta = bifurcation[1]

        one = max(1.0, max(alpha.vector[:]))
        upperbound = one
        lowerbound = alpha_lb

        # positive
        mask = np.int32(np.where(beta.vector[:] > 0)[0])

        hp2 = (
            (one - alpha.vector[mask]) / beta.vector[mask]
            if len(mask) > 0
            else [np.inf]
        )
        hp1 = (
            (alpha_lb.vector[mask] - alpha.vector[mask]) / beta.vector[mask]
            if len(mask) > 0
            else [-np.inf]
        )
        hp = (max(hp1), min(hp2))

        # negative
        mask = np.int32(np.where(beta.vector[:] < 0)[0])

        hn2 = (
            (one - alpha.vector[mask]) / beta.vector[mask]
            if len(mask) > 0
            else [-np.inf]
        )
        hn1 = (
            (alpha_lb.vector[mask] - alpha.vector[mask]) / beta.vector[mask]
            if len(mask) > 0
            else [np.inf]
        )
        hn = (max(hn2), min(hn1))

        hmax = np.array(np.min([hp[1], hn[1]]))
        hmin = np.array(np.max([hp[0], hn[0]]))

        hmax_glob = np.array(0.0, "d")
        hmin_glob = np.array(0.0, "d")

        comm.Allreduce(hmax, hmax_glob, op=mpi4py.MPI.MIN)
        comm.Allreduce(hmin, hmin_glob, op=mpi4py.MPI.MAX)

        hmax = float(hmax_glob)
        hmin = float(hmin_glob)

        if hmin > 0:
            log(LogLevel.INFO, "Line search troubles: found hmin>0")
            return (0.0, 0.0)
        if hmax == 0 and hmin == 0:
            log(LogLevel.INFO, "Line search failed: found zero step size")
            # import pdb; pdb.set_trace()
            return (0.0, 0.0)
        if hmax < hmin:
            log(LogLevel.INFO, "Line search failed: optimal h* not admissible")
            return (0.0, 0.0)
            # get next perturbation mode

        assert hmax > hmin, "hmax > hmin"

        return (hmin, hmax)

    def get_unilateral_interval(self, state, perturbation):
        """Computes the admissible interval for the line search, based on
        the positive perturbation solution to the cone-problem. This is a unilateral interval,
        the upper bound given by the condition
            h: alpha + h*perturbation <= 1."""

        alpha = state["alpha"]
        beta = perturbation["beta"]
        assert (beta.vector[:] >= 0).all(), "beta non-negative"

        one = max(1.0, max(alpha.vector[:]))
        mask = np.int32(np.where(beta.vector[:] > 0)[0])

        _hmax = (
            (one - alpha.vector[mask]) / beta.vector[mask]
            if len(mask) > 0
            else [np.inf]
        )
        hmax_glob = np.array(0.0, "d")

        comm.Allreduce(np.min(_hmax), hmax_glob, op=mpi4py.MPI.MIN)

        hmax = float(hmax_glob)
        assert hmax > 0, "hmax > 0"

        return (0, hmax)
