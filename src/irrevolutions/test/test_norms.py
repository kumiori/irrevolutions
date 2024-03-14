import os
import sys
sys.path.append("../")
sys.path.append("../playground/nb")
import irrevolutions.solvers.restriction as restriction
from algorithms.so import StabilitySolver
import test_binarydataio as bio
from test_extend import test_extend_vector
from test_cone_project import _cone_project_restricted
from test_spa import load_minimal_constraints
from irrevolutions.utils import _logger
import dolfinx
import ufl
import numpy as np
from dolfinx.io import XDMFFile
import random

from petsc4py import PETSc
from mpi4py import MPI
import pickle 
import logging

import eigenspace

from dolfinx.fem import Function, functionspace, assemble_scalar, form

from dolfinx.mesh import (CellType, compute_midpoints, create_unit_cube,
                          create_unit_interval, meshtags)

def test_norms(N):

    mesh = create_unit_interval(MPI.COMM_WORLD, N)
    V = functionspace(mesh, ("CG", 1))
    v = Function(V, dtype=np.float64)
    dx = ufl.Measure("dx", domain=mesh) #-> volume measure
    
    logging.critical("""
        typedef enum {
            NORM_1         = 0,
            NORM_2         = 1,
            NORM_FROBENIUS = 2,
            NORM_INFINITY  = 3,
            NORM_1_AND_2   = 4
            } NormType;
    """)

    test_functions = {
        "sin(pi*x)": lambda x: np.sin(np.pi * x[0]),
        "sin(x)": lambda x: np.sin(x[0]),
        "constant 1": lambda x: np.ones_like(x[0]),
        "cos(x)": lambda x: np.cos(x[0]),
        "exp(x)": lambda x: np.exp(x[0]),
        "1/(x-x0), x0=-0.5": lambda x: 1 / (x[0]+0.5),
    }
    
    for description, function in test_functions.items():
        v = dolfinx.fem.Function(dolfinx.fem.FunctionSpace(mesh, ("CG", 1)))
        v.interpolate(function)
        print_norms(v, description)
        scaling = compute_scaling_factor(v, function, mesh)
        logging.critical(f"Scaling factor: {scaling}\n")
        
def compute_scaling_factor(u, f, mesh):
    dx = ufl.Measure("dx", domain=mesh)
    analytic_norm = np.sqrt(dolfinx.fem.assemble_scalar(form(ufl.inner(u, u) * dx)))

    vector_norm = u.vector.norm(PETSc.NormType.NORM_2)
    
    scaling_factor = analytic_norm / vector_norm

    return scaling_factor
  
def print_norms(u, description):
    mesh = u.function_space.mesh
    dx = ufl.Measure("dx", domain=mesh)
    function_norm = np.sqrt(dolfinx.fem.assemble_scalar(form(ufl.inner(u, u) * dx)))
    vector_norm = u.vector.norm(PETSc.NormType.NORM_2)

    comm = MPI.COMM_WORLD
    function_norm = comm.allreduce(function_norm, op=MPI.SUM)

    logging.critical(description)
    logging.critical(f"Function Norm ||u_h||_L^2: {function_norm}")
    logging.critical(f"Vector Norm   ||u_i||_l^2: {vector_norm}\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test norms.')
    parser.add_argument("-N", help="The number of dofs.", type=int, default=10)
    args = parser.parse_args()

    test_norms(N=args.N)