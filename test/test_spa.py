import os
import sys
sys.path.append("../")
import solvers.restriction as restriction
import test_binarydataio
from test_extend import test_extend_vector
from test_cone_project import _cone_project_restricted
from utils import _logger
import dolfinx
import ufl
import numpy as np
from dolfinx.io import XDMFFile
import random

from petsc4py import PETSc
from mpi4py import MPI
import pickle 

def update_lambda_and_y(xk, Ar):
    # Update λ_t and y computing:
    # λ_k = <x_k, A x_k> / <x_k, x_k>
    # y_k = A x_k - λ_k x_k
    _Axr = xk.copy()
    _y = xk.copy()
    Ar.mult(xk, _Axr)
    
    xAx_r = xk.dot(_Axr)
    
    _logger.debug(f'xk view in update at iteration')
    
    _lmbda_t = xAx_r / xk.dot(xk)
    _y.waxpy(-_lmbda_t, xk, Axr)
    residual_norm = _y.norm()

    return _lmbda_t, _y


def update_xk(xk, y, s):
    # Update _xk based on the scaling and projection algorithm
    _xoldr = xk.copy()
    xk.copy(result=_xoldr)
    # x_k = x_k + (-s * y) 

    xk.axpy(-s, y)

    _cone_restricted = _cone_project_restricted(xk, F, constraints)
    n2 = _cone_restricted.normalize()

    return _cone_restricted
    

def load_minimal_constraints(filename):
    with open(filename, 'rb') as file:
        minimal_constraints = pickle.load(file)

def load_minimal_constraints(filename, spaces):
    with open(filename, 'rb') as file:
        minimal_constraints = pickle.load(file)

    # Assuming you have a constructor for your class
    # Modify this accordingly based on your actual class structure
    reconstructed_obj = restriction.Restriction(spaces, np.array([[], []]))
    for key, value in minimal_constraints.items():
        setattr(reconstructed_obj, key, value)

    return reconstructed_obj

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

_s = 1e-3

A = test_binarydataio.load_binary_matrix('data/solvers/A.mat')
Ar = test_binarydataio.load_binary_matrix('data/solvers/Ar.mat')
x0r = test_binarydataio.load_binary_vector('data/solvers/x0r.vec')
Axr = x0r.copy()

with XDMFFile(comm, "data/solvers/1d.xdmf", "r") as file: 
    mesh = file.read_mesh(name='mesh')

element_u = ufl.FiniteElement("Lagrange", mesh.ufl_cell(),
                            degree=1)
element_alpha = ufl.FiniteElement("Lagrange", mesh.ufl_cell(),
                                degree=1)

V_u = dolfinx.fem.FunctionSpace(mesh, element_u)
V_alpha = dolfinx.fem.FunctionSpace(mesh, element_alpha)
u = dolfinx.fem.Function(V_u, name="Displacement")
alpha = dolfinx.fem.Function(V_alpha, name="Damage")
dx = ufl.Measure("dx", alpha.function_space.mesh)

energy = (1-alpha)**2*ufl.inner(ufl.grad(u),ufl.grad(u)) * dx

F_ = [
    ufl.derivative(
        energy, u, ufl.TestFunction(u.ufl_function_space())
    ),
    ufl.derivative(
        energy,
        alpha,
        ufl.TestFunction(alpha.ufl_function_space()),
    ),
]
F = dolfinx.fem.form(F_)

# constraints = restriction.Restriction([V_u, V_alpha], np.array([[], []]))
constraints = load_minimal_constraints('data/solvers/constraints.pkl', [V_u, V_alpha])
A.assemble()

Ar.mult(x0r, Axr)

while True:
    lmbda_t, y = update_lambda_and_y(x0r, Ar)
    x0r = update_xk(x0r, y, _s)

__import__('pdb').set_trace()
_Ar = constraints.restrict_matrix(A)
