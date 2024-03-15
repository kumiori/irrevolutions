import os

from irrevolutions.algorithms.so import StabilitySolver

from . import test_binarydataio as bio
from .test_spa import load_minimal_constraints
from irrevolutions.utils import _logger
import dolfinx
import ufl
import numpy as np
from dolfinx.io import XDMFFile

from mpi4py import MPI
import logging

_logger.setLevel(logging.CRITICAL)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class StabilitySolverTester(StabilitySolver):
    def __init__(self, errors, Ar, xk, constraints, F):
        self.errors = errors
        self.Ar = Ar
        self.xk = xk
        self.F = F
        self._v = dolfinx.fem.petsc.create_vector_block(self.F)
        self.constraints = constraints
        self.V_u = self.constraints.function_spaces[0]
        self.V_alpha = self.constraints.function_spaces[1]
        self._xoldr = xk.duplicate()
        self.parameters = { "cone":
                                {"cone_atol": 1.0e-06,
                                "cone_max_it": 300000,
                                "cone_rtol": 1.0e-06,
                                "maxmodes": 3,
                                "scaling": 1.e-3}
                                }
        self._reasons = {'0': 'converged',
                    '-1': 'non-converged, check the logs',
                    '1': 'converged atol',
                    '2': 'converged residual'
                    }
        self.iterations = 0
        self._aerrors = []
        self._residual_norm = 1.
        self.data = {
            "error_x_L2": [],
            "lambda_k": [],
            "y_norm_L2": [],
        }
        self._converged = False
        
    def run_convergence_test(self):
        return self.convergence_loop(self.errors, self.Ar, self.xk)


with XDMFFile(comm, os.path.join(os.path.dirname(__file__), "data/input_data.xdmf"), "r") as file: 
    mesh = file.read_mesh(name='mesh')

element_u = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)
element_alpha = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)

V_u = dolfinx.fem.FunctionSpace(mesh, element_u)
V_alpha = dolfinx.fem.FunctionSpace(mesh, element_alpha)
u = dolfinx.fem.Function(V_u, name="Displacement")
alpha = dolfinx.fem.Function(V_alpha, name="Damage")
dx = ufl.Measure("dx", alpha.function_space.mesh)

state = {"u": u, "alpha": alpha}

F_ = [
    ufl.derivative(
        (1-alpha)**2. * ufl.inner(ufl.grad(u), ufl.grad(u))* dx , u, ufl.TestFunction(
            u.ufl_function_space())
    ),
    ufl.derivative(
        (alpha + ufl.dot(ufl.grad(alpha), ufl.grad(alpha))) * dx,
        alpha,
        ufl.TestFunction(alpha.ufl_function_space()),
    ),
]
F = dolfinx.fem.form(F_)

constraints = load_minimal_constraints(os.path.join(os.path.dirname(__file__), 'data/constraints.pkl'), [V_u, V_alpha])
A = bio.load_binary_matrix(os.path.join(os.path.dirname(__file__), 'data/A_hessian.mat'))
Ar = bio.load_binary_matrix(os.path.join(os.path.dirname(__file__), 'data/Ar_hessian.mat'))
x0 = bio.load_binary_vector(os.path.join(os.path.dirname(__file__), 'data/x0.vec'))

# zero vector, compatible with the linear system
_x = x0.duplicate()
errors = []

tester = StabilitySolverTester(errors, Ar, x0, constraints, F)
_y, _xk, _lmbda_k = tester.run_convergence_test()

tester.store_results(_lmbda_k, _xk, _y)

atol = tester.parameters["cone"]["cone_atol"]

assert tester._isin_cone(_xk) == True
assert np.isclose(_lmbda_k, -0.044659195907104675, atol=1e-4) == True
