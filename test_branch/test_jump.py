from irrevolutions.algorithms.gf import JumpSolver

from dolfinx import fem, nls, la
from petsc4py import PETSc
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx.mesh import create_unit_square
from dolfinx.fem import (
    functionspace,
    Function,
    dirichletbc,
    locate_dofs_topological,
    Constant,
)
import basix

from dolfinx.fem.petsc import LinearProblem

if __name__ == "__main__":
    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
    element_u = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1, shape=(2,))
    element_alpha = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1)
    V_u = functionspace(mesh, element_u)
    V_alpha = functionspace(mesh, element_alpha)

    u = Function(V_u, name="u")
    alpha = Function(V_alpha, name="alpha")
    alpha.x.array[:] = 0.1  # initial value

    beta = Function(V_alpha, name="beta")
    beta.x.array[:] = 1.0  # perturbation direction

    # Simple energy functional: just a penalty on alpha
    def energy_form(u, alpha):
        return ufl.inner(alpha, alpha) * ufl.dx

    state = {"u": u, "alpha": alpha}
    perturbation = {"v": u, "beta": beta}

    bcs = []  # no boundary conditions
    jump_parameters = {"tau": 1e-2, "max_steps": 100, "rtol": 1e-8, "verbose": True}

    flow = JumpSolver(energy_form, state, bcs, jump_parameters)
    state = flow.solve(perturbation, h=0.1)

    print(
        "Final alpha min/max:",
        state["alpha"].x.array.min(),
        state["alpha"].x.array.max(),
    )
