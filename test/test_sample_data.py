import os
import sys
sys.path.append("../")
import solvers.restriction as restriction
from utils import _logger
import dolfinx
import ufl
import numpy as np
import random

from petsc4py import PETSc
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from dolfinx.cpp.la.petsc import get_local_vectors, scatter_local_vectors

def init_data(N):
    _N = 3

    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, N)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    element_u = ufl.FiniteElement("Lagrange", mesh.ufl_cell(),
                                degree=1)
    element_alpha = ufl.FiniteElement("Lagrange", mesh.ufl_cell(),
                                    degree=1)

    V_u = dolfinx.fem.FunctionSpace(mesh, element_u)
    V_alpha = dolfinx.fem.FunctionSpace(mesh, element_alpha)
    u = dolfinx.fem.Function(V_u, name="Displacement")
    alpha = dolfinx.fem.Function(V_alpha, name="Damage")
    dx = ufl.Measure("dx", alpha.function_space.mesh)

    energy = (1-alpha)**2*ufl.inner(u,u) * dx

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

    v = dolfinx.fem.petsc.create_vector_block(F)
    v.array = [np.around(random.uniform(0.1, 1.5), decimals=1) for r in range(v.local_size)]

    maps = [(V.dofmap.index_map, V.dofmap.index_map_bs) for V in [V_u, V_alpha]]
    u, alpha = get_local_vectors(v, maps)
    # for visibility
    u *= 10
    scatter_local_vectors(v, [u, alpha], maps)
    v.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    return F, v


if __name__ == "__main__":
    F, v = init_data(10)
    _logger.info(f"F: {F}")
    _logger.info(f"v: {v}")
    v.view()
    