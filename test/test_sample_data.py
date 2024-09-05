import random
import sys

import dolfinx
import numpy as np
import ufl
from dolfinx.cpp.la.petsc import get_local_vectors, scatter_local_vectors
from irrevolutions.utils import _logger
from mpi4py import MPI
from petsc4py import PETSc
import basix.ufl

sys.path.append("../")


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def init_data(N, positive=True):
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, N - 1)
    comm = MPI.COMM_WORLD
    comm.Get_rank()

    element_u = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1)
    element_alpha = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1)

    V_u = dolfinx.fem.functionspace(mesh, element_u)
    V_alpha = dolfinx.fem.functionspace(mesh, element_alpha)
    u = dolfinx.fem.Function(V_u, name="Displacement")
    alpha = dolfinx.fem.Function(V_alpha, name="Damage")
    dx = ufl.Measure("dx", alpha.function_space.mesh)

    energy = (1 - alpha) ** 2 * ufl.inner(u, u) * dx

    F_ = [
        ufl.derivative(energy, u, ufl.TestFunction(u.ufl_function_space())),
        ufl.derivative(
            energy,
            alpha,
            ufl.TestFunction(alpha.ufl_function_space()),
        ),
    ]
    F = dolfinx.fem.form(F_)

    v = dolfinx.fem.petsc.create_vector_block(F)
    v.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    if positive:
        v.array = [
            np.around(random.uniform(0.1, 1.5), decimals=1) for r in range(v.local_size)
        ]
    else:
        v.array = [
            np.around(random.uniform(-1.5, 1.5), decimals=1)
            for r in range(v.local_size)
        ]

    maps = [(V.dofmap.index_map, V.dofmap.index_map_bs) for V in [V_u, V_alpha]]
    u, alpha = get_local_vectors(v, maps)
    # for visibility
    u *= 100
    scatter_local_vectors(v, [u, alpha], maps)
    v.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    return F, v


if __name__ == "__main__":
    F, v = init_data(10)
    _logger.info(f"F: {F}")
    _logger.info(f"v: {v}")
    v.view()
