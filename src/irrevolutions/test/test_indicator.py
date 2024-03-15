import ufl
from petsc4py import PETSc
import dolfinx
from mpi4py import MPI
import numpy as np
import sys

sys.path.append("../")


def indicator_function(v):
    # Create the indicator function
    w = dolfinx.fem.Function(v.function_space)
    with w.vector.localForm() as w_loc, v.vector.localForm() as v_loc:
        w_loc[:] = np.where(v_loc[:] > 0, 1.0, 0.0)

    w.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    return w


if __name__ == "__main__":
    # Create a mesh and a function space
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)
    V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))

    # Create a function
    v = dolfinx.fem.Function(V)
    # v.interpolate(lambda x: np.sin(np.pi * x[0]))
    v.interpolate(lambda x: 0.5 - x[0])

    # Compute the indicator function
    w = indicator_function(v)
    dx = ufl.Measure("dx", mesh)

    _D = dolfinx.fem.form(w * dx)
    D = dolfinx.fem.assemble_scalar(_D)

    print(D)
    # Save the indicator function to a file
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "output/indicator.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(w)
