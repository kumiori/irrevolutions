import ufl
from dolfinx.fem import assemble_scalar, form
import numpy as np
import mpi4py

def norm_L2(u):
    """
    Returns the L2 norm of the function u
    """
    comm = u.function_space.mesh.comm
    dx = ufl.Measure("dx", u.function_space.mesh)
    norm_form = form(ufl.inner(u, u) * dx)
    norm = np.sqrt(comm.allreduce(
        assemble_scalar(norm_form), op=mpi4py.MPI.SUM))
    return norm


def norm_H1(u):
    """
    Returns the H1 norm of the function u
    """
    comm = u.function_space.mesh.comm
    dx = ufl.Measure("dx", u.function_space.mesh)
    norm_form = form(
        (ufl.inner(u, u) + ufl.inner(ufl.grad(u), ufl.grad(u))) * dx)
    norm = np.sqrt(comm.allreduce(
        assemble_scalar(norm_form), op=mpi4py.MPI.SUM))
    return norm
