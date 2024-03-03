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

import test_binarydataio
from test_extend import test_extend_vector

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from test_restriction import (
    __log_incipit,
    get_inactive_dofset,
)

from dolfinx.cpp.la.petsc import get_local_vectors, scatter_local_vectors
from test_sample_data import init_data


def _cone_project_restricted(v, _x, constraints):
    """
    Projects a vector into the relevant cone, handling restrictions. Not in place.

    Args:
        v: Vector to be projected.
        _x: A full vector.

    Returns:
        Vector: The projected vector.
    """
    with dolfinx.common.Timer(f"~Second Order: Cone Project"):
        maps = [
            (V.dofmap.index_map, V.dofmap.index_map_bs)
            for V in constraints.function_spaces
        ]
        # _x = dolfinx.fem.petsc.create_vector_block(F)

        test_extend_vector(v, _x, constraints)

        # _logger.critical(f"rank {rank} viewing _x")
        # _x.view()

        with _x.localForm() as x_local:
            _dofs = constraints.bglobal_dofs_vec[1]
            x_local.array[_dofs] = np.maximum(x_local.array[_dofs], 0)

            _logger.debug(f"Local dofs: {_dofs}")
            _logger.debug(f"x_local")
            _logger.debug(f"x_local truncated")

        _x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        # x_u, x_alpha = get_local_vectors(_x, maps)

        # _logger.info(f"Cone Project: Local data of the subvector x_u: {x_u}")
        # _logger.info(f"Cone Project: Local data of the subvector x_alpha: {x_alpha}")

        x = constraints.restrict_vector(_x)

        # _x.copy(result=x)
        # _x.destroy()

    return x


if __name__ == "__main__":
    full_matrix = test_binarydataio.load_binary_matrix("data/solver/A.mat")
    matrix = test_binarydataio.load_binary_matrix("data/solver/Ar.mat")
    guess = test_binarydataio.load_binary_vector("data/solver/x0r.vec")

    F, v = init_data(10, positive=False)
    V_u, V_alpha = F[0].function_spaces[0], F[1].function_spaces[0]
    _x = dolfinx.fem.petsc.create_vector_block(F)
    x = dolfinx.fem.petsc.create_vector_block(F)
    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    restricted_dofs = get_inactive_dofset(v, F)

    constraints = restriction.Restriction([V_u, V_alpha], restricted_dofs)

    vr = constraints.restrict_vector(v)
    # x = test_cone_project_restricted(vr, constraints, x)
    x = _cone_project_restricted(vr, _x, constraints)

    # _logger.info(f"The vr vector")
    # vr.view()

    # _logger.info(f"The x vector")
    # x.view()
