import sys

import dolfinx
import irrevolutions.solvers.restriction as restriction
import numpy as np
from irrevolutions.utils import (
    _logger,
    load_binary_matrix,
    load_binary_vector,
    sample_data,
)
from mpi4py import MPI
from petsc4py import PETSc
from test_restriction import (
    get_inactive_dofset,
)

sys.path.append("../")


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def extend_vector(vres, vext, constraints):
    """
    Extends a restricted vector vr into v, not in place.

    Args:
        vres: Restricted vector to be extended.
        vext: Extended vector.

    Returns:
        None
    """

    vext.zeroEntries()

    vext.array[constraints.bglobal_dofs_vec_stacked] = vres.array

    return vext


def _cone_project_restricted(v, _x, constraints):
    """
    Projects a vector into the relevant cone, handling restrictions. Not in place.

    Args:
        v: Vector to be projected.
        _x: A full vector.

    Returns:
        Vector: The projected vector.
    """
    with dolfinx.common.Timer("~Second Order: Cone Project"):
        [
            (V.dofmap.index_map, V.dofmap.index_map_bs)
            for V in constraints.function_spaces
        ]
        # _x = dolfinx.fem.petsc.create_vector_block(F)

        extend_vector(v, _x, constraints)

        # _logger.critical(f"rank {rank} viewing _x")
        # _x.view()

        with _x.localForm() as x_local:
            _dofs = constraints.bglobal_dofs_vec[1]
            x_local.array[_dofs] = np.maximum(x_local.array[_dofs], 0)

            _logger.debug(f"Local dofs: {_dofs}")
            _logger.debug("x_local")
            _logger.debug("x_local truncated")

        _x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        # x_u, x_alpha = get_local_vectors(_x, maps)

        # _logger.info(f"Cone Project: Local data of the subvector x_u: {x_u}")
        # _logger.info(f"Cone Project: Local data of the subvector x_alpha: {x_alpha}")

        x = constraints.restrict_vector(_x)

        # _x.copy(result=x)
        # _x.destroy()

    return x


def test_cone_project():
    full_matrix = load_binary_matrix("data/solver/A.mat")
    matrix = load_binary_matrix("data/solver/Ar.mat")
    guess = load_binary_vector("data/solver/x0r.vec")

    F, v = sample_data(10, positive=False)
    V_u, V_alpha = F[0].function_spaces[0], F[1].function_spaces[0]
    _x = dolfinx.fem.petsc.create_vector_block(F)
    x = dolfinx.fem.petsc.create_vector_block(F)
    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    restricted_dofs = get_inactive_dofset(v, F)

    constraints = restriction.Restriction([V_u, V_alpha], restricted_dofs)

    vr = constraints.restrict_vector(v)
    # x = test_cone_project_restricted(vr, constraints, x)
    x = _cone_project_restricted(vr, _x, constraints)


if __name__ == "__main__":
    test_cone_project()

    # _logger.info(f"The vr vector")
    # vr.view()

    # _logger.info(f"The x vector")
    # x.view()
