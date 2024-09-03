from logging import getLevelName

import dolfinx
from dolfinx.cpp.la.petsc import get_local_vectors
from mpi4py import MPI
from test_restriction import __log_incipit, get_inactive_dofset

import irrevolutions.solvers.restriction as restriction
from irrevolutions.utils import _logger, sample_data

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def test_extend_vector():
    F, v = sample_data(10)
    V_u, V_alpha = F[0].function_spaces[0], F[1].function_spaces[0]
    x = dolfinx.fem.petsc.create_vector_block(F)

    restricted_dofs = get_inactive_dofset(v, F)

    constraints = restriction.Restriction([V_u, V_alpha], restricted_dofs)

    _logger.debug(f"{__log_incipit} constraints.blocal_dofs {constraints.blocal_dofs}")
    _logger.debug(
        f"{__log_incipit} constraints.bglobal_dofs_vec {constraints.bglobal_dofs_vec}"
    )
    _logger.debug(
        f"{__log_incipit} constraints.bglobal_dofs_vec_stacked {constraints.bglobal_dofs_vec_stacked}"
    )

    vr = constraints.restrict_vector(v)
    # test_extend_vector(vr, x, constraints)

    V_u, V_alpha = constraints.function_spaces[0], constraints.function_spaces[1]
    maps = [(V.dofmap.index_map, V.dofmap.index_map_bs) for V in [V_u, V_alpha]]

    if getLevelName(_logger.getEffectiveLevel()) == "DEBUG":
        _logger.debug("x")
        x.view()

        _logger.debug("vr")
        vr.view()

    _logger.info("Setting up dofs for extension")
    _logger.debug(
        f'{__log_incipit} The "good" dofs: constraints.bglobal_dofs_vec_stacked {constraints.bglobal_dofs_vec_stacked}'
    )

    x.zeroEntries()
    x.array[constraints.bglobal_dofs_vec_stacked] = vr.array

    _logger.debug(f"{__log_incipit} Local data of the x: {x.array}")

    if getLevelName(_logger.getEffectiveLevel()) == "DEBUG":
        _logger.debug("x")
        x.view()

    x_u, x_alpha = get_local_vectors(x, maps)

    _logger.info("The local vectors")
    _logger.debug(f"{__log_incipit} Local data of the subvector x_u: {x_u}")
    _logger.debug(f"{__log_incipit} Local data of the subvector x_alpha: {x_alpha}")
