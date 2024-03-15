
import dolfinx
from logging import getLevelName
from mpi4py import MPI


import irrevolutions.solvers.restriction as restriction
from irrevolutions.utils import _logger

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from .test_restriction import (
    __log_incipit,
    get_inactive_dofset,
)

from dolfinx.cpp.la.petsc import get_local_vectors
from .test_sample_data import init_data  

def test_extend_vector():
    F, v = init_data(10)
    V_u, V_alpha = F[0].function_spaces[0], F[1].function_spaces[0]
    x = dolfinx.fem.petsc.create_vector_block(F)
    
    restricted_dofs = get_inactive_dofset(v, F)
    
    constraints = restriction.Restriction([V_u, V_alpha], restricted_dofs)

    _logger.debug(f"{__log_incipit} constraints.blocal_dofs {constraints.blocal_dofs}")
    _logger.debug(f"{__log_incipit} constraints.bglobal_dofs_vec {constraints.bglobal_dofs_vec}")
    _logger.debug(f"{__log_incipit} constraints.bglobal_dofs_vec_stacked {constraints.bglobal_dofs_vec_stacked}")
    
    vr = constraints.restrict_vector(v)
    # test_extend_vector(vr, x, constraints)
    
    V_u, V_alpha = constraints.function_spaces[0], constraints.function_spaces[1]
    maps = [(V.dofmap.index_map, V.dofmap.index_map_bs) for V in [V_u, V_alpha]]

    if getLevelName(_logger.getEffectiveLevel()) == 'DEBUG':
        _logger.debug(f"x")
        x.view()

        _logger.debug(f"vr")
        vr.view()
    
    _logger.info(f"Setting up dofs for extension")
    _logger.debug(f"{__log_incipit} The \"good\" dofs: constraints.bglobal_dofs_vec_stacked {constraints.bglobal_dofs_vec_stacked}")

    x.zeroEntries()
    x.array[constraints.bglobal_dofs_vec_stacked] = vr.array

    _logger.debug(f"{__log_incipit} Local data of the x: {x.array}")
    
    if getLevelName(_logger.getEffectiveLevel()) == 'DEBUG':
        _logger.debug(f"x")
        x.view()

    x_u, x_alpha = get_local_vectors(x, maps)

    _logger.info(f"The local vectors")
    _logger.debug(f"{__log_incipit} Local data of the subvector x_u: {x_u}")
    _logger.debug(f"{__log_incipit} Local data of the subvector x_alpha: {x_alpha}")
