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

from test_restriction import (
    __log_incipit,
    get_inactive_dofset,
)

from dolfinx.cpp.la.petsc import get_local_vectors, scatter_local_vectors
from test_sample_data import init_data  
from test_extend import test_extension

def test_cone_project_restricted(v, constraints, x):
    V_u, V_alpha = constraints.function_spaces[0], constraints.function_spaces[1]
    maps = [(V.dofmap.index_map, V.dofmap.index_map_bs) for V in [V_u, V_alpha]]
    # x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    
    test_extension(v, x, constraints)
    _logger.info(f"v extended into x")
    x.view()

    with x.localForm() as x_local:
        _dofs = constraints.bglobal_dofs_vec[1]
        _logger.debug(f"x_local")
        x_local.view()
        _logger.debug(f"{__log_incipit} Local dofs: {_dofs}")
        x_local.array[_dofs] = np.maximum(x_local.array[_dofs], 0)
        _logger.debug(f"x_local projected")
        x_local.view()

    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    x_u, x_alpha = get_local_vectors(x, maps)
    comm.barrier()
    _logger.info(f"The local vectors")
    _logger.critical(f"{__log_incipit} Local data of the subvector x_u: {x_u}")
    _logger.critical(f"{__log_incipit} Local data of the subvector x_alpha: {x_alpha}")
    comm.barrier()

    constraints.restrict_vector(x)
    
    return x
     
if __name__ == "__main__":
    F, v = init_data(10, positive=False)
    V_u, V_alpha = F[0].function_spaces[0], F[1].function_spaces[0]
    x = dolfinx.fem.petsc.create_vector_block(F)
    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    
    restricted_dofs = get_inactive_dofset(v, F)
    
    constraints = restriction.Restriction([V_u, V_alpha], restricted_dofs)

    # _logger.critical(f"{__log_incipit} constraints.blocal_dofs {constraints.blocal_dofs}")
    # _logger.critical(f"{__log_incipit} constraints.bglobal_dofs_vec {constraints.bglobal_dofs_vec}")
    # _logger.critical(f"{__log_incipit} constraints.bglobal_dofs_vec_stacked {constraints.bglobal_dofs_vec_stacked}")
    
    vr = constraints.restrict_vector(v)
    x = test_cone_project_restricted(vr, constraints, x)
    
    _logger.info(f"The vr vector")
    vr.view()
    
    _logger.info(f"The x vector")
    x.view()
    