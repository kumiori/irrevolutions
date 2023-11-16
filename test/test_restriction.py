from test_scatter_MPI import (
    mesh,
    element_alpha,
    element_u,
    V_alpha,
    V_u,
    alpha,
    u,
    dofs_alpha_left,
    dofs_alpha_right,
    dofs_u_right,
    dofs_u_left,
    energy)

import os
import sys
sys.path.append("../")
import solvers.restriction as restriction
from utils import _logger
import dolfinx
import ufl
import numpy as np
import random
from test_sample_data import init_data  

from petsc4py import PETSc
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
__log_incipit = f"rank {rank}#{size}/"

from dolfinx.cpp.la.petsc import get_local_vectors, scatter_local_vectors

def get_inactive_dofset(v, V_u = V_u, V_alpha = V_alpha):
    """docstring for get_inactive_dofset"""
    _logger.info(f"inactive dofset")

    V_u_size = V_u.dofmap.index_map_bs * (V_u.dofmap.index_map.size_local)
    V_alpha_size = V_alpha.dofmap.index_map_bs * (V_alpha.dofmap.index_map.size_local)
    maps = [(V.dofmap.index_map, V.dofmap.index_map_bs) for V in [V_u, V_alpha]]
    u, alpha = get_local_vectors(v, maps)
    
    # simil to: admissibility
    idx_alpha_local = np.array(np.where(alpha<=1)[0], dtype=np.int32)
    idx_u_local = np.arange(V_u_size, dtype=np.int32)
    dofs_u_all = idx_u_local

    # Access the local data
    local_data = v.array
    
    restricted_dofs = [dofs_u_all, idx_alpha_local]

    # Print information about the vector
    _logger.critical(f"{__log_incipit} Len of subvector u: {len(u)}")
    _logger.critical(f"{__log_incipit} Len of subvector alpha: {len(alpha)}")
    _logger.critical(f"{__log_incipit} Local data of the subvector u: {u}")
    _logger.critical(f"{__log_incipit} Local data of the subvector alpha: {alpha}")
    _logger.critical(f"{__log_incipit} restricted_dofs: {restricted_dofs}")
    _logger.critical(f"{__log_incipit} idx_alpha_local: {idx_alpha_local}")
    _logger.critical(f"{__log_incipit} idx_u_local: {idx_u_local}")
    _logger.critical(f"{__log_incipit} Size of the vector: {v.getSize()}")
    _logger.critical(f"{__log_incipit} Local data of the vector: {local_data}")
    _logger.critical(f"{__log_incipit} Nonzero entries in the local data: {len(local_data.nonzero()[0])}")
    _logger.critical(f"{__log_incipit} Global indices of nonzero entries: {v.getOwnershipRange()}")
    _logger.critical(f"{__log_incipit} Global indices of nonzero entries: {v.getOwnershipRanges()}")
    
    return restricted_dofs
    
def test_restriction():
    F, v = init_data(3)

    x = dolfinx.fem.petsc.create_vector_block(F)
    
    restricted_dofs = get_inactive_dofset(v)

    constraints = restriction.Restriction([V_u, V_alpha], restricted_dofs)
    vr = constraints.restrict_vector(v)

    comm.Barrier()
    _logger.critical(f"{__log_incipit} constraints.blocal_dofs {constraints.blocal_dofs}")
    _logger.critical(f"{__log_incipit} constraints.bglobal_dofs_vec {constraints.bglobal_dofs_vec}")
    _logger.critical(f"{__log_incipit} constraints.bglobal_dofs_vec_stacked {constraints.bglobal_dofs_vec_stacked}")
    
    _logger.info(f"v")
    v.view()
    _logger.info(f"vr")
    vr.view()
    
    # return v, vr, constraints, restricted_dofs, F, x
    return v, vr, constraints

if __name__ == '__main__':
    test_restriction()