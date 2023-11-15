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
    get_inactive_dofset,
    energy)

import os
import sys
sys.path.append("../")
import solvers.restriction as restriction
from utils import _logger
# from algorithms.so import _extend_vector
import dolfinx
import ufl
import numpy as np
import random

from petsc4py import PETSc
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
__log_incipit = f"rank {rank}#{size}/"

def get_inactive_dofset(V_u = V_u, V_alpha = V_alpha):
    """docstring for get_inactive_dofset"""
    _logger.info(f"inactive dofset")
    V_u_size = V_u.dofmap.index_map_bs * (V_u.dofmap.index_map.size_local)
    V_alpha_size = V_alpha.dofmap.index_map_bs * (V_alpha.dofmap.index_map.size_local)
    # simil to: localisation
    if rank == 0:
        idx_alpha_local = set(np.arange(1, 3))
    else: 
        idx_alpha_local = np.arange(V_alpha_size, dtype=np.int32)
    idx_u_local = np.arange(V_u_size, dtype=np.int32)
    _logger.warning(f"{__log_incipit} idx_alpha_local {idx_alpha_local}")
    _logger.warning(f"{__log_incipit} idx_u_local {idx_u_local}")
    # simil to: homogeneous response
    # idx_alpha_local = np.arange(V_alpha_size, dtype=np.int32)
    dofs_u_all = idx_u_local
    dofs_alpha_inactive = np.array(list(idx_alpha_local), dtype=np.int32)
    
    restricted_dofs = [dofs_u_all, dofs_alpha_inactive]
    return restricted_dofs
    
def _extend_vector(vres, vext, global_dofs_stacked):
    """
    Extends a restricted vector vres into vext, extending by zero.

    Args:
        vres: Restricted vector to be extended.
        vext: Extended vector.

    Returns:
        None
    """
    _isall = PETSc.IS().createGeneral(global_dofs_stacked)
    _subvector = vext.getSubVector(_isall)
    # _logger.critical(f"{__log_incipit} # len _isall {len(_isall)}")
    # _logger.critical(f"{__log_incipit} # _subvector {_subvector.array}")
    _logger.info(f"isall")
    _isall.view()
    
    _logger.info(f"suball")
    _subvector.view()
    
    _logger.info(f"vres copies into subvector")
    _logger.warning(f"subvector")
    vres.copy(_subvector)
    _subvector.view()
    
    vext.restoreSubVector(_isall, _subvector)
    
    return

def test_extend_vector():
    restricted_dofs = get_inactive_dofset(V_u, V_alpha)
    constraints = restriction.Restriction([V_u, V_alpha], restricted_dofs)

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
    x = dolfinx.fem.petsc.create_vector_block(F)

    V_u_size = V_u.dofmap.index_map_bs * (V_u.dofmap.index_map.size_local)
    V_alpha_size = V_alpha.dofmap.index_map_bs * (V_alpha.dofmap.index_map.size_local)
    alpha_dofs = constraints.bglobal_dofs_vec[1]
    
    _logger.critical(f"{__log_incipit} V_u_size {V_u_size}")
    _logger.critical(f"{__log_incipit} V_u_size_local {V_u.dofmap.index_map.size_local}")
    
    _logger.critical(f"{__log_incipit} V_alpha_size {V_alpha_size}")
    _logger.critical(f"{__log_incipit} V_alpha_size_local {V_alpha.dofmap.index_map.size_local}")
    alpha.vector.view()
    u.vector.view()
    _logger.critical(f"{__log_incipit} restricted_dofs {restricted_dofs}")
    _logger.critical(f"{__log_incipit} restricted % {len(np.hstack(restricted_dofs))} / {(V_u_size + V_alpha_size)}")
    _logger.critical(f"{__log_incipit} alpha bglobal_dofs_vec {alpha_dofs}")
    
    for i, space in enumerate([V_u, V_alpha]):
        bs = space.dofmap.index_map_bs

        size_local = space.dofmap.index_map.size_local
        num_ghosts = space.dofmap.index_map.num_ghosts
        # _logger.info(f"space {space}")
        _logger.critical(f"{__log_incipit}  subspace_{i}, bs {bs}")
        _logger.critical(f"{__log_incipit}  subspace_{i}, size_local {size_local}")
        _logger.critical(f"{__log_incipit}  subspace_{i}, num_ghosts {num_ghosts}")
        
    v.array = [np.around(random.uniform(0, 2.5), decimals=1) for r in range(v.local_size)]
    # v.array = [r for r in range(v.local_size)]
    
    vr = constraints.restrict_vector(v)
    _logger.critical(f"{__log_incipit} bglobal_dofs_vec_stacked {constraints.bglobal_dofs_vec_stacked}")

    _is = PETSc.IS().createGeneral(alpha_dofs)
    _sub = v.getSubVector(_is)

    _logger.critical(f"{__log_incipit} constraints.blocal_dofs {constraints.blocal_dofs}")
    _logger.critical(f"{__log_incipit} constraints.bglobal_dofs_vec {constraints.bglobal_dofs_vec}")
    _logger.critical(f"{__log_incipit} constraints.bglobal_dofs_vec_stacked {constraints.bglobal_dofs_vec_stacked}")

    _extend_vector(vr, x, global_dofs_stacked=constraints.bglobal_dofs_vec_stacked)

    comm.Barrier()

    _logger.critical(f"{__log_incipit} v {v.array}")
    _logger.critical(f"{__log_incipit} vr {vr.array}")
    _logger.critical(f"{__log_incipit} x {x.array}")


    # with PETSc.Viewer().createASCII("vec.txt") as viewer:
    #     x.view(viewer)
    
    _logger.info(f"v")
    v.view()
    _logger.info(f"vr")
    vr.view()
    _logger.info(f"x")
    x.view()
    
    return

if __name__ == '__main__':
    test_extend_vector()