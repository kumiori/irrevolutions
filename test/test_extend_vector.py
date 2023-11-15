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

from petsc4py import PETSc
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def _extend_vector(self, vres, vext):
    """
    Extends a restricted vector vr into v, in place.

    Args:
        vres: Restricted vector to be extended.
        vext: Extended vector.

    Returns:
        None
    """
    _isall = PETSc.IS().createGeneral(self.eigen.restriction.bglobal_dofs_vec_stacked)
    _suball = vext.getSubVector(_isall)

    vres.copy(_suball)
    
    vext.restoreSubVector(_isall, _suball)
    
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
    V_alpha_size = V_alpha.dofmap.index_map_bs * (V_u.dofmap.index_map.size_local)
    alpha_dofs = constraints.bglobal_dofs_vec[1]
    __log_incipit = f"rank {rank}#{size}/"
    _logger.critical(f"{__log_incipit} V_u_size {V_u_size}")
    _logger.critical(f"{__log_incipit} V_alpha_size {V_alpha_size}")
    _logger.critical(f"{__log_incipit} restricted_dofs {restricted_dofs}")
    _logger.critical(f"{__log_incipit} restricted % {len(restricted_dofs)} / {(V_u_size + V_alpha_size)}")
    _logger.critical(f"{__log_incipit} alpha bglobal_dofs_vec {alpha_dofs}")
    
    for i, space in enumerate([V_u, V_alpha]):
        bs = space.dofmap.index_map_bs

        size_local = space.dofmap.index_map.size_local
        num_ghosts = space.dofmap.index_map.num_ghosts
        _logger.info(f"space {space}")
        _logger.critical(f"{__log_incipit}  i {i}, bs {bs}")
        _logger.critical(f"{__log_incipit}  i {i}, size_local {size_local}")
        _logger.critical(f"{__log_incipit}  i {i}, num_ghosts {num_ghosts}")

    return

if __name__ == '__main__':
    test_extend_vector()