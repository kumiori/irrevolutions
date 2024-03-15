from test_scatter_MPI import (
    V_alpha,
    V_u,
    alpha,
    u,
    energy)

import sys
sys.path.append("../")
from irrevolutions.utils import _logger
# from algorithms.so import _extend_vector
import dolfinx
import ufl

from test_restriction import (
    __log_incipit,
    test_restriction,
    
)

from petsc4py import PETSc
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
__log_incipit = f"rank {rank}#{size}/"

from dolfinx.cpp.la.petsc import get_local_vectors
 
def _extend_vector(vres, x, constraints):
    """
    Extends a restricted vector vres into x, extending by zero.

    Args:
        vres: Restricted vector to be extended.
        x: Extended vector.

    Returns:
        None
    """
    _islocal = PETSc.IS().createGeneral(constraints.bglobal_dofs_vec_stacked)
    scatter = PETSc.Scatter().create(vres, is_from = None, vec_to = x, is_to = _islocal)
    scatter.scatter(vres, x, False, PETSc.ScatterMode.FORWARD)

    return

def test_extend_vector(vr, constraints):
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

    x = dolfinx.fem.petsc.create_vector_block(F)
    
    _extend_vector(vr, x, constraints)
    
    # _logger.info(f"The restricted vector")
    # vr.view()
    
    _logger.info(f"The extended vector")
    x.view()

    maps = [(V.dofmap.index_map, V.dofmap.index_map_bs) for V in [V_u, V_alpha]]

    x_u, x_alpha = get_local_vectors(x, maps)
    
    _logger.info(f"The local vectors")
    _logger.debug(f"{__log_incipit} Local data of the subvector x_u: {x_u}")
    _logger.debug(f"{__log_incipit} Local data of the subvector x_alpha: {x_alpha}")
    
    
    return

if __name__ == '__main__':
    
    v, vr, constraints = test_restriction()
    _logger.info(f"The original vector")
    v.view()

    test_extend_vector(vr, constraints)
    
    maps = [(V.dofmap.index_map, V.dofmap.index_map_bs) for V in [V_u, V_alpha]]

    x_u, x_alpha = get_local_vectors(v, maps)
    
    _logger.info(f"The local vectors")
    _logger.critical(f"{__log_incipit} Local data of the subvector v_u: {x_u}")
    _logger.critical(f"{__log_incipit} Local data of the subvector v_alpha: {x_alpha}")
