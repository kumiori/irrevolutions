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

from petsc4py import PETSc
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from test_restriction import (
    __log_incipit,
    test_restriction,
    get_inactive_dofset,
    
)

def test_extension(v, vr, constraints):
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
    _logger.info(f"x")
    x.view()
    
    _logger.info(f"setting up dofs for extension")
    _logger.critical(f"{__log_incipit} The \"good\" dofs: constraints.bglobal_dofs_vec_stacked {constraints.bglobal_dofs_vec_stacked}")
    
    _islocal = PETSc.IS().createGeneral(constraints.bglobal_dofs_vec_stacked)
    _logger.info(f"_islocal")
    _islocal.view()
    # _islocal is the inverse mapping from the global dofs of the entire vector to the local dofs of the restricted vector 
    # I need, not only the target degrees of freedom, but also the index set mapping
    
    _subvector = v.getSubVector(_islocal)
    _logger.info(f"_subvector contains the values on the \"good\" dofs")
    _logger.info(f"_subvector")
    _subvector.view()
    # now I want to scatter the values from _subvector into x
    # scatter(_islocal, _subvector, x)
    scatter = PETSc.Scatter().create(vr, is_from = None, vec_to = x, is_to = _islocal)
    scatter.view()
    scatter.scatter(vr, x, False, PETSc.ScatterMode.FORWARD)
    _logger.info(f"we have scattered the \"good\" values into x")
    x.view()
    
if __name__ == "__main__":
    v, vr, constraints = test_restriction()
    test_extension(v, vr, constraints)