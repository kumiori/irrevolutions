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

def test_extend_vector(vr, x, constraints):
    V_u, V_alpha = constraints.function_spaces[0], constraints.function_spaces[1]
    maps = [(V.dofmap.index_map, V.dofmap.index_map_bs) for V in [V_u, V_alpha]]

    _logger.info(f"x")
    x.view()

    _logger.info(f"vr")
    vr.view()
    
    _logger.info(f"Setting up dofs for extension")
    _logger.critical(f"{__log_incipit} The \"good\" dofs: constraints.bglobal_dofs_vec_stacked {constraints.bglobal_dofs_vec_stacked}")

    x.zeroEntries()
    x.array[constraints.bglobal_dofs_vec_stacked] = vr.array

    _logger.critical(f"{__log_incipit} Local data of the x: {x.array}")
    _logger.info(f"x")

    x.view()

    x_u, x_alpha = get_local_vectors(x, maps)

    _logger.info(f"The local vectors")
    _logger.critical(f"{__log_incipit} Local data of the subvector x_u: {x_u}")
    _logger.critical(f"{__log_incipit} Local data of the subvector x_alpha: {x_alpha}")

def sandbox():
    
    extension = PETSc.Scatter().create(vr, is_from = None, vec_to = x, is_to = _islocal)
    extension.view()
    # _islocal = PETSc.IS().createGeneral(constraints.bglobal_dofs_vec_stacked)
    # _logger.info(f"_islocal")
    # _islocal.view()


    # extension.scatter(vr, x, False, PETSc.ScatterMode.REVERSE)
    # _logger.info(f"x: we have scattered the \"good\" values into x REVERSE")
    # x.view()

    # x_u, x_alpha = get_local_vectors(x, maps)
    # comm.Barrier()
    # _logger.info(f"The local vectors")
    # _logger.critical(f"{__log_incipit} Local data of the subvector x_u: {x_u}")
    # _logger.critical(f"{__log_incipit} Local data of the subvector x_alpha: {x_alpha}")
    
    # extension.scatter(vr, x, False, PETSc.ScatterMode.FORWARD)
    extension.scatter(vr, x, addv = PETSc.InsertMode.INSERT, mode = PETSc.ScatterMode.FORWARD)
    _logger.info(f"x: we have scattered the \"good\" values into x with INSERT & FORWARD")
    x.view()
    x_u, x_alpha = get_local_vectors(x, maps)
    comm.Barrier()
    _logger.info(f"The local vectors")
    _logger.critical(f"{__log_incipit} Local data of the subvector x_u: {x_u}")
    _logger.critical(f"{__log_incipit} Local data of the subvector x_alpha: {x_alpha}")
    comm.Barrier()

    
    x.zeroEntries()
    extension.scatter(vr, x, addv = PETSc.InsertMode.INSERT, mode = PETSc.ScatterMode.REVERSE)
    _logger.info(f"x: we have scattered the \"good\" values into x with INSERT & REVERSE")
    x.view()
    
    x_u, x_alpha = get_local_vectors(x, maps)

    _logger.info(f"The local vectors")
    _logger.critical(f"{__log_incipit} Local data of the subvector x_u: {x_u}")
    _logger.critical(f"{__log_incipit} Local data of the subvector x_alpha: {x_alpha}")
    comm.Barrier()

    # x.zeroEntries()
    # extension.scatter(vr, x, False, PETSc.ScatterMode.SCATTER_FORWARD_LOCAL)
    # _logger.info(f"x: we have scattered the \"good\" values into x SCATTER_FORWARD_LOCAL")
    # x.view()
 
    # x_u, x_alpha = get_local_vectors(x, maps)
    # comm.Barrier()
    # _logger.info(f"The local vectors")
    # _logger.critical(f"{__log_incipit} Local data of the subvector x_u: {x_u}")
    # _logger.critical(f"{__log_incipit} Local data of the subvector x_alpha: {x_alpha}")
 
    
    _logger.info(f"The original vector")
    v.view()
    v_u, v_alpha = get_local_vectors(v, maps)
    comm.Barrier()
    _logger.info(f"The local vectors")
    _logger.critical(f"{__log_incipit} Local data of the subvector v_u: {v_u}")
    _logger.critical(f"{__log_incipit} Local data of the subvector v_alpha: {v_alpha}")
   
if __name__ == "__main__":
    F, v = init_data(10)
    V_u, V_alpha = F[0].function_spaces[0], F[1].function_spaces[0]
    x = dolfinx.fem.petsc.create_vector_block(F)
    
    restricted_dofs = get_inactive_dofset(v, F)
    
    constraints = restriction.Restriction([V_u, V_alpha], restricted_dofs)

    _logger.critical(f"{__log_incipit} constraints.blocal_dofs {constraints.blocal_dofs}")
    _logger.critical(f"{__log_incipit} constraints.bglobal_dofs_vec {constraints.bglobal_dofs_vec}")
    _logger.critical(f"{__log_incipit} constraints.bglobal_dofs_vec_stacked {constraints.bglobal_dofs_vec_stacked}")
    
    vr = constraints.restrict_vector(v)
    test_extend_vector(vr, x, constraints)