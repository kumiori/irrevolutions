from mpi4py import MPI
from petsc4py import PETSc


import os
import sys
sys.path.append("../")
import irrevolutions.solvers.restriction as restriction
# from irrevolutions.utils import _logger
import dolfinx
import ufl
import numpy as np
import random

from petsc4py import PETSc
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from test_sample_data import init_data
from test_restriction import (
    __log_incipit,
    get_inactive_dofset,
)
from test_logging_mpi import setup_logger_mpi
from test_vector_info import display_vector_info

_logger = setup_logger_mpi()

def is_restricted(v, constraints):
    """
    Check if the vector v is a restricted vector based on the given constraints.

    Parameters:
        - v: PETSc.Vec, the vector to check
        - constraints: constraints object with bglobal_dofs_vec_stacked attribute

    Returns:
        - bool: True if v is restricted, False otherwise
    """
    comm = constraints.comm
    _logger.info(f"v sizes: {v.sizes}")
    global_size = comm.allreduce(len(v.array), op=MPI.SUM)
    global_size = comm.allreduce(v.sizes[0], op=MPI.SUM)
    return global_size == len(constraints.bglobal_dofs_vec_stacked)

if __name__ == "__main__":
    F, v = init_data(10, positive=False)
    # _logger.info(f"F: {F}")
    # _logger.info(f"v: {v}")
    # v.view()
    V_u, V_alpha = F[0].function_spaces[0], F[1].function_spaces[0]
    x = dolfinx.fem.petsc.create_vector_block(F)
    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    
    restricted_dofs = get_inactive_dofset(v, F)
    
    constraints = restriction.Restriction([V_u, V_alpha], restricted_dofs)
    vr = constraints.restrict_vector(v)

    # Example usage
    # Assuming you have a constraints object named 'constraints' and a PETSc.Vec named 'v'
    # display_vector_info(v)
    # comm.barrier()
    _ans = []
    # ans = 
    for _v, name in zip([v, vr], ["v", "vr"]):
        _logger.debug(f"       ~ Vector  {name}")
        reason = is_restricted(_v, constraints)
        if reason:
            print(f"The vector {name} is restricted.")
        else:
            print(f"The vector {name} is not restricted.")
        _ans.append(reason)
        
        display_vector_info(_v)
        
    print(f"Are v and vr restricted? {_ans}")
    # assert _ans == [False, True]
