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
        _logger.info(f"x_local")
        x_local.view()
        _logger.critical(f"{__log_incipit} Local dofs: {_dofs}")
        x_local.array[_dofs] = np.maximum(x_local.array[_dofs], 0)
        _logger.info(f"x_local truncated")
        x_local.view()

    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    x_u, x_alpha = get_local_vectors(x, maps)
    _logger.info(f"The local vectors")
    _logger.critical(f"{__log_incipit} Local data of the subvector x_u: {x_u}")
    _logger.critical(f"{__log_incipit} Local data of the subvector x_alpha: {x_alpha}")

def sandbox():


    __import__('pdb').set_trace()

    with x.localForm() as x_local:
        _dofs = constraints.bglobal_dofs_vec[1]
        _is = PETSc.IS().createGeneral(_dofs)
        _logger.info(f"is with bglobal_dofs_vec")
        _is.view()
        comm.Barrier()
        _sub = x_local.getSubVector(_is)
        zero = _sub.duplicate()
        zero.zeroEntries()
        _logger.info(f"sub with blocal_dofs")
        _sub.view()
        comm.Barrier()
        
        _sub.pointwiseMax(_sub, zero)
        _logger.info(f"sub pointwiseMax")
        _sub.view()
        _logger.info(f"x_local")
        x_local.view()
        x_local.restoreSubVector(_is, zero)
        _logger.info(f"x_local restored")
        x_local.view()
        
    __import__('pdb').set_trace()
    
    
    with x.localForm() as x_local:
        _dofs = constraints.bglobal_dofs_vec[1]
        _is = PETSc.IS().createGeneral(_dofs)
        _is.view()
        _sub = x.getSubVector(_is)
        zero = _sub.duplicate()
        zero.zeroEntries()
        _sub.view()
        
        _sub.pointwiseMax(_sub, zero)
        _sub.view()
        x_local.view()
        x_local.restoreSubVector(_is, zero)
        x_local.view()
        
    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
    x_u, x_alpha = get_local_vectors(x, maps)
    _logger.info(f"The local vectors")
    _logger.critical(f"{__log_incipit} Local data of the subvector x_u: {x_u}")
    _logger.critical(f"{__log_incipit} Local data of the subvector x_alpha: {x_alpha}")

    __import__('pdb').set_trace()

    _dofs = constraints.bglobal_dofs_vec[1]
    _is = PETSc.IS().createGeneral(_dofs)
    _logger.info(f"Index set _is of alpha subvector of global vector x")
    _is.view()

    _sub = x.getSubVector(_is)
    zero = _sub.duplicate()
    _sub.view()
    
    zero.zeroEntries()
    _sub.pointwiseMax(_sub, zero)
    
    _logger.info(f"Local pointwiseMax _sub")
    _sub.view()

    x.restoreSubVector(_is, _sub)
    
    x.view()


    __import__('pdb').set_trace()

    with x.localForm() as x_local, v.localForm() as v_local:
        # x_local.view()
        x_u, x_alpha = get_local_vectors(x_local, maps)
        _logger.critical(f"{__log_incipit} x_alpha: {x_alpha}")
        _local_dofs = constraints.blocal_dofs[1]
        _local_is = PETSc.IS().createGeneral(_local_dofs)
        _logger.info(f"Local _is")
        _local_is.view()
        _sub = x_local.getSubVector(_local_is)
        _logger.info(f"Local _sub")
        _sub.view()
                
    __import__('pdb').set_trace()
    
    # sum([space.dofmap.index_map_bs * space.dofmap])
    # is the vector restricted?
    
    bs = V_alpha.dofmap.index_map_bs
    _logger.critical(f"{__log_incipit} V_alpha.dofmap.index_map.num_ghosts {V_alpha.dofmap.index_map.num_ghosts}")
    _logger.critical(f"{__log_incipit} V_alpha.dofmap.index_map.size_global {V_alpha.dofmap.index_map.size_global}")
    _logger.critical(f"{__log_incipit} V_alpha.dofmap.index_map.size_local {V_alpha.dofmap.index_map.size_local}")
    
    V_alpha_local_size = V_alpha.dofmap.index_map_bs * (V_alpha.dofmap.index_map.size_local + V_alpha.dofmap.index_map.num_ghosts)
    _logger.critical(f"{__log_incipit} V_alpha_local_size {V_alpha_local_size}")
    
    v_u, v_alpha = get_local_vectors(x, maps)
    _logger.info(f"The local vectors")
    _logger.critical(f"{__log_incipit} Local data of the subvector v_u: {v_u}")
    _logger.critical(f"{__log_incipit} Local data of the subvector v_alpha: {v_alpha}")

    _logger.critical(f"{__log_incipit} constraints.blocal_dofs {constraints.blocal_dofs[1]}")

    if v.size != x.size:
        _logger.info(f"{__log_incipit} v is restricted")
        test_extension(v, x, constraints)
        # _v = self._v
    else:
        _logger.info(f"v is not restricted")
        x = v
    
    _local_dofs = constraints.blocal_dofs[1]
    _local_is = PETSc.IS().createGeneral(_local_dofs)
    _logger.info(f"Local _is")
    _local_is.view()
    
    __import__('pdb').set_trace()
    _dofs = constraints.bglobal_dofs_vec[1]
    _is = PETSc.IS().createGeneral(_dofs)

    _sub = x.getSubVector(_is)
    zero = _sub.duplicate()

    _logger.info(f"Global alpha subvector _is")
    _is.view()
    
    _logger.info(f"Global alpha subvector _sub")
    _sub.view()

    zero.zeroEntries()
    _sub.pointwiseMax(_sub, zero)
    
    _logger.info(f"Local pointwiseMax _sub")
    _sub.view()

    x.restoreSubVector(_is, _sub)
    # x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    
    _logger.info(f"x after substitution")
    x.view()

    # with: local vectors
    #   local pointwise max
    #   local value set
    #   local scatter

if __name__ == "__main__":
    F, v = init_data(10, positive=False)
    V_u, V_alpha = F[0].function_spaces[0], F[1].function_spaces[0]
    x = dolfinx.fem.petsc.create_vector_block(F)
    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    
    restricted_dofs = get_inactive_dofset(v, F)
    
    constraints = restriction.Restriction([V_u, V_alpha], restricted_dofs)

    _logger.critical(f"{__log_incipit} constraints.blocal_dofs {constraints.blocal_dofs}")
    _logger.critical(f"{__log_incipit} constraints.bglobal_dofs_vec {constraints.bglobal_dofs_vec}")
    _logger.critical(f"{__log_incipit} constraints.bglobal_dofs_vec_stacked {constraints.bglobal_dofs_vec_stacked}")
    
    vr = constraints.restrict_vector(v)
    test_cone_project_restricted(vr, constraints, x)
    