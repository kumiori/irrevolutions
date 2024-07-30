from test_sample_data import init_data
from dolfinx.cpp.la.petsc import get_local_vectors
from mpi4py import MPI
import numpy as np
import dolfinx
from irrevolutions.utils import _logger
import irrevolutions.solvers.restriction as restriction
import sys

sys.path.append("../")


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
__log_incipit = f"rank {rank}#{size}/"


def get_inactive_dofset(v, F):
    """docstring for get_inactive_dofset"""
    _logger.info(f"inactive dofset")
    V_u, V_alpha = F[0].function_spaces[0], F[1].function_spaces[0]

    __names = ["u", "alpha"]

    for i, space in enumerate([V_u, V_alpha]):

        bs = space.dofmap.index_map_bs

        size_local = space.dofmap.index_map.size_local
        num_ghosts = space.dofmap.index_map.num_ghosts

        _logger.debug(f"{__log_incipit} space {__names[i]}, bs {bs}")
        _logger.debug(f"{__log_incipit} space {__names[i]}, size_local {size_local}")
        _logger.debug(f"{__log_incipit} space {__names[i]}, num_ghosts {num_ghosts}")
        comm.Barrier()

    V_u_size = V_u.dofmap.index_map_bs * (V_u.dofmap.index_map.size_local)
    V_alpha_size = V_alpha.dofmap.index_map_bs * (V_alpha.dofmap.index_map.size_local)
    maps = [(V.dofmap.index_map, V.dofmap.index_map_bs) for V in [V_u, V_alpha]]
    u, alpha = get_local_vectors(v, maps)

    # simil to: admissibility
    idx_alpha_local = np.array(np.where(alpha <= 1)[0], dtype=np.int32)
    idx_u_local = np.arange(V_u_size, dtype=np.int32)
    dofs_u_all = idx_u_local

    # Access the local data
    local_data = v.array

    restricted_dofs = [dofs_u_all, idx_alpha_local]
    # if len(idx_alpha_local)==0:
    #     _logger.critical(f"{__log_incipit} no inactive constraints found")
    #     raise RuntimeWarning("no inactive constraints found")

    # Print information about the vector
    _logger.critical(f"{__log_incipit} Size of local V_u_size: {V_u_size}")
    _logger.critical(f"{__log_incipit} Size of local V_alpha_size: {V_alpha_size}")
    comm.Barrier()

    _logger.critical(f"{__log_incipit} Len of subvector u: {len(u)}")
    _logger.critical(f"{__log_incipit} Len of subvector alpha: {len(alpha)}")
    comm.Barrier()
    _logger.critical(f"{__log_incipit} Local data of the subvector u: {u}")
    _logger.critical(f"{__log_incipit} Local data of the subvector alpha: {alpha}")
    comm.Barrier()
    _logger.critical(f"{__log_incipit} restricted_dofs: {restricted_dofs}")
    comm.Barrier()
    _logger.critical(f"{__log_incipit} idx_alpha_local: {idx_alpha_local}")
    _logger.critical(f"{__log_incipit} idx_u_local: {idx_u_local}")
    comm.Barrier()
    _logger.critical(f"{__log_incipit} Size of the v vector: {v.getSize()}")
    _logger.critical(f"{__log_incipit} Local data of the vector: {local_data}")
    comm.Barrier()
    # _logger.critical(f"{__log_incipit} Nonzero entries in the local data: {len(local_data.nonzero()[0])}")
    # _logger.critical(f"{__log_incipit} Global indices of nonzero entries: {v.getOwnershipRange()}")
    # _logger.critical(f"{__log_incipit} Global indices of nonzero entries: {v.getOwnershipRanges()}")

    return restricted_dofs


def test_restriction():
    F, v = init_data(5)
    V_u, V_alpha = F[0].function_spaces[0], F[1].function_spaces[0]

    dolfinx.fem.petsc.create_vector_block(F)

    restricted_dofs = get_inactive_dofset(v, F)

    constraints = restriction.Restriction([V_u, V_alpha], restricted_dofs)
    vr = constraints.restrict_vector(v)

    comm.Barrier()
    _logger.critical(
        f"{__log_incipit} constraints.blocal_dofs {constraints.blocal_dofs}"
    )
    _logger.critical(
        f"{__log_incipit} constraints.bglobal_dofs_vec {constraints.bglobal_dofs_vec}"
    )
    _logger.critical(
        f"{__log_incipit} constraints.bglobal_dofs_vec_stacked {constraints.bglobal_dofs_vec_stacked}"
    )

    _logger.info(f"v")
    v.view()
    _logger.info(f"vr")
    vr.view()

    # assert we get the right number of restricted dofs
    assert len(np.concatenate(restricted_dofs)) == vr.getSize()
    
    # return v, vr, constraints


if __name__ == "__main__":
    test_restriction()
