import pdb
import sys
import os

from mpi4py import MPI
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import dolfinx
import ufl
import numpy as np
import random

sys.path.append("../")
import solvers.restriction as restriction
_N = 3

mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, _N)

outdir = "output"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# if rank == 0:
#     data = {'a': 7, 'b': 3.14}
#     comm.send(data, dest=1, tag=11)
# elif rank == 1:
#     data = comm.recv(source=0, tag=11)

prefix = os.path.join(outdir, "test_cone")

element_u = ufl.FiniteElement("Lagrange", mesh.ufl_cell(),
                              degree=1)

element_alpha = ufl.FiniteElement("Lagrange", mesh.ufl_cell(),
                                  degree=1)

V_u = dolfinx.fem.FunctionSpace(mesh, element_u)
V_alpha = dolfinx.fem.FunctionSpace(mesh, element_alpha)
u = dolfinx.fem.Function(V_u, name="Displacement")
alpha = dolfinx.fem.Function(V_alpha, name="Damage")
from dolfinx.fem import locate_dofs_geometrical, dirichletbc

dofs_alpha_left = locate_dofs_geometrical(
    V_alpha, lambda x: np.isclose(x[0], 0.))
dofs_alpha_right = locate_dofs_geometrical(
    V_alpha, lambda x: np.isclose(x[0], 1.))

dofs_u_left = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], 0.))
dofs_u_right = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], 1.))

def get_inactive_dofset(V_u = V_u, V_alpha =  V_alpha):
    """docstring for get_inactive_dofset"""
    V_u_size = V_u.dofmap.index_map_bs * (V_u.dofmap.index_map.size_local)
    # simil to: localisation
    idx_alpha_local = set(np.arange(1, 4))
    # simil to: homogeneous response
    # idx_alpha_local = np.arange(V_alpha_size, dtype=np.int32)
    dofs_u_all = np.arange(V_u_size, dtype=np.int32)
    dofs_alpha_inactive = np.array(list(idx_alpha_local), dtype=np.int32)
    
    restricted_dofs = [dofs_u_all, dofs_alpha_inactive]
    
    return restricted_dofs
    
V_u_size = V_u.dofmap.index_map_bs * (V_u.dofmap.index_map.size_local)
V_alpha_size = V_alpha.dofmap.index_map_bs * (V_u.dofmap.index_map.size_local)

restricted_dofs = get_inactive_dofset()

restriction = restriction.Restriction([V_u, V_alpha], restricted_dofs)
dx = ufl.Measure("dx", alpha.function_space.mesh)

energy = (1-alpha)**2*ufl.inner(u,u) * dx

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

def test():

    v = dolfinx.fem.petsc.create_vector_block(F)
    x = dolfinx.fem.petsc.create_vector_block(F)
    # scatter_local_vectors(x, [u.vector.array_r, p.vector.array_r],
    #                         [(u.function_space.dofmap.index_map, u.function_space.dofmap.index_map_bs),
    #                         (p.function_space.dofmap.index_map, p.function_space.dofmap.index_map_bs)])
    # x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    from dolfinx.cpp.la.petsc import scatter_local_vectors

    print(f"{rank}) v original (unadmissible!) {v.array}")
    print(f"{rank}) V_u_size {V_u_size}")
    print(f"{rank}) V_alpha_size {V_alpha_size}")
    print(f"{rank}) restricted_dofs {restricted_dofs}")
    print(f"{rank}) bglobal_dofs_vec {restriction.bglobal_dofs_vec}")
    alpha_dofs = restriction.bglobal_dofs_vec[1]

    print(f"{rank}) restricted _dofs in state vector {alpha_dofs}")

    _is_alpha = PETSc.IS().createGeneral(alpha_dofs)
    _is = PETSc.IS().createGeneral(alpha_dofs)

    maps = [(form.function_spaces[0].dofmap.index_map, form.function_spaces[0].dofmap.index_map_bs) for form in F]
    # this is a pointer
    _sub = v.getSubVector(_is)

    a = _sub.duplicate()
    # a.array = [1]*len(alpha_dofs)
    # a.array = [.5*k for k in [-2,.5, -1, 2, 3, 4, 5, 3, ][0:len(alpha_dofs)]]
    # a.assemble()
    # _sub.pointwiseMax(_sub, a)
    # _sub.assemble()

    # print(f"{rank}) a array {a.array}")
    print(f"{rank}) v (_sub) {v.array}")
    print(f"{rank}) sub array {_sub.array}")
    v.restoreSubVector(_is, _sub)
    # _sub is destroyed

    urandom = v.duplicate()
    urandom.array = [random.uniform(0, 1.5) for r in range(v.local_size)]


    urandom.copy(v)

    for i, space in enumerate([V_u, V_alpha]):

        bs = space.dofmap.index_map_bs

        size_local = space.dofmap.index_map.size_local
        num_ghosts = space.dofmap.index_map.num_ghosts
        print(space)
        print(f"{rank}) ", i, "bs", bs)
        print(f"{rank}) ", i, "size_local", size_local)
        print(f"{rank}) ", i, "num_ghosts", num_ghosts)


    v_r = restriction.restrict_vector(v)
    print(f"{rank}) v_restricted.size {v_r.size}")
    print(f"{rank}) v_restricted.array_r {v_r.array_r}")
    c_dofs = restriction.bglobal_dofs_vec[1]

    print(f"{rank}) bglobal_dofs_vec {restriction.bglobal_dofs_vec}")
    print(f"{rank}) blocal_dofs {restriction.blocal_dofs}")
    print(f"{rank}) boffsets_vec {restriction.boffsets_vec}")
    print(f"{rank}) bglobal_dofs_vec_stacked {restriction.bglobal_dofs_vec_stacked}")

    def converged(x):
        # computes error of x with respect to xold
        _converged = bool(np.int32(random.uniform(0, 1.5)))
        print(f"{rank}) xold.array {xold.array}")
        diff = xold.duplicate()
        diff.zeroEntries()
        diff.waxpy(-1., xold, x)
        error_x_L2 = diff.norm()
        error = error_x_L2
        print(f"{rank}) err {error_x_L2}")

        # update xold   
        # x.copy(_xold)
        # x.vector.ghostUpdate(
        #     addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        # )

        # if not converged:
        #     its += 1
        
        print("converged" if _converged else f" converging")

        return _converged

    def _cone_project(v):
        """Projection vector into the cone

            takes arguments:
            - v: vector in a mixed space

            returns
        """

        _dofs = restriction.bglobal_dofs_vec[1]
        _is = PETSc.IS().createGeneral(_dofs)
        
        # new vector
        w = v.copy()
        _sub = w.getSubVector(_is)
        
        zero = _sub.duplicate()
        zero.zeroEntries()

        _sub.pointwiseMax(_sub, zero)
        w.restoreSubVector(_is, _sub)
        return w

    def _cone_rproject(v):
        """returns the projection of a full state vector v
        (considering the restriction), onto the positive cone
        the returned vector (new) is defined on the same space as v"""

        vk = v.copy()
        _is = PETSc.IS().createGeneral(restriction.bglobal_dofs_vec[1])
        _sub = vk.getSubVector(_is)
        print(f"{rank}) _sub-.array_r {_sub.array_r}")
        _subzero = zero.getSubVector(_is)
        _sub.pointwiseMax(_sub, _subzero)
        print(f"{rank}) _sub+.array_r {_sub.array_r}")
        vk.restoreSubVector(_is, _sub)

        return vk

    def _cone_project_restricted(vr):
        """returns the projection of a full state vector v
        (considering the restriction), onto the positive cone
        the returned vector (new) is defined on the same space as v"""

        vk = vr.copy()
        _is = PETSc.IS().createGeneral(restriction.bglobal_dofs_vec[1])
        _sub = vk.getSubVector(_is)
        print(f"{rank}) _sub-.array_r {_sub.array_r}")
        _subzero = zero.getSubVector(_is)
        _sub.pointwiseMax(_sub, _subzero)
        print(f"{rank}) _sub+.array_r {_sub.array_r}")
        vk.restoreSubVector(_is, _sub)

        return vk

    print(f"{rank}) v_restricted.array_r {v_r.array_r}")
    print(f"{rank}) v.array_r {v.array_r}")
    print(f"{rank}) blocal_dofs {restriction.blocal_dofs}")
    print(f"{rank}) bglobal_dofs_vec_stacked {restriction.bglobal_dofs_vec_stacked}")
    print(f"{rank}) bglobal_dofs_vec {restriction.bglobal_dofs_vec}")
    print(f"{rank}) bglobal_dofs_vec[1] {restriction.bglobal_dofs_vec[1]}")
    print(f"{rank}) blocal_dofs[1] {restriction.blocal_dofs[1]}")

    print(f"{rank}) v.array[restriction.blocal_dofs[1]] {v.array[restriction.blocal_dofs[1]]}")

    urandom.copy(v)
    zero = urandom.duplicate()
    zero.zeroEntries()

    vk = v.copy()
    _is = PETSc.IS().createGeneral(restriction.bglobal_dofs_vec[1])
    _sub = vk.getSubVector(_is)

    _subzero = zero.getSubVector(_is)
    _sub.pointwiseMax(_sub, _subzero)

    print(f"{rank}) pointwiseMax(_sub, _subzero) {_sub.array}")

    _sub.pointwiseMin(_sub, _subzero)

    print(f"{rank}) pointwiseMin(_sub, _subzero) {_sub.array}")

    vk.restoreSubVector(_is, _sub)

    print(f"{rank}) vk.array_r {vk.array_r}")

    urandom = v.duplicate()
    urandom.array = [random.uniform(-1., 1.) for r in range(v.local_size)]

    v = urandom.copy()
    vk = _cone_rproject(v)

    print(f"{rank}) vk.array_r {vk.array_r}")

    v_r = restriction.restrict_vector(vk)

    print(f"{rank}) v_r.array_r {v_r.array_r}")


    urandom.copy(x)
    xold = dolfinx.fem.petsc.create_vector_block(F)

    while not converged(x):
        # if restriction is not None:
        # this is the general case
        # update xk
        print(f"{rank}) x.array_r {x.array_r}")
        xk = _cone_rproject(x)
        urandom.array = [random.uniform(-1., 1.) for r in range(v.local_size)]
        xk = urandom.copy()
        xk = _cone_rproject(xk)

        print(f"{rank}) xk.array_r {xk.array_r}")
        xk.array[-1] = 0.
        print(f"{rank}) xk.array_r {xk.array_r}")
        # update full field
        # update x old full
        xk.copy(xold)
        # xold.copy(...)

        print(converged(x))

    _is = PETSc.IS().createGeneral(restriction.bglobal_dofs_vec[1])
    _sub = vk.getSubVector(_is)

    v = urandom.copy()
    vk = _cone_rproject(v)

    v_r = restriction.restrict_vector(vk)

    print(f"{rank}) vk.array_r {vk.array_r}")
    print(f"{rank}) v_r.array_r {v_r.array_r}")

    _isall = PETSc.IS().createGeneral(restriction.bglobal_dofs_vec_stacked)
    _suball = v.getSubVector(_isall)
    v_r.zeroEntries()
    v_r.copy(_suball)
    v.restoreSubVector(_isall, _suball)

    def extend_vector(vres, vext):
        """extends restricted vector vr into v, in place"""
        # v = dolfinx.fem.petsc.create_vector_block(F)

        _isall = PETSc.IS().createGeneral(restriction.bglobal_dofs_vec_stacked)
        _suball = vext.getSubVector(_isall)

        # v_r.zeroEntries()
        __import__('pdb').set_trace()
        vres.copy(_suball)
        vext.restoreSubVector(_isall, _suball)
        
        return

    v = urandom.copy()
    vk = _cone_rproject(v)
    vr = restriction.restrict_vector(vk)

    print(f"{rank}) xold.array_r {xold.array_r}")
    print(f"{rank}) vr.array_r {vr.array_r}")
    print(f"{rank}) vk.array_r {vk.array_r}")

    xold.zeroEntries()
    extend_vector(vr, xold)

    __import__('pdb').set_trace()


    while not converged(v_r):
        # if restriction is not None:
        # this is the general case
        # update xk
        print(f"{rank}) v_r.array_r {v_r.array_r}")
        v_k = _cone_rproject(v_r)
        urandom.array = [random.uniform(-1., 1.) for r in range(v.local_size)]
        xk = urandom.copy()
        xk = _cone_rproject(xk)

        print(f"{rank}) xk.array_r {xk.array_r}")
        xk.array[-1] = 0.
        print(f"{rank}) xk.array_r {xk.array_r}")
        # update full field
        # update x old full
        xk.copy(xold)
        # xold.copy(...)

        print(converged(x))


if __name__ == "__main__":
    test()