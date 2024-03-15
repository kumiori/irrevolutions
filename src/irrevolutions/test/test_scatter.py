import sys
import os

import petsc4py
petsc4py.init(sys.argv)
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import ufl
sys.path.append("../")
import irrevolutions.solvers.restriction as restriction

"""Discrete endommageable springs in series
        1         2        i        k
0|----[WWW]--*--[WWW]--*--...--*--{WWW} |========> t
u_0         u_1       u_2     u_i      u_k


[WWW]: endommageable spring, alpha_i
load: displacement hard-t

"""
_N = 10

mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, _N)

outdir = os.path.join(os.path.dirname(__file__), "output")
prefix = os.path.join(outdir, "test_cone")

element_u = ufl.FiniteElement("Lagrange", mesh.ufl_cell(),
                              degree=1)

element_alpha = ufl.FiniteElement("Lagrange", mesh.ufl_cell(),
                                  degree=1)

V_u = dolfinx.fem.FunctionSpace(mesh, element_u)
V_alpha = dolfinx.fem.FunctionSpace(mesh, element_alpha)
u = dolfinx.fem.Function(V_u, name="Displacement")
alpha = dolfinx.fem.Function(V_alpha, name="Damage")
from dolfinx.fem import locate_dofs_geometrical
import numpy as np

dofs_alpha_left = locate_dofs_geometrical(
    V_alpha, lambda x: np.isclose(x[0], 0.))
dofs_alpha_right = locate_dofs_geometrical(
    V_alpha, lambda x: np.isclose(x[0], 1.))

dofs_u_left = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], 0.))
dofs_u_right = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], 1.))


def get_inactive_dofset():
    """docstring for get_inactive_dofset"""
    V_u_size = V_u.dofmap.index_map_bs * (V_u.dofmap.index_map.size_local)
    
    idx_alpha_local = set(np.arange(5, 8))
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

v = dolfinx.fem.petsc.create_vector_block(F)
x = dolfinx.fem.petsc.create_vector_block(F)
# scatter_local_vectors(x, [u.vector.array_r, p.vector.array_r],
#                         [(u.function_space.dofmap.index_map, u.function_space.dofmap.index_map_bs),
#                         (p.function_space.dofmap.index_map, p.function_space.dofmap.index_map_bs)])
# x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

print(f"v original (unadmissible!) {v.array}")

print(f"V_u_size {V_u_size}")
print(f"V_alpha_size {V_alpha_size}")
print(f"restricted_dofs {restricted_dofs}")
print(f"bglobal_dofs_vec {restriction.bglobal_dofs_vec}")
alpha_dofs = restriction.bglobal_dofs_vec[1]

print(f"restricted _dofs in state vector {alpha_dofs}")

_is_alpha = PETSc.IS().createGeneral(alpha_dofs)
_is = PETSc.IS().createGeneral(alpha_dofs)


# this is a pointer
_sub = v.getSubVector(_is)

a = _sub.duplicate()
# a.interpolate(1.)

maps = [(form.function_spaces[0].dofmap.index_map, form.function_spaces[0].dofmap.index_map_bs) for form in F]

print(f"a array {a.array}")

# with a.localForm() as loc:
#     loc.set(1.0)
# *** AttributeError: attribute 'array_r' of 'petsc4py.PETSc.Vec' objects is not writable
a.array = [1]*len(alpha_dofs)
a.array = [.5*k for k in [-2,.5, -1, 2, 3, 4, 5, 3][0:len(alpha_dofs)]]
a.assemble()

_sub.pointwiseMax(_sub, a)
_sub.assemble()
print(f"one array {a.array}")

print(f"v (_sub) {v.array}")
print(f"sub array {_sub.array}")
v.restoreSubVector(_is, _sub)
# _sub is destroyed

print(f"v restored (projected) {v.array}")

for i, space in enumerate([V_u, V_alpha]):

    bs = space.dofmap.index_map_bs

    size_local = space.dofmap.index_map.size_local
    num_ghosts = space.dofmap.index_map.num_ghosts

    print(i, space, "bs", bs)
    print(i, space, "size_local", size_local)
    print(i, space, "num_ghosts", num_ghosts)


from dolfinx import cpp as _cpp
x0_local = _cpp.la.petsc.get_local_vectors(x, maps)

print(f"this should scatter x0_local into the global vector v")

_cpp.la.petsc.scatter_local_vectors(v, x0_local, maps)
v.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.FORWARD)

print(f"v should now be zero")
print(f"v restored (projected) {v.array}")

_sub = v.getSubVector(_is)
_sub.pointwiseMax(_sub, a)

# _cpp.la.petsc.scatter_local_vectors(v, x0_local, maps)
print(f"v should now be harmless")

print(f"v restored {v.array}")

v_r = restriction.restrict_vector(v)
print(f"v_restricted.size {v_r.size}")
print(f"v_restricted.array_r {v_r.array_r}")
c_dofs = restriction.bglobal_dofs_vec[1]

# its=0

import random
def converged(x):
    _converged = bool(np.int32(random.uniform(0, 1.5)))
    
    # update xold
    # x.copy(_xold)
    # x.vector.ghostUpdate(
    #     addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    # )

    # if not converged:
    #     its += 1
    
    print("converged" if _converged else f" converging")

    return _converged

def _cone_project(v, v_r):
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


_sub = v.getSubVector(_is)
_sub.array = [.5*k for k in [-2,.5, -1, 2, 3, 4, 5, 3][0:len(alpha_dofs)]]
_sub.assemble()
v.restoreSubVector(_is, _sub)


urandom = v.duplicate()
urandom.array = [random.uniform(0, 1.5) for r in range(v.local_size)]

# get initial guess (full)
urandom.copy(x)

# restrict 
# project component
x_r = restriction.restrict_vector(x)
# x_rp = _cone_project(x, x_r)

while not converged(x):
    # if restriction is not None:
    # this is the general case
    # update xk
    # update full field
    # update x old full
    # xold.copy(...)

    print(converged(x))


comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

print(rank)
v_local = _cpp.la.petsc.get_local_vectors(v, maps)
v1_local = v_local[1]
print(f"v1_local {v1_local}")

print(f"{comm.rank}, {rank}/{size} restriction.bglobal_dofs_vec {restriction.bglobal_dofs_vec}")
print(f"{comm.rank}, {rank}/{size} restriction.bglobal_dofs_vec {restriction.blocal_dofs}")
# scatters block_local vectors into v
_cpp.la.petsc.scatter_local_vectors(v, v_local, maps)
v.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
