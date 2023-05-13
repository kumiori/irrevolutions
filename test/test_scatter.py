from petsc4py import PETSc
from mpi4py import MPI
import sys
import os
import dolfinx
import ufl
sys.path.append("../")
import solvers.restriction as restriction

"""Discrete endommageable springs in series
        1         2        i        k
0|----[WWW]--*--[WWW]--*--...--*--{WWW} |========> t
u_0         u_1       u_2     u_i      u_k


[WWW]: endommageable spring, alpha_i
load: displacement hard-t

"""
_N = 10

mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, _N)

outdir = "output"
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
from dolfinx.cpp.la.petsc import scatter_local_vectors

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

one = _sub.duplicate()
# one.interpolate(1.)

maps = [(form.function_spaces[0].dofmap.index_map, form.function_spaces[0].dofmap.index_map_bs) for form in F]

print(f"one array {one.array}")

# with one.localForm() as loc:
#     loc.set(1.0)
# *** AttributeError: attribute 'array_r' of 'petsc4py.PETSc.Vec' objects is not writable
one.array = [1]*len(alpha_dofs)
one.assemble()

_sub.pointwiseMax(_sub, one)
print(f"one array {one.array}")

print(f"v original (_sub) {v.array}")
print(f"sub array {_sub.array}")
v.restoreSubVector(_is, _sub)

print(f"v restored (projected) {v.array}")

for i, space in enumerate([V_u, V_alpha]):

    bs = space.dofmap.index_map_bs

    size_local = space.dofmap.index_map.size_local
    num_ghosts = space.dofmap.index_map.num_ghosts

    print(i, space, "bs", bs)
    print(i, space, "size_local", size_local)
    print(i, space, "num_ghosts", num_ghosts)

__import__('pdb').set_trace()

x0_local = _cpp.la.petsc.get_local_vectors(x, maps)

_cpp.la.petsc.scatter_local_vectors(v, x0_local, maps)
v.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

__import__('pdb').set_trace()
