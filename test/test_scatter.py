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


print(f"V_u_size {V_u_size}")
print(f"V_alpha_size {V_alpha_size}")






__import__('pdb').set_trace()