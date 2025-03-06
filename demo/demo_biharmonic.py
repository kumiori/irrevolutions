import json
import logging
import os
import pdb
import sys

import numpy as np
from irrevolutions.meshes.primitives import mesh_circle_gmshapi
from irrevolutions.solvers import SNESSolver

import petsc4py
from dolfinx import log
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc

logging.basicConfig(level=logging.INFO)

import sys

import dolfinx
import dolfinx.io
import dolfinx.mesh
import dolfinx.plot
import ufl
import yaml
from dolfinx.fem import Constant, dirichletbc
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI
from ufl import (
    avg,
    div,
    grad,
    inner,
    dot,
    jump,
)

petsc4py.init(sys.argv)
log.set_log_level(log.LogLevel.WARNING)

comm = MPI.COMM_WORLD


def monitor(snes, it, norm):
    logging.info(f"Iteration {it}, residual {norm}")
    return PETSc.SNES.ConvergedReason.ITERATING


with open("default_parameters.yml") as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)

mesh_size = parameters["geometry"]["lc"]
parameters["geometry"]["radius"] = 1
parameters["geometry"]["geom_type"] = "circle"

model_rank = 0
tdim = 2

gmsh_model, tdim = mesh_circle_gmshapi(
    parameters["geometry"]["geom_type"], 1, mesh_size, tdim
)
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

outdir = "output"
prefix = os.path.join(outdir, "biharmonic")
order = 3

V = dolfinx.fem.functionspace(mesh, ("Lagrange", order))
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
bndry_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

dofs = dolfinx.fem.locate_dofs_topological(V=V, entity_dim=1, entities=bndry_facets)

bcs_u = [
    dolfinx.fem.dirichletbc(value=np.array(0, dtype=PETSc.ScalarType), dofs=dofs, V=V)
]
bcs = {"bcs_u": bcs_u}
u = dolfinx.fem.Function(V)
D = dolfinx.fem.Constant(mesh, 1.0)
α = dolfinx.fem.Constant(mesh, 10.0)
load = dolfinx.fem.Constant(mesh, 1.0)
h = ufl.CellDiameter(mesh)
h_avg = (h("+") + h("-")) / 2.0

tdim = mesh.topology.dim
num_cells = mesh.topology.index_map(tdim).size_local
_h = dolfinx.cpp.mesh.h(mesh._cpp_object, tdim, np.arange(num_cells, dtype=np.int32))

n = ufl.FacetNormal(mesh)

dx = ufl.Measure("dx")  # volume measure
ds = ufl.Measure("ds")  # boundary measure
dS = ufl.Measure("dS")  # interior facet measure

bending = (D / 2 * (inner(div(grad(u)), div(grad(u))))) * dx
W_ext = load * u * dx

# DG terms implemented as in the paper
# https://arxiv.org/abs/2501.15959


def dg1(u):
    return 1 / 2 * dot(jump(grad(u)), avg(grad(grad(u)) * n)) * dS


def dg2(u):
    return 1 / 2 * α / avg(h) * inner(jump(grad(u)), jump(grad(u))) * dS


def dg3(u):
    return 1 / 2 * α / h * inner(grad(u), grad(u)) * ds


L = bending + dg1(u) + dg2(u) + dg3(u) - W_ext
F = ufl.derivative(L, u, ufl.TestFunction(V))


solver_parameters = {
    "type": "SNES",
    "prefix": "biharmonic_",
    "snes": {
        # "snes_type": "vinewtonrsls",
        # "snes_type": "vinewtonssls",
        "snes_type": "newtontr",
        "snes_linesearch_type": "basic",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_atol": 1.0e-8,
        "snes_rtol": 1.0e-8,
        "snes_max_it": 50,
        "snes_monitor": "",
    },
}
biharmonic = SNESSolver(
    F,
    u,
    bcs.get("bcs_u"),
    petsc_options=solver_parameters.get("snes"),
    prefix=solver_parameters.get("prefix"),
)
biharmonic.solve()

convergedreason = biharmonic.solver.getConvergedReason()
iterationnumber = biharmonic.solver.getIterationNumber()
functionnorm = biharmonic.solver.getFunctionNorm()

assert convergedreason > 0, "Convergence failed"
assert functionnorm < 1.0e-8, "Function norm too large"

print("Converged reason:", convergedreason)
print("Iterations:", iterationnumber)
print("Residual norm:", functionnorm)
print("Norm of u:", u.x.petsc_vec.norm())
