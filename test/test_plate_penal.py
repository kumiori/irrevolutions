# Solving a plate problem with mixed formulation
# and Regge elements (ie. tensors with n-n continuity
# across facets)

#!/usr/bin/env python3
import numpy as np
import yaml
import json
from pathlib import Path
import sys
import os
from mpi4py import MPI
import petsc4py
from petsc4py import PETSc
import dolfinx
import dolfinx.plot
from dolfinx import log
import ufl
sys.path.append("../")
import pdb

from dolfinx.io import XDMFFile
from meshes import gmsh_model_to_mesh
# from damage.utils import ColorPrint
from models import ElasticityModel
from solvers import SNESSolver as ElasticitySolver

from meshes.primitives import mesh_bar_gmshapi

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

import dolfinx
import dolfinx.plot
import dolfinx.io
from dolfinx.fem import (
    Constant,
    Function,
    FunctionSpace,
    assemble_scalar,
    dirichletbc,
    form,
    locate_dofs_geometrical,
    set_bc,
)
import dolfinx.mesh
from dolfinx.mesh import CellType
import ufl

from ufl import (CellDiameter, FacetNormal, SpatialCoordinate, TestFunction,
                 TrialFunction, avg, div, ds, dS, dx, grad, inner, sym, jump, outer)

from mpi4py import MPI
import petsc4py
from petsc4py import PETSc
import sys
import yaml

sys.path.append("../")
from solvers import SNESSolver


petsc4py.init(sys.argv)
log.set_log_level(log.LogLevel.WARNING)

comm = MPI.COMM_WORLD

with open("parameters.yml") as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)

# Get mesh parameters
Lx = parameters["geometry"]["Lx"]
Ly = parameters["geometry"]["Ly"]
tdim = parameters["geometry"]["geometric_dimension"]
lc = parameters["geometry"]["lc"]

# Get geometry model
geom_type = parameters["geometry"]["geom_type"]

# Create the mesh of the specimen with given dimensions
gmsh_model, tdim = mesh_bar_gmshapi(geom_type, Lx, Ly, lc, tdim)

# Get mesh and meshtags
mesh, mts = gmsh_model_to_mesh(gmsh_model,
                               cell_data=False,
                               facet_data=True,
                               gdim=2)

outdir = "output"
if comm.rank == 0:
    Path(outdir).mkdir(parents=True, exist_ok=True)

prefix = os.path.join(outdir, "plate")

with XDMFFile(comm, f"{prefix}.xdmf", "w",
              encoding=XDMFFile.Encoding.HDF5) as file:
    file.write_mesh(mesh)

    # Function spaces

    # x_extents = mesh_bounding_box(mesh, 0)
    # y_extents = mesh_bounding_box(mesh, 1)
element_W = ufl.FiniteElement("Lagrange", "triangle", 2)


W = dolfinx.fem.FunctionSpace(mesh, element_W)

w_ = dolfinx.fem.Function(W)
import logging

logging.info(f"Problem dimension (W): { W.dofmap.index_map.size_global} dofs")

E = Constant(mesh, 1.0)
nu = Constant(mesh, 0.3)
t = Constant(mesh, 0.1)

theta = ufl.grad(w_)
k = ufl.variable(sym(grad(theta)))


def A_inner(e1, e2):
    coeff = nu / (1.0 - nu)
    return ufl.inner(e1, e2) + coeff * ufl.tr(e1) * ufl.tr(e2)

D = Constant(mesh, 1.0)
psi_b = 1.0 / 12.0 * A_inner(k, k)

# p = Expression(
#         ('exp(-pow(beta, 2) * (pow(x[0]-x0, 2.)+pow(x[1]-y0, 2.)))'),
#     beta = 30.,
#     x0 = np.average(x_extents), y0 = np.average(y_extents),
#     degree = 3)

psi = psi_b

M = ufl.diff(psi, k)



alpha = E * t ** 3
h = CellDiameter(mesh)

n = FacetNormal(mesh)

M_nn = inner(M, outer(n, n))

L_CDG = (
    -inner(jump(theta, n), avg(M_nn)) * dS
    + 0.5 * (alpha("+") / avg(h)) * inner(jump(theta, n), jump(theta, n)) * dS
)
pdb.set_trace()
bcs_w = DirichletBC(W, Constant(0.0), f"near(x[0], {x_extents[0]}) and on_boundary")

bcs = [bcs_w]
# bcs = [bcs_w]

all_boundary = CompiledSubDomain("on_boundary")
# left = CompiledSubDomain(f"near(x[0], {x_extents[0]}) and on_boundary")
right = CompiledSubDomain(f"near(x[0], {x_extents[1]}) and on_boundary")
# dirichlet = CompiledSubDomain(f"(near(x[0], {x_extents[0]}) or near(x[0], {x_extents[1]})) and on_boundary")
dirichlet = CompiledSubDomain(f"near(x[0], {x_extents[0]}) and on_boundary")
facet_function = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
facet_function.set_all(0)
# all_boundary.mark(facet_function, 1)
# left.mark(facet_function, 2)
right.mark(facet_function, 2)
dirichlet.mark(facet_function, 0)

ds = Measure("ds")(subdomain_data=facet_function)
dx = Measure("dx")

Mnn0 = Constant(1.0)

theta_d = Constant((0.0, 0.0))
theta_effective = theta - theta_d
L_BC = (
    -inner(inner(theta_effective, n), M_nn) * ds(0)
    + 0.5
    * (alpha / h)
    * inner(inner(theta_effective, n), inner(theta_effective, n))
    * ds(0)
    - inner(inner(theta_effective, n), Mnn0) * ds(2)
)

# f = Constant(1.)
# W_ext = f0*f * w_
# W_ext = f0*f*p * w_
W_ext = Constant(0.0) * w_

L = psi * dx - W_ext * dx + L_CDG + L_BC

DL_CDG = derivative(L_CDG, w_, TestFunction(W))
DL_BC = derivative(L_BC, w_, TrialFunction(W))
F = derivative(L, w_, TestFunction(W))
J = derivative(F, w_, TrialFunction(W))







# SREG = ufl.FiniteElement('HHJ', mesh.ufl_cell(), r)
# CG = ufl.FiniteElement('CG', mesh.ufl_cell(), r + 1)
pdb.set_trace()
element_u = ufl.VectorElement("Lagrange", mesh.ufl_cell(), degree=1, dim=tdim)
V_u = dolfinx.fem.FunctionSpace(mesh, element_u)

# Define the state
u = dolfinx.fem.Function(V_u, name="Displacement")
u_ = dolfinx.fem.Function(V_u, name="Boundary Displacement")
ux_ = dolfinx.fem.Function(V_u.sub(0).collapse(), name="Boundary Displacement")
zero_u = dolfinx.fem.Function(V_u, name="   Boundary Displacement")

state = {"u": u}

# Measures
dx = ufl.Measure("dx", domain=mesh)
ds = ufl.Measure("ds", domain=mesh)

dofs_u_left = dolfinx.fem.locate_dofs_geometrical(
    V_u, lambda x: np.isclose(x[0], 0.0))
dofs_u_right = dolfinx.fem.locate_dofs_geometrical(
    V_u, lambda x: np.isclose(x[0], Lx))
dofs_ux_right = dolfinx.fem.locate_dofs_geometrical(
    (V_u.sub(0), V_u.sub(0).collapse()), lambda x: np.isclose(x[0], Lx))

# Set Bcs Function
zero_u.interpolate(lambda x: (np.zeros_like(x[0]), np.zeros_like(x[1])))
u_.interpolate(lambda x: (np.ones_like(x[0]), 0 * np.ones_like(x[1])))
ux_.interpolate(lambda x: np.ones_like(x[0]))

for f in [zero_u, ux_]:
    f.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                         mode=PETSc.ScatterMode.FORWARD)

bcs_u = [
    dolfinx.fem.dirichletbc(zero_u, dofs_u_left),
    dolfinx.fem.dirichletbc(u_, dofs_u_right),
    # dolfinx.fem.dirichletbc(ux_, dofs_ux_right, V_u.sub(0)),
]

bcs = {"bcs_u": bcs_u}
# Define the model
model = ElasticityModel(parameters["model"])

# Energy functional
f = dolfinx.fem.Constant(mesh, np.array([0, 0], dtype=PETSc.ScalarType))
external_work = ufl.dot(f, state["u"]) * dx
total_energy = model.total_energy_density(state) * dx - external_work
energy_u = ufl.derivative(total_energy, u, ufl.TestFunction(V_u))

load_par = parameters["loading"]
loads = np.linspace(load_par["min"], load_par["max"], load_par["steps"])

solver = ElasticitySolver(
    energy_u,
    u,
    bcs_u,
    bounds=None,
    petsc_options=parameters.get("solvers").get("elasticity").get("snes"),
    prefix=parameters.get("solvers").get("elasticity").get("prefix"),
)

history_data = {
    "load": [],
    "elastic_energy": [],
}

for i_t, t in enumerate(loads):
    u_.interpolate(lambda x: (t * np.ones_like(x[0]), 0 * np.ones_like(x[1])))
    u_.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                          mode=PETSc.ScatterMode.FORWARD)

    logging.info(f"-- Solving for t = {t:3.2f} --")

    solver.solve()

    elastic_energy = comm.allreduce(
        dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(model.elastic_energy_density(state) * dx)),
        op=MPI.SUM,
    )

    history_data["load"].append(t)
    history_data["elastic_energy"].append(elastic_energy)

    with XDMFFile(comm, f"{prefix}.xdmf", "a",
                  encoding=XDMFFile.Encoding.HDF5) as file:
        file.write_function(u, t)

    if comm.rank == 0:
        a_file = open(f"{prefix}_data.json", "w")
        json.dump(history_data, a_file)
        a_file.close()

from utils.viz import plot_mesh, plot_vector, plot_scalar
import pyvista

plotter = pyvista.Plotter(
    title="Displacement",
    window_size=[1600, 600],
    shape=(1, 2),
)

# _plt = plot_scalar(u_.sub(0), plotter, subplot=(0, 0))
_plt = plot_vector(u, plotter, subplot=(0, 1))
_plt.screenshot(f"output/elasticity_displacement_MPI{comm.size}.png")