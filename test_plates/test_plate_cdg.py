# Solving a plate problem with a penalised formulation

import json
import os
import sys
from pathlib import Path

import dolfinx
import dolfinx.plot
import numpy as np
import petsc4py
import ufl
import yaml
from dolfinx import log
from mpi4py import MPI
from petsc4py import PETSc

sys.path.append("../")
import logging
import pdb

import numpy as np
from dolfinx.io import XDMFFile
from meshes import mesh_bounding_box
from meshes.primitives import mesh_bar_gmshapi
# from damage.utils import ColorPrint
from models import ElasticityModel
from solvers import SNESSolver as PlateSolver
logging.basicConfig(level=logging.INFO)

import sys

import dolfinx
import dolfinx.io
import dolfinx.mesh
import dolfinx.plot
import ufl
import yaml
from dolfinx.io import XDMFFile, gmshio
from dolfinx.fem import (Constant, Function, FunctionSpace, assemble_scalar,
                         dirichletbc, form, locate_dofs_geometrical, set_bc)
from dolfinx.mesh import CellType
from mpi4py import MPI
from ufl import (CellDiameter, FacetNormal, SpatialCoordinate, TestFunction,
                 TrialFunction, avg, div, ds, dS, dx, grad, inner, jump, outer,
                 sym)

sys.path.append("../")
from solvers import SNESSolver




# class PlateModel(ElasticityModel):

#     def total_energy_density(self, state):



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
model_rank = 0

# Create the mesh of the specimen with given dimensions
gmsh_model, tdim = mesh_bar_gmshapi(geom_type, Lx, Ly, lc, tdim)

# # Get mesh and meshtags
# mesh, mts = gmsh_model_to_mesh(gmsh_model,
#                                cell_data=False,
#                                facet_data=True,
#                                gdim=2)

# mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)


# Create the mesh of the specimen with given dimensions
gmsh_model, tdim = mesh_bar_gmshapi(geom_type, Lx, Ly, lc, tdim)

# Get mesh and meshtags
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)



outdir = "output"
if comm.rank == 0:
    Path(outdir).mkdir(parents=True, exist_ok=True)

prefix = os.path.join(outdir, "plate")

with XDMFFile(comm, f"{prefix}.xdmf", "w",
              encoding=XDMFFile.Encoding.HDF5) as file:
    file.write_mesh(mesh)

    # Function spaces

x_extents = mesh_bounding_box(mesh, 0)
y_extents = mesh_bounding_box(mesh, 1)

element_w = ufl.FiniteElement("Lagrange", "triangle", 2)
element_u = ufl.VectorElement("Lagrange", mesh.ufl_cell(), degree=1, dim=tdim)

V_u = dolfinx.fem.FunctionSpace(mesh, element_u)
V_w = dolfinx.fem.FunctionSpace(mesh, element_w)

u = dolfinx.fem.Function(V_u, name="Displacement")
u_ = dolfinx.fem.Function(V_u, name="Boundary Displacement")
ux_ = dolfinx.fem.Function(V_u.sub(0).collapse()[0], name="Boundary Displacement")
zero_u = dolfinx.fem.Function(V_u, name="   Boundary Displacement")

w_ = dolfinx.fem.Function(V_w)

theta = ufl.grad(w_)
k = ufl.variable(sym(grad(theta)))


logging.info(f"Problem dimension (W): {V_w.dofmap.index_map.size_global} dofs")

E = Constant(mesh, 1.0)
nu = Constant(mesh, 0.3)
t = Constant(mesh, 0.1)

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

# Boundary sets

dofs_u_left = dolfinx.fem.locate_dofs_geometrical(
    V_u, lambda x: np.isclose(x[0], 0.0))
dofs_u_right = dolfinx.fem.locate_dofs_geometrical(
    V_u, lambda x: np.isclose(x[0], Lx))
dofs_ux_right = dolfinx.fem.locate_dofs_geometrical(
    (V_u.sub(0), V_u.sub(0).collapse()[0]), lambda x: np.isclose(x[0], Lx))

# Bcs

bcs_w = dirichletbc(
    np.array(0, dtype=PETSc.ScalarType),
    dofs_u_left, V_w
)
bcs = [bcs_w]

ds = ufl.Measure("ds")(subdomain_data=mts)
dx = ufl.Measure("dx")

Mnn0 = Constant(mesh, np.array(1.0, dtype = PETSc.ScalarType))

theta_d = Constant(mesh, np.array([0.0, 0.0], dtype=PETSc.ScalarType))

theta_e = theta - theta_d

_right = 7; _left = 6

L_BC = (
    - inner(inner(theta_e, n), M_nn) * ds(_left)
    + 0.5 * (alpha / h) * inner(inner(theta_e, n),
                                inner(theta_e, n)) * ds(_left)
    - inner(inner(theta_e, n), Mnn0) * ds(_right)
)

W_ext = Constant(mesh, np.array(0.0, dtype=PETSc.ScalarType)) * w_

L = psi * dx - W_ext * dx + L_CDG + L_BC

DL_CDG = ufl.derivative(L_CDG, w_, TestFunction(V_w))
DL_BC = ufl.derivative(L_BC, w_, TrialFunction(V_w))
F = ufl.derivative(L, w_, TestFunction(V_w))
J = ufl.derivative(F, w_, TrialFunction(V_w))

state = {"u": u}

# Set Bcs Function
zero_u.interpolate(lambda x: (np.zeros_like(x[0]), np.zeros_like(x[1])))
u_.interpolate(lambda x: (np.ones_like(x[0]), np.zeros_like(x[1])))
ux_.interpolate(lambda x: np.ones_like(x[0]))

for f in [zero_u, ux_]:
    f.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                         mode=PETSc.ScatterMode.FORWARD)

bcs_u = [
    dolfinx.fem.dirichletbc(zero_u, dofs_u_left),
    dolfinx.fem.dirichletbc(u_, dofs_u_right),
]

bcs = {"bcs_u": bcs_u}

# Define the model

loads = np.linspace(
    parameters.get('loading').get("min"),
    parameters.get('loading').get("max"),
    parameters.get('loading').get("steps")
    )

pdb.set_trace()

solver = PlateSolver(
    L,
    w_,
    bcs_w,
    bounds=None,
    petsc_options=parameters.get("solvers").get("elasticity").get("snes"),
    prefix='plate_cdg',
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

import pyvista
from utils.viz import plot_mesh, plot_scalar, plot_vector

plotter = pyvista.Plotter(
    title="Displacement",
    window_size=[1600, 600],
    shape=(1, 2),
)

_plt = plot_vector(u, plotter, subplot=(0, 1))
_plt.screenshot(f"output/plate_displacement_MPI{comm.size}.png")
