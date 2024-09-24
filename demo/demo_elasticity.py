#!/usr/bin/env python3
import json
import logging
import os
import sys
from pathlib import Path

import dolfinx
import dolfinx.mesh
import dolfinx.plot
import numpy as np
import petsc4py
import pyvista
import ufl
import yaml
from dolfinx import log
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI
from petsc4py import PETSc
from pyvista.utilities import xvfb

from irrevolutions.meshes.primitives import mesh_bar_gmshapi
from irrevolutions.models import ElasticityModel
from irrevolutions.solvers import SNESSolver as ElasticitySolver
from irrevolutions.utils.viz import plot_vector

logging.basicConfig(level=logging.INFO)


# ///////////


petsc4py.init(sys.argv)
log.set_log_level(log.LogLevel.WARNING)

comm = MPI.COMM_WORLD

# Mesh on node model_rank and then distribute
model_rank = 0

with open(os.path.join(os.path.dirname(__file__), "parameters.yml")) as f:
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
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

outdir = os.path.join(os.path.dirname(__file__), "output")
if comm.rank == 0:
    Path(outdir).mkdir(parents=True, exist_ok=True)

prefix = os.path.join(outdir, "elasticity")

with XDMFFile(comm, f"{prefix}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as file:
    file.write_mesh(mesh)

# Function spaces
element_u = ufl.VectorElement("Lagrange", mesh.ufl_cell(), degree=1, dim=tdim)
V_u = dolfinx.fem.FunctionSpace(mesh, element_u)
V_ux = dolfinx.fem.FunctionSpace(
    mesh, ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)
)

# Define the state
u = dolfinx.fem.Function(V_u, name="Displacement")
u_ = dolfinx.fem.Function(V_u, name="Boundary Displacement")
ux_ = dolfinx.fem.Function(V_ux, name="Boundary Displacement")
zero_u = dolfinx.fem.Function(V_u, name="   Boundary Displacement")

state = {"u": u}

# Measures
dx = ufl.Measure("dx", domain=mesh)
ds = ufl.Measure("ds", domain=mesh)

dofs_u_left = dolfinx.fem.locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], 0.0))
dofs_u_right = dolfinx.fem.locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], Lx))
dofs_ux_right = dolfinx.fem.locate_dofs_geometrical(
    V_ux, lambda x: np.isclose(x[0], Lx)
)

# Set Bcs Function
zero_u.interpolate(lambda x: (np.zeros_like(x[0]), np.zeros_like(x[1])))
u_.interpolate(lambda x: (np.ones_like(x[0]), 0 * np.ones_like(x[1])))
ux_.interpolate(lambda x: np.ones_like(x[0]))

for f in [zero_u, ux_]:
    f.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

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
    u_.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    logging.info(f"-- Solving for t = {t:3.2f} --")

    solver.solve()

    elastic_energy = comm.allreduce(
        dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(model.elastic_energy_density(state) * dx)
        ),
        op=MPI.SUM,
    )

    history_data["load"].append(t)
    history_data["elastic_energy"].append(elastic_energy)

    with XDMFFile(comm, f"{prefix}.xdmf", "a", encoding=XDMFFile.Encoding.HDF5) as file:
        file.write_function(u, t)

    if comm.rank == 0:
        a_file = open(f"{prefix}_data.json", "w")
        json.dump(history_data, a_file)
        a_file.close()

xvfb.start_xvfb(wait=0.05)
pyvista.OFF_SCREEN = True

plotter = pyvista.Plotter(
    title="Displacement",
    window_size=[1600, 600],
    shape=(1, 2),
)

# _plt = plot_scalar(u_.sub(0), plotter, subplot=(0, 0))
_plt = plot_vector(u, plotter, subplot=(0, 1))
_plt.screenshot(os.path.join(prefix, f"elasticity_displacement_MPI{comm.size}.png"))
