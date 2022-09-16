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
import numpy as np
sys.path.append("../")
from utils.plots import plot_energies
from utils import ColorPrint

from models import DamageElasticityModel as Brittle
from algorithms.am import AlternateMinimisation

from meshes.primitives import mesh_bar_gmshapi
from dolfinx.common import Timer, list_timings, TimingType

import logging

logging.basicConfig(level=logging.INFO)

import dolfinx
import dolfinx.plot
from dolfinx.io import XDMFFile, gmshio
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

from mpi4py import MPI
import petsc4py
from petsc4py import PETSc
import sys
import yaml

sys.path.append("../")
from solvers import SNESSolver
from algorithms.so import StabilitySolver


class ConeSolver:

# ///////////




petsc4py.init(sys.argv)
comm = MPI.COMM_WORLD

# Mesh on node model_rank and then distribute
model_rank = 0


with open("parameters.yml") as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)

# Get mesh parameters
Lx = parameters["geometry"]["Lx"]
Ly = parameters["geometry"]["Ly"]
tdim = parameters["geometry"]["geometric_dimension"]
_nameExp = parameters["geometry"]["geom_type"]
ell_ = parameters["model"]["ell"]
lc = ell_ / 5.0


# Get geometry model
geom_type = parameters["geometry"]["geom_type"]

# Create the mesh of the specimen with given dimensions
gmsh_model, tdim = mesh_bar_gmshapi(geom_type, Lx, Ly, lc, tdim)

# Get mesh and meshtags
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)


outdir = "output"
prefix = os.path.join(outdir, "traction")
if comm.rank == 0:
    Path(prefix).mkdir(parents=True, exist_ok=True)

with XDMFFile(comm, f"{prefix}/{_nameExp}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as file:
    file.write_mesh(mesh)

# Function spaces
# element_u = ufl.VectorElement("Lagrange", mesh.ufl_cell(), degree=1, dim=tdim)
# V_u = FunctionSpace(mesh, element_u)

element_u = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)
V = FunctionSpace(mesh, element_u)

# Define the state
u = Function(V, name="Unknown")
u_ = Function(V, name="Boundary Unknown")
zero_u = Function(V, name="Boundary Unknown")

state = {"u": u}

# need upper/lower bound for the damage field
u_lb = Function(V, name="Lower bound")
u_ub = Function(V, name="Upper bound")

# Measures
dx = ufl.Measure("dx", domain=mesh)
ds = ufl.Measure("ds", domain=mesh)

dofs_u_left = locate_dofs_geometrical(
    V, lambda x: np.isclose(x[0], 0.0))
dofs_u_right = locate_dofs_geometrical(
    V, lambda x: np.isclose(x[0], Lx))

dofs_u_left = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
dofs_u_right = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], Lx))

# Set Bcs Function
zero_u.interpolate(lambda x: (np.zeros_like(x[0]), np.zeros_like(x[1])))
u_.interpolate(lambda x: (np.ones_like(x[0]), 0 * np.ones_like(x[1])))
u_lb.interpolate(lambda x: np.zeros_like(x[0]))
u_ub.interpolate(lambda x: np.ones_like(x[0]))

for f in [zero_u, u_, u_lb, u_ub]:
    f.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                         mode=PETSc.ScatterMode.FORWARD)

bc_u_left = dirichletbc(
    np.array([0, 0], dtype=PETSc.ScalarType), dofs_u_left, V)

bc_u_right = dirichletbc(
    u_, dofs_u_right)
bcs_u = [bc_u_left, bc_u_right]

bcs = {"bcs_u": bcs_u}
# Define the model

model = Brittle(parameters["model"])

# Energy functional
f = Constant(mesh, np.array([0, 0], dtype=PETSc.ScalarType))
external_work = ufl.dot(f, state["u"]) * dx
total_energy = model.total_energy_density(state) * dx - external_work

load_par = parameters["loading"]
loads = np.linspace(load_par["min"],
                    load_par["max"], load_par["steps"])

solver = VariationalInequality(
    total_energy, state, bcs, parameters.get("solvers"), bounds=(u_lb, u_ub)
)

stability = ConeSolver(
    total_energy, state, bcs, stability_parameters=parameters.get("stability")
)

history_data = {
    "load": [],
    "elastic_energy": [],
    "dissipated_energy": [],
    "total_energy": [],
    "solver_data": [],
    "eigs": [],
    "stable": [],
}

check_stability = []

for i_t, t in enumerate(loads):
    u_.interpolate(lambda x: (t * np.ones_like(x[0]),  np.zeros_like(x[1])))
    u_.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                          mode=PETSc.ScatterMode.FORWARD)

    # update the lower bound
    alpha.vector.copy(alpha_lb.vector)
    alpha_lb.vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )

    logging.critical(f"-- Solving for t = {t:3.2f} --")

    solver.solve()

    n_eigenvalues = 10
    is_stable = stability.solve(alpha_lb, n_eigenvalues)
    is_elastic = stability.is_elastic()
    inertia = stability.get_inertia()
    stability.save_eigenvectors(filename=f"{prefix}_eigv_{t:3.2f}.xdmf")
    check_stability.append(is_stable)

    ColorPrint.print_bold(f"State is elastic: {is_elastic}")
    ColorPrint.print_bold(f"State's inertia: {inertia}")
    ColorPrint.print_bold(f"State is stable: {is_stable}")

    dissipated_energy = comm.allreduce(
        assemble_scalar(form(model.damage_dissipation_density(state) * dx)),
        op=MPI.SUM,
    )
    elastic_energy = comm.allreduce(
        assemble_scalar(form(model.elastic_energy_density(state) * dx)),
        op=MPI.SUM,
    )

    history_data["load"].append(t)
    history_data["dissipated_energy"].append(dissipated_energy)
    history_data["elastic_energy"].append(elastic_energy)
    history_data["total_energy"].append(elastic_energy+dissipated_energy)
    history_data["solver_data"].append(solver.data)
    history_data["eigs"].append(stability.data["eigs"])
    history_data["stable"].append(stability.data["stable"])

    with XDMFFile(comm, f"{prefix}/{_nameExp}.xdmf", "a", encoding=XDMFFile.Encoding.HDF5) as file:
        file.write_function(u, t)
        file.write_function(alpha, t)

    if comm.rank == 0:
        a_file = open(f"{prefix}/{_nameExp}-data.json", "w")
        json.dump(history_data, a_file)
        a_file.close()

    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

import pandas as pd
df = pd.DataFrame(history_data)
print(df)

if comm.rank == 0:
    plot_energies(history_data, file=f"{prefix}_energies.pdf")

# Viz