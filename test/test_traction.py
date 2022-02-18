#!/usr/bin/env python3
from pyvista.utilities import xvfb
import pyvista
import sys
sys.path.append("../")
from utils.viz import plot_mesh, plot_vector, plot_scalar
# 
from models import DamageElasticityModel as Brittle
from algorithms.am import AlternateMinimisation
from solvers import SNESSolver
from dolfinx.mesh import CellType
import dolfinx.mesh

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

import dolfinx.io
from dolfinx.common import list_timings

import logging
from meshes.primitives import mesh_bar_gmshapi
from meshes import gmsh_model_to_mesh
from dolfinx.io import XDMFFile
import numpy as np
import yaml
import json
from pathlib import Path
import os
from mpi4py import MPI
import petsc4py
from petsc4py import PETSc
import dolfinx
import dolfinx.plot
from dolfinx import log
import ufl
sys.path.append("../")

# from damage.utils import ColorPrint


logging.basicConfig(level=logging.INFO)


sys.path.append("../")



petsc4py.init(sys.argv)
log.set_log_level(log.LogLevel.WARNING)


comm = MPI.COMM_WORLD

with open("parameters.yml") as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)

# Get mesh parameters
Lx = parameters["geometry"]["Lx"]
Ly = parameters["geometry"]["Ly"]
tdim = parameters["geometry"]["geometric_dimension"]
ell_ = parameters["model"]["ell"]
lc = ell_ / 5.0


# Get geometry model
geom_type = parameters["geometry"]["geom_type"]

# Create the mesh of the specimen with given dimensions
gmsh_model, tdim = mesh_bar_gmshapi(geom_type, Lx, Ly, lc, tdim)

mesh, mts = gmsh_model_to_mesh(
    gmsh_model, cell_data=False, facet_data=True, gdim=2)

outdir = "output"
if comm.rank == 0:
    Path(outdir).mkdir(parents=True, exist_ok=True)
prefix = os.path.join(outdir, "traction")

# check = parameters["solvers"]["damage_elasticity"]["check"]
# if check:
#     check_load = parameters["solvers"]["damage_elasticity"]["check_load"]
#     out_subdir = f"{outdir}/fields_check"
#     if comm.rank == 0:
#         Path(out_subdir).mkdir(parents=True, exist_ok=True)

with XDMFFile(comm, f"{prefix}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as file:
    file.write_mesh(mesh)

# Function spaces
element_u = ufl.VectorElement("Lagrange", mesh.ufl_cell(), degree=1, dim=tdim)
V_u = FunctionSpace(mesh, element_u)

element_alpha = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)
V_alpha = FunctionSpace(mesh, element_alpha)

# Define the state
u = Function(V_u, name="Displacement")
u_ = Function(V_u, name="Boundary Displacement")
ux_ = Function(V_u.sub(0).collapse(), name="Boundary Displacement")
zero_u = Function(V_u, name="   Boundary Displacement")
alpha = Function(V_alpha, name="Damage")
zero_alpha = Function(V_alpha, name="Damage Boundary Field")

state = {"u": u, "alpha": alpha}

# need upper/lower bound for the damage field
alpha_lb = Function(V_alpha, name="Lower bound")
alpha_ub = Function(V_alpha, name="Upper bound")

# Measures
dx = ufl.Measure("dx", domain=mesh)
ds = ufl.Measure("ds", domain=mesh)

dofs_alpha_left = locate_dofs_geometrical(
    V_alpha, lambda x: np.isclose(x[0], 0.0))
dofs_alpha_right = locate_dofs_geometrical(
    V_alpha, lambda x: np.isclose(x[0], Lx))

dofs_u_left = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], 0.0))
dofs_u_right = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], Lx))
dofs_ux_right = locate_dofs_geometrical(
    (V_u.sub(0), V_u.sub(0).collapse()), lambda x: np.isclose(x[0], Lx)
)
# Set Bcs Function
zero_u.interpolate(lambda x: (np.zeros_like(x[0]), np.zeros_like(x[1])))
zero_alpha.interpolate((lambda x: np.zeros_like(x[0])))
u_.interpolate(lambda x: (np.ones_like(x[0]), 0 * np.ones_like(x[1])))
ux_.interpolate(lambda x: np.ones_like(x[0]))
alpha_lb.interpolate(lambda x: np.zeros_like(x[0]))
alpha_ub.interpolate(lambda x: np.ones_like(x[0]))

for f in [zero_u, zero_alpha, u_, ux_, alpha_lb, alpha_ub]:
    f.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                         mode=PETSc.ScatterMode.FORWARD)

bc_u_left = dirichletbc(
    np.array([0, 0], dtype=PETSc.ScalarType), dofs_u_left, V_u)

# import pdb; pdb.set_trace()

bc_u_right = dirichletbc(
    u_, dofs_u_right)
bcs_u = [bc_u_left, bc_u_right]

bcs_alpha = [
    dirichletbc(
        np.array(0, dtype=PETSc.ScalarType),
        np.concatenate([dofs_alpha_left, dofs_alpha_right]),
        V_alpha,
    )
]

set_bc(alpha_ub.vector, bcs_alpha)
alpha_ub.vector.ghostUpdate(
    addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
)


bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}
# Define the model

model = Brittle(parameters["model"])

# Energy functional
f = Constant(mesh, np.array([0, 0], dtype=PETSc.ScalarType))
external_work = ufl.dot(f, state["u"]) * dx
total_energy = model.total_energy_density(state) * dx - external_work

load_par = parameters["loading"]
loads = np.linspace(load_par["min"],
                    load_par["max"], load_par["steps"])

solver = AlternateMinimisation(
    total_energy, state, bcs, parameters.get("solvers"), bounds=(alpha_lb, alpha_ub)
)

history_data = {
    "load": [],
    "elastic_energy": [],
    "total_energy": [],
    "dissipated_energy": [],
    "solver_data": [],
}

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

    with XDMFFile(comm, f"{prefix}.xdmf", "a", encoding=XDMFFile.Encoding.HDF5) as file:
        file.write_function(u, t)
        file.write_function(alpha, t)

    if comm.rank == 0:
        a_file = open(f"{prefix}-data.json", "w")
        json.dump(history_data, a_file)
        a_file.close()

    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

import pandas as pd
df = pd.DataFrame(history_data)
print(df)

# Viz
xvfb.start_xvfb(wait=0.05)
pyvista.OFF_SCREEN = True


plotter = pyvista.Plotter(
    title="Displacement",
    window_size=[1600, 600],
    shape=(1, 2),
)
_plt = plot_scalar(alpha, plotter, subplot=(0, 0))
_plt = plot_vector(u, plotter, subplot=(0, 1))
_plt.screenshot(f"{outdir}/traction-state.png")

tol = 1e-3
xs = np.linspace(0 + tol, Lx - tol, 101)
points = np.zeros((3, 101))
points[0] = xs
ell = parameters.get("model").get("ell")
_plt, data = plot_profile(alpha, points=points,
                        #   plotter=plotter, subplot=(0, 0),
                          lineproperties={
                              "c": "k",
                              "label": "$\\alpha(x)$"
                          },)
ax = _plt.gca()
ax.axvline(Lx/2, c="k", ls=":", label="$x_0=L/2$")
ax.axvline(Lx/2 + 2*ell, c="k", label='D=$4\ell$')
ax.axvline(Lx/2 - 2*ell, c="k")
_plt.legend()
_plt.title(f"Damage profile, traction bar, $\ell$ = {ell:.2f}")
_plt.fill_between(data[0], data[1].reshape(len(data[1])))
_plt.savefig(
    f"{outdir}/traction-profile.png")

# if comm.rank == 0:
#     plot_energies(history_data, file=f"{prefix}_energies.pdf")
#     plot_AMit_load(history_data, file=f"{prefix}_it_load.pdf")
