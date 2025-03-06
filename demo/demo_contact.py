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
import pandas as pd
import petsc4py
import pyvista
import ufl
import yaml
from dolfinx.common import list_timings
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
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI
from petsc4py import PETSc
from pyvista.plotting.utilities import xvfb
from irrevolutions.algorithms.am import ContactAlternateMinimisation
from irrevolutions.meshes.primitives import create_arc_ring_mesh
from irrevolutions.models import DamageElasticityModel as Brittle
from irrevolutions.utils.plots import plot_energies, plot_force_displacement
from irrevolutions.utils.viz import plot_scalar, plot_vector
import basix.ufl

logging.basicConfig(level=logging.INFO)

petsc4py.init(sys.argv)
comm = MPI.COMM_WORLD

# Mesh on node model_rank and then distribute
model_rank = 0


with open(os.path.join(os.path.dirname(__file__), "parameters.yml")) as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)

# Get mesh parameters

parameters["geometry"] = {
    "R_outer": 1.0,  # Outer disk radius
    "R_inner": 0.3,  # Inner hole radius (set to 0.0 for no hole)
    "lc": 0.05,  # Mesh element size
    "a": 0.1,  # Half-width of the refined region (-a < x < a)
    "geometric_dimension": 2,
    "geom_type": "ring_arc",
    "angle": 180,
}
_nameExp = "ring_arc"
ell_ = parameters["model"]["ell"]

lc = ell_ / 3.0

parameters["model"]["w1"] = 100

parameters["loading"]["max"] = 1
parameters["loading"]["steps"] = 30

# Get geometry model
geom_type = parameters["geometry"]["geom_type"]

# Create the mesh of the specimen with given dimensions
gmsh_model, tdim = create_arc_ring_mesh(parameters["geometry"])
# Get mesh and meshtags
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)


outdir = os.path.join(os.path.dirname(__file__), "output")
prefix = os.path.join(outdir, "contact")
if comm.rank == 0:
    Path(prefix).mkdir(parents=True, exist_ok=True)

with XDMFFile(
    comm, f"{prefix}/{_nameExp}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5
) as file:
    file.write_mesh(mesh)

# Function spaces
element_u = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1, shape=(tdim,))
V_u = dolfinx.fem.functionspace(mesh, element_u)

element_alpha = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1)
V_alpha = dolfinx.fem.functionspace(mesh, element_alpha)
# Define the state
u = Function(V_u, name="Displacement")
u_lb = dolfinx.fem.Function(V_u, name="Displacement_lb")
u_ub = dolfinx.fem.Function(V_u, name="Displacement_ub")

u_ = Function(V_u, name="Boundary Displacement")
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


def on_outer_boundary(x):
    return np.isclose(
        x[0] ** 2 + x[1] ** 2, parameters["geometry"]["R_outer"] ** 2, atol=1e-5
    )


def bottom_boundary(x):
    return np.isclose(x[1], 0, atol=1e-2)


# Locate DOFs on the outer boundary
outer_dofs_alpha = dolfinx.fem.locate_dofs_geometrical(V_alpha, on_outer_boundary)

bottom_dofs = dolfinx.fem.locate_dofs_topological(
    V_u,
    mesh.topology.dim - 1,
    dolfinx.mesh.locate_entities_boundary(mesh, 1, bottom_boundary),
)

# Set Bcs Function
zero_u.interpolate(lambda x: (np.zeros_like(x[0]), np.zeros_like(x[1])))
zero_alpha.interpolate((lambda x: np.zeros_like(x[0])))
u_.interpolate(lambda x: (np.ones_like(x[0]), 0 * np.ones_like(x[1])))

# Bounds
alpha_lb.interpolate(lambda x: np.zeros_like(x[0]))
alpha_ub.interpolate(lambda x: np.ones_like(x[0]))

g = 1.1 * max(mesh.geometry.x[:, 1])  # The coordinate of the obstacle
u_lb.interpolate(
    lambda x: np.stack([np.full_like(x[0], -np.inf), np.full_like(x[1], -x[1])])
)
u_ub.interpolate(
    lambda x: np.stack([np.full_like(x[0], np.inf), np.full_like(x[1], g - x[1])])
)

for f in [zero_u, zero_alpha, u_, alpha_lb, alpha_ub, u_lb, u_ub]:
    f.x.petsc_vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )

# import pdb; pdb.set_trace()
bottom_disp = dolfinx.fem.Constant(mesh, np.array([0.0, 0.0], dtype=PETSc.ScalarType))
bcs_u = [
    dirichletbc(bottom_disp, bottom_dofs, V_u),
]

bcs_alpha = [
    dirichletbc(
        np.array(0, dtype=PETSc.ScalarType),
        outer_dofs_alpha,
        V_alpha,
    )
]
# bcs_alpha = [
#     dirichletbc(
#         np.array(0, dtype=PETSc.ScalarType),
#         np.concatenate([dofs_alpha_left, dofs_alpha_right]),
#         V_alpha,
#     )
# ]

for bc in bcs_u:
    bc.set(u_lb.x.array)
    bc.set(u_ub.x.array)

set_bc(alpha_ub.x.petsc_vec, bcs_alpha)
alpha_ub.x.petsc_vec.ghostUpdate(
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
loads = np.linspace(load_par["min"], load_par["max"], load_par["steps"])

parameters["solvers"]["contact"] = {
    "type": "SNES",
    "prefix": "contact_",
    "snes": {
        # "snes_type": "vinewtonrsls",
        "snes_type": "vinewtonssls",
        "snes_linesearch_type": "basic",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_atol": 1.0e-8,
        "snes_rtol": 1.0e-8,
        "snes_max_it": 50,
        "snes_monitor": "",
        "snes_vi_monitor": True,
    },
}

equilibrium = ContactAlternateMinimisation(
    total_energy,
    state,
    bcs,
    parameters.get("solvers"),
    bounds_u=(u_lb, u_ub),
    bounds_alpha=(alpha_lb, alpha_ub),
)

# hybrid = HybridSolver(
#     total_energy,
#     state,
#     bcs,
#     bounds=(alpha_lb, alpha_ub),
#     solver_parameters=parameters.get("solvers"),
# )

history_data = {
    "load": [],
    "elastic_energy": [],
    "total_energy": [],
    "fracture_energy": [],
    "solver_data": [],
    # "solver_HY_data": [],
    "F": [],
}

for i_t, t in enumerate(loads):
    # u_.interpolate(lambda x: (t * np.ones_like(x[0]), np.zeros_like(x[1])))
    # u_.x.petsc_vec.ghostUpdate(
    #     addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    # )
    u_ub.interpolate(
        lambda x: np.stack(
            [np.full_like(x[0], np.inf), np.full_like(x[1], g - np.abs(t) - x[1])]
        )
    )
    # update the lower bound
    alpha.x.petsc_vec.copy(alpha_lb.x.petsc_vec)
    alpha_lb.x.petsc_vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )

    logging.critical(f"-- Solving for t = {t:3.2f} --")
    equilibrium.solve()
    # hybrid.solve(alpha_lb)

    fracture_energy = comm.allreduce(
        assemble_scalar(form(model.damage_energy_density(state) * dx)),
        op=MPI.SUM,
    )
    elastic_energy = comm.allreduce(
        assemble_scalar(form(model.elastic_energy_density(state) * dx)),
        op=MPI.SUM,
    )
    _stress = model.stress(model.eps(u), alpha)

    stress = comm.allreduce(
        assemble_scalar(form(_stress[0, 0] * dx)),
        op=MPI.SUM,
    )
    history_data["load"].append(t)
    history_data["fracture_energy"].append(fracture_energy)
    history_data["elastic_energy"].append(elastic_energy)
    history_data["total_energy"].append(elastic_energy + fracture_energy)
    history_data["solver_data"].append([])
    # history_data["solver_HY_data"].append(hybrid.newton_data)
    history_data["F"].append(stress)

    with XDMFFile(
        comm, f"{prefix}/{_nameExp}.xdmf", "a", encoding=XDMFFile.Encoding.HDF5
    ) as file:
        file.write_function(u, t)
        file.write_function(alpha, t)

    if comm.rank == 0:
        a_file = open(f"{prefix}/{_nameExp}-data.json", "w")
        json.dump(history_data, a_file)
        a_file.close()

list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

df = pd.DataFrame(history_data)
print(df)

#
if comm.Get_size() == 1:
    # xvfb.start_xvfb(wait=0.05)
    pyvista.OFF_SCREEN = True

    plotter = pyvista.Plotter(
        title="Displacement",
        window_size=[1600, 600],
        shape=(1, 2),
    )
    _plt, grid = plot_scalar(alpha, plotter, subplot=(0, 0))
    _plt, grid = plot_vector(u, plotter, subplot=(0, 1))
    _plt.screenshot(f"{prefix}/contact-state.png")


if comm.rank == 0:
    plot_energies(history_data, file=f"{prefix}/{_nameExp}_energies.pdf")
    # plot_AMit_load(history_data, file=f"{prefix}/{_nameExp}_it_load.pdf")
    plot_force_displacement(history_data, file=f"{prefix}/{_nameExp}_stress-load.pdf")
