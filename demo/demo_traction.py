#!/usr/bin/env python3
from dataclasses import asdict
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
    functionspace,
    assemble_scalar,
    dirichletbc,
    form,
    locate_dofs_geometrical,
    set_bc,
)
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI
from petsc4py import PETSc
from irrevolutions.algorithms.am import AlternateMinimisation, HybridSolver
from irrevolutions.contracts import (
    EquilibriumResult,
    ExperimentSetup,
    History,
    Manifest,
    StepRecord,
    get_bounds_pair,
    legacy_bcs_from_contract,
    make_field_bounds,
    normalise_bcs,
)
from irrevolutions.meshes.primitives import mesh_bar_gmshapi
from irrevolutions.models import DamageElasticityModel as Brittle
from irrevolutions.utils.plots import plot_energies, plot_force_displacement
from irrevolutions.utils.viz import (
    plot_scalar,
    plot_vector,
    safe_screenshot,
    setup_pyvista_offscreen,
)
import basix.ufl

logging.basicConfig(level=logging.INFO)


# ///////////


petsc4py.init(sys.argv)
comm = MPI.COMM_WORLD

# Mesh on node model_rank and then distribute
model_rank = 0


def _float_or_none(value):
    if value is None:
        return None
    if np.isscalar(value) and np.isnan(value):
        return None
    return float(value)


with open(os.path.join(os.path.dirname(__file__), "parameters.yml")) as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)

# Get mesh parameters
Lx = parameters["geometry"]["Lx"]
Ly = parameters["geometry"]["Ly"]
tdim = parameters["geometry"]["geometric_dimension"]
_nameExp = parameters["geometry"]["geom_type"]
_nameExp = "bar"
ell_ = parameters["model"]["ell"]


lc = ell_ / 3.0

parameters["loading"]["max"] = 3
parameters["loading"]["steps"] = 100

# Get geometry model
geom_type = parameters["geometry"]["geom_type"]

# Create the mesh of the specimen with given dimensions
gmsh_model, tdim = mesh_bar_gmshapi(geom_type, Lx, Ly, lc, tdim)

# Get mesh and meshtags
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)


outdir = os.path.join(os.path.dirname(__file__), "output")
prefix = os.path.join(outdir, "traction")
if comm.rank == 0:
    Path(prefix).mkdir(parents=True, exist_ok=True)

with XDMFFile(
    comm, f"{prefix}/{_nameExp}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5
) as file:
    file.write_mesh(mesh)

# Function spaces
element_u = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1, shape=(tdim,))
V_u = functionspace(mesh, element_u)

element_alpha = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1)
V_alpha = functionspace(mesh, element_alpha)
# Define the state
u = Function(V_u, name="Displacement")
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

dofs_alpha_left = locate_dofs_geometrical(V_alpha, lambda x: np.isclose(x[0], 0.0))
dofs_alpha_right = locate_dofs_geometrical(V_alpha, lambda x: np.isclose(x[0], Lx))

dofs_u_left = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], 0.0))
dofs_u_right = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], Lx))
# Set Bcs Function
zero_u.interpolate(lambda x: (np.zeros_like(x[0]), np.zeros_like(x[1])))
zero_alpha.interpolate((lambda x: np.zeros_like(x[0])))
u_.interpolate(lambda x: (np.ones_like(x[0]), 0 * np.ones_like(x[1])))
alpha_lb.interpolate(lambda x: np.zeros_like(x[0]))
alpha_ub.interpolate(lambda x: np.ones_like(x[0]))

for f in [zero_u, zero_alpha, u_, alpha_lb, alpha_ub]:
    f.x.petsc_vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )

bc_u_left = dirichletbc(np.array([0, 0], dtype=PETSc.ScalarType), dofs_u_left, V_u)

# import pdb; pdb.set_trace()

bc_u_right = dirichletbc(u_, dofs_u_right)
bcs_u = [bc_u_left, bc_u_right]

# bcs_alpha = [
#     dirichletbc(
#         np.array(0, dtype=PETSc.ScalarType),
#         np.concatenate([dofs_alpha_left, dofs_alpha_right]),
#         V_alpha,
#     )
# ]

bcs_alpha = []

set_bc(alpha_ub.x.petsc_vec, bcs_alpha)
alpha_ub.x.petsc_vec.ghostUpdate(
    addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
)


bcs = normalise_bcs(
    {
        "u": {
            "dirichlet": bcs_u,
            "loading": {
                "type": "displacement_control",
                "parameter": None,
                "component": 0,
                "region": "right",
            },
        },
        "alpha": {"dirichlet": bcs_alpha, "loading": None},
    }
)
# Define the model

model = Brittle(parameters["model"])

# Energy functional
f = Constant(mesh, np.array([0, 0], dtype=PETSc.ScalarType))
external_work = ufl.dot(f, state["u"]) * dx
total_energy = model.total_energy_density(state) * dx - external_work

load_par = parameters["loading"]
loads = np.linspace(load_par["min"], load_par["max"], load_par["steps"])

bounds = {"alpha": make_field_bounds(alpha_lb, alpha_ub)}
setup = ExperimentSetup(
    state=state,
    bcs=bcs,
    bounds=bounds,
    parameters=parameters,
    energy=total_energy,
    mesh=mesh,
    spaces={"u": V_u, "alpha": V_alpha},
    metadata={"geom_type": _nameExp},
)
solver_bcs = legacy_bcs_from_contract(setup.bcs)
alpha_bounds = get_bounds_pair(setup.bounds, "alpha")
history = History()
manifest = Manifest(
    parameters=parameters,
    mesh={"cell_name": mesh.topology.cell_name(), "tdim": mesh.topology.dim},
    spaces={"u": str(V_u.element), "alpha": str(V_alpha.element)},
)

solver = AlternateMinimisation(
    total_energy, state, solver_bcs, parameters.get("solvers"), bounds=alpha_bounds
)

hybrid = HybridSolver(
    total_energy,
    state,
    solver_bcs,
    bounds=alpha_bounds,
    solver_parameters=parameters.get("solvers"),
)

for i_t, t in enumerate(loads):
    u_.interpolate(lambda x: (t * np.ones_like(x[0]), np.zeros_like(x[1])))
    u_.x.petsc_vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )

    # update the lower bound
    alpha.x.petsc_vec.copy(alpha_lb.x.petsc_vec)
    alpha_lb.x.petsc_vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )

    logging.critical(f"-- Solving for t = {t:3.2f} --")

    hybrid.solve(alpha_lb)

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
    equilibrium_result = EquilibriumResult(
        step=i_t,
        load=float(t),
        time=float(t),
        state=setup.state,
        bounds=setup.bounds,
        converged=True,
        solver_name="hybrid",
        iterations=(
            hybrid.data["iteration"][-1] if hybrid.data["iteration"] else None
        ),
        residual_norm=(
            float(hybrid.data["error_residual_F"][-1])
            if hybrid.data["error_residual_F"]
            else None
        ),
        total_energy=elastic_energy + fracture_energy,
    )
    history.append(
        StepRecord(
            step=equilibrium_result.step,
            load=equilibrium_result.load,
            time=equilibrium_result.time,
            elastic_energy=elastic_energy,
            fracture_energy=fracture_energy,
            total_energy=equilibrium_result.total_energy,
            solver_converged=True,
            n_iterations=equilibrium_result.iterations,
            extra={
                "solver_data": [],
                "solver_HY_data": hybrid.newton_data,
                "F": stress,
            },
        )
    )
    history_data = history.to_columns()

    with XDMFFile(
        comm, f"{prefix}/{_nameExp}.xdmf", "a", encoding=XDMFFile.Encoding.HDF5
    ) as file:
        file.write_function(u, t)
        file.write_function(alpha, t)

    if comm.rank == 0:
        a_file = open(f"{prefix}/{_nameExp}-data.json", "w")
        json.dump(history_data, a_file, default=str)
        a_file.close()

if comm.rank == 0:
    with open(f"{prefix}/{_nameExp}-manifest.json", "w") as file:
        json.dump(asdict(manifest), file, default=str)

list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

df = pd.DataFrame(history_data)
print(df)

#
if comm.Get_size() == 1:
    setup_pyvista_offscreen()

    plotter = pyvista.Plotter(
        title="Displacement",
        window_size=[1600, 600],
        shape=(1, 2),
    )
    plotter, _ = plot_scalar(alpha, plotter, subplot=(0, 0))
    plotter, _ = plot_vector(u, plotter, subplot=(0, 1))
    safe_screenshot(plotter, f"{prefix}/traction-state.png")


if comm.rank == 0:
    plot_energies(history_data, file=f"{prefix}/{_nameExp}_energies.pdf")
    # plot_AMit_load(history_data, file=f"{prefix}/{_nameExp}_it_load.pdf")
    plot_force_displacement(history_data, file=f"{prefix}/{_nameExp}_stress-load.pdf")
