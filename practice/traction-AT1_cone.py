#!/usr/bin/env python3
from dataclasses import asdict
from irrevolutions.utils.viz import plot_scalar, plot_vector
from pyvista.plotting.utilities import xvfb
import pyvista
from irrevolutions.utils.plots import plot_AMit_load, plot_force_displacement
import hashlib
from irrevolutions.utils import ColorPrint
from irrevolutions.utils.compat import initial_mode_from_spectrum
from irrevolutions.utils.plots import plot_energies
from irrevolutions.models import DamageElasticityModel as Brittle
from irrevolutions.meshes.primitives import mesh_bar_gmshapi
from irrevolutions.algorithms.so import BifurcationSolver, StabilitySolver
from irrevolutions.algorithms.am import AlternateMinimisation, HybridSolver
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
import basix.ufl

from _paths import repo_path
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


"""Traction endommageable bar

0|WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW|========> t


[WWW]: endommageable bar, alpha
load: displacement hard-t

"""


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

with open(repo_path("test", "parameters.yml")) as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)

# parameters["cone"] = ""
parameters["stability"]["cone"]["cone_max_it"] = 400000
parameters["stability"]["cone"]["cone_atol"] = 1e-6
parameters["stability"]["cone"]["cone_rtol"] = 1e-5
parameters["stability"]["cone"]["scaling"] = 0.3

parameters["model"]["ell"] = 0.1
parameters["model"]["model_dimension"] = 2
parameters["model"]["model_type"] = "2D"
parameters["model"]["w1"] = 1
parameters["model"]["k_res"] = 0.0

parameters["loading"]["min"] = 0.98
parameters["loading"]["max"] = 1.4
parameters["loading"]["steps"] = 100

parameters["geometry"]["geom_type"] = "traction-bar"
parameters["geometry"]["ell_lc"] = 3
# Get mesh parameters
Lx = parameters["geometry"]["Lx"]
Ly = parameters["geometry"]["Ly"]
tdim = parameters["geometry"]["geometric_dimension"]

_nameExp = parameters["geometry"]["geom_type"]
ell_ = parameters["model"]["ell"]
# lc = ell_ / 5.0

# Get geometry model
geom_type = parameters["geometry"]["geom_type"]

# Create the mesh of the specimen with given dimensions

outdir = os.path.join(os.path.dirname(__file__), "output")
prefix = os.path.join(outdir, "traction_AT1_cone")

if comm.rank == 0:
    Path(prefix).mkdir(parents=True, exist_ok=True)
_lc = ell_ / parameters["geometry"]["ell_lc"]
# _lc = Lx/2

gmsh_model, tdim = mesh_bar_gmshapi(geom_type, Lx, Ly, _lc, tdim)

# Get mesh and meshtags
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)


signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

if comm.rank == 0:
    with open(f"{prefix}/parameters.yaml", "w") as file:
        yaml.dump(parameters, file)

if comm.rank == 0:
    with open(f"{prefix}/signature.md5", "w") as f:
        f.write(signature)

with XDMFFile(
    comm, f"{prefix}/{_nameExp}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5
) as file:
    file.write_mesh(mesh)

# Functional Setting

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
alphadot = dolfinx.fem.Function(V_alpha, name="Damage rate")

state = {"u": u, "alpha": alpha}

z = [u, alpha]
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

bcs_alpha = []
# bcs_alpha = [
#     dolfinx.fem.dirichletbc(zero_alpha, dofs_alpha_left),
#     dolfinx.fem.dirichletbc(zero_alpha, dofs_alpha_right),
# ]

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

# Pack state
state = {"u": u, "alpha": alpha}

# Material behaviour

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
    run_id=signature,
    solver_options={
        "solvers": parameters.get("solvers"),
        "stability": parameters.get("stability"),
    },
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

bifurcation = BifurcationSolver(
    total_energy, state, solver_bcs, bifurcation_parameters=parameters.get("stability")
)

cone = StabilitySolver(
    total_energy, state, solver_bcs, cone_parameters=parameters.get("stability")
)


# logging.basicConfig(level=logging.INFO)
# logging.getLogger().setLevel(logging.ERROR)
# logging.getLogger().setLevel(logging.INFO)
# logging.getLogger().setLevel(logging.DEBUG)

for i_t, t in enumerate(loads):
    # for i_t, t in enumerate([0., .99, 1.0, 1.01]):
    u_.interpolate(lambda x: (t * np.ones_like(x[0]), np.zeros_like(x[1])))
    u_.x.petsc_vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )

    # update the lower bound
    alpha.x.petsc_vec.copy(alpha_lb.x.petsc_vec)
    alpha_lb.x.petsc_vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )

    logging.critical("--  --")
    logging.critical("")
    logging.critical("")
    logging.critical("")

    ColorPrint.print_bold("   Solving first order: AM   ")
    ColorPrint.print_bold("===================-=========")

    logging.critical(f"-- {i_t}/{len(loads)}: Solving for t = {t:3.2f} --")

    solver.solve()

    ColorPrint.print_bold("   Solving first order: Hybrid   ")
    ColorPrint.print_bold("===================-=============")

    logging.info(f"-- {i_t}/{len(loads)}: Solving for t = {t:3.2f} --")
    hybrid.solve(alpha_lb)

    # compute the rate
    alpha.x.petsc_vec.copy(alphadot.x.petsc_vec)
    alphadot.x.petsc_vec.axpy(-1, alpha_lb.x.petsc_vec)
    alphadot.x.petsc_vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )

    logging.info(f"alpha vector norm: {alpha.x.petsc_vec.norm()}")
    logging.info(f"alpha lb norm: {alpha_lb.x.petsc_vec.norm()}")
    logging.info(f"alphadot norm: {alphadot.x.petsc_vec.norm()}")
    logging.info(f"vector norms [u, alpha]: {[zi.x.petsc_vec.norm() for zi in z]}")

    rate_12_norm = hybrid.scaled_rate_norm(alpha, parameters)
    urate_12_norm = hybrid.unscaled_rate_norm(alpha)
    logging.info(f"scaled rate state_12 norm: {rate_12_norm}")
    logging.info(f"unscaled scaled rate state_12 norm: {urate_12_norm}")

    ColorPrint.print_bold("   Solving second order: Rate Pb.    ")
    ColorPrint.print_bold("===================-=================")

    # n_eigenvalues = 10
    is_stable = bifurcation.solve(alpha_lb)
    is_elastic = bifurcation.is_elastic()
    inertia = bifurcation.get_inertia()
    # bifurcation.save_eigenvectors(filename=f"{prefix}/{_nameExp}_eigv_{t:3.2f}.xdmf")

    ColorPrint.print_bold(f"State is elastic: {is_elastic}")
    ColorPrint.print_bold(f"State's inertia: {inertia}")
    # ColorPrint.print_bold(f"State is stable: {is_stable}")

    ColorPrint.print_bold("   Solving second order: Cone Pb.    ")
    ColorPrint.print_bold("===================-=================")

    z0 = initial_mode_from_spectrum(bifurcation._spectrum)
    stable = cone.solve(alpha_lb, eig0=z0, inertia=inertia)

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
    _unique = True if inertia[0] == 0 and inertia[1] == 0 else False

    equilibrium_result = EquilibriumResult(
        step=i_t,
        load=float(t),
        time=float(t),
        state=setup.state,
        bounds=setup.bounds,
        converged=True,
        solver_name="alternate_minimisation",
        iterations=(
            solver.data["iteration"][-1] if solver.data["iteration"] else None
        ),
        residual_norm=(
            float(solver.data["error_residual_F"][-1])
            if solver.data["error_residual_F"]
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
            inertia=inertia,
            stability_attempted=True,
            stability_converged=stable is not None,
            stable=stable,
            lambda_stab_min=_float_or_none(cone.data.get("lambda_0")),
            bifurcation_attempted=True,
            bifurcation_converged=True,
            unique=_unique,
            lambda_bif_min=(
                min(float(np.real(value)) for value in bifurcation.data.get("eigs", []))
                if bifurcation.data.get("eigs")
                else None
            ),
            extra={
                "solver_data": solver.data,
                "equilibrium_data": solver.data,
                "eigs": bifurcation.data.get("eigs", []),
                "F": stress,
                "cone_data": cone.data,
                "alphadot_norm": alphadot.x.petsc_vec.norm(),
                "rate_12_norm": rate_12_norm,
                "unscaled_rate_12_norm": urate_12_norm,
                "cone-stable": stable,
                "cone-eig": cone.data.get("lambda_0", np.nan),
                "uniqueness": _unique,
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
        a_file = open(f"{prefix}/time_data.json", "w")
        json.dump(history_data, a_file, default=str)
        a_file.close()

    ColorPrint.print_bold("   Written timely data.    ")
    print()
    print()
    print()
    print()
list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])
# print(history_data)


history_data = history.to_columns()
df = pd.DataFrame(history_data)
print(df.drop(["solver_data", "cone_data"], axis=1))

# Viz


if comm.rank == 0:
    plot_energies(history_data, file=f"{prefix}/{_nameExp}_energies.pdf")
    plot_AMit_load(history_data, file=f"{prefix}/{_nameExp}_it_load.pdf")
    plot_force_displacement(history_data, file=f"{prefix}/{_nameExp}_stress-load.pdf")
    with open(f"{prefix}/manifest.json", "w") as file:
        json.dump(asdict(manifest), file, default=str)


#
xvfb.start_xvfb(wait=0.05)
pyvista.OFF_SCREEN = True


plotter = pyvista.Plotter(
    title="Traction test",
    window_size=[1600, 600],
    shape=(1, 2),
)
_plt = plot_scalar(alpha, plotter, subplot=(0, 0))
_plt = plot_vector(u, plotter, subplot=(0, 1))
_plt.screenshot(f"{prefix}/traction-state.png")
