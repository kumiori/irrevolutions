#!/usr/bin/env python3
from re import A
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
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from models import DamageElasticityModel as Brittle
from algorithms.am import AlternateMinimisation

from meshes.pacman import mesh_pacman
from dolfinx.common import Timer, list_timings, TimingType

from ufl import Circumradius, FacetNormal, SpatialCoordinate

import logging

logging.basicConfig(level=logging.INFO)

import dolfinx
import dolfinx.plot
from dolfinx.io import XDMFFile, gmshio
from dolfinx.fem import (
    Constant,
    Expression,
    Function,
    FunctionSpace,
    assemble_scalar,
    dirichletbc,
    form,
    locate_dofs_geometrical,
    set_bc,
)
import dolfinx.mesh
from dolfinx.mesh import CellType, locate_entities_boundary

import ufl

from mpi4py import MPI
import petsc4py
from petsc4py import PETSc
import sys
import yaml

sys.path.append("../")
from solvers import SNESSolver
from algorithms.so import StabilitySolver

# ///////////


def plot_mesh(mesh, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.set_aspect("equal")
    points = mesh.geometry.x
    cells = mesh.geometry.dofmap.array.reshape((-1, mesh.topology.dim + 1))
    tria = tri.Triangulation(points[:, 0], points[:, 1], cells)
    ax.triplot(tria, color="k")
    return ax


petsc4py.init(sys.argv)
comm = MPI.COMM_WORLD

# Mesh on node model_rank and then distribute
model_rank = 0

with open("output/test_notch/parameters.yml") as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)


# Get mesh parameters
_r = parameters["geometry"]["r"]
_omega = parameters["geometry"]["omega"]
tdim = parameters["geometry"]["geometric_dimension"]
_nameExp = parameters["geometry"]["geom_type"]
ell_ = parameters["model"]["ell"]
lc = ell_ / 3.0

parameters["geometry"]["lc"] = lc


# Get geometry model
geom_type = parameters["geometry"]["geom_type"]

# Create the mesh of the specimen with given dimensions

gmsh_model, tdim = mesh_pacman(geom_type, parameters["geometry"], tdim)

# Get mesh and meshtags
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)


outdir = "output"
prefix = os.path.join(outdir, "test_notch")
if comm.rank == 0:
    Path(prefix).mkdir(parents=True, exist_ok=True)

with XDMFFile(comm, f"{prefix}/{_nameExp}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as file:
    file.write_mesh(mesh)



if comm.rank == 0:
    plt.figure()
    ax = plot_mesh(mesh)
    fig = ax.get_figure()
    fig.savefig(f"{prefix}/mesh.png")


# Function spaces
element_u = ufl.VectorElement("Lagrange", mesh.ufl_cell(), degree=1, dim=tdim)
V_u = FunctionSpace(mesh, element_u)

element_alpha = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)
V_alpha = FunctionSpace(mesh, element_alpha)

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

# Data

uD = Function(V_u)

class Asymptotic(Expression):
    def __init__(self, omega, **kwargs):
        self.omega = omega

    def value_shape(self):
        return (2,)
    
    def eval(self, value, x):
        self.theta = ufl.atan_2(x[1], x[0])
        print(self.theta)
        value[0] = x[0]
        value[1] = x[1]
        

bd_facets = locate_entities_boundary(
    mesh, dim=1, marker=lambda x: np.isclose((x[0]**2 + x[1]**2), _r**2))
    # mesh, dim=1, marker=lambda x: np.greater((x[0]**2 + x[1]**2), .9*_r**2))
    # mesh, dim=1, marker=lambda x: np.greater(x[0], .1))

_asym = Asymptotic(omega = parameters["geometry"]["omega"])

uD.interpolate(_asym, cells = bd_facets)

__import__('pdb').set_trace()

uD.interpolate(lambda x: [np.zeros_like(x[0]), np.zeros_like(x[1])])
uD.interpolate(asymptotic_displ)



# Measures
dx = ufl.Measure("dx", domain=mesh)
ds = ufl.Measure("ds", domain=mesh)

dofs_alpha_left = locate_dofs_geometrical(
    V_alpha, lambda x: np.isclose(x[0], 0.0))
dofs_alpha_right = locate_dofs_geometrical(
    V_alpha, lambda x: np.isclose(x[0], Lx))

dofs_u_left = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], 0.0))
dofs_u_right = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], Lx))
# Set Bcs Function
zero_u.interpolate(lambda x: (np.zeros_like(x[0]), np.zeros_like(x[1])))
zero_alpha.interpolate((lambda x: np.zeros_like(x[0])))
u_.interpolate(lambda x: (np.ones_like(x[0]), 0 * np.ones_like(x[1])))
alpha_lb.interpolate(lambda x: np.zeros_like(x[0]))
alpha_ub.interpolate(lambda x: np.ones_like(x[0]))

for f in [zero_u, zero_alpha, u_, alpha_lb, alpha_ub]:
    f.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                         mode=PETSc.ScatterMode.FORWARD)

bc_u_left = dirichletbc(
    np.array([0, 0], dtype=PETSc.ScalarType), dofs_u_left, V_u)


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


stability = StabilitySolver(
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