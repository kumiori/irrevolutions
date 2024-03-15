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
import ufl
import numpy as np
sys.path.append("../")
from utils.plots import plot_energies
from irrevolutions.utils import ColorPrint
import matplotlib.pyplot as plt

from models import DamageElasticityModel as Brittle
from algorithms.am import AlternateMinimisation

from meshes.pacman import mesh_pacman
from dolfinx.common import list_timings


import logging

# logging.basicConfig(level=logging.INFO)
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

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
    set_bc,
)
import dolfinx.mesh
from dolfinx.mesh import locate_entities_boundary

import ufl
import pyvista
from pyvista.utilities import xvfb

from mpi4py import MPI
import petsc4py
from petsc4py import PETSc
import sys
import yaml

sys.path.append("../")
from algorithms.so import BifurcationSolver

# ///////////

from utils.viz import (
    plot_mesh,
    plot_scalar,
    plot_vector
)


petsc4py.init(sys.argv)
comm = MPI.COMM_WORLD

# Mesh on node model_rank and then distribute
model_rank = 0

outdir = os.path.join(os.path.dirname(__file__), "output")
prefix = os.path.join(outdir, "test_notch")


with open(f"{prefix}/parameters.yml") as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)


# Get mesh parameters
_r = parameters["geometry"]["r"]
_omega = parameters["geometry"]["omega"]
tdim = parameters["geometry"]["geometric_dimension"]
_nameExp = parameters["geometry"]["geom_type"]
_nameExp = 'pacman'
ell_ = parameters["model"]["ell"]
lc = ell_ / 1.

parameters["geometry"]["lc"] = lc

parameters["loading"]["min"] = 0.
parameters["loading"]["max"] = .5
# Get geometry model
geom_type = parameters["geometry"]["geom_type"]

# Create the mesh of the specimen with given dimensions

gmsh_model, tdim = mesh_pacman(geom_type, parameters["geometry"], tdim)

# Get mesh and meshtags
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)


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

uD = Function(V_u, name="Asymptotic Notch Displacement")

import sympy as sp

from dolfinx.fem import (
    Constant,
    Function,
    FunctionSpace,
    assemble_scalar,
    dirichletbc,
    form,
    locate_dofs_topological,
    set_bc,
)
def singularity_exp(omega):
    """Exponent of singularity, λ\in [1/2, 1]
    lmbda : = sin(2*lmbda*(pi - omega)) + lmbda*sin(2(pi-lmbda)) = 0"""
    from sympy import nsolve, pi, sin, symbols

    x = symbols('x')

    return nsolve(
        sin(2*x*(pi - omega)) + x*sin(2*(pi-omega)), 
        x, .5)

def ext_boundary_marker(x):
    return np.isclose(x[0]**2. + x[1]**2. - _r**2, 0., atol = 1.e-4)

ext_bd_facets = locate_entities_boundary(
    mesh, dim=1, marker=lambda x: np.isclose(x[0]**2. + x[1]**2. - _r**2, 0., atol = 1.e-4)
    )

# boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
boundary_dofs_u = locate_dofs_topological(V_u, mesh.topology.dim - 1, ext_bd_facets)
boundary_dofs_alpha = locate_dofs_topological(V_alpha, mesh.topology.dim - 1, ext_bd_facets)

# locate_dofs_geometrical(V_alpha, ext_boundary_marker)
# locate_dofs_geometrical(V_u, ext_boundary_marker)

# also: boundary_facets = locate_entities_boundary(mesh, dim=1, marker = lambda x: np.full(x.shape[1], True, dtype=bool))


def _local_notch_asymptotic(x, ω=np.deg2rad(_omega / 2.), t=1., par = parameters["material"]):
    from sympy import cos, pi, pi, sin, symbols
    λ = singularity_exp(ω)
    Θ = symbols('Θ')
    _E = par['E']
    ν = par['ν']
    Θv = np.arctan2(x[1], x[0])
        
    coeff = ( (1+λ) * sin( (1+λ) * (pi - ω) ) ) / ( (1-λ) * sin( (1-λ) * (pi - ω) ) )

    _f = (2*np.pi)**(λ - 1) * ( cos( (1+λ) * Θ) - coeff * cos((1-λ) * Θ) ) / (1-coeff)

    f = sp.lambdify(Θ, _f, "numpy")
    fp = sp.lambdify(Θ, sp.diff(_f, Θ, 1), "numpy")
    fpp = sp.lambdify(Θ, sp.diff(_f, Θ, 2), "numpy")
    fppp = sp.lambdify(Θ, sp.diff(_f, Θ, 3), "numpy")

    # print("F(0)", f(0))
    # print("F(pi - ω)", f(np.float16(pi.n() - ω)))
    # print("F(-pi + ω)", f(np.float16(-pi.n() + ω)))

    # __import__('pdb').set_trace()
    # assert(np.isclose(f(np.float16(pi.n() - ω)), 0., atol=1.0e-5))
    # assert(np.isclose(f(np.float16(-pi.n() + ω)), 0., atol=1.0e-5))

    r = np.sqrt(x[0]**2. + x[1]**2.)
    _c1 = (λ+1)*(1- ν*λ - ν**2.*(λ+1))
    _c2 = 1-ν**2.
    _c3 = 2.*(1+ν)*λ**2. + _c1
    _c4 = _c2
    _c5 = λ**2. * (1-λ**2.)

    ur = t * ( r**λ / _E * (_c1*f(Θv) + _c2*fpp(Θv)) ) / _c5
    uΘ = t * ( r**λ / _E * (_c3*fp(Θv) + _c4*fppp(Θv)) ) / _c5

    values = np.zeros((tdim, x.shape[1]))
    values[0] = ur * np.cos(Θv) - uΘ * np.sin(Θv)
    values[1] = ur * np.sin(Θv) + uΘ * np.cos(Θv)
    return values

uD.interpolate(_local_notch_asymptotic)

# Measures
dx = ufl.Measure("dx", domain=mesh)
ds = ufl.Measure("ds", domain=mesh)

# Set Bcs Function
zero_u.interpolate(lambda x: (np.zeros_like(x[0]), np.zeros_like(x[1])))
zero_alpha.interpolate((lambda x: np.zeros_like(x[0])))
alpha_lb.interpolate(lambda x: np.zeros_like(x[0]))
alpha_ub.interpolate(lambda x: np.ones_like(x[0]))

for f in [zero_u, zero_alpha, alpha_lb, alpha_ub]:
    f.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                         mode=PETSc.ScatterMode.FORWARD)

bcs_u = [dirichletbc(value=uD, dofs=boundary_dofs_u)]

bcs_alpha = [
    dirichletbc(
        np.array(0, dtype=PETSc.ScalarType),
        boundary_dofs_alpha,
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

stability = BifurcationSolver(
    total_energy, state, bcs, stability_parameters=parameters.get("stability")
)

history_data = {
    "load": [],
    "elastic_energy": [],
    "fracture_energy": [],
    "total_energy": [],
    "solver_data": [],
    "eigs": [],
    "stable": [],
    "residual_F": [],
}

check_stability = []

for i_t, t in enumerate(loads):

    uD.interpolate(lambda x: _local_notch_asymptotic(
        x,
        ω=np.deg2rad(_omega / 2.),
        t=t,
        par = parameters["material"]
    ))

    uD.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                          mode=PETSc.ScatterMode.FORWARD)

    # update the lower bound
    alpha.vector.copy(alpha_lb.vector)
    alpha_lb.vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )

    logging.critical(f"-- Solving for t = {t:3.2f} --")

    solver.solve(outdir=prefix)

    n_eigenvalues = 10
    is_stable = stability.solve(alpha_lb, n_eigenvalues)
    is_elastic = stability.is_elastic()
    inertia = stability.get_inertia()
    stability.save_eigenvectors(filename=f"{prefix}_eigv_{t:3.2f}.xdmf")
    check_stability.append(is_stable)

    ColorPrint.print_bold(f"State is elastic: {is_elastic}")
    ColorPrint.print_bold(f"State's inertia: {inertia}")
    ColorPrint.print_bold(f"State is stable: {is_stable}")

    fracture_energy = comm.allreduce(
        assemble_scalar(form(model.damage_energy_density(state) * dx)),
        op=MPI.SUM,
    )
    elastic_energy = comm.allreduce(
        assemble_scalar(form(model.elastic_energy_density(state) * dx)),
        op=MPI.SUM,
    )

    history_data["load"].append(t)
    history_data["fracture_energy"].append(fracture_energy)
    history_data["elastic_energy"].append(elastic_energy)
    history_data["total_energy"].append(elastic_energy+fracture_energy)
    history_data["solver_data"].append(solver.data)
    history_data["eigs"].append(stability.data["eigs"])
    history_data["stable"].append(stability.data["stable"])
    history_data["residual_F"].append(solver.data["error_residual_F"])

    # with XDMFFile(comm, f"{prefix}/{_nameExp}.xdmf", "a", encoding=XDMFFile.Encoding.HDF5) as file:
    #     file.write_function(u, t)
    #     file.write_function(alpha, t)

    if comm.rank == 0:
        a_file = open(f"{prefix}/history-data.json", "w")
        json.dump(history_data, a_file)
        a_file.close()

    xvfb.start_xvfb(wait=0.05)
    pyvista.OFF_SCREEN = True

    plotter = pyvista.Plotter(
        title="Local notch",
        window_size=[1600, 600],
        shape=(1, 2),
    )

    _plt = plot_scalar(alpha, plotter, subplot=(0, 0))
    logging.critical('plotted scalar')
    _plt = plot_vector(u, plotter, subplot=(0, 1))
    logging.critical('plotted vector')

    _plt.screenshot(f"{prefix}/fields-{i_t}.png")


list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

import pandas as pd
df = pd.DataFrame(history_data)
print(df)

if comm.rank == 0:
    plot_energies(history_data, file=f"{prefix}/energies.pdf")

# Viz