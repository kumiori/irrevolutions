#!/usr/bin/env python3
import pdb
import pandas as pd
import numpy as np
from sympy import derive_by_array
import yaml
import json
from pathlib import Path
import sys
import os

from dolfinx.fem import locate_dofs_geometrical, dirichletbc
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
from mpi4py import MPI
import petsc4py
from petsc4py import PETSc
import dolfinx
import dolfinx.plot
from dolfinx import log
import ufl

from dolfinx.fem.petsc import (
    set_bc,
    )
from dolfinx.io import XDMFFile, gmshio
import logging
from dolfinx.common import Timer, list_timings, TimingType

sys.path.append("../")
from models import DamageElasticityModel as Brittle
from algorithms.am import AlternateMinimisation, HybridFractureSolver
from algorithms.so import BifurcationSolver, StabilitySolver
from meshes.primitives import mesh_bar_gmshapi
from utils import ColorPrint
from utils.plots import plot_energies
from utils import norm_H1, norm_L2


"""Traction endommageable bar

0|WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW|========> t


[WWW]: endommageable bar, alpha
load: displacement hard-t

"""


petsc4py.init(sys.argv)
comm = MPI.COMM_WORLD

# Mesh on node model_rank and then distribute
model_rank = 0

with open("../test/parameters.yml") as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)

# parameters["cone"] = ""
parameters["stability"]["cone"]["cone_max_it"] = 400000
parameters["stability"]["cone"]["cone_atol"] = 1e-6
parameters["stability"]["cone"]["cone_rtol"] = 1e-5
parameters["stability"]["cone"]["scaling"] = 0.3

parameters["model"]["ell"] = .1
parameters["model"]["model_dimension"] = 2
parameters["model"]["model_type"] = '1D'
parameters["model"]["w1"] = 1
parameters["model"]["k_res"] = 0.

parameters["loading"]["min"] = .98
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

outdir = "output"
prefix = os.path.join(outdir, "traction_AT1_cone")

if comm.rank == 0:
    Path(prefix).mkdir(parents=True, exist_ok=True)
_lc = ell_ / parameters["geometry"]["ell_lc"] 
# _lc = Lx/2

gmsh_model, tdim = mesh_bar_gmshapi(geom_type, Lx, Ly, _lc, tdim)

# Get mesh and meshtags
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)


import hashlib
signature = hashlib.md5(str(parameters).encode('utf-8')).hexdigest()

if comm.rank == 0:
    with open(f"{prefix}/parameters.yaml", 'w') as file:
        yaml.dump(parameters, file)

if comm.rank == 0:
    with open(f"{prefix}/signature.md5", 'w') as f:
        f.write(signature)

with XDMFFile(comm, f"{prefix}/{_nameExp}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as file:
    file.write_mesh(mesh)

# Functional Setting

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
alphadot = dolfinx.fem.Function(V_alpha, name="Damage rate")

state = {"u": u, "alpha": alpha}

z = [u, alpha]
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

# import pdb; pdb.set_trace()

bc_u_right = dirichletbc(
    u_, dofs_u_right)
bcs_u = [bc_u_left, bc_u_right]

bcs_alpha = []
# bcs_alpha = [
#     dolfinx.fem.dirichletbc(zero_alpha, dofs_alpha_left),
#     dolfinx.fem.dirichletbc(zero_alpha, dofs_alpha_right),
# ]

set_bc(alpha_ub.vector, bcs_alpha)
alpha_ub.vector.ghostUpdate(
    addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
)

bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}
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
loads = np.linspace(load_par["min"],
                    load_par["max"], load_par["steps"])

solver = AlternateMinimisation(
    total_energy, state, bcs, parameters.get("solvers"), 
    bounds=(alpha_lb, alpha_ub)
)

hybrid = HybridFractureSolver(
    total_energy,
    state,
    bcs,
    bounds=(alpha_lb, alpha_ub),
    solver_parameters=parameters.get("solvers"),
)

bifurcation = BifurcationSolver(
    total_energy, state, bcs, bifurcation_parameters=parameters.get("stability")
)

cone = StabilitySolver(
    total_energy, state, bcs,
    cone_parameters=parameters.get("stability")
)

history_data = {
    "load": [],
    "elastic_energy": [],
    "fracture_energy": [],
    "total_energy": [],
    "solver_data": [],
    "cone_data": [],
    "cone-eig": [],
    "eigs": [],
    "uniqueness": [],
    "inertia": [],
    "F": [],    
    "alphadot_norm" : [],
    "rate_12_norm" : [], 
    "unscaled_rate_12_norm" : [],
    "cone-stable": []
}


# logging.basicConfig(level=logging.INFO)
# logging.getLogger().setLevel(logging.ERROR)
# logging.getLogger().setLevel(logging.INFO)
# logging.getLogger().setLevel(logging.DEBUG)

for i_t, t in enumerate(loads):
# for i_t, t in enumerate([0., .99, 1.0, 1.01]):
    u_.interpolate(lambda x: (t * np.ones_like(x[0]),  np.zeros_like(x[1])))
    u_.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                          mode=PETSc.ScatterMode.FORWARD)

    # update the lower bound
    alpha.vector.copy(alpha_lb.vector)
    alpha_lb.vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )

    logging.critical("--  --")
    logging.critical("")
    logging.critical("")
    logging.critical("")
    
    ColorPrint.print_bold(f"   Solving first order: AM   ")
    ColorPrint.print_bold(f"===================-=========")


    logging.critical(f"-- {i_t}/{len(loads)}: Solving for t = {t:3.2f} --")

    solver.solve()

    ColorPrint.print_bold(f"   Solving first order: Hybrid   ")
    ColorPrint.print_bold(f"===================-=============")

    logging.info(f"-- {i_t}/{len(loads)}: Solving for t = {t:3.2f} --")
    hybrid.solve(alpha_lb)

    # compute the rate
    alpha.vector.copy(alphadot.vector)
    alphadot.vector.axpy(-1, alpha_lb.vector)
    alphadot.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

    logging.info(f"alpha vector norm: {alpha.vector.norm()}")
    logging.info(f"alpha lb norm: {alpha_lb.vector.norm()}")
    logging.info(f"alphadot norm: {alphadot.vector.norm()}")
    logging.info(f"vector norms [u, alpha]: {[zi.vector.norm() for zi in z]}")

    rate_12_norm = hybrid.scaled_rate_norm(alpha, parameters)
    urate_12_norm = hybrid.unscaled_rate_norm(alpha)
    logging.info(f"scaled rate state_12 norm: {rate_12_norm}")
    logging.info(f"unscaled scaled rate state_12 norm: {urate_12_norm}")


    ColorPrint.print_bold(f"   Solving second order: Rate Pb.    ")
    ColorPrint.print_bold(f"===================-=================")

    # n_eigenvalues = 10
    is_stable = bifurcation.solve(alpha_lb)
    is_elastic = bifurcation.is_elastic()
    inertia = bifurcation.get_inertia()
    # bifurcation.save_eigenvectors(filename=f"{prefix}/{_nameExp}_eigv_{t:3.2f}.xdmf")

    ColorPrint.print_bold(f"State is elastic: {is_elastic}")
    ColorPrint.print_bold(f"State's inertia: {inertia}")
    # ColorPrint.print_bold(f"State is stable: {is_stable}")
    
    ColorPrint.print_bold(f"   Solving second order: Cone Pb.    ")
    ColorPrint.print_bold(f"===================-=================")
    
    stable = cone.my_solve(alpha_lb, eig0=bifurcation._spectrum)
    
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

    history_data["load"].append(t)
    history_data["fracture_energy"].append(fracture_energy)
    history_data["elastic_energy"].append(elastic_energy)
    history_data["total_energy"].append(elastic_energy+fracture_energy)
    history_data["solver_data"].append(solver.data)
    history_data["eigs"].append(bifurcation.data["eigs"])
    history_data["F"].append(stress)
    history_data["cone_data"].append(cone.data)
    history_data["alphadot_norm"].append(alphadot.vector.norm())
    history_data["rate_12_norm"].append(rate_12_norm)
    history_data["unscaled_rate_12_norm"].append(urate_12_norm)
    history_data["cone-stable"].append(stable)
    history_data["cone-eig"].append(cone.data["lambda_0"])
    history_data["uniqueness"].append(_unique)
    history_data["inertia"].append(inertia)

    with XDMFFile(comm, f"{prefix}/{_nameExp}.xdmf", "a", encoding=XDMFFile.Encoding.HDF5) as file:
        file.write_function(u, t)
        file.write_function(alpha, t)

    if comm.rank == 0:
        a_file = open(f"{prefix}/time_data.json", "w")
        json.dump(history_data, a_file)
        a_file.close()

    ColorPrint.print_bold(f"   Written timely data.    ")
    print()
    print()
    print()
    print()
list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])
# print(history_data)



df = pd.DataFrame(history_data)
print(df.drop(['solver_data', 'cone_data'], axis=1))

# Viz

from utils.plots import plot_energies, plot_AMit_load, plot_force_displacement

if comm.rank == 0:
    plot_energies(history_data, file=f"{prefix}/{_nameExp}_energies.pdf")
    plot_AMit_load(history_data, file=f"{prefix}/{_nameExp}_it_load.pdf")
    plot_force_displacement(history_data, file=f"{prefix}/{_nameExp}_stress-load.pdf")



from pyvista.utilities import xvfb
import pyvista
import sys
from utils.viz import plot_mesh, plot_vector, plot_scalar
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


