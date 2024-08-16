#!/usr/bin/env python3
import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
import sys
import os
import hashlib

from dolfinx.fem import dirichletbc
import dolfinx.mesh
from dolfinx.fem import (
    Function,
    FunctionSpace,
    assemble_scalar,
    dirichletbc,
    form,
    set_bc,
)
from mpi4py import MPI
import petsc4py
from petsc4py import PETSc
import dolfinx
import dolfinx.plot
import ufl

from dolfinx.fem.petsc import set_bc
from dolfinx.io import XDMFFile, gmshio
import logging

import pyvista
from pyvista.utilities import xvfb
from dolfinx.mesh import locate_entities_boundary
from dolfinx.fem import locate_dofs_topological

from irrevolutions.algorithms.so import BifurcationSolver, StabilitySolver
from irrevolutions.algorithms.am import HybridSolver
from irrevolutions.models import DamageElasticityModel as Brittle
from irrevolutions.meshes.pacman import mesh_pacman
from irrevolutions.utils import (
    ColorPrint,
    _write_history_data,
    history_data,
    set_vector_to_constant,
)
from irrevolutions.utils import _logger
from irrevolutions.utils.lib import _local_notch_asymptotic
from irrevolutions.utils.viz import plot_mesh
from irrevolutions.utils.viz import plot_mesh, plot_scalar, plot_vector

description = """We solve here a basic 2d of a notched specimen.
Imagine a dinner a pizza which is missing a slice, and lots of hungry friends
that pull from the sides of the pizza. Will a real pizza will break at the centre?

We solve this problem as an example of localisation with singularity.
"""

petsc4py.init(sys.argv)
comm = MPI.COMM_WORLD

# Mesh on node model_rank and then distribute
model_rank = 0


def run_computation(parameters, storage):
    # Load mesh

    _r = parameters["geometry"]["r"]
    _omega = parameters["geometry"]["omega"]
    _nameExp = parameters["geometry"]["geom_type"]
    tdim = parameters["geometry"]["geometric_dimension"]
    ell = parameters["model"]["ell"]
    geom_type = parameters["geometry"]["geom_type"]

    parameters["geometry"]["meshsize"] = (
        ell / parameters["geometry"]["mesh_size_factor"]
    )
    gmsh_model, tdim = mesh_pacman(geom_type, parameters["geometry"], tdim)

    # Get mesh and meshtags
    mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

    outdir = os.path.join(os.path.dirname(__file__), "output")
    if storage is None:
        prefix = os.path.join(outdir, f"test_1d-N{parameters['model']['N']}")
    else:
        prefix = storage

    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    with XDMFFile(
        comm, f"{prefix}/{_nameExp}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5
    ) as file:
        file.write_mesh(mesh)

    if comm.rank == 0:
        ax = plot_mesh(mesh)
        fig = ax.get_figure()
        fig.savefig(f"{prefix}/mesh.png")
    hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    # Function spaces
    element_u = ufl.VectorElement("Lagrange", mesh.ufl_cell(), degree=1, dim=2)
    V_u = FunctionSpace(mesh, element_u)

    element_alpha = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)
    V_alpha = FunctionSpace(mesh, element_alpha)

    # Define the state
    u = Function(V_u, name="Displacement")
    alpha = Function(V_alpha, name="Damage")
    alphadot = Function(V_alpha, name="Damage rate")

    # upper/lower bound for the damage field
    alpha_lb = Function(V_alpha, name="Lower bound")
    alpha_ub = Function(V_alpha, name="Upper bound")

    state = {"u": u, "alpha": alpha}

    # Data

    uD = Function(V_u, name="Asymptotic Notch Displacement")

    # Measures
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh)

    # Set Bcs Function

    ext_bd_facets = locate_entities_boundary(
        mesh,
        dim=1,
        marker=lambda x: np.isclose(
            x[0] ** 2.0 + x[1] ** 2.0 - _r**2, 0.0, atol=1.0e-4
        ),
    )

    boundary_dofs_u = locate_dofs_topological(V_u, mesh.topology.dim - 1, ext_bd_facets)
    boundary_dofs_alpha = locate_dofs_topological(
        V_alpha, mesh.topology.dim - 1, ext_bd_facets
    )

    uD.interpolate(
        lambda x: _local_notch_asymptotic(
            x, ω=np.deg2rad(_omega / 2.0), par=parameters["material"]
        )
    )

    alpha_lb.interpolate(lambda x: np.zeros_like(x[0]))
    alpha_ub.interpolate(lambda x: np.ones_like(x[0]))

    for f in [alpha_lb, alpha_ub]:
        f.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

    bcs_u = [dirichletbc(value=uD, dofs=boundary_dofs_u)]

    bcs_alpha = [
        dirichletbc(
            np.array(0, dtype=PETSc.ScalarType),
            boundary_dofs_alpha,
            V_alpha,
        )
    ]
    bcs_alpha = []
    set_bc(alpha_ub.vector, bcs_alpha)
    alpha_ub.vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )

    bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}

    # Bounds for Newton solver

    u_lb = Function(V_u, name="displacement lower bound")
    u_ub = Function(V_u, name="displacement upper bound")
    alpha_lb = Function(V_alpha, name="damage lower bound")
    alpha_ub = Function(V_alpha, name="damage upper bound")
    set_vector_to_constant(u_lb.vector, PETSc.NINFINITY)
    set_vector_to_constant(u_ub.vector, PETSc.PINFINITY)
    set_vector_to_constant(alpha_lb.vector, 0)
    set_vector_to_constant(alpha_ub.vector, 1)

    model = Brittle(parameters["model"])

    # Energy functional
    f = dolfinx.fem.Constant(mesh, np.array([0, 0], dtype=PETSc.ScalarType))
    external_work = ufl.dot(f, state["u"]) * dx
    total_energy = model.total_energy_density(state) * dx - external_work

    load_par = parameters["loading"]
    loads = np.linspace(load_par["min"], load_par["max"], load_par["steps"])
    # loads = [0., 0.5, 1.01, 1.3, 2.]

    equilibrium = HybridSolver(
        total_energy,
        state,
        bcs,
        bounds=(alpha_lb, alpha_ub),
        solver_parameters=parameters.get("solvers"),
    )

    bifurcation = BifurcationSolver(
        total_energy, state, bcs, bifurcation_parameters=parameters.get("stability")
    )

    stability = StabilitySolver(
        total_energy, state, bcs, cone_parameters=parameters.get("stability")
    )

    mode_shapes_data = {
        "time_steps": [],
        "point_values": {
            "x_values": [],
        },
    }

    _logger.setLevel(level=logging.CRITICAL)

    for i_t, t in enumerate(loads):

        uD.interpolate(
            lambda x: _local_notch_asymptotic(
                x, ω=np.deg2rad(_omega / 2.0), t=t, par=parameters["material"]
            )
        )

        # update the lower bound
        alpha.vector.copy(alpha_lb.vector)
        alpha_lb.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        logging.info(f"-- Solving for t = {t:3.2f} --")
        equilibrium.solve(alpha_lb)

        bifurcation.solve(alpha_lb)
        not bifurcation._is_critical(alpha_lb)

        inertia = bifurcation.get_inertia()

        stable = stability.solve(alpha_lb, eig0=bifurcation._spectrum, inertia=inertia)

        with dolfinx.common.Timer(f"~Postprocessing and Vis") as timer:

            fracture_energy = comm.allreduce(
                assemble_scalar(form(model.damage_energy_density(state) * dx)),
                op=MPI.SUM,
            )
            elastic_energy = comm.allreduce(
                assemble_scalar(form(model.elastic_energy_density(state) * dx)),
                op=MPI.SUM,
            )

            _write_history_data(
                equilibrium,
                bifurcation,
                stability,
                history_data,
                t,
                inertia,
                stable,
                [fracture_energy, elastic_energy],
            )

            with XDMFFile(
                comm, f"{prefix}/{_nameExp}.xdmf", "a", encoding=XDMFFile.Encoding.HDF5
            ) as file:
                file.write_function(u, t)
                file.write_function(alpha, t)

            if comm.rank == 0:
                a_file = open(f"{prefix}/time_data.json", "w")
                json.dump(history_data, a_file)
                a_file.close()

            xvfb.start_xvfb(wait=0.05)
            pyvista.OFF_SCREEN = True
            plotter = pyvista.Plotter(
                title="State of the System",
                window_size=[1600, 600],
                shape=(1, 2),
            )
            _plt = plot_scalar(alpha, plotter, subplot=(0, 0))
            _plt = plot_vector(u, plotter, subplot=(0, 1))
            if comm.rank == 0:
                Path("output").mkdir(parents=True, exist_ok=True)
            _plt.screenshot(f"{prefix}/{_nameExp}-{comm.size}-{i_t}.png")
            _plt.close()

    from utils.plots import plot_energies

    if comm.rank == 0:
        plot_energies(history_data, file=f"{prefix}/{_nameExp}_energies.pdf")
        # plot_AMit_load(history_data, file=f"{prefix}/{_nameExp}_it_load.pdf")

    return history_data, stability.data, state

def load_parameters(file_path, ndofs, model="at1"):
    """
    Load parameters from a YAML file.

    Args:
        file_path (str): Path to the YAML parameter file.

    Returns:
        dict: Loaded parameters.
    """
    import hashlib

    with open(file_path) as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    parameters["model"]["model_dimension"] = 2
    parameters["model"]["model_type"] = "2D"
    # parameters["model"]["mu"] = 1
    parameters["model"]["w1"] = 1

    parameters["geometry"]["geom_type"] = "brittle-damageable"
    # Get mesh parameters

    if model == "at2":
        parameters["loading"]["min"] = 0.9
        parameters["loading"]["max"] = 0.9
        parameters["loading"]["steps"] = 1

    elif model == "at1":
        parameters["loading"]["min"] = 0.0
        parameters["loading"]["max"] = 1.5
        parameters["loading"]["steps"] = 20

    parameters["geometry"]["geom_type"] = "local_notch"
    parameters["geometry"]["mesh_size_factor"] = 1
    parameters["geometry"]["refinement"] = 10
    # parameters["geometry"]["N"] = ndofs

    parameters["stability"]["cone"]["cone_max_it"] = 400000
    parameters["stability"]["cone"]["cone_atol"] = 1e-6
    parameters["stability"]["cone"]["cone_rtol"] = 1e-6
    parameters["stability"]["cone"]["scaling"] = 1e-4

    parameters["model"]["model_dimension"] = 2
    parameters["model"]["w1"] = 1
    parameters["model"]["ell"] = 0.1
    parameters["model"]["k_res"] = 0.0

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    return parameters, signature

def test_2d():
    # import argparse
    from mpi4py import MPI

    # parser = argparse.ArgumentParser(description="Process evolution.")
    # parser.add_argument("-N", help="The number of dofs.", type=int, default=10)
    # args = parser.parse_args()
    N = 1000
    parameters, signature = load_parameters("data/pacman/parameters.yaml", ndofs=N)
    pretty_parameters = json.dumps(parameters, indent=2)
    print(pretty_parameters)

    _storage = (
        f"output/two-dimensional-pizza/MPI-{MPI.COMM_WORLD.Get_size()}/{signature}"
    )
    ColorPrint.print_bold(f"===================-{_storage}-=================")

    with dolfinx.common.Timer(f"~Computation Experiment") as timer:
        history_data, stability_data, state = run_computation(parameters, _storage)

    ColorPrint.print_bold(history_data["eigs-cone"])
    from irrevolutions.utils import ResultsStorage, Visualization

    storage = ResultsStorage(MPI.COMM_WORLD, _storage)
    storage.store_results(parameters, history_data, state)
    visualization = Visualization(_storage)
    # visualization.visualise_results(pd.DataFrame(history_data), drop = ["solver_data", "cone_data"])
    visualization.save_table(pd.DataFrame(history_data), "history_data")
    # visualization.save_table(pd.DataFrame(stability_data), "stability_data")
    pd.DataFrame(stability_data).to_json(f"{_storage}/stability_data.json")

    ColorPrint.print_bold(f"===================-{signature}-=================")
    ColorPrint.print_bold(f"===================-{_storage}-=================")

    print(pd.DataFrame(history_data))
    ColorPrint.print_bold(f"===================-{signature}-=================")
    print(pd.DataFrame(stability_data))


if __name__ == "__main__":
    test_2d()