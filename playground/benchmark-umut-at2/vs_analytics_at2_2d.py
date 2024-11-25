#!/usr/bin/env python3
import hashlib
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
from dolfinx.fem import (Constant, Function, assemble_scalar, form,
                         locate_dofs_geometrical)
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI
from petsc4py import PETSc
from pyvista.utilities import xvfb
import basix.ufl

from irrevolutions.algorithms.am import HybridSolver
from irrevolutions.algorithms.so import BifurcationSolver, StabilitySolver
from irrevolutions.meshes.primitives import mesh_bar_gmshapi
from irrevolutions.models import \
    BrittleMembraneOverElasticFoundation as ThinFilm
from irrevolutions.utils import (ColorPrint, Visualization, _logger,
                                 _write_history_data, history_data)
from irrevolutions.utils.plots import (plot_AMit_load, plot_energies,
                                       plot_force_displacement)
from irrevolutions.utils.viz import plot_profile, plot_scalar, plot_vector

petsc4py.init(sys.argv)
comm = MPI.COMM_WORLD

# Mesh on node model_rank and then distribute
model_rank = 0


class ThinFilmAT2(ThinFilm):
    def w(self, alpha):
        """
        Return the dissipated energy function as a function of the state
        (only depends on damage).
        """
        # Return w(alpha) function
        return alpha**2


def stress(state):
    """
    Return the one-dimensional stress
    """
    u = state["u"]
    alpha = state["alpha"]
    dx = ufl.Measure("dx", domain=u.function_space.mesh)

    return parameters["model"]["E"] * ThinFilmAT2.a(alpha) * u.dx() * dx


def run_computation(parameters, storage=None):
    Lx = parameters["geometry"]["Lx"]
    Ly = parameters["geometry"]["Ly"]
    geom_type = parameters["geometry"]["geom_type"]
    tdim = parameters["geometry"]["geometric_dimension"]
    lc = parameters["model"]["ell"] / parameters["geometry"]["mesh_size_factor"]

    _nameExp = parameters["geometry"]["geom_type"]

    # Get geometry model
    outdir = os.path.join(os.path.dirname(__file__), "output")

    if storage is None:
        prefix = os.path.join(outdir, "thin-film-at2-2d")
    else:
        prefix = storage

    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    if comm.rank == 0:
        with open(f"{prefix}/parameters.yaml", "w") as file:
            yaml.dump(parameters, file)

        with open(f"{prefix}/signature.md5", "w") as f:
            f.write(signature)

    gmsh_model, tdim = mesh_bar_gmshapi(geom_type, Lx, Ly, lc, tdim)
    model_rank = 0
    mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

    with XDMFFile(
        comm, f"{prefix}/{_nameExp}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5
    ) as file:
        file.write_mesh(mesh)

    # Functional Setting
    element_u = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1, shape=(tdim,))
    element_alpha = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1)

    V_u = dolfinx.fem.functionspace(mesh, element_u)
    V_alpha = dolfinx.fem.functionspace(mesh, element_alpha)

    u = dolfinx.fem.Function(V_u, name="Displacement")
    dolfinx.fem.Function(V_u, name="BoundaryDisplacement")
    u_zero = Function(V_u, name="InelasticDisplacement")
    zero_u = Function(V_u, name="BoundaryUnknown")

    alpha = dolfinx.fem.Function(V_alpha, name="Damage")
    dolfinx.fem.Function(V_alpha, name="Damage_rate")

    # Perturbations
    Function(V_alpha, name="DamagePerturbation")
    Function(V_u, name="DisplacementPerturbation")

    # Pack state
    state = {"u": u, "alpha": alpha}

    # Bounds
    alpha_ub = dolfinx.fem.Function(V_alpha, name="UpperBoundDamage")
    alpha_lb = dolfinx.fem.Function(V_alpha, name="LowerBoundDamage")

    dx = ufl.Measure("dx", domain=mesh)
    ufl.Measure("ds", domain=mesh)

    # Useful references
    Lx = parameters.get("geometry").get("Lx")

    # Measures
    dx = ufl.Measure("dx", domain=mesh)
    ufl.Measure("ds", domain=mesh)

    locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], 0.0))
    locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], Lx))

    alpha_lb.interpolate(lambda x: np.zeros_like(x[0]))
    alpha_ub.interpolate(lambda x: np.ones_like(x[0]))

    # eps_t = dolfinx.fem.Constant(mesh, np.array(1., dtype=PETSc.ScalarType))

    for f in [zero_u, u_zero, alpha_lb, alpha_ub]:
        f.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

    # bcs_u = [dirichletbc(u_zero, dofs_u_right),
    #          dirichletbc(u_zero, dofs_u_left)]

    bcs_u = []
    bcs_alpha = []

    bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}

    tau = Constant(mesh, np.array(0.0, dtype=PETSc.ScalarType))
    eps_t = tau * ufl.as_tensor([[1.0, 0], [0, 0]])

    model = ThinFilmAT2(parameters["model"], eps_0=eps_t)

    f = Constant(mesh, np.array([0, 0], dtype=PETSc.ScalarType))
    external_work = ufl.dot(f, state["u"]) * dx
    total_energy = model.total_energy_density(state) * dx - external_work
    _stress = model.stress(model.eps(u), alpha)

    load_par = parameters["loading"]
    loads = np.linspace(load_par["min"], load_par["max"], load_par["steps"])

    hybrid = HybridSolver(
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

    history_data["F"] = []

    logging.basicConfig(level=logging.INFO)

    for i_t, t in enumerate(loads):
        tau.value = t

        # update the lower bound
        alpha.x.petsc_vec.copy(alpha_lb.x.petsc_vec)
        alpha_lb.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        _logger.critical(f"-- Solving Equilibrium (Criticality) for t = {t:3.2f} --")
        hybrid.solve(alpha_lb)

        _logger.critical(f"-- Solving Bifurcation (Uniqueness) for t = {t:3.2f} --")
        is_unique = bifurcation.solve(alpha_lb)
        is_elastic = not bifurcation._is_critical(alpha_lb)
        inertia = bifurcation.get_inertia()

        ColorPrint.print_bold(f"State is elastic: {is_elastic}")
        ColorPrint.print_bold(f"State's inertia: {inertia}")
        ColorPrint.print_bold(f"Evolution is unique: {is_unique}")

        z0 = (
            bifurcation._spectrum[0]["xk"]
            if bifurcation._spectrum and "xk" in bifurcation._spectrum[0]
            else None
        )

        _logger.critical(f"-- Solving Stability (Stability) for t = {t:3.2f} --")
        stable = stability.solve(alpha_lb, eig0=z0, inertia=inertia)

        with dolfinx.common.Timer("~Postprocessing and Vis"):
            if comm.rank == 0:
                plot_energies(history_data, file=f"{prefix}/{_nameExp}_energies.pdf")
                plot_AMit_load(history_data, file=f"{prefix}/{_nameExp}_it_load.pdf")
                plot_force_displacement(
                    history_data, file=f"{prefix}/{_nameExp}_stress-load.pdf"
                )

                xvfb.start_xvfb(wait=0.05)
                pyvista.OFF_SCREEN = True

                plotter = pyvista.Plotter(
                    title="Thin film",
                    window_size=[1600, 600],
                    shape=(1, 2),
                )
                _plt = plot_scalar(alpha, plotter, subplot=(0, 0))
                _plt = plot_scalar(u, plotter, subplot=(0, 1))
                _plt.screenshot(f"{prefix}/thinfilm-state.png")

                plotter = pyvista.Plotter(
                    title="Test Profile",
                    window_size=[800, 600],
                    shape=(1, 1),
                )

                tol = 1e-3
                xs = np.linspace(0 + tol, parameters["geometry"]["Lx"] - tol, 101)
                points = np.zeros((3, 101))
                points[0] = xs

                _plt, data = plot_profile(
                    alpha,
                    points,
                    plotter,
                    lineproperties={
                        "c": "k",
                        "label": f"$\\alpha$ with $\ell$ = {parameters['model']['ell']:.2f}",
                    },
                )
                ax = _plt.gca()
                _plt.legend()
                _plt.fill_between(data[0], data[1].reshape(len(data[1])))
                _plt.title("Damage profile")
                ax.set_ylim(-0.1, 1.1)

                _plt, data = plot_profile(
                    u_zero,
                    points,
                    plotter,
                    fig=_plt,
                    ax=ax,
                    lineproperties={"c": "r", "label": "$u_0$"},
                )

                _plt, data = plot_profile(
                    u,
                    points,
                    plotter,
                    fig=_plt,
                    ax=ax,
                    lineproperties={"c": "g", "label": "$u$"},
                )

                _plt.savefig(f"{prefix}/damage_profile-{i_t}.png")

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

            _write_history_data(
                hybrid,
                bifurcation,
                stability,
                history_data,
                t,
                inertia,
                stable,
                [elastic_energy, fracture_energy],
            )
            history_data["F"].append(stress)

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
                title="Thin Film",
                window_size=[1600, 600],
                shape=(1, 2),
            )
            _plt = plot_scalar(alpha, plotter, subplot=(0, 0))
            _plt = plot_vector(u, plotter, subplot=(0, 1))
            _plt.screenshot(f"{prefix}/traction-state.png")

            _plt.close()

    # df = pd.DataFrame(history_data)
    print(pd.DataFrame(history_data))

    return history_data, stability.data, state


def load_parameters(file_path, ndofs, model="at2"):
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
    parameters["model"]["model_type"] = "2d"

    parameters["geometry"]["geom_type"] = "thinfilm"
    # Get mesh parameters

    if model == "at2":
        parameters["loading"]["min"] = 0.0
        parameters["loading"]["max"] = 1.5
        parameters["loading"]["steps"] = 50

    parameters["geometry"]["geom_type"] = "bar"
    parameters["geometry"]["mesh_size_factor"] = 3
    parameters["geometry"]["Lx"] = 3
    parameters["geometry"]["Ly"] = 5e-2

    parameters["stability"]["cone"]["cone_max_it"] = 400000
    parameters["stability"]["cone"]["cone_atol"] = 1e-6
    parameters["stability"]["cone"]["cone_rtol"] = 1e-6
    parameters["stability"]["cone"]["scaling"] = 1e-2

    parameters["model"]["w1"] = 1
    parameters["model"]["ell"] = (0.158114) ** 2 / 2
    # parameters["model"]["ell"] = .1
    parameters["model"]["k_res"] = 0.0
    parameters["model"]["E"] = 1
    parameters["model"]["ell_e"] = 0.34

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    return parameters, signature


if __name__ == "__main__":
    # Set the logging level
    logging.basicConfig(level=logging.INFO)

    # Load parameters
    parameters, signature = load_parameters(
        os.path.join(os.path.dirname(__file__), "parameters.yaml"),
        ndofs=100,
        model="at2",
    )

    # Run computation
    _storage = f"output/thinfilm-bar/MPI-{MPI.COMM_WORLD.Get_size()}/{signature}"
    visualization = Visualization(_storage)
    ColorPrint.print_bold(f"===================- {_storage} -=================")

    with dolfinx.common.Timer("~Computation Experiment") as timer:
        history_data, stability_data, state = run_computation(parameters, _storage)

    from irrevolutions.utils import table_timing_data

    _timings = table_timing_data()
    visualization.save_table(_timings, "timing_data")
    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

    ColorPrint.print_bold(f"===================- {signature} -=================")
    ColorPrint.print_bold(f"===================- {_storage} -=================")
