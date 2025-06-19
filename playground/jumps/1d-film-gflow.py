#!/usr/bin/env python3
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

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
    assemble_scalar,
    dirichletbc,
    form,
    locate_dofs_geometrical,
    set_bc,
)
from dolfinx.fem.petsc import assemble_vector, set_bc
from dolfinx.io import XDMFFile
from irrevolutions.algorithms.am import HybridSolver
from irrevolutions.algorithms.so import BifurcationSolver, StabilitySolver
from irrevolutions.algorithms.ls import StabilityStepper, LineSearch
from irrevolutions.algorithms.gf import JumpSolver

# import irrevolutions.utils.postprocess as pp
import crunchy.plots as cp
from irrevolutions.models.one_dimensional import FilmModel1D as ThinFilm

from irrevolutions.solvers.function import vec_to_functions
from irrevolutions.algorithms.am import AlternateMinimisation1D as AlternateMinimisation
from irrevolutions.utils import (
    ColorPrint,
    ResultsStorage,
    Visualization,
    _logger,
    _write_history_data,
    history_data,
    norm_H1,
    norm_L2,
)
from irrevolutions.utils.plots import (
    plot_AMit_load,
    plot_energies,
)
import matplotlib
from irrevolutions.utils.viz import plot_mesh, plot_profile, plot_scalar, plot_vector
from irrevolutions.utils import create_function_spaces_nd, initialize_functions
from mpi4py import MPI
from petsc4py import PETSc
from pyvista.plotting.utilities import xvfb
import random
import matplotlib.pyplot as plt

# from irrevolutions.utils.viz import _plot_bif_spectrum_profiles
from irrevolutions.utils import setup_logger_mpi

petsc4py.init(sys.argv)
comm = MPI.COMM_WORLD

logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
logger = setup_logger_mpi(logging.INFO)
# Mesh on node model_rank and then distribute
model_rank = 0


def run_computation(parameters, storage=None):
    Lx = parameters["geometry"]["Lx"]
    _nameExp = parameters["geometry"]["geom_type"]
    parameters["model"]["ell"]

    # Get geometry model
    parameters["geometry"]["geom_type"]

    # N = max(_N, parameters["model"]["ell"] / parameters["geometry"]["mesh_size_factor"])
    N = parameters["model"]["ell"] / parameters["geometry"]["mesh_size_factor"]
    logger.info(f"Mesh size: {N}")

    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, int(1 / N))
    outdir = os.path.join(os.path.dirname(__file__), "output")

    if MPI.COMM_WORLD.rank == 0:
        Path(storage).mkdir(parents=True, exist_ok=True)

    prefix = storage

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    if MPI.COMM_WORLD.rank == 0:
        with open(f"{prefix}/parameters.yaml", "w") as file:
            yaml.dump(parameters, file)
        with open(f"{prefix}/signature.md5", "w") as f:
            f.write(signature)

    with XDMFFile(
        comm, f"{prefix}/{_nameExp}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5
    ) as file:
        file.write_mesh(mesh)

    # Functional Setting
    V_u, V_alpha = create_function_spaces_nd(mesh, dim=1)
    u, u_, alpha, β, v, state = initialize_functions(V_u, V_alpha)

    # Bounds
    alpha_ub = dolfinx.fem.Function(V_alpha, name="UpperBoundDamage")
    alpha_lb = dolfinx.fem.Function(V_alpha, name="LowerBoundDamage")

    # Useful references
    Lx = parameters.get("geometry").get("Lx")

    # Define the state
    zero_u = Function(V_u, name="BoundaryUnknown")
    zero_u.interpolate(lambda x: np.zeros_like(x[0]))

    u_zero = Function(V_u, name="InelasticDisplacement")
    eps_t = dolfinx.fem.Constant(mesh, np.array(1.0, dtype=PETSc.ScalarType))
    u_zero.interpolate(lambda x: eps_t / 2.0 * (2 * x[0] - Lx))

    tilde_u = Function(V_u, name="BoundaryDatum")
    tilde_u.interpolate(lambda x: np.ones_like(x[0]))

    dofs_u_left = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], 0.0))
    dofs_u_right = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], Lx))

    alpha_lb.interpolate(lambda x: np.zeros_like(x[0]))
    alpha_ub.interpolate(lambda x: np.ones_like(x[0]))

    for f in [u, zero_u, tilde_u, u_zero, alpha_lb, alpha_ub]:
        f.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

    # Natural boundary conditions
    bcs_u = []

    # Redundant boundary conditions
    bcs_u = [dirichletbc(u_zero, dofs_u_right), dirichletbc(u_zero, dofs_u_left)]

    bcs_alpha = []

    bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}

    # Measures
    dx = ufl.Measure("dx", domain=mesh)
    model = ThinFilm(parameters["model"])

    total_energy = (
        model.elastic_energy_density(state, u_zero) + model.damage_energy_density(state)
    ) * dx

    load_par = parameters["loading"]
    loads = np.linspace(load_par["min"], load_par["max"], load_par["steps"])
    iterator = StabilityStepper(loads)

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

    linesearch = LineSearch(
        total_energy,
        state,
        linesearch_parameters=parameters.get("stability").get("linesearch"),
    )

    arclength = []

    while True:
        try:
            i_t = next(iterator)
            # next increments the self index
        except StopIteration:
            break

        t = loads[i_t - 1]

        # Perform your time step with t
        eps_t.value = t
        u_zero.interpolate(lambda x: eps_t / 2.0 * (2 * x[0] - Lx))
        u_zero.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        alpha.x.petsc_vec.copy(alpha_lb.x.petsc_vec)
        alpha_lb.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        # Log current load
        logger.critical(f"-- Solving for t = {t:3.2f} --")
        with dolfinx.common.Timer(f"~First Order: Equilibrium") as timer:
            equilibrium.solve(alpha_lb)

        is_unique = bifurcation.solve(alpha_lb)
        is_elastic = not bifurcation._is_critical(alpha_lb)
        inertia = bifurcation.get_inertia()

        z0 = (
            bifurcation._spectrum[0]["xk"]
            if bifurcation._spectrum and "xk" in bifurcation._spectrum[0]
            else None
        )

        stable = stability.solve(alpha_lb, eig0=z0, inertia=inertia)

        logger.info(f"Stability of state at load {t:.2f}: {stable}")
        ColorPrint.print_bold(f"Evolution is unique: {is_unique}")
        ColorPrint.print_bold(f"State's inertia: {inertia}")
        ColorPrint.print_bold(f"State's stable: {stable}")

        # equilibrium.log()
        bifurcation.log()
        stability.log()
        ColorPrint.print_bold(f"===================- {prefix} -=================")

        if not stable:
            iterator.pause_time()
            logger.info(f"Time paused at {t:.2f}")
            __import__("pdb").set_trace()

            vec_to_functions(stability.solution["xt"], [v, β])
            perturbation = {"v": v, "beta": β}
            interval = linesearch.get_unilateral_interval(state, perturbation)

            order = 4
            h_opt, energies_1d, p, _ = linesearch.search(
                state, perturbation, interval, m=order
            )
            arclength.append((t, h_opt))

            with dolfinx.common.Timer(f"~Visualisation") as timer:
                logger.critical(f" *> State is unstable: {not stable}")
                logger.critical(f"line search interval is {interval}")
                logger.critical(f"perturbation energies: {energies_1d}")
                logger.critical(f"hopt: {h_opt}")
                logger.critical(f"lambda_t: {stability.solution['lambda_t']}")

                h_steps = np.linspace(interval[0], interval[1], order + 1)
                fig, axes = plt.subplots(1, 1)
                plt.scatter(h_steps, energies_1d)
                plt.scatter(
                    h_opt, 0, c="k", s=40, marker="|", label=f"$h^*={h_opt:.2f}$"
                )
                plt.scatter(h_opt, p(h_opt), c="k", s=40, alpha=0.5)
                xs = np.linspace(interval[0], interval[1], 30)
                axes.plot(xs, p(xs), label="Energy slice along perturbation")
                axes.set_xlabel("h")
                axes.set_ylabel("$E_h - E_0$")
                axes.set_title(f"Polynomial Interpolation - order {order}")
                axes.legend()
                axes.spines["top"].set_visible(False)
                axes.spines["right"].set_visible(False)
                axes.spines["left"].set_visible(False)
                axes.spines["bottom"].set_visible(False)
                axes.set_yticks([0])
                axes.axhline(0, c="k")
                fig.savefig(f"{prefix}/energy_interpolation-order-{order}.png")
                plt.close()

                tol = 1e-3
                xs = np.linspace(0 + tol, Lx - tol, 101)
                points = np.zeros((3, 101))
                points[0] = xs

                plotter = pyvista.Plotter(
                    title="Perturbation profile",
                    window_size=[800, 600],
                    shape=(1, 1),
                )

                fig, data = plot_profile(
                    perturbation["beta"],
                    points,
                    plotter,
                    lineproperties={"c": "k", "label": f"$\\beta$"},
                )
                fig.legend()
                fig.gca().fill_between(data[0], data[1].reshape(len(data[1])))
                fig.gca().set_title("Perturbation")
                fig.savefig(f"{prefix}/perturbation-profile-{i_t}.png")
                plt.close()

            linesearch.perturb(state, perturbation, h_opt)
            fracture_energy, elastic_energy = postprocess(
                parameters,
                _nameExp,
                prefix,
                v,
                β,
                state,
                u_zero,
                dx,
                bifurcation,
                stability,
                i_t,
                model=model,
            )
        else:
            # If stable, postprocess and dump
            fracture_energy, elastic_energy = postprocess(
                parameters,
                _nameExp,
                prefix,
                v,
                β,
                state,
                u_zero,
                dx,
                bifurcation,
                stability,
                i_t,
                model=model,
            )

            dump_output(
                _nameExp,
                prefix,
                history_data,
                u,
                alpha,
                equilibrium,
                bifurcation,
                stability,
                t,
                fracture_energy,
                elastic_energy,
            )

    logger.info(f"Arclengths: {arclength}")

    print(pd.DataFrame(history_data).drop(columns=["equilibrium_data"]))
    return history_data, stability.data, state


def dump_output(
    _nameExp,
    prefix,
    history_data,
    u,
    alpha,
    equilibrium,
    bifurcation,
    stability,
    t,
    fracture_energy,
    elastic_energy,
):
    logger.info(f"Dumping output at {t:.2f}")

    with dolfinx.common.Timer(f"~Output and Storage") as timer:
        with XDMFFile(
            comm, f"{prefix}/{_nameExp}.xdmf", "a", encoding=XDMFFile.Encoding.HDF5
        ) as file:
            file.write_function(u, t)
            file.write_function(alpha, t)

        if comm.rank == 0:
            a_file = open(f"{prefix}/time_data.json", "w")
            json.dump(history_data, a_file)
            a_file.close()

        _write_history_data(
            equilibrium=equilibrium,
            bifurcation=bifurcation,
            stability=stability,
            history_data=history_data,
            inertia=bifurcation.get_inertia(),
            t=t,
            stable=np.nan,
            energies=[elastic_energy, fracture_energy],
        )


def postprocess(
    parameters,
    _nameExp,
    prefix,
    β,
    v,
    state,
    u_zero,
    dx,
    bifurcation,
    stability,
    i_t,
    model,
):
    with dolfinx.common.Timer(f"~Postprocessing and Vis") as timer:
        fracture_energy = comm.allreduce(
            assemble_scalar(form(model.damage_energy_density(state) * dx)),
            op=MPI.SUM,
        )
        elastic_energy = comm.allreduce(
            assemble_scalar(form(model.elastic_energy_density(state, u_zero) * dx)),
            op=MPI.SUM,
        )

        fig_state, ax1 = matplotlib.pyplot.subplots()

        if comm.rank == 0:
            plot_energies(history_data, file=f"{prefix}/{_nameExp}_energies.pdf")
            plot_AMit_load(history_data, file=f"{prefix}/{_nameExp}_it_load.pdf")
            # plot_force_displacement(
            #     history_data, file=f"{prefix}/{_nameExp}_stress-load.pdf"
            # )

        # xvfb.start_xvfb(wait=0.05)
        pyvista.OFF_SCREEN = True

        plotter = pyvista.Plotter(
            title="Profiles",
            window_size=[800, 600],
            shape=(1, 1),
        )

        tol = 1e-3
        xs = np.linspace(0 + tol, parameters["geometry"]["Lx"] - tol, 101)
        points = np.zeros((3, 101))
        points[0] = xs

        fig, data = plot_profile(
            state["alpha"],
            points,
            plotter,
            lineproperties={
                "c": "k",
                "label": f"$\\alpha$ with $\\ell$ = {parameters['model']['ell']:.2f}",
            },
        )
        ax = fig.gca()
        ax.set_ylim(0, 1)
        fig, data = plot_profile(
            state["u"],
            points,
            plotter,
            fig=fig,
            ax=ax,
            lineproperties={
                "c": "g",
                "label": "$u$",
                "marker": "o",
            },
        )

        fig, data = plot_profile(
            u_zero,
            points,
            plotter,
            fig=fig,
            ax=ax,
            lineproperties={"c": "r", "lw": 3, "label": "$u_0$"},
        )
        fig.legend()
        fig.gca().set_title("Solution state")
        # ax.set_ylim(-2.1, 2.1)
        ax.axhline(0, color="k", lw=0.5)
        fig.savefig(f"{prefix}/state_profile-{i_t}.png")

        if bifurcation._spectrum:
            fig_bif, ax = matplotlib.pyplot.subplots()

            vec_to_functions(bifurcation._spectrum[0]["xk"], [v, β])

            fig, data = plot_profile(
                β,
                points,
                plotter,
                fig=fig_bif,
                ax=ax,
                lineproperties={
                    "c": "k",
                    "label": f"$\\beta, \\lambda = {bifurcation._spectrum[0]['lambda']:.0e}$",
                },
            )
            fig.legend()
            # fig.fill_between(data[0], data[1].reshape(len(data[1])))

            if hasattr(stability, "perturbation"):
                if stability.perturbation["λ"] < 0:
                    _colour = "r"
                    _style = "--"
                else:
                    _colour = "b"
                    _style = ":"

                fig, data = plot_profile(
                    stability.perturbation["β"],
                    points,
                    plotter,
                    fig=fig,
                    ax=ax,
                    lineproperties={
                        "c": _colour,
                        "ls": _style,
                        "lw": 3,
                        "label": f"$\\beta^+, \\lambda = {stability.perturbation['λ']:.0e}$",
                    },
                )

                fig.legend()
                # fig.fill_between(data[0], data[1].reshape(len(data[1])))
                fig.gca().set_title("Perturbation profiles")
                # ax.set_ylim(-2.1, 2.1)
                ax.axhline(0, color="k", lw=0.5)
                fig_bif.savefig(f"{prefix}/second_order_profiles-{i_t}.png")
        try:
            fig, ax = cp.plot_spectrum(history_data)
            ax.set_ylim(-0.01, 0.1)
            ax.set_xlim(parameters["loading"]["min"], parameters["loading"]["max"])
            fig.savefig(f"{prefix}/spectrum.png")
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error plotting spectrum: {e}")

    return fracture_energy, elastic_energy


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

    parameters["model"]["model_dimension"] = 1
    parameters["model"]["model_type"] = "1D"

    L = 1

    if model == "at2":
        parameters["model"]["at_number"] = 2
        parameters["loading"]["min"] = 0.0
        parameters["loading"]["max"] = 3.0
        parameters["loading"]["steps"] = 30
    else:
        parameters["model"]["at_number"] = 1
        parameters["loading"]["min"] = 0.99
        parameters["loading"]["max"] = 1.5
        parameters["loading"]["steps"] = 30

    parameters["geometry"]["geom_type"] = "1d-film"
    parameters["geometry"]["Lx"] = L
    parameters["geometry"]["mesh_size_factor"] = 3

    parameters["stability"]["maxmodes"] = 3

    parameters["stability"]["eigen"]["shift"] = 1e-2

    parameters["stability"]["cone"]["cone_max_it"] = 1000000
    parameters["stability"]["cone"]["cone_atol"] = 1e-8
    parameters["stability"]["cone"]["cone_rtol"] = 1e-8
    parameters["stability"]["cone"]["scaling"] = 1e-2

    parameters["model"]["w1"] = 1
    parameters["model"]["ell"] = 0.1 / L
    parameters["model"]["k_res"] = 0.0
    parameters["model"]["mu"] = 1
    parameters["model"]["kappa"] = (0.5 / L) ** (-2)

    parameters["solvers"]["damage_elasticity"]["alpha_rtol"] = 1e-5
    parameters["solvers"]["newton"]["snes_atol"] = 1e-8
    parameters["solvers"]["newton"]["snes_rtol"] = 1e-8

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    return parameters, signature


if __name__ == "__main__":
    # Set the logging level
    logging.basicConfig(level=logging.INFO)

    # Load parameters
    parameters, signature = load_parameters(
        os.path.join(os.path.dirname(__file__), "parameters_1d.yaml"),
        ndofs=100,
        model="at1",
    )

    # Run computation
    storage = f"output/jumps-1d/{signature[0:6]}"
    visualization = Visualization(storage)

    with dolfinx.common.Timer(f"~Computation Experiment") as timer:
        history_data, stability_data, state = run_computation(parameters, storage)

    from irrevolutions.utils import table_timing_data

    tasks = [
        "~First Order: Equilibrium",
        "~First Order: AltMin-Damage solver",
        "~First Order: AltMin-Elastic solver",
        "~Postprocessing and Vis",
        "~Output and Storage",
        "~Computation Experiment",
    ]

    _timings = table_timing_data(tasks)
    visualization.save_table(_timings, "timing_data")
    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

    ColorPrint.print_bold(f"===================- {signature} -=================")
    ColorPrint.print_bold(f"===================- {storage} -=================")
