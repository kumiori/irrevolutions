#!/usr/bin/env python3
from irrevolutions.utils import ColorPrint, setup_logger_mpi, table_timing_data
from utils.viz import plot_profile, plot_scalar, plot_vector
from utils.plots import plot_AMit_load, plot_energies, plot_force_displacement
from utils.parametric import parameters_vs_elle
from solvers.function import vec_to_functions
from models import BrittleMembraneOverElasticFoundation as ThinFilm
from meshes.primitives import mesh_bar_gmshapi
from default import ResultsStorage, Visualization
from algorithms.so import BifurcationSolver, StabilitySolver
from algorithms.am import HybridSolver
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
import pyvista
import ufl
import yaml
from dolfinx.common import list_timings
from dolfinx.fem import (Constant, Function, FunctionSpace, assemble_scalar,
                         dirichletbc, form, locate_dofs_geometrical, set_bc)
from dolfinx.io import XDMFFile, gmshio
#
from mpi4py import MPI
from petsc4py import PETSc
from pyvista.utilities import xvfb
import basix.ufl

sys.path.append("../")
# from meshes.pacman import mesh_pacman

# logging.basicConfig(level=logging.DEBUG)


# ------------------------------------------------------------------
class ConvergenceError(Exception):
    """Error raised when a solver fails to converge"""


def _make_reasons(reasons):
    return dict(
        [(getattr(reasons, r), r) for r in dir(reasons) if not r.startswith("_")]
    )


SNESReasons = _make_reasons(PETSc.SNES.ConvergedReason())
KSPReasons = _make_reasons(PETSc.KSP.ConvergedReason())


def check_snes_convergence(snes):
    r = snes.getConvergedReason()
    try:
        reason = SNESReasons[r]
        inner = False
        logging.info(f"snes converged with reason {r}: {reason}")
    except KeyError:
        r = snes.getKSP().getConvergedReason()
        try:
            inner = True
            reason = KSPReasons[r]
        except KeyError:
            reason = "unknown reason (petsc4py enum incomplete?), try with -snes_converged_reason and -ksp_converged_reason"
    if r < 0:
        if inner:
            msg = (
                "Inner linear solve failed to converge after %d iterations with reason: %s"
                % (snes.getKSP().getIterationNumber(), reason)
            )
        else:
            msg = reason
        raise ConvergenceError(
            r"""Nonlinear solve failed to converge after %d nonlinear iterations.
                Reason:
                %s"""
            % (snes.getIterationNumber(), msg)
        )


# ------------------------------------------------------------------


comm = MPI.COMM_WORLD

logger = setup_logger_mpi(logging.INFO)

outdir = os.path.join(os.path.dirname(__file__), "output")
prefix = os.path.join(outdir, "thinfilm-bar")
if comm.rank == 0:
    Path(prefix).mkdir(parents=True, exist_ok=True)


def main(parameters, storage=None):
    """Testing nucleation of patterns"""

    Lx = parameters["geometry"]["Lx"]
    Ly = parameters["geometry"]["Ly"]

    tdim = parameters["geometry"]["geometric_dimension"]
    lc = parameters["model"]["ell"] / parameters["geometry"]["mesh_size_factor"]
    geom_type = parameters["geometry"]["geom_type"]
    _nameExp = "thinfilm-" + parameters["geometry"]["geom_type"]

    import hashlib

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    outdir = os.path.join(os.path.dirname(__file__), "output")
    if storage is None:
        prefix = os.path.join(outdir, "thinfilm-bar", signature)
    else:
        prefix = storage

    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    if comm.rank == 0:
        with open(f"{prefix}/signature.md5", "w") as f:
            f.write(signature)

    # generate mesh
    gmsh_model, tdim = mesh_bar_gmshapi(geom_type, Lx, Ly, lc, tdim)

    # Get mesh and meshtags
    model_rank = 0
    mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

    # functional space
    element_u = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1, shape=(tdim,))
    V_u = FunctionSpace(mesh, element_u)

    element_alpha = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1)
    V_alpha = FunctionSpace(mesh, element_alpha)

    u = Function(V_u, name="Displacement")
    zero_u = Function(V_u, name="Boundary Displacement")
    alpha = Function(V_alpha, name="Damage")
    Function(V_alpha, name="Damage Boundary Field")
    alphadot = dolfinx.fem.Function(V_alpha, name="Damage_rate")

    state = {"u": u, "alpha": alpha}

    # Perturbation
    β = Function(V_alpha, name="DamagePerturbation")
    v = Function(V_u, name="DisplacementPerturbation")

    [u, alpha]
    # need upper/lower bound for the damage field
    alpha_lb = Function(V_alpha, name="Lower bound")
    alpha_ub = Function(V_alpha, name="Upper bound")

    # Measures
    dx = ufl.Measure("dx", domain=mesh)
    ufl.Measure("ds", domain=mesh)

    dofs_u_left = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], 0.0))
    dofs_u_right = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], Lx))

    zero_u = Function(V_u)
    u_ = Function(V_u, name="Boundary Displacement")

    Function(V_alpha)

    bc_u_left = dirichletbc(zero_u, dofs_u_left)
    bc_u_right = dirichletbc(u_, dofs_u_right)
    bcs_u = [bc_u_left, bc_u_right]

    # boundary conditions
    bcs_u = []
    bcs_alpha = []
    set_bc(alpha_ub.x.petsc_vec, bcs_alpha)
    alpha_ub.x.petsc_vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )
    bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}

    # loading
    tau = Constant(mesh, np.array(0.0, dtype=PETSc.ScalarType))
    eps_0 = tau * ufl.as_tensor([[1.0, 0], [0, 0]])

    load_par = parameters["loading"]
    loads = np.linspace(load_par["min"], load_par["max"], load_par["steps"])
    # energy (model)
    model = ThinFilm(parameters["model"], eps_0=eps_0)
    f = Constant(mesh, np.array([0, 0], dtype=PETSc.ScalarType))
    external_work = ufl.dot(f, state["u"]) * dx
    total_energy = model.total_energy_density(state) * dx - external_work

    # solvers

    equilibrium = HybridSolver(
        total_energy,
        state,
        bcs,
        bounds=(alpha_lb, alpha_ub),
        solver_parameters=parameters.get("solvers"),
    )

    _stress = model.stress(model.eps(u), alpha)

    bifurcation = BifurcationSolver(
        total_energy, state, bcs, bifurcation_parameters=parameters.get("stability")
    )

    stability = StabilitySolver(
        total_energy, state, bcs, cone_parameters=parameters.get("stability")
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
        "alphadot_norm": [],
        "rate_12_norm": [],
        "unscaled_rate_12_norm": [],
        "cone-stable": [],
    }
    # timestepping

    with XDMFFile(
        comm, f"{prefix}/{_nameExp}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5
    ) as file:
        file.write_mesh(mesh)

    for i_t, t in enumerate(loads):
        tau.value = t

        # update the lower bound
        alpha.x.petsc_vec.copy(alpha_lb.x.petsc_vec)
        alpha_lb.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        ColorPrint.print_bold("   Solving first order: AM   ")
        ColorPrint.print_bold("===================-=========")

        logger.critical(f"{i_t}/{len(loads)}: Solving for t = {t:3.2f} --")
        ColorPrint.print_bold("   Solving first order: Hybrid   ")
        ColorPrint.print_bold("===================-=============")

        logger.info(f"-- {i_t}/{len(loads)}: Solving for t = {t:3.2f} --")

        equilibrium.solve(alpha_lb)

        fracture_energy = comm.allreduce(
            dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(model.damage_energy_density(state) * dx)
            ),
            op=MPI.SUM,
        )
        elastic_energy = comm.allreduce(
            dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(model.elastic_energy_density(state) * dx)
            ),
            op=MPI.SUM,
        )

        # compute rate
        alpha.x.petsc_vec.copy(alphadot.x.petsc_vec)
        alphadot.x.petsc_vec.axpy(-1, alpha_lb.x.petsc_vec)
        alphadot.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        rate_12_norm = equilibrium.scaled_rate_norm(alphadot, parameters)
        rate_12_norm_unscaled = equilibrium.unscaled_rate_norm(alphadot)

        ColorPrint.print_bold("   Solving second order: Rate Pb.    ")
        ColorPrint.print_bold("===================-=================")

        bifurcation.solve(alpha_lb)
        bifurcation.is_elastic()
        inertia = bifurcation.get_inertia()

        ColorPrint.print_bold("   Solving second order: Stability Pb.    ")
        ColorPrint.print_bold("===================-=================")

        stable = stability.my_solve(
            alpha_lb, eig0=bifurcation._spectrum, inertia=inertia
        )

        if bifurcation._spectrum:
            vec_to_functions(bifurcation._spectrum[0]["xk"], [v, β])

            tol = 1e-3
            xs = np.linspace(0 + tol, Lx - tol, 101)
            points = np.zeros((3, 101))
            points[0] = xs

            plotter = pyvista.Plotter(
                title="Perturbation profile",
                window_size=[800, 600],
                shape=(1, 1),
            )
            _plt, data = plot_profile(
                β,
                points,
                plotter,
                subplot=(0, 0),
                lineproperties={"c": "k", "label": "$\\beta$"},
            )
            _plt.gca()
            _plt.legend()
            _plt.fill_between(data[0], data[1].reshape(len(data[1])))
            _plt.title("Perurbation")
            _plt.savefig(f"{prefix}/perturbation-profile-{i_t}.png")
            _plt.close()

            plotter = pyvista.Plotter(
                title="Cone-Perturbation profile",
                window_size=[800, 600],
                shape=(1, 1),
            )

            _plt, data = plot_profile(
                stability.perturbation["beta"],
                points,
                plotter,
                subplot=(0, 0),
                lineproperties={"c": "k", "label": "$\\beta$"},
            )
            _plt.gca()
            _plt.legend()
            _plt.fill_between(data[0], data[1].reshape(len(data[1])))
            _plt.title("Perurbation from the Cone")
            _plt.savefig(f"{prefix}/perturbation-profile-cone-{i_t}.png")
            _plt.close()

        # if stability.perturbation:
        # pass

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
        history_data["total_energy"].append(elastic_energy + fracture_energy)
        history_data["solver_data"].append(equilibrium.data)
        history_data["eigs"].append(bifurcation.data["eigs"])
        history_data["F"].append(stress)
        history_data["cone_data"].append(stability.data)
        history_data["alphadot_norm"].append(alphadot.x.petsc_vec.norm())
        history_data["rate_12_norm"].append(rate_12_norm)
        history_data["unscaled_rate_12_norm"].append(rate_12_norm_unscaled)
        history_data["cone-stable"].append(stable)
        history_data["cone-eig"].append(stability.data["lambda_0"])
        history_data["uniqueness"].append(_unique)
        history_data["inertia"].append(inertia)

        # postprocessing
        with dolfinx.common.Timer("~Postprocessing and Vis"):
            if comm.rank == 0:
                plot_energies(history_data, file=f"{prefix}/{_nameExp}_energies.pdf")
                plot_AMit_load(history_data, file=f"{prefix}/{_nameExp}_it_load.pdf")
                plot_force_displacement(
                    history_data, file=f"{prefix}/{_nameExp}_stress-load.pdf"
                )

        with XDMFFile(
            comm, f"{prefix}/{_nameExp}.xdmf", "a", encoding=XDMFFile.Encoding.HDF5
        ) as file:
            file.write_function(u, t)
            file.write_function(alpha, t)

    # postprocessing
    with dolfinx.common.Timer("~Postprocessing and Vis"):
        if comm.Get_rank == 1:
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

    return history_data, state


def load_parameters(file_path):
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

    parameters["stability"]["cone"]["cone_max_it"] = 400000
    parameters["stability"]["cone"]["cone_atol"] = 1e-6
    parameters["stability"]["cone"]["cone_rtol"] = 1e-6
    parameters["stability"]["cone"]["scaling"] = 1.0e-5

    parameters["geometry"]["Lx"] = 5
    parameters["geometry"]["Ly"] = 5e-1
    # parameters["model"]["model_dimension"] = 2
    # parameters["model"]["model_type"] = '1D'
    # parameters["model"]["w1"] = 1
    parameters["model"]["nu"] = 0
    parameters["model"]["ell_e"] = 0.2
    parameters["model"]["ell"] = parameters["model"]["ell_e"] / 3
    # parameters["model"]["ell"] = .05
    # parameters["model"]["k_res"] = 0.
    parameters["loading"]["min"] = 0.99
    parameters["loading"]["max"] = 1.1
    parameters["loading"]["steps"] = 3

    # parameters["geometry"]["geom_type"] = "traction-bar"
    parameters["geometry"]["mesh_size_factor"] = 3
    # # Get mesh parameters
    # Lx = parameters["geometry"]["Lx"]
    # Ly = parameters["geometry"]["Ly"]
    # tdim = parameters["geometry"]["geometric_dimension"]

    # _nameExp = parameters["geometry"]["geom_type"]
    # ell_ = parameters["model"]["ell"]

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    return parameters, signature


if __name__ == "__main__":
    import argparse

    base_parameters, base_signature = load_parameters("../data/thinfilm/parameters.yml")
    # _storage = f"output/thinfilm-bar/{signature}"
    parser = argparse.ArgumentParser(description="Process evolution.")

    parser.add_argument(
        "--ell_e", type=float, default=0.3, help="internal elastic length"
    )
    args = parser.parse_args()

    if "--ell_e" in sys.argv:
        parameters, signature = parameters_vs_elle(
            parameters= base_parameters, elle=np.float(args.ell_e)
        )
        _storage = (
            f"output/parametric/thinfilm-bar/vs_ell_e/{base_signature}/{signature}"
        )

    else:
        parameters, signature = base_parameters, base_signature
        _storage = f"output/thinfilm-bar/{signature}"

    ColorPrint.print_bold(f"===================-{_storage}-=================")
    pretty_parameters = json.dumps(parameters, indent=2)
    print(pretty_parameters)
    ColorPrint.print_bold(f"===================-{_storage}-=================")

    with dolfinx.common.Timer("~Computation Experiment") as timer:
        history_data, state = main(parameters, _storage)
    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

    df = pd.DataFrame(history_data)

    print(df.drop(["solver_data", "cone_data"], axis=1))
    ColorPrint.print_bold(f"===================-{signature}-=================")
    ColorPrint.print_bold(f"===================-{_storage}-=================")

    # Store and visualise results
    storage = ResultsStorage(MPI.COMM_WORLD, _storage)
    storage.store_results(parameters, history_data, state)

    visualization = Visualization(_storage)

    visualization.visualise_results(history_data)
    visualization.save_table(pd.DataFrame(history_data), "_history_data")

    _timings = table_timing_data()

    visualization.save_table(_timings, "timing_data")
