#!/usr/bin/env python3
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
import basix.ufl

import dolfinx
import dolfinx.mesh
import dolfinx.plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import petsc4py
import pyvista
import ufl
import yaml
from dolfinx.common import list_timings
from dolfinx.fem import (Constant, Function, functionspace, assemble_scalar,
                         dirichletbc, form, locate_dofs_geometrical, set_bc)
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI
from petsc4py import PETSc
from pyvista.utilities import xvfb
import basix.ufl

from irrevolutions.algorithms.am import AlternateMinimisation, HybridSolver
from irrevolutions.algorithms.ls import LineSearch
from irrevolutions.algorithms.so import BifurcationSolver, StabilitySolver
from irrevolutions.meshes.primitives import mesh_bar_gmshapi
from irrevolutions.models import DamageElasticityModel as Brittle
from irrevolutions.solvers.function import vec_to_functions
from irrevolutions.utils import (ColorPrint, _write_history_data, history_data,
                                 norm_H1)
from irrevolutions.utils.plots import plot_energies
from irrevolutions.utils.viz import plot_profile, plot_scalar, plot_vector

logging.basicConfig(level=logging.INFO)

petsc4py.init(sys.argv)
comm = MPI.COMM_WORLD
size = comm.Get_size()

model_rank = 0
test_dir = os.path.dirname(__file__)


def test_linsearch():
    parameters, signature = load_parameters(os.path.join(test_dir, "./parameters.yml"))
    storage = f"output/linesearch/{signature}"
    comm = MPI.COMM_WORLD

    model_rank = 0
    Lx = parameters["geometry"]["Lx"]
    Ly = parameters["geometry"]["Ly"]
    tdim = parameters["geometry"]["geometric_dimension"]
    _nameExp = parameters["geometry"]["geom_type"]
    ell_ = parameters["model"]["ell"]
    _lc = ell_ / parameters["geometry"]["mesh_size_factor"]
    geom_type = parameters["geometry"]["geom_type"]
    kick = parameters["stability"]["continuation"]

    gmsh_model, tdim = mesh_bar_gmshapi(geom_type, Lx, Ly, _lc, tdim)
    mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()
    outdir = os.path.join(os.path.dirname(__file__), "output")

    if storage is None:
        prefix = os.path.join(outdir, "traction_AT2_cone", signature)
    else:
        prefix = storage

    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

        with open(f"{prefix}/signature.md5", "w") as f:
            f.write(signature)

        with open(f"{prefix}/parameters.yaml", "w") as file:
            yaml.dump(parameters, file)

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
    v = Function(V_u, name="Displacement perturbation")
    zero_u = Function(V_u, name="   Boundary Displacement")
    alpha = Function(V_alpha, name="Damage")
    β = Function(V_alpha, name="Damage perturbation")
    zero_alpha = Function(V_alpha, name="Damage Boundary Field")
    alphadot = dolfinx.fem.Function(V_alpha, name="Damage rate")

    state = {"u": u, "alpha": alpha}
    z = [u, alpha]

    # need upper/lower bound for the damage field
    alpha_lb = Function(V_alpha, name="Lower bound")
    alpha_ub = Function(V_alpha, name="Upper bound")

    # Measures
    dx = ufl.Measure("dx", domain=mesh)
    ufl.Measure("ds", domain=mesh)

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

    bc_u_right = dirichletbc(u_, dofs_u_right)
    bcs_u = [bc_u_left, bc_u_right]

    bcs_alpha = [
        dirichletbc(
            np.array(0, dtype=PETSc.ScalarType),
            np.concatenate([dofs_alpha_left, dofs_alpha_right]),
            V_alpha,
        )
    ]
    bcs_alpha = []

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

    equilibrium = AlternateMinimisation(
        total_energy, state, bcs, parameters.get("solvers"), bounds=(alpha_lb, alpha_ub)
    )

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

    linesearch = LineSearch(
        total_energy,
        state,
        linesearch_parameters=parameters.get("stability").get("linesearch"),
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

        ColorPrint.print_bold("   Solving first order: AM   ")
        ColorPrint.print_bold("===================-=========")

        logging.critical(f"-- {i_t}/{len(loads)}: Solving for t = {t:3.2f} --")

        equilibrium.solve()

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

        rate_12_norm = hybrid.scaled_rate_norm(alpha, parameters)
        urate_12_norm = hybrid.unscaled_rate_norm(alpha)

        ColorPrint.print_bold("   Solving second order: Rate Pb.    ")
        ColorPrint.print_bold("===================-=================")

        # n_eigenvalues = 10
        is_unique = bifurcation.solve(alpha_lb)
        is_elastic = not bifurcation._is_critical(alpha_lb)
        inertia = bifurcation.get_inertia()

        ColorPrint.print_bold("   Solving second order: Cone Pb.    ")
        ColorPrint.print_bold("===================-=================")

        z0 = (
            bifurcation._spectrum[0]["xk"]
            if bifurcation._spectrum and "xk" in bifurcation._spectrum[0]
            else None
        )

        stable = stability.solve(alpha_lb, eig0=z0, inertia=inertia)

        if not stable:
            vec_to_functions(stability.solution["xt"], [v, β])
            perturbation = {"v": v, "beta": β}

            interval = linesearch.get_unilateral_interval(state, perturbation)

            order = 3
            h_opt, energies_1d, p, _ = linesearch.search(
                state, perturbation, interval, m=order
            )
            logging.critical(f" *> State is unstable: {not stable}")
            logging.critical(f"line search interval is {interval}")
            logging.critical(f"perturbation energies: {energies_1d}")
            logging.critical(f"hopt: {h_opt}")
            logging.critical(f"lambda_t: {stability.solution['lambda_t']}")
            x_plot = np.linspace(interval[0], interval[1], order + 1)
            fig, axes = plt.subplots(1, 1)
            plt.scatter(x_plot, energies_1d)
            plt.scatter(h_opt, 0, c="k", s=40, marker="|", label=f"$h^*={h_opt:.2f}$")
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
            fig.savefig(f"{prefix}/energy_interpolation-{order}.png")
            plt.close()

            orders = [2, 3, 4, 10, 30]

            fig, axes = plt.subplots(1, 1, figsize=(5, 8))

            for order in orders:
                x_plot = np.linspace(interval[0], interval[1], order + 1)
                h_opt, energies_1d, p, _ = linesearch.search(
                    state, perturbation, interval, m=order
                )
                xs = np.linspace(interval[0], interval[1], 30)

                plt.scatter(x_plot, energies_1d)
                axes.plot(xs, p(xs), label=f"Energy slice order {order}")
                plt.scatter(
                    h_opt, 0, s=60, label=f"$h^*-{ order }={h_opt:.2f}$", alpha=0.5
                )
                plt.scatter(h_opt, p(h_opt), c="k", s=40, alpha=0.5)

            axes.legend()
            axes.spines["top"].set_visible(False)
            axes.spines["right"].set_visible(False)
            axes.spines["left"].set_visible(False)
            axes.spines["bottom"].set_visible(False)
            axes.set_yticks([0])
            axes.set_xlabel("h")
            axes.set_ylabel("$E_h - E_0$")
            axes.axhline(0, c="k")
            fig.savefig(f"{prefix}/energy_interpolation-orders.png")

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

            _plt, data = plot_profile(
                β,
                points,
                plotter,
                lineproperties={"c": "k", "label": "$\\beta$"},
            )
            _plt.gca()
            _plt.legend()
            _plt.fill_between(data[0], data[1].reshape(len(data[1])))
            _plt.title("Perurbation")
            _plt.savefig(f"{prefix}/perturbation-profile.png")
            _plt.close()

            # perturb the state
            # compute norm of state
            norm_state_pre = sum([norm_H1(v) for v in state.values()])

            if kick:
                linesearch.perturb(state, perturbation, h_opt)
                hybrid.solve(alpha_lb)

            norm_state_post = sum([norm_H1(v) for v in state.values()])
            # print norms and relative norms difference
            logging.critical(f"norm state pre: {norm_state_pre}")
            logging.critical(f"norm state post: {norm_state_post}")
            logging.critical(
                f"relative norm difference: {(norm_state_post-norm_state_pre) / norm_state_pre}"
            )

        ColorPrint.print_bold(f"State is elastic: {is_elastic}")
        ColorPrint.print_bold(f"Evolution is unique: {is_unique}")
        ColorPrint.print_bold(f"State's inertia: {inertia}")
        logging.critical(f"alpha vector norm: {alpha.x.petsc_vec.norm()}")
        logging.critical(f"alpha lb norm: {alpha_lb.x.petsc_vec.norm()}")
        logging.critical(f"alphadot norm: {alphadot.x.petsc_vec.norm()}")
        logging.critical(f"vector norms [u, alpha]: {[zi.x.petsc_vec.norm() for zi in z]}")
        logging.critical(f"scaled rate state_12 norm: {rate_12_norm}")
        logging.critical(f"unscaled scaled rate state_12 norm: {urate_12_norm}")

        fracture_energy = comm.allreduce(
            assemble_scalar(form(model.damage_energy_density(state) * dx)),
            op=MPI.SUM,
        )
        elastic_energy = comm.allreduce(
            assemble_scalar(form(model.elastic_energy_density(state) * dx)),
            op=MPI.SUM,
        )
        _stress = model.stress(model.eps(u), alpha)

        comm.allreduce(
            assemble_scalar(form(_stress[0, 0] * dx)),
            op=MPI.SUM,
        )

        _unique = True if inertia[0] == 0 and inertia[1] == 0 else False

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

        ColorPrint.print_bold("   Written timely data.    ")
        print()
    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])
    df = pd.DataFrame(history_data)
    print(df.drop(["equilibrium_data", "cone_data"], axis=1))

    # Viz

    #
    xvfb.start_xvfb(wait=0.05)
    pyvista.OFF_SCREEN = True

    # if size == 1:
    if comm.rank == 0:
        plotter = pyvista.Plotter(
            title="Displacement",
            window_size=[1600, 600],
            shape=(1, 2),
        )
        _plt = plot_scalar(alpha, plotter, subplot=(0, 0))
        _plt = plot_vector(u, plotter, subplot=(0, 1))
        _plt.screenshot(f"{prefix}/traction-state.png")

    if comm.rank == 0:
        plot_energies(history_data, file=f"{prefix}/{_nameExp}_energies.pdf")


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

    # parameters["stability"]["cone"]["cone_max_it"] = 400000
    parameters["stability"]["cone"]["cone_atol"] = 1e-6
    parameters["stability"]["cone"]["cone_rtol"] = 1e-6
    parameters["stability"]["cone"]["scaling"] = 1e-4
    parameters["stability"]["continuation"] = True

    parameters["model"]["model_dimension"] = 2
    parameters["model"]["model_type"] = "2D"
    parameters["model"]["w1"] = 1
    parameters["model"]["ell"] = 0.1
    parameters["model"]["k_res"] = 0.0
    parameters["loading"]["min"] = 0.0
    parameters["loading"]["max"] = 1.3
    parameters["loading"]["steps"] = 30

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    return parameters, signature


if __name__ == "__main__":
    test_linsearch()
