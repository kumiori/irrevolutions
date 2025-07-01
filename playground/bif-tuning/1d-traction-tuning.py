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
# from irrevolutions.algorithms.gf import JumpSolver

# import irrevolutions.utils.postprocess as pp
import crunchy.plots as cp
from irrevolutions.models.one_dimensional import Brittle1D as Bar
# from irrevolutions.models.one_dimensional import FilmModel1D as ThinFilm

from irrevolutions.utils import indicator_function

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
mode_shapes_data = {
    "time_steps": [],
    "mesh": [],
    "point_values": {},
    "global_values": {
        "R_vector": [],
        "R_cone": [],
        "D_theory": [],
        "D_support": [],
        "load": [],
    },
}


def save_mode_shapes_to_npz(
    load,
    timestep,
    state,
    bifurcation,
    stability,
    parameters,
    model,
    points,
    D_theory,
    D_support,
    num_modes=1,
    prefix="output",
):
    # Extract spatial points
    data_stability = get_datapoints(stability.perturbation["β"], points)
    mode_shapes_data["point_values"]["x_values"] = data_stability[0]

    # Compute Rayleigh ratios
    _R_vector = rayleigh_ratio(
        (bifurcation.perturbation["v"], bifurcation.perturbation["β"]), state, model
    )
    _R_cone = rayleigh_ratio(
        (stability.perturbation["v"], stability.perturbation["β"]), state, model
    )

    for mode in range(1, num_modes + 1):
        bifurcation_values_mode_β = get_datapoints(
            bifurcation.perturbation["β"], points
        )[1].flatten()
        bifurcation_values_mode_v = get_datapoints(
            bifurcation.perturbation["v"], points
        )[1].flatten()
        stability_values_mode_β = get_datapoints(stability.perturbation["β"], points)[
            1
        ].flatten()
        stability_values_mode_v = get_datapoints(stability.perturbation["v"], points)[
            1
        ].flatten()
        # stability_residual_w = get_datapoints(stability.residual["w"], points)[
        #     1
        # ].flatten()
        # stability_residual_ζ = get_datapoints(stability.residual["ζ"], points)[
        #     1
        # ].flatten()

        mode_key = f"mode_{mode}"

        if mode_key not in mode_shapes_data["point_values"]:
            mode_shapes_data["point_values"][mode_key] = []

        mode_shapes_data["point_values"][mode_key].append(
            {
                "bifurcation_β": bifurcation_values_mode_β,
                "bifurcation_v": bifurcation_values_mode_v,
                "stability_β": stability_values_mode_β,
                "stability_v": stability_values_mode_v,
            }
        )
        # "stability_residual_w": stability_residual_w,
        # "stability_residual_ζ": stability_residual_ζ,
        # }

    # Global metadata
    mode_shapes_data["time_steps"].append(timestep)

    if "global_values" not in mode_shapes_data:
        mode_shapes_data["global_values"] = {
            "R_vector": [],
            "R_cone": [],
            "D_theory": [],
            "D_support": [],
            "load": [],
        }

    mode_shapes_data["global_values"]["R_vector"].append(_R_vector)
    mode_shapes_data["global_values"]["R_cone"].append(_R_cone)
    mode_shapes_data["global_values"]["D_theory"].append(D_theory)
    mode_shapes_data["global_values"]["D_support"].append(D_support)
    mode_shapes_data["global_values"]["load"].append(load)

    mode_shapes_data["mesh"] = get_datapoints(stability.perturbation["β"], points)[0][
        :, 0
    ]
    # Save
    np.savez(f"{prefix}/mode_shapes_data.npz", **mode_shapes_data)
    print(f"Saved mode shape data to {prefix}/mode_shapes_data.npz")


def compute_coefficients_from_state(state, model, parameters, tol_hom=1e-6):
    """
    Compute coefficients a, b, c for the Rayleigh quotient based on a homogeneous state.

    Parameters
    ----------
    state : dict
        Dictionary containing FEM functions: {'u': Function, 'alpha': Function}
    parameters : dict
        Dictionary containing 'model' and 'geometry' parameters, including:
        E, ell, mu(alpha), mu'(alpha), mu''(alpha), w''(alpha)
    tol_hom : float
        Tolerance to detect homogeneity via H1 seminorm of alpha

    Returns
    -------
    dict with keys 'a', 'b', 'c'
    """

    u = state["u"]
    alpha = state["alpha"]

    mesh = alpha.function_space.mesh
    dx = ufl.Measure("dx", domain=mesh)
    L = parameters["geometry"]["Lx"]
    ell = parameters["model"]["ell"]

    # Check homogeneity of alpha (via H1 seminorm)
    alpha_dx = ufl.grad(alpha)
    seminorm_form = form(ufl.inner(alpha_dx, alpha_dx) * dx)
    seminorm = assemble_scalar(seminorm_form)
    if seminorm > tol_hom:
        raise ValueError(f"Damage field is not homogeneous: H1 seminorm = {seminorm}")

    # Average strain and damage
    # u_avg = assemble_scalar(form(u[0] * dx)) / L
    u_vals = u.x.array
    # Compute effective strain
    strain = (np.max(u_vals) - np.min(u_vals)) / L
    alpha_avg = assemble_scalar(form(alpha * dx)) / L

    E = model.parameters["E"]

    # Symbolic variable
    alpha_sym = ufl.variable(alpha_avg)

    mu_expr = E * model.a(alpha_sym)
    mu_prime = ufl.diff(mu_expr, alpha_sym)
    mu_dd = ufl.diff(mu_prime, alpha_sym)

    w_expr = model.w(alpha_sym)
    w_prime = ufl.diff(w_expr, alpha_sym)
    w_dd = ufl.diff(w_prime, alpha_sym)

    # Evaluate
    mu_val = float(mu_expr)
    mu_prime_val = float(mu_prime)
    mu_dd_val = float(mu_dd)
    w_val = float(w_expr)
    w_prime_val = float(w_prime)
    w_dd_val = float(w_dd)
    # Compute denominator N
    N = 0.5 * mu_dd(alpha_avg) * strain**2 + w_dd(alpha_avg)

    a_val = ell**2 / N
    b_val = mu_val / N
    c_val = -mu_prime_val / mu_val * strain

    return {"a": a_val, "b": b_val, "c": c_val}


def rayleigh_ratio(
    z,
    state,
    model,
):
    (v, β) = z
    dx = ufl.Measure("dx", v.function_space.mesh)

    u = state["u"]
    alpha = state["alpha"]

    a, b, c = compute_coefficients_from_state(
        {"u": u, "alpha": alpha}, model, parameters, tol_hom=1e-6
    ).values()
    numerator = (
        a * ufl.inner(β.dx(0), β.dx(0))
        + b * ufl.inner(v.dx(0)[0] - c * β, v.dx(0)[0] - c * β)
    ) * dx
    denominator = ufl.inner(β, β) * dx
    _den = assemble_scalar(form(denominator))

    if _den == 0:
        logging.critical(
            "Denominator in Rayleigh quotient is zero, cannot compute ratio."
        )
        return np.nan, (a, b, c)

    R = assemble_scalar(form(numerator)) / _den

    return R, (a, b, c)


def parallel_assemble_scalar(ufl_form):
    compiled_form = dolfinx.fem.form(ufl_form)
    comm = compiled_form.mesh.comm
    local_scalar = dolfinx.fem.assemble_scalar(compiled_form)
    return comm.allreduce(local_scalar, op=MPI.SUM)


def run_computation(parameters, storage=None):
    Lx = parameters["geometry"]["Lx"]
    _nameExp = parameters["geometry"]["geom_type"]
    parameters["model"]["ell"]

    # Get geometry model
    parameters["geometry"]["geom_type"]

    # N = max(_N, parameters["model"]["ell"] / parameters["geometry"]["mesh_size_factor"])
    N = parameters["model"]["ell"] / parameters["geometry"]["mesh_size_factor"]
    logger.info(f"Mesh size: {N}")

    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 100)
    # mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, int(1 / N))
    # outdir = os.path.join(os.path.dirname(__file__), "output")

    if MPI.COMM_WORLD.rank == 0:
        Path(storage).mkdir(parents=True, exist_ok=True)

    prefix = storage

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
    alpha_zero = Function(V_alpha, name="Damage0")
    alpha_zero.interpolate(lambda x: np.zeros_like(x[0]))

    eps_t = dolfinx.fem.Constant(mesh, np.array(1.0, dtype=PETSc.ScalarType))
    u_zero.interpolate(lambda x: eps_t / 2.0 * (2 * x[0] - Lx))

    tilde_u = Function(V_u, name="BoundaryDatum")
    tilde_u.interpolate(lambda x: np.ones_like(x[0]))

    dofs_u_left = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], 0.0))
    dofs_u_right = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], Lx))

    dofs_alpha_left = locate_dofs_geometrical(V_alpha, lambda x: np.isclose(x[0], 0.0))
    dofs_alpha_right = locate_dofs_geometrical(V_alpha, lambda x: np.isclose(x[0], Lx))

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
    # bcs_alpha = [
    #     dirichletbc(alpha_zero, dofs_alpha_right),
    #     dirichletbc(alpha_zero, dofs_alpha_left),
    # ]

    bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}
    parameters["notes"] = (
        # "bcs_u: substrate Dirichlet BC on u, bcs_alpha: homogeneous Dirichlet BC on alpha"
        "bcs_u: substrate Dirichlet, bcs_alpha: natural"
    )
    # Measures
    dx = ufl.Measure("dx", domain=mesh)
    # model = ThinFilm(parameters["model"])

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
    history_data["Rayleigh"] = []
    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    if MPI.COMM_WORLD.rank == 0:
        with open(f"{prefix}/parameters.yaml", "w") as file:
            yaml.dump(parameters, file)
        with open(f"{prefix}/signature.md5", "w") as f:
            f.write(signature)

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

        constraints = bifurcation.setup_constraints(alpha_lb)
        bifurcation.inertia_setup(constraints)
        inertia = bifurcation.get_inertia()
        logger.info(f"State inertia: {inertia}")

        is_unique = bifurcation.solve(alpha_lb, inertia=inertia)
        # is_elastic = not bifurcation._is_critical(alpha_lb)

        z0 = (
            bifurcation._spectrum[0]["xk"]
            if bifurcation._spectrum and "xk" in bifurcation._spectrum[0]
            else None
        )

        stable = stability.solve(alpha_lb, eig0=z0, inertia=inertia)

        # stable = True

        logger.info(f"Stability of state at load {t:.2f}: {stable}")
        ColorPrint.print_bold(f"Evolution is unique: {is_unique}")
        ColorPrint.print_bold(f"State's inertia: {inertia}")
        ColorPrint.print_bold(f"State's stable: {stable}")

        # equilibrium.log()
        bifurcation.log()
        stability.log()
        ColorPrint.print_bold(f"===================- {prefix} -=================")

        # if not stable:
        #     iterator.pause_time()
        #     logger.info(f"Time paused at {t:.2f}")

        #     vec_to_functions(stability.solution["xt"], [v, β])
        #     perturbation = {"v": v, "beta": β}
        #     interval = linesearch.get_unilateral_interval(state, perturbation)

        #     order = 4
        #     h_opt, energies_1d, p, _ = linesearch.search(
        #         state, perturbation, interval, m=order
        #     )

        # add coefficients to history data and log Rayleigh quotients

        compute_coefficients_from_state(state, model, parameters)

        R_ball, (a, b, c) = rayleigh_ratio(
            (bifurcation.perturbation["v"], bifurcation.perturbation["β"]), state, model
        )

        if stability.solution["xt"] is None:
            logging.warning(
                "Stability perturbation is None, skipping Rayleigh ratio for cone."
            )
            R_cone = np.nan
            _support = np.nan
            D_support = np.nan

        else:
            tol = 1e-3
            xs = np.linspace(0 + tol, 1 - tol, 101)
            points = np.zeros((3, 101))
            points[0] = xs

            _support = indicator_function(stability.perturbation["β"])
            D_support = parallel_assemble_scalar(_support * dx)
            R_cone, (a, b, c) = rayleigh_ratio(
                (stability.perturbation["v"], stability.perturbation["β"]), state, model
            )

            D_theory = (np.pi**2 * a / (b * c**2)) ** (1 / 3)
            save_mode_shapes_to_npz(
                t,
                i_t,
                state,
                bifurcation,
                stability,
                parameters,
                model,
                points=points,
                D_theory=D_theory,
                D_support=D_support,
                num_modes=1,
                prefix=prefix,
            )

        logging.info(f"Rayleigh quotients: R_ball={R_ball:.4f}, R_cone={R_cone:.4f}")
        # history_data["Rayleigh"].append(
        # )
        rayleigh = {"R_ball": R_ball, "R_cone": R_cone, "a": a, "b": b, "c": c}

        plot_bifurcation_spectrum(bifurcation.spectrum, Lx, i_t, prefix, inertia)

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

        with dolfinx.common.Timer(f"~Output and Storage") as timer:
            dump_output(
                _nameExp,
                prefix,
                history_data,
                u,
                alpha,
                equilibrium,
                bifurcation,
                stability,
                rayleigh,
                t,
                fracture_energy,
                elastic_energy,
            )

        # if stable:

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
    rayleigh,
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
        history_data["Rayleigh"].append(rayleigh)

        if comm.rank == 0:
            a_file = open(f"{prefix}/time_data.json", "w")
            json.dump(history_data, a_file)
            a_file.close()


from matplotlib.cm import get_cmap


def plot_bifurcation_spectrum(spectrum, Lx, i_t, prefix, inertia=None, n_points=101):
    tol = 1e-3
    xs = np.linspace(0 + tol, Lx - tol, n_points)
    points = np.zeros((3, n_points))
    points[0] = xs

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax_beta, ax_v = axes

    # Colormaps for progression
    stable_cmap = get_cmap("Blues")
    unstable_cmap = get_cmap("Reds")
    max_n = max(mode["n"] for mode in spectrum)
    neg_count = 0

    for mode in spectrum:
        n = mode["n"]
        λ = mode["lambda"]
        v, β = mode["v"], mode["beta"]
        xk = mode["xk"]

        vec_to_functions(xk, [v, β])

        is_unstable = λ < 0
        if is_unstable:
            neg_count += 1

        # Choose color and linestyle
        cmap = unstable_cmap if is_unstable else stable_cmap
        color = cmap(n / max_n)
        linestyle = "-" if is_unstable else "--"

        label_beta = f"$\\beta_{{{n}}}$, λ={λ:.2e}"
        label_v = f"$v_{{{n}}}$"

        _plot_profile(
            β,
            points,
            ax_beta,
            lineproperties={
                "label": label_beta,
                "color": color,
                "linestyle": linestyle,
                "linewidth": 2,
            },
        )
        _plot_profile(
            v,
            points,
            ax_v,
            lineproperties={
                "label": label_v,
                "color": color,
                "linestyle": linestyle,
                "linewidth": 1.5,
            },
        )

    ax_beta.set_title("Damage modes $\\beta_n$")
    ax_v.set_title("Displacement modes $v_n$")
    ax_v.set_xlabel("x")
    ax_beta.legend(ncol=2, fontsize="small")
    ax_v.legend(ncol=2, fontsize="small")

    if inertia:
        expected_neg = inertia[0]
        success = "\u2713" if neg_count == expected_neg else "\u2717"
        # success = "[OK]" if neg_count == expected_neg else "[FAIL]"
        title = f"Bifurcation spectrum at step {i_t} ({prefix}) — expected {expected_neg}/{sum(inertia)} unstable modes, got {neg_count} {success}, coverage {len(spectrum) / sum(inertia):.0%} spectrum"
    else:
        title = f"Bifurcation spectrum at step {i_t} ({prefix})"
    fig.suptitle(title)

    plt.tight_layout()
    fig.savefig(f"{prefix}/perturbation-spectrum-{i_t}.png")
    # close fig
    plt.close(fig)


from irrevolutions.utils.viz import get_datapoints


def _plot_profile(u, points, ax, lineproperties={}):
    points_on_proc, u_values = get_datapoints(u, points)
    ax.plot(points_on_proc[:, 0], u_values, **lineproperties)
    ax.legend()
    return ax, (points_on_proc[:, 0], u_values)


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

        # fig, data = plot_profile(
        #     state["alpha"],
        #     points,
        #     plotter,
        #     lineproperties={
        #         "c": "k",
        #         "label": f"$\\alpha$ with $\\ell$ = {parameters['model']['ell']:.2f}",
        #     },
        # )
        # ax = fig.gca()
        # ax.set_ylim(0, 1)
        # fig, data = plot_profile(
        #     state["u"],
        #     points,
        #     plotter,
        #     fig=fig,
        #     ax=ax,
        #     lineproperties={
        #         "c": "g",
        #         "label": "$u$",
        #         "marker": "o",
        #     },
        # )

        # fig, data = plot_profile(
        #     u_zero,
        #     points,
        #     plotter,
        #     fig=fig,
        #     ax=ax,
        #     lineproperties={"c": "r", "lw": 3, "label": "$u_0$"},
        # )
        # fig.legend()
        # fig.gca().set_title("Solution state")
        # # ax.set_ylim(-2.1, 2.1)
        # ax.axhline(0, color="k", lw=0.5)
        # fig.savefig(f"{prefix}/state_profile-{i_t}.png")

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
        parameters["loading"]["max"] = 3
        parameters["loading"]["steps"] = 30

    parameters["geometry"]["geom_type"] = "1d-traction"
    parameters["geometry"]["Lx"] = L
    # parameters["geometry"]["mesh_size_factor"] = 10

    parameters["stability"]["maxmodes"] = 3
    parameters["stability"]["eigen"]["shift"] = -5e-1
    parameters["stability"]["eigen"]["eps_tol"] = 1e-8

    parameters["stability"]["cone"]["cone_max_it"] = 1000000
    parameters["stability"]["cone"]["cone_atol"] = 1e-6
    parameters["stability"]["cone"]["cone_rtol"] = 1e-6
    parameters["stability"]["cone"]["scaling"] = 1e-3
    parameters["stability"]["mode"] = "always"

    parameters["model"]["w1"] = 1
    parameters["model"]["ell"] = 0.3 / L
    parameters["model"]["k_res"] = 0.0
    parameters["model"]["mu"] = 1
    parameters["model"]["kappa"] = (1 / L) ** (-2)

    parameters["solvers"]["damage_elasticity"]["alpha_rtol"] = 1e-5
    parameters["solvers"]["newton"]["snes_atol"] = 1e-8
    parameters["solvers"]["newton"]["snes_rtol"] = 1e-8

    jump_parameters = {
        "tau": 1,
        "max_steps": 30,
        "rtol": 1e-8,
        "verbose": True,
        "save_state": False,
    }
    parameters["solvers"]["jump"] = jump_parameters

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    return parameters, signature


if __name__ == "__main__":
    # Set the logging level
    logging.basicConfig(level=logging.INFO)

    # Load parameters
    parameters, signature = load_parameters(
        os.path.join(os.path.dirname(__file__), "parameters.yaml"),
        ndofs=100,
        model="at1",
    )

    # Run computation
    storage = f"output/{signature[0:6]}"
    storage = f"output/last"
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
