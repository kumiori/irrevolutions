#!/usr/bin/env python3
from dataclasses import asdict
import json
import logging
import os
import sys
import basix.ufl

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
from dolfinx.fem import (
    Constant,
    Function,
    assemble_scalar,
    dirichletbc,
    form,
    locate_dofs_geometrical,
)
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc

from irrevolutions.algorithms.am import AlternateMinimisation1D, HybridSolver
from irrevolutions.algorithms.so import BifurcationSolver, StabilitySolver
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
from irrevolutions.solvers.function import vec_to_functions
from irrevolutions.utils import (
    ColorPrint,
    setup_logger_mpi,
)
from irrevolutions.utils.plots import plot_AMit_load, plot_energies

#
from irrevolutions.utils.viz import plot_profile, setup_pyvista_offscreen

"""The fundamental problem of a 1d bar in traction.
0|(WWWWWWWWWWWWWWWWWWWWWW)|========> t

(WWW): endommageable bar, y=(u, alpha)
load: displacement hard-t
"""


# logging.getLogger().setLevel(logging.INFO)


logger = setup_logger_mpi(logging.INFO)


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


def run_computation(parameters, storage=None):
    Lx = parameters["geometry"]["Lx"]

    _nameExp = parameters["geometry"]["geom_type"]
    parameters["model"]["ell"]

    # Get geometry model
    parameters["geometry"]["geom_type"]
    _N = int(parameters["geometry"]["N"])

    # Create the mesh of the specimen with given dimensions
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, _N)

    outdir = os.path.join(os.path.dirname(__file__), "output")
    if storage is None:
        prefix = os.path.join(outdir, f"test_1d-N{parameters['model']['N']}")
    else:
        prefix = storage

    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    import hashlib

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    if comm.rank == 0:
        with open(f"{prefix}/parameters.yaml", "w") as file:
            yaml.dump(parameters, file)

        with open(f"{prefix}/signature.md5", "w") as f:
            f.write(signature)

    with XDMFFile(
        comm, f"{prefix}/{_nameExp}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5
    ) as file:
        file.write_mesh(mesh)

    # Functional Setting
    element_u = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1)

    element_alpha = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1)

    V_u = dolfinx.fem.functionspace(mesh, element_u)
    V_alpha = dolfinx.fem.functionspace(mesh, element_alpha)

    u = dolfinx.fem.Function(V_u, name="Displacement")
    u_ = dolfinx.fem.Function(V_u, name="BoundaryDisplacement")

    alpha = dolfinx.fem.Function(V_alpha, name="Damage")

    # Perturbations
    β = Function(V_alpha, name="DamagePerturbation")
    v = Function(V_u, name="DisplacementPerturbation")

    # Pack state
    state = {"u": u, "alpha": alpha}

    # Bounds
    alpha_ub = dolfinx.fem.Function(V_alpha, name="UpperBoundDamage")
    alpha_lb = dolfinx.fem.Function(V_alpha, name="LowerBoundDamage")

    dx = ufl.Measure("dx", domain=mesh)
    ufl.Measure("ds", domain=mesh)

    # Useful references
    Lx = parameters.get("geometry").get("Lx")

    # Define the state
    u = Function(V_u, name="Unknown")
    u_ = Function(V_u, name="Boundary Unknown")
    zero_u = Function(V_u, name="Boundary Unknown")

    # Boundary sets

    # dofs_alpha_left = locate_dofs_geometrical(
    #     V_alpha, lambda x: np.isclose(x[0], 0.))
    # dofs_alpha_right = locate_dofs_geometrical(
    #     V_alpha, lambda x: np.isclose(x[0], Lx))

    dofs_u_left = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], 0.0))
    dofs_u_right = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], Lx))

    # Boundary data

    u_.interpolate(lambda x: np.ones_like(x[0]))

    # Bounds (nontrivial)

    alpha_lb.interpolate(lambda x: np.zeros_like(x[0]))
    alpha_ub.interpolate(lambda x: np.ones_like(x[0]))

    # Set Bcs Function
    zero_u.interpolate(lambda x: np.zeros_like(x[0]))
    u_.interpolate(lambda x: np.ones_like(x[0]))

    for f in [zero_u, u_, alpha_lb, alpha_ub]:
        f.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

    bc_u_left = dirichletbc(np.array(0, dtype=PETSc.ScalarType), dofs_u_left, V_u)

    bc_u_right = dirichletbc(u_, dofs_u_right)
    bcs_u = [bc_u_left, bc_u_right]

    bcs_alpha = []

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

    # Material behaviour

    def a(alpha):
        # k_res = parameters["model"]['k_res']
        return (1 - alpha) ** 2

    def a_atk(alpha):
        parameters["model"]["k_res"]
        _k = parameters["model"]["k"]
        return (1 - alpha) / ((_k - 1) * alpha + 1)

    def w(alpha):
        """
        Return the homogeneous damage energy term,
        as a function of the state
        (only depends on damage).
        """
        # Return w(alpha) function
        return alpha

    def elastic_energy_density_atk(state):
        """
        Returns the elastic energy density from the state.
        """
        # Parameters
        alpha = state["alpha"]
        u = state["u"]
        eps = ufl.grad(u)

        _mu = parameters["model"]["E"]
        energy_density = _mu / 2.0 * a_atk(alpha) * ufl.inner(eps, eps)
        return energy_density

    def elastic_energy_density(state):
        """
        Returns the elastic energy density from the state.
        """
        # Parameters
        alpha = state["alpha"]
        u = state["u"]
        eps = ufl.grad(u)

        _mu = parameters["model"]["E"]
        energy_density = _mu / 2.0 * a(alpha) * ufl.inner(eps, eps)
        return energy_density

    def damage_energy_density(state):
        """
        Return the damage dissipation density from the state.
        """
        # Get the material parameters
        _w1 = parameters["model"]["w1"]
        _ell = parameters["model"]["ell"]
        # Get the damage
        alpha = state["alpha"]
        # Compute the damage gradient
        grad_alpha = ufl.grad(alpha)
        # Compute the damage dissipation density
        D_d = _w1 * w(alpha) + _w1 * _ell**2 * ufl.dot(grad_alpha, grad_alpha)
        return D_d

    def stress(state):
        """
        Return the one-dimensional stress
        """
        u = state["u"]
        alpha = state["alpha"]

        return parameters["model"]["E"] * a(alpha) * u.dx() * dx

    total_energy = (elastic_energy_density(state) + damage_energy_density(state)) * dx

    # Energy functional
    # f = Constant(mesh, 0)
    f = Constant(mesh, np.array(0, dtype=PETSc.ScalarType))

    load_par = parameters["loading"]
    loads = np.linspace(load_par["min"], load_par["max"], load_par["steps"])

    loads = [0.0, 0.5, 0.99, 1.01, 1.3]

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

    equilibrium = AlternateMinimisation1D(
        setup.energy,
        setup.state,
        solver_bcs,
        parameters.get("solvers"),
        bounds=alpha_bounds,
    )

    hybrid = HybridSolver(
        setup.energy,
        setup.state,
        solver_bcs,
        bounds=alpha_bounds,
        solver_parameters=parameters.get("solvers"),
    )

    bifurcation = BifurcationSolver(
        setup.energy,
        setup.state,
        solver_bcs,
        bifurcation_parameters=parameters.get("stability"),
    )

    stability = StabilitySolver(
        setup.energy,
        setup.state,
        solver_bcs,
        cone_parameters=parameters.get("stability"),
    )

    mode_shapes_data = {
        "time_steps": [],
        "point_values": {
            "x_values": [],
        },
    }
    num_modes = 1

    # Extra data fields
    # history_data["F"] = []

    logging.basicConfig(level=logging.INFO)

    for i_t, t in enumerate(loads):
        u_.interpolate(lambda x: t * np.ones_like(x[0]))
        u_.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        # update the lower bound
        alpha.x.petsc_vec.copy(alpha_lb.x.petsc_vec)
        alpha_lb.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        logger.critical(f"-- Solving for t = {t:3.2f} --")

        equilibrium.solve()
        hybrid.solve(alpha_lb)

        # n_eigenvalues = 10
        is_unique = bifurcation.solve(alpha_lb)
        is_elastic = not bifurcation._is_critical(alpha_lb)
        inertia = bifurcation.get_inertia()
        # stability.save_eigenvectors(filename=f"{prefix}/{_nameExp}_eigv_{t:3.2f}.xdmf")

        ColorPrint.print_bold(f"State is elastic: {is_elastic}")
        ColorPrint.print_bold(f"State's inertia: {inertia}")
        ColorPrint.print_bold(f"Evolution is unique: {is_unique}")

        z0 = (
            bifurcation._spectrum[0]["xk"]
            if bifurcation._spectrum and "xk" in bifurcation._spectrum[0]
            else None
        )

        stable = stability.solve(alpha_lb, eig0=z0, inertia=inertia)

        with dolfinx.common.Timer("~Postprocessing and Vis"):
            if comm.Get_size() == 1:
                if bifurcation._spectrum:
                    vec_to_functions(bifurcation._spectrum[0]["xk"], [v, β])

                    tol = 1e-3
                    xs = np.linspace(0 + tol, Lx - tol, 101)
                    points = np.zeros((3, 101))
                    points[0] = xs

                    setup_pyvista_offscreen()
                    plotter = pyvista.Plotter(
                        title="Perturbation profile",
                        window_size=[800, 600],
                        shape=(1, 2),
                    )
                    _plt, data_bifurcation = plot_profile(
                        β,
                        points,
                        plotter,
                        subplot=(0, 0),
                        lineproperties={"c": "k", "label": "$\\beta$"},
                        subplotnumber=1,
                    )
                    ax = _plt.gca()
                    ax.set_xlabel("x")
                    ax.set_yticks([-1, 0, 1])
                    ax.set_ylabel("$\\beta$")
                    handles, labels = ax.get_legend_handles_labels()
                    if labels:
                        _plt.legend()
                    _plt.fill_between(
                        data_bifurcation[0],
                        data_bifurcation[1].reshape(len(data_bifurcation[1])),
                    )
                    _plt.title("Perurbation in Vector Space")
                    _plt.savefig(f"{prefix}/perturbation-profile-{i_t}.png")
                    _plt.close()

                    # plotter = pyvista.Plotter(
                    #     title="Cone-Perturbation profile",
                    #     window_size=[800, 600],
                    #     shape=(1, 1),
                    # )

                    _plt, data_stability = plot_profile(
                        stability.perturbation["β"],
                        points,
                        plotter,
                        subplot=(0, 1),
                        lineproperties={"c": "k", "label": "$\\beta$"},
                        subplotnumber=2,
                        ax=ax,
                    )

                    # # Set custom ticks and tick locations
                    # plotter.x_tick_labels = np.arange(0, 11, 2)  # Set X-axis tick labels
                    # plotter.y_tick_locations = np.arange(-5, 6, 1)  # Set
                    # Y-axis tick locations

                    ax = _plt.gca()
                    ax.set_xlabel("x")
                    ax.set_yticks([0, 1])
                    ax.set_ylabel("$\\beta$")
                    handles, labels = ax.get_legend_handles_labels()
                    if labels:
                        _plt.legend()
                    _plt.fill_between(
                        data_stability[0],
                        data_stability[1].reshape(len(data_stability[1])),
                    )
                    _plt.title("Perurbation in the Cone")
                    # _plt.screenshot(f"{prefix}/perturbations-{i_t}.png")
                    _plt.savefig(f"{prefix}/perturbation-profile-cone-{i_t}.png")
                    _plt.close()

                    len(data_stability[0])

                    mode_shapes_data["time_steps"].append(t)
                    mode_shapes_data["point_values"]["x_values"] = data_stability[0]

                    for mode in range(1, num_modes + 1):
                        bifurcation_values_mode = data_bifurcation[
                            1
                        ].flatten()  # Replace with actual values
                        stability_values_mode = data_stability[
                            1
                        ].flatten()  # Replace with actual values
                        # Append mode-specific fields to the data structure
                        mode_key = f"mode_{mode}"
                        mode_shapes_data["point_values"][mode_key] = {
                            "bifurcation": mode_shapes_data["point_values"]
                            .get(mode_key, {})
                            .get("bifurcation", []),
                            "stability": mode_shapes_data["point_values"]
                            .get(mode_key, {})
                            .get("stability", []),
                        }
                        mode_shapes_data["point_values"][mode_key][
                            "bifurcation"
                        ].append(bifurcation_values_mode)
                        mode_shapes_data["point_values"][mode_key]["stability"].append(
                            stability_values_mode
                        )

        np.savez(f"{prefix}/mode_shapes_data.npz", **mode_shapes_data)

        fracture_energy = comm.allreduce(
            assemble_scalar(form(damage_energy_density(setup.state) * dx)),
            op=MPI.SUM,
        )
        elastic_energy = comm.allreduce(
            assemble_scalar(form(elastic_energy_density(setup.state) * dx)),
            op=MPI.SUM,
        )
        # _F = assemble_scalar(form(stress(state)))

        ColorPrint.print_bold(stability.solution["lambda_t"])

        equilibrium_result = EquilibriumResult(
            step=i_t,
            load=float(t),
            time=float(t),
            state=setup.state,
            bounds=setup.bounds,
            converged=True,
            solver_name="alternate_minimisation_1d",
            iterations=(
                equilibrium.data["iteration"][-1]
                if equilibrium.data["iteration"]
                else None
            ),
            residual_norm=(
                float(equilibrium.data["error_residual_F"][-1])
                if equilibrium.data["error_residual_F"]
                else None
            ),
            total_energy=elastic_energy + fracture_energy,
            diagnostics={"inertia": inertia},
        )
        lambda_bif_min = (
            min(float(np.real(value)) for value in bifurcation.data.get("eigs", []))
            if bifurcation.data.get("eigs")
            else None
        )
        lambda_stab_min = _float_or_none(stability.solution.get("lambda_t"))

        history.append(
            StepRecord(
                step=equilibrium_result.step,
                load=equilibrium_result.load,
                time=equilibrium_result.time,
                elastic_energy=elastic_energy,
                fracture_energy=fracture_energy,
                total_energy=equilibrium_result.total_energy,
                solver_converged=equilibrium_result.converged,
                n_iterations=equilibrium_result.iterations,
                inertia=inertia,
                stability_attempted=True,
                stability_converged=stable is not None,
                stable=stable,
                lambda_stab_min=lambda_stab_min,
                bifurcation_attempted=True,
                bifurcation_converged=True,
                unique=bool(is_unique),
                lambda_bif_min=lambda_bif_min,
                extra={
                    "solver_data": equilibrium.data,
                    "equilibrium_data": equilibrium.data,
                    "cone_data": stability.data,
                    "eigs_ball": bifurcation.data.get("eigs", []),
                    "eigs_cone": stability.solution.get("lambda_t"),
                },
            )
        )

        # history_data["F"].append(_F)

        with XDMFFile(
            comm, f"{prefix}/{_nameExp}.xdmf", "a", encoding=XDMFFile.Encoding.HDF5
        ) as file:
            file.write_function(u, t)
            file.write_function(alpha, t)

        if comm.rank == 0:
            history_data = history.to_columns()
            a_file = open(f"{prefix}/time_data.json", "w")
            json.dump(history_data, a_file, default=str)
            a_file.close()

    history_data = history.to_columns()
    df = pd.DataFrame(history_data)
    print(df)

    if comm.rank == 0:
        plot_energies(history_data, file=f"{prefix}/{_nameExp}_energies.pdf")
        plot_AMit_load(history_data, file=f"{prefix}/{_nameExp}_it_load.pdf")
        # plot_force_displacement(
        #     history_data, file=f"{prefix}/{_nameExp}_stress-load.pdf"
        # )

    return history_data, stability.data, setup.state


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
    # parameters["model"]["mu"] = 1
    parameters["model"]["w1"] = 1

    parameters["geometry"]["geom_type"] = "discrete-damageable"
    # Get mesh parameters

    if model == "at2":
        parameters["loading"]["min"] = 0.9
        parameters["loading"]["max"] = 0.9
        parameters["loading"]["steps"] = 1

    elif model == "at1":
        parameters["loading"]["min"] = 0.0
        parameters["loading"]["max"] = 1.5
        parameters["loading"]["steps"] = 20

    parameters["geometry"]["geom_type"] = "traction-bar"
    parameters["geometry"]["mesh_size_factor"] = 4
    parameters["geometry"]["N"] = ndofs

    parameters["stability"]["cone"]["cone_max_it"] = 400000
    parameters["stability"]["cone"]["cone_atol"] = 1e-6
    parameters["stability"]["cone"]["cone_rtol"] = 1e-6
    parameters["stability"]["cone"]["scaling"] = 1e-2

    # parameters["model"]["model_dimension"] = 2
    parameters["model"]["w1"] = 1
    parameters["model"]["ell"] = 0.2
    parameters["model"]["k_res"] = 0.0

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    return parameters, signature


def test_1d():
    from mpi4py import MPI

    # parser = argparse.ArgumentParser(description="Process evolution.")
    # parser.add_argument("-N", help="The number of dofs.", type=int, default=10)
    # args = parser.parse_args()
    _N = 30
    parameters, signature = load_parameters(
        os.path.join(os.path.dirname(__file__), "parameters.yml"), ndofs=_N
    )
    json.dumps(parameters, indent=2)
    # print(pretty_parameters)
    # _storage = f"output/one-dimensional-bar/MPI-{MPI.COMM_WORLD.Get_size()}/{args.N}/{signature}"
    _storage = (
        f"output/one-dimensional-bar/MPI-{MPI.COMM_WORLD.Get_size()}/{_N}/{signature}"
    )
    ColorPrint.print_bold(f"===================-{_storage}-=================")

    with dolfinx.common.Timer("~Computation Experiment"):
        history_data, stability_data, state = run_computation(parameters, _storage)

    from irrevolutions.utils import ResultsStorage, Visualization

    manifest = Manifest(
        parameters=parameters,
        run_id=signature,
        solver_options={
            "solvers": parameters.get("solvers"),
            "stability": parameters.get("stability"),
        },
        mesh={"tdim": 1},
        spaces={"u": "CG1", "alpha": "CG1"},
    )
    storage = ResultsStorage(MPI.COMM_WORLD, _storage)
    storage.store_results(parameters, history_data, state)
    if MPI.COMM_WORLD.rank == 0:
        with open(f"{_storage}/manifest.json", "w") as file:
            json.dump(asdict(manifest), file)
    visualization = Visualization(_storage)
    # visualization.visualise_results(pd.DataFrame(history_data), drop = ["solver_data", "cone_data"])
    visualization.save_table(pd.DataFrame(history_data), "history_data")
    # visualization.save_table(pd.DataFrame(stability_data), "stability_data")
    pd.DataFrame(stability_data).to_json(f"{_storage}/stability_data.json")

    ColorPrint.print_bold(f"===================-{signature}-=================")
    ColorPrint.print_bold(f"===================-{_storage}-=================")

    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

    from irrevolutions.utils import table_timing_data

    _timings = table_timing_data()

    visualization.save_table(_timings, "timing_data")
    _neg_eigen_ball = [d[0] for d in pd.DataFrame(history_data).inertia.values]
    _stability = pd.DataFrame(history_data).stable.values
    _uniqueness = pd.DataFrame(history_data).unique.values

    np.testing.assert_array_equal(_neg_eigen_ball, [0, 0, 0, 1, 2])
    np.testing.assert_array_equal(
        _stability, np.array([True, True, True, False, False])
    )
    np.testing.assert_array_equal(
        _uniqueness, np.array([True, True, True, False, False])
    )


if __name__ == "__main__":
    test_1d()
