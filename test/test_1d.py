#!/usr/bin/env python3
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
from dolfinx.fem import (Constant, Function, assemble_scalar, dirichletbc,
                         form, locate_dofs_geometrical, set_bc)
from dolfinx.fem.petsc import assemble_vector
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc

from irrevolutions.algorithms.am import HybridSolver
from irrevolutions.algorithms.so import BifurcationSolver, StabilitySolver
from irrevolutions.solvers import SNESSolver
from irrevolutions.solvers.function import vec_to_functions
from irrevolutions.utils import (ColorPrint, _logger, _write_history_data,
                                 history_data, norm_H1, norm_L2)
from irrevolutions.utils.plots import plot_AMit_load, plot_energies
#
from irrevolutions.utils.viz import plot_profile

"""The fundamental problem of a 1d bar in traction.
0|(WWWWWWWWWWWWWWWWWWWWWW)|========> t

(WWW): endommageable bar, y=(u, alpha)
load: displacement hard-t
"""


# logging.getLogger().setLevel(logging.INFO)


class _AlternateMinimisation1D:
    def __init__(
        self,
        total_energy,
        state,
        bcs,
        solver_parameters={},
        bounds=(dolfinx.fem.function.Function, dolfinx.fem.function.Function),
    ):
        self.state = state
        self.alpha = state["alpha"]
        self.alpha_old = dolfinx.fem.function.Function(self.alpha.function_space)
        self.u = state["u"]
        self.alpha_lb = bounds[0]
        self.alpha_ub = bounds[1]
        self.total_energy = total_energy
        self.solver_parameters = solver_parameters

        V_u = state["u"].function_space
        V_alpha = state["alpha"].function_space

        energy_u = ufl.derivative(self.total_energy, self.u, ufl.TestFunction(V_u))
        energy_alpha = ufl.derivative(
            self.total_energy, self.alpha, ufl.TestFunction(V_alpha)
        )

        self.F = [energy_u, energy_alpha]

        self.elasticity = SNESSolver(
            energy_u,
            self.u,
            bcs.get("bcs_u"),
            bounds=None,
            petsc_options=self.solver_parameters.get("elasticity").get("snes"),
            prefix=self.solver_parameters.get("elasticity").get("prefix"),
        )

        self.damage = SNESSolver(
            energy_alpha,
            self.alpha,
            bcs.get("bcs_alpha"),
            bounds=(self.alpha_lb, self.alpha_ub),
            petsc_options=self.solver_parameters.get("damage").get("snes"),
            prefix=self.solver_parameters.get("damage").get("prefix"),
        )

    def solve(self, outdir=None):
        alpha_diff = dolfinx.fem.Function(self.alpha.function_space)

        self.data = {
            "iteration": [],
            "error_alpha_L2": [],
            "error_alpha_H1": [],
            "F_norm": [],
            "error_alpha_max": [],
            "error_residual_F": [],
            "solver_alpha_reason": [],
            "solver_alpha_it": [],
            "solver_u_reason": [],
            "solver_u_it": [],
            "total_energy": [],
        }
        for iteration in range(
            self.solver_parameters.get("damage_elasticity").get("max_it")
        ):
            with dolfinx.common.Timer(
                "~First Order: Alternate Minimization : Elastic solver"
            ):
                (solver_u_it, solver_u_reason) = self.elasticity.solve()
            with dolfinx.common.Timer(
                "~First Order: Alternate Minimization : Damage solver"
            ):
                (solver_alpha_it, solver_alpha_reason) = self.damage.solve()

            # Define error function
            self.alpha.x.petsc_vec.copy(alpha_diff.vector)
            alpha_diff.vector.axpy(-1, self.alpha_old.vector)
            alpha_diff.vector.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )

            error_alpha_H1 = norm_H1(alpha_diff)
            error_alpha_L2 = norm_L2(alpha_diff)

            Fv = [assemble_vector(form(F)) for F in self.F]

            Fnorm = np.sqrt(
                np.array([comm.allreduce(Fvi.norm(), op=MPI.SUM) for Fvi in Fv]).sum()
            )

            error_alpha_max = alpha_diff.vector.max()[1]
            total_energy_int = comm.allreduce(
                assemble_scalar(form(self.total_energy)), op=MPI.SUM
            )
            residual_F = assemble_vector(self.elasticity.F_form)
            residual_F.ghostUpdate(
                addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
            )
            set_bc(residual_F, self.elasticity.bcs, self.u.vector)
            error_residual_F = ufl.sqrt(residual_F.dot(residual_F))

            self.alpha.x.petsc_vec.copy(self.alpha_old.vector)
            self.alpha_old.vector.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )

            logging.critical(
                f"AM - Iteration: {iteration:3d}, res F Error: {error_residual_F:3.4e}, alpha_max: {self.alpha.vector.max()[1]:3.4e}"
            )

            logging.critical(
                f"AM - Iteration: {iteration:3d}, H1 Error: {error_alpha_H1:3.4e}, alpha_max: {self.alpha.vector.max()[1]:3.4e}"
            )

            logging.critical(
                f"AM - Iteration: {iteration:3d}, L2 Error: {error_alpha_L2:3.4e}, alpha_max: {self.alpha.vector.max()[1]:3.4e}"
            )

            logging.critical(
                f"AM - Iteration: {iteration:3d}, Linfty Error: {error_alpha_max:3.4e}, alpha_max: {self.alpha.vector.max()[1]:3.4e}"
            )

            self.data["iteration"].append(iteration)
            self.data["error_alpha_L2"].append(error_alpha_L2)
            self.data["error_alpha_H1"].append(error_alpha_H1)
            self.data["F_norm"].append(Fnorm)
            self.data["error_alpha_max"].append(error_alpha_max)
            self.data["error_residual_F"].append(error_residual_F)
            self.data["solver_alpha_it"].append(solver_alpha_it)
            self.data["solver_alpha_reason"].append(solver_alpha_reason)
            self.data["solver_u_reason"].append(solver_u_reason)
            self.data["solver_u_it"].append(solver_u_it)
            self.data["total_energy"].append(total_energy_int)

            if (
                self.solver_parameters.get("damage_elasticity").get("criterion")
                == "residual_u"
            ):
                if error_residual_F <= self.solver_parameters.get(
                    "damage_elasticity"
                ).get("alpha_rtol"):
                    break
            if (
                self.solver_parameters.get("damage_elasticity").get("criterion")
                == "alpha_H1"
            ):
                if error_alpha_H1 <= self.solver_parameters.get(
                    "damage_elasticity"
                ).get("alpha_rtol"):
                    break
        else:
            raise RuntimeError(
                f"Could not converge after {iteration:3d} iterations, error {error_alpha_H1:3.4e}"
            )


petsc4py.init(sys.argv)
comm = MPI.COMM_WORLD

# Mesh on node model_rank and then distribute
model_rank = 0


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
        f.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

    bc_u_left = dirichletbc(np.array(0, dtype=PETSc.ScalarType), dofs_u_left, V_u)

    bc_u_right = dirichletbc(u_, dofs_u_right)
    bcs_u = [bc_u_left, bc_u_right]

    bcs_alpha = []

    bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}
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

    equilibrium = _AlternateMinimisation1D(
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
        u_.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        # update the lower bound
        alpha.vector.copy(alpha_lb.vector)
        alpha_lb.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        _logger.critical(f"-- Solving for t = {t:3.2f} --")

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

                    plotter = pyvista.Plotter(
                        title="Perturbation profile",
                        window_size=[800, 600],
                        shape=(1, 2),
                    )
                    _plt, data_bifurcation = plot_profile(
                        β,
                        points,
                        plotter,
                        subplot=(1, 2),
                        lineproperties={"c": "k", "label": "$\\beta$"},
                        subplotnumber=1,
                    )
                    ax = _plt.gca()
                    ax.set_xlabel("x")
                    ax.set_yticks([-1, 0, 1])
                    ax.set_ylabel("$\\beta$")
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
                        subplot=(1, 2),
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
            assemble_scalar(form(damage_energy_density(state) * dx)),
            op=MPI.SUM,
        )
        elastic_energy = comm.allreduce(
            assemble_scalar(form(elastic_energy_density(state) * dx)),
            op=MPI.SUM,
        )
        # _F = assemble_scalar(form(stress(state)))

        ColorPrint.print_bold(stability.solution["lambda_t"])

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

        # history_data["F"].append(_F)

        with XDMFFile(
            comm, f"{prefix}/{_nameExp}.xdmf", "a", encoding=XDMFFile.Encoding.HDF5
        ) as file:
            file.write_function(u, t)
            file.write_function(alpha, t)

        if comm.rank == 0:
            a_file = open(f"{prefix}/time_data.json", "w")
            json.dump(history_data, a_file)
            a_file.close()

    df = pd.DataFrame(history_data)
    print(df)

    if comm.rank == 0:
        plot_energies(history_data, file=f"{prefix}/{_nameExp}_energies.pdf")
        plot_AMit_load(history_data, file=f"{prefix}/{_nameExp}_it_load.pdf")
        # plot_force_displacement(
        #     history_data, file=f"{prefix}/{_nameExp}_stress-load.pdf"
        # )

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

    storage = ResultsStorage(MPI.COMM_WORLD, _storage)
    storage.store_results(parameters, history_data, state)
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
