#!/usr/bin/env python3

import hashlib
import os

import basix.ufl
import dolfinx
import dolfinx.mesh
import numpy as np
import pandas as pd
import ufl
import yaml
from dolfinx.common import list_timings
from dolfinx.fem import (
    Function,
    assemble_scalar,
    dirichletbc,
    form,
    locate_dofs_geometrical,
)
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


comm = MPI.COMM_WORLD


def load_parameters(file_path, ndofs, model="at1"):
    with open(file_path) as stream:
        parameters = yaml.load(stream, Loader=yaml.FullLoader)

    parameters["model"]["model_dimension"] = 1
    parameters["model"]["model_type"] = "1D"
    parameters["model"]["w1"] = 1

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

    parameters["model"]["w1"] = 1
    parameters["model"]["ell"] = 0.2
    parameters["model"]["k_res"] = 0.0

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()
    return parameters, signature


def build_1d_setup(parameters):
    mesh = dolfinx.mesh.create_unit_interval(
        MPI.COMM_WORLD, int(parameters["geometry"]["N"])
    )
    element_u = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1)
    element_alpha = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1)

    V_u = dolfinx.fem.functionspace(mesh, element_u)
    V_alpha = dolfinx.fem.functionspace(mesh, element_alpha)

    u = Function(V_u, name="Displacement")
    u_boundary = Function(V_u, name="BoundaryDisplacement")
    zero_u = Function(V_u, name="ZeroDisplacement")
    alpha = Function(V_alpha, name="Damage")

    alpha_lb = Function(V_alpha, name="LowerBoundDamage")
    alpha_ub = Function(V_alpha, name="UpperBoundDamage")

    state = {"u": u, "alpha": alpha}

    Lx = parameters["geometry"]["Lx"]
    dofs_u_left = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], 0.0))
    dofs_u_right = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], Lx))

    zero_u.interpolate(lambda x: np.zeros_like(x[0]))
    u_boundary.interpolate(lambda x: np.ones_like(x[0]))
    alpha_lb.interpolate(lambda x: np.zeros_like(x[0]))
    alpha_ub.interpolate(lambda x: np.ones_like(x[0]))

    for func in [zero_u, u_boundary, alpha_lb, alpha_ub]:
        func.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

    bc_u_left = dirichletbc(np.array(0, dtype=PETSc.ScalarType), dofs_u_left, V_u)
    bc_u_right = dirichletbc(u_boundary, dofs_u_right)

    bcs = normalise_bcs(
        {
            "u": {
                "dirichlet": [bc_u_left, bc_u_right],
                "loading": {
                    "type": "displacement_control",
                    "parameter": None,
                    "component": 0,
                    "region": "right",
                },
            },
            "alpha": {"dirichlet": [], "loading": None},
        }
    )

    bounds = {"alpha": make_field_bounds(alpha_lb, alpha_ub)}
    dx = ufl.Measure("dx", domain=mesh)

    def degradation(alpha_field):
        return (1 - alpha_field) ** 2

    def elastic_energy_density(current_state):
        alpha_field = current_state["alpha"]
        displacement = current_state["u"]
        eps = ufl.grad(displacement)
        return (
            parameters["model"]["E"]
            / 2.0
            * degradation(alpha_field)
            * ufl.inner(eps, eps)
        )

    def damage_energy_density(current_state):
        alpha_field = current_state["alpha"]
        grad_alpha = ufl.grad(alpha_field)
        w1 = parameters["model"]["w1"]
        ell = parameters["model"]["ell"]
        return w1 * alpha_field + w1 * ell**2 * ufl.dot(grad_alpha, grad_alpha)

    total_energy = (elastic_energy_density(state) + damage_energy_density(state)) * dx

    setup = ExperimentSetup(
        state=state,
        bcs=bcs,
        bounds=bounds,
        parameters=parameters,
        energy=total_energy,
        mesh=mesh,
        spaces={"u": V_u, "alpha": V_alpha},
        metadata={"geom_type": parameters["geometry"]["geom_type"]},
    )

    runtime = {
        "u_boundary": u_boundary,
        "alpha_lb": alpha_lb,
        "alpha_ub": alpha_ub,
        "dx": dx,
        "elastic_energy_density": elastic_energy_density,
        "damage_energy_density": damage_energy_density,
    }

    return setup, runtime


def _float_or_none(value):
    if value is None:
        return None
    if np.isscalar(value) and np.isnan(value):
        return None
    return float(value)


def test_1d():
    columns = run_contract_1d()

    neg_eigs = [entry[0] for entry in columns["inertia"]]
    np.testing.assert_array_equal(neg_eigs, [0, 0, 0, 1, 2])
    np.testing.assert_array_equal(columns["stable"], [True, True, True, False, False])
    np.testing.assert_array_equal(columns["unique"], [True, True, True, False, False])


def run_contract_1d():
    parameters, signature = load_parameters(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "parameters.yml"),
        ndofs=30,
    )

    setup, runtime = build_1d_setup(parameters)
    solver_bcs = legacy_bcs_from_contract(setup.bcs)
    alpha_bounds = get_bounds_pair(setup.bounds, "alpha")

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

    manifest = Manifest(
        parameters=parameters,
        run_id=signature,
        solver_options={
            "solvers": parameters.get("solvers"),
            "stability": parameters.get("stability"),
        },
        mesh={
            "cell_name": setup.mesh.topology.cell_name(),
            "tdim": setup.mesh.topology.dim,
        },
        spaces={name: str(space.element) for name, space in setup.spaces.items()},
    )

    history = History()
    loads = [0.0, 0.5, 0.99, 1.01, 1.3]

    for step, load in enumerate(loads):
        runtime["u_boundary"].interpolate(lambda x, t=load: t * np.ones_like(x[0]))
        runtime["u_boundary"].x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        setup.state["alpha"].x.petsc_vec.copy(runtime["alpha_lb"].x.petsc_vec)
        runtime["alpha_lb"].x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        equilibrium.solve()
        hybrid.solve(runtime["alpha_lb"])

        unique = bifurcation.solve(runtime["alpha_lb"])
        inertia = bifurcation.get_inertia()
        eig0 = (
            bifurcation._spectrum[0]["xk"]
            if bifurcation._spectrum and "xk" in bifurcation._spectrum[0]
            else None
        )
        stable = stability.solve(runtime["alpha_lb"], eig0=eig0, inertia=inertia)

        fracture_energy = comm.allreduce(
            assemble_scalar(
                form(runtime["damage_energy_density"](setup.state) * runtime["dx"])
            ),
            op=MPI.SUM,
        )
        elastic_energy = comm.allreduce(
            assemble_scalar(
                form(runtime["elastic_energy_density"](setup.state) * runtime["dx"])
            ),
            op=MPI.SUM,
        )
        total_energy = elastic_energy + fracture_energy

        equilibrium_result = EquilibriumResult(
            step=step,
            load=float(load),
            time=float(load),
            state=setup.state,
            bounds=setup.bounds,
            converged=True,
            solver_name="hybrid",
            iterations=(
                equilibrium.data["iteration"][-1] + 1
                if equilibrium.data["iteration"]
                else None
            ),
            residual_norm=(
                float(equilibrium.data["error_residual_F"][-1])
                if equilibrium.data["error_residual_F"]
                else None
            ),
            total_energy=total_energy,
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
                unique=bool(unique),
                lambda_bif_min=lambda_bif_min,
                extra={"solver_name": equilibrium_result.solver_name},
            )
        )

    columns = history.to_columns()

    assert manifest.run_id == signature
    assert len(history) == len(loads)
    assert all(len(column) == len(loads) for column in columns.values())
    assert set(setup.state.keys()) == {"u", "alpha"}
    assert set(setup.bcs.keys()) == {"u", "alpha"}
    assert set(setup.bounds.keys()) == {"alpha"}

    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])
    return columns


if __name__ == "__main__":
    columns = run_contract_1d()
    if comm.rank == 0:
        df = pd.DataFrame(columns)
        print(df[["step", "load", "total_energy", "stable", "unique"]])
