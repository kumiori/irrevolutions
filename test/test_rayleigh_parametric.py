import argparse
import json
import logging
import sys
from pathlib import Path

import dolfinx
import numpy as np
import ufl
import yaml
from dolfinx.fem import (assemble_scalar, dirichletbc, form,
                         locate_dofs_geometrical)
from mpi4py import MPI
from petsc4py import PETSc

from irrevolutions.algorithms.so import BifurcationSolver, StabilitySolver
from irrevolutions.utils import ColorPrint, _logger
from irrevolutions.utils import eigenspace as eig
from irrevolutions.utils import indicator_function
from irrevolutions.utils.viz import get_datapoints

sys.path.append("../")
sys.path.append("../playground/nb")

# from test_extend import test_extend_vector
# from test_cone_project import _cone_project_restricted


_logger.setLevel(logging.CRITICAL)


def rayleigh_ratio_reduced(β, parameters):
    dx = ufl.Measure("dx", β.function_space.mesh)
    a, b, c = (
        parameters["model"]["a"],
        parameters["model"]["b"],
        parameters["model"]["c"],
    )

    sq_int_beta = (β * dx) ** 2
    numerator = (a * β.dx(0) ** 2 * dx) + b * c**2 * sq_int_beta
    denominator = ufl.inner(β, β) * dx

    R = assemble_scalar(form(numerator)) / assemble_scalar(form(denominator))

    # Create the dolfinx form

    return R


def rayleigh_ratio(z, parameters):
    (v, β) = z
    dx = ufl.Measure("dx", v.function_space.mesh)

    a, b, c = (
        parameters["model"]["a"],
        parameters["model"]["b"],
        parameters["model"]["c"],
    )

    numerator = (
        a * ufl.inner(β.dx(0), β.dx(0))
        + b * ufl.inner(v.dx(0) - c * β, v.dx(0) - c * β)
    ) * dx
    denominator = ufl.inner(β, β) * dx

    R = assemble_scalar(form(numerator)) / assemble_scalar(form(denominator))

    return R


def rayleigh(parameters, storage=None):
    comm = MPI.COMM_WORLD
    comm.Get_rank()
    comm.Get_size()

    # with XDMFFile(comm, "data/input_data.xdmf", "r") as file:
    #     mesh = file.read_mesh(name='mesh')

    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 30)

    if storage is None:
        prefix = "output/rayleigh-benchmark"
    else:
        prefix = storage

    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    if comm.rank == 0:
        with open(f"{prefix}/parameters.yaml", "w") as file:
            yaml.dump(parameters, file)

        with open(f"{prefix}/signature.md5", "w") as f:
            f.write(signature)

    _a = parameters["model"]["a"]
    _b = parameters["model"]["b"]
    _c = parameters["model"]["c"]

    a = dolfinx.fem.Constant(mesh, PETSc.ScalarType(_a))
    b = dolfinx.fem.Constant(mesh, PETSc.ScalarType(_b))
    c = dolfinx.fem.Constant(mesh, PETSc.ScalarType(_c))

    # (size of the) support of the cone-eigenfunction - if any.
    #
    _D = (
        (np.pi**2 * _a / (_b * _c**2)) ** (1 / 3)
        if np.pi**2 * _a - _b * _c**2 < 0
        else 1
    )

    element_u = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)
    element_alpha = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)

    V_u = dolfinx.fem.FunctionSpace(mesh, element_u)
    V_alpha = dolfinx.fem.FunctionSpace(mesh, element_alpha)
    u = dolfinx.fem.Function(V_u, name="Displacement")

    alpha = dolfinx.fem.Function(V_alpha, name="Damage")

    zero_u = dolfinx.fem.Function(V_u, name="Boundary condition")
    zero_alpha = dolfinx.fem.Function(V_u, name="Lower bound")
    one_alpha = dolfinx.fem.Function(V_u, name="Upper bound")

    alpha.interpolate(lambda x: 1e-4 * np.ones_like(x[0]))

    for zero in [zero_u, zero_alpha]:
        zero.interpolate(lambda x: np.zeros_like(x[0]))
        zero.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

    one_alpha.interpolate(lambda x: np.zeros_like(x[0]))
    one_alpha.vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )

    dx = ufl.Measure("dx", alpha.function_space.mesh)

    # G = 1/2 * (a * alpha.dx(0)**2 + b * (u.dx(0) - c * alpha)**2) \
    #         * dx

    G = 1 / 2 * (a * alpha.dx(0) ** 2 + b * (u.dx(0) - c * alpha) ** 2) * dx

    locate_dofs_geometrical(V_alpha, lambda x: np.isclose(x[0], 0.0))
    locate_dofs_geometrical(V_alpha, lambda x: np.isclose(x[0], 1))

    dofs_u_left = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], 0.0))
    dofs_u_right = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], 1))

    bc_u_left = dirichletbc(np.array(0, dtype=PETSc.ScalarType), dofs_u_left, V_u)

    bc_u_right = dirichletbc(zero, dofs_u_right)

    bcs_u = [bc_u_left, bc_u_right]

    bcs = {"bcs_u": bcs_u, "bcs_alpha": []}

    # Perturbations
    dolfinx.fem.Function(V_alpha, name="DamagePerturbation")
    dolfinx.fem.Function(V_u, name="DisplacementPerturbation")

    # Pack state
    state = {"u": u, "alpha": alpha}

    mode_shapes_data = {
        "time_steps": [],
        "mesh": [],
        "point_values": {
            # 'x_values': [],
        },
        "global_values": {
            "R_vector": [],
            "R_cone": [],
            "D_theory": [],
            "D_support": [],
        },
    }

    _logger.setLevel(level=logging.INFO)

    bifurcation = BifurcationSolver(
        G, state, bcs, bifurcation_parameters=parameters.get("stability")
    )

    stability = StabilitySolver(
        G, state, bcs, cone_parameters=parameters.get("stability")
    )

    bifurcation.solve(zero_alpha)
    bifurcation.get_inertia()
    stability.solve(zero_alpha, eig0=bifurcation.spectrum, inertia=(1, 0, 10))
    # (size of the) support of the cone-eigenfunction - if any.
    #

    _logger.setLevel(level=logging.INFO)

    if bifurcation.spectrum:
        # vec_to_functions(bifurcation.spectrum[0]['xk'], [v, β])

        tol = 1e-3
        xs = np.linspace(0 + tol, 1 - tol, 101)
        points = np.zeros((3, 101))
        points[0] = xs

        data_bifurcation_v = get_datapoints(bifurcation.perturbation["v"], points)
        data_bifurcation_β = get_datapoints(bifurcation.perturbation["β"], points)
        data_stability_v = get_datapoints(stability.perturbation["v"], points)
        data_stability_β = get_datapoints(stability.perturbation["β"], points)
        data_stability_residual_w = get_datapoints(stability.residual["w"], points)
        data_stability_residual_ζ = get_datapoints(stability.residual["ζ"], points)

        mode_shapes_data["time_steps"].append(0)
        mode_shapes_data["mesh"] = data_stability_β[0][:, 0]

        _R_vector = rayleigh_ratio(
            (bifurcation.perturbation["v"], bifurcation.perturbation["β"]), parameters
        )
        _R_cone = rayleigh_ratio(
            (stability.perturbation["v"], stability.perturbation["β"]), parameters
        )

        _support = indicator_function(stability.perturbation["β"])
        D_support = dolfinx.fem.assemble_scalar(dolfinx.fem.form(_support * dx))

        # for mode in range(1, num_modes + 1):
        mode = 1

        bifurcation_values_mode_β = data_bifurcation_β[1].flatten()
        bifurcation_values_mode_v = data_bifurcation_v[1].flatten()
        stability_values_mode_β = data_stability_β[1].flatten()
        stability_values_mode_v = data_stability_v[1].flatten()
        stability_values_residual_w = data_stability_residual_w[1].flatten()
        stability_values_residual_ζ = data_stability_residual_ζ[1].flatten()

        # Append mode-specific fields to the data structure
        mode_key = f"mode_{mode}"
        mode_shapes_data["point_values"][mode_key] = {
            "bifurcation_β": mode_shapes_data["point_values"]
            .get(mode_key, {})
            .get("bifurcation_β", []),
            "bifurcation_v": mode_shapes_data["point_values"]
            .get(mode_key, {})
            .get("bifurcation_v", []),
            "stability_β": mode_shapes_data["point_values"]
            .get(mode_key, {})
            .get("stability_β", []),
            "stability_v": mode_shapes_data["point_values"]
            .get(mode_key, {})
            .get("stability_v", []),
            "stability_residual_w": mode_shapes_data["point_values"]
            .get(mode_key, {})
            .get("stability_residual_w", []),
            "stability_residual_ζ": mode_shapes_data["point_values"]
            .get(mode_key, {})
            .get("stability_residual_ζ", []),
        }
        mode_shapes_data["point_values"][mode_key]["bifurcation_β"].append(
            bifurcation_values_mode_β
        )
        mode_shapes_data["point_values"][mode_key]["bifurcation_v"].append(
            bifurcation_values_mode_v
        )
        mode_shapes_data["point_values"][mode_key]["stability_β"].append(
            stability_values_mode_β
        )
        mode_shapes_data["point_values"][mode_key]["stability_v"].append(
            stability_values_mode_v
        )
        mode_shapes_data["point_values"][mode_key]["stability_residual_w"].append(
            stability_values_residual_w
        )
        mode_shapes_data["point_values"][mode_key]["stability_residual_ζ"].append(
            stability_values_residual_ζ
        )

        mode_shapes_data["global_values"]["R_vector"] = _R_vector
        mode_shapes_data["global_values"]["R_cone"] = _R_cone
        mode_shapes_data["global_values"]["D_theory"] = _D
        mode_shapes_data["global_values"]["D_support"] = D_support

    np.savez(f"{prefix}/mode_shapes_data.npz", **mode_shapes_data)

    return None, None, None


def load_parameters(file_path, ndofs, model="rayleigh"):
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

    parameters["model"] = {}
    parameters["model"]["model_dimension"] = 1
    parameters["model"]["model_type"] = "1D"

    _numerical_parameters = eig.book_of_the_numbers(scale_c=2, scale_b=3)

    # parameters["model"].update({'a': 1,
    #                             'b': 1,
    #                             'c': 8})

    parameters["model"].update(_numerical_parameters)
    parameters["geometry"]["geom_type"] = "infinite-dimensional-unit-test"

    # Get mesh parameters
    parameters["geometry"]["mesh_size_factor"] = 4
    parameters["geometry"]["N"] = ndofs

    parameters["stability"]["inactiveset_gatol"] = 1e-1

    parameters["stability"]["cone"]["cone_max_it"] = 400000
    parameters["stability"]["cone"]["cone_atol"] = 1e-6
    parameters["stability"]["cone"]["cone_rtol"] = 1e-6
    parameters["stability"]["cone"]["scaling"] = 1e-3

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    return parameters, signature


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process evolution.")
    parser.add_argument("-N", help="The number of dofs.", type=int, default=50)
    parser.add_argument(
        "-M", help="The number of simulation runs.", type=int, default=10
    )

    args = parser.parse_args()

    # run M simulations:

    input(
        f"About to run {args.M} random tests, from the Book of the Numbers \nPress any key to continue..."
    )

    for i in range(args.M):
        parameters, signature = load_parameters("parameters.yml", ndofs=args.N)
        pretty_parameters = json.dumps(parameters, indent=2)

        _storage = f"output/rayleigh-benchmark-parametric/MPI-{MPI.COMM_WORLD.Get_size()}/{signature}"
        ColorPrint.print_bold(f"===================-{_storage}-=================")

        with dolfinx.common.Timer("~Random Computation Experiment") as timer:
            history_data, stability_data, state = rayleigh(parameters, _storage)
