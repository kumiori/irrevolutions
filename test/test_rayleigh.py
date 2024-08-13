import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
import os

import dolfinx
import matplotlib.pyplot as plt
import numpy as np
import pyvista
import ufl
import yaml
from dolfinx.fem import dirichletbc, locate_dofs_geometrical
from dolfinx.fem import form, assemble_scalar
from irrevolutions.algorithms.so import BifurcationSolver, StabilitySolver
from irrevolutions.solvers.function import vec_to_functions
from irrevolutions.utils import ColorPrint, _logger, indicator_function
from irrevolutions.utils.viz import get_datapoints, plot_profile
from mpi4py import MPI
from petsc4py import PETSc

test_dir = os.path.dirname(__file__)

_logger.setLevel(logging.CRITICAL)

def parallel_assemble_scalar(ufl_form):
      compiled_form = dolfinx.fem.form(ufl_form)
      comm = compiled_form.mesh.comm
      local_scalar = dolfinx.fem.assemble_scalar(compiled_form)
      return comm.allreduce(local_scalar, op=MPI.SUM)
  
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

    R = parallel_assemble_scalar(numerator) / parallel_assemble_scalar(denominator)

    return R


def test_rayleigh(parameters = None, storage=None):

    if parameters is None:
        parameters, signature = load_parameters(os.path.join(test_dir, "parameters.yml"), ndofs=50)
        pretty_parameters = json.dumps(parameters, indent=2)
        storage = f"output/rayleigh-benchmark/MPI-{MPI.COMM_WORLD.Get_size()}/{signature}"
    else:
        signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    if storage is None:
        prefix = "output/rayleigh-benchmark"
    else:
        prefix = storage

    comm = MPI.COMM_WORLD
    comm.Get_rank()
    comm.Get_size()

    # with XDMFFile(comm, "data/input_data.xdmf", "r") as file:
    #     mesh = file.read_mesh(name='mesh')
    N = parameters["geometry"]["N"]
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, N)


    ColorPrint.print_bold(f"===================-{storage}-=================")

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
    _D = (np.pi**2 * _a / (_b * _c**2)) ** (1 / 3)

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

    G = 1 / 2 * (a * alpha.dx(0) ** 2 + b * (u.dx(0) - c * alpha) ** 2) * dx

    F_ = [
        ufl.derivative(G, u, ufl.TestFunction(u.ufl_function_space())),
        ufl.derivative(
            G,
            alpha,
            ufl.TestFunction(alpha.ufl_function_space()),
        ),
    ]
    dolfinx.fem.form(F_)

    dofs_alpha_left = locate_dofs_geometrical(V_alpha, lambda x: np.isclose(x[0], 0.0))
    dofs_alpha_right = locate_dofs_geometrical(V_alpha, lambda x: np.isclose(x[0], 1))

    dofs_u_left = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], 0.0))
    dofs_u_right = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], 1))

    bc_u_left = dirichletbc(np.array(0, dtype=PETSc.ScalarType), dofs_u_left, V_u)

    bc_u_right = dirichletbc(zero, dofs_u_right)

    bcs_u = [bc_u_left, bc_u_right]

    bcs = {"bcs_u": bcs_u, "bcs_alpha": []}

    # Perturbations
    β = dolfinx.fem.Function(V_alpha, name="DamagePerturbation")
    v = dolfinx.fem.Function(V_u, name="DisplacementPerturbation")
    perturbation = {"v": v, "β": β}

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
    num_modes = 1

    _logger.setLevel(level=logging.INFO)

    bifurcation = BifurcationSolver(
        G, state, bcs, bifurcation_parameters=parameters.get("stability")
    )
    stability = StabilitySolver(
        G, state, bcs, cone_parameters=parameters.get("stability")
    )
    bifurcation.solve(zero_alpha)
    bifurcation.get_inertia()
    stable = stability.solve(
        zero_alpha, eig0=bifurcation.spectrum[0]["xk"], inertia=(1, 0, 10)
    )

    _logger.setLevel(level=logging.INFO)

    if bifurcation.spectrum:
        vec_to_functions(bifurcation.spectrum[0]["xk"], [v, β])

        _support = indicator_function(stability.perturbation["β"])
        D_support = parallel_assemble_scalar(_support * dx)

        tol = 1e-3
        xs = np.linspace(0 + tol, 1 - tol, 101)
        points = np.zeros((3, 101))
        points[0] = xs

        plotter = pyvista.Plotter(
            title="Perturbation profile",
            window_size=[1000, 600],
            shape=(1, 3),
        )
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

        _plt, data_bifurcation = plot_profile(
            β,
            points,
            plotter,
            subplot=(1, 3),
            fig=fig,
            ax=axes[0],
            lineproperties={"c": "k", "label": f"$\\beta$"},
            subplotnumber=1,
        )
        axes[0] = _plt.gca()
        axes[0].set_xlabel("x")
        axes[0].set_yticks([-1, 0, 1])

        _plt.legend()
        _plt.fill_between(
            data_bifurcation[0], data_bifurcation[1].reshape(len(data_bifurcation[1]))
        )
        _plt.title("Perurbation in Vector Space")

        _plt, data_bifurcation = plot_profile(
            v,
            points,
            plotter,
            subplot=(1, 3),
            fig=fig,
            ax=axes[0],
            lineproperties={"c": "k", "label": f"$v$", "ls": "--"},
            subplotnumber=1,
        )
        axes[0].set_ylabel("$v,\\beta$")

        _plt, data_stability = plot_profile(
            stability.perturbation["β"],
            points,
            plotter,
            fig=fig,
            ax=axes[1],
            subplot=(1, 3),
            lineproperties={"c": "k", "label": f"$\\beta$"},
            subplotnumber=2,
        )
        _plt.fill_between(
            data_stability[0], data_stability[1].reshape(len(data_stability[1]))
        )

        _plt, data_stability = plot_profile(
            stability.perturbation["v"],
            points,
            plotter,
            fig=fig,
            ax=axes[1],
            subplot=(1, 3),
            lineproperties={"c": "k", "label": f"$v$", "ls": "--"},
            subplotnumber=2,
        )

        axes[1] = _plt.gca()
        axes[1].set_xlabel("x")
        axes[1].set_xticks(
            [0, _D, D_support, 1 - _D, 1], [0, r"$D$", r"D^*", r"$1-D$", 1]
        )
        axes[1].set_yticks([0], [0])
        axes[1].set_ylabel("$v,\\beta$")
        _plt.legend()
        _plt.title("Perurbation in the Cone")

        _plt, data_stability = plot_profile(
            stability.residual["ζ"],
            points,
            plotter,
            fig=fig,
            ax=axes[2],
            subplot=(1, 3),
            lineproperties={"c": "k", "label": f"$\\zeta$"},
            subplotnumber=3,
        )
        _plt.fill_between(
            data_stability[0], data_stability[1].reshape(len(data_stability[1]))
        )

        _plt, data_stability = plot_profile(
            stability.residual["w"],
            points,
            plotter,
            fig=fig,
            ax=axes[2],
            subplot=(1, 3),
            lineproperties={"c": "k", "label": f"$w$", "ls": "--"},
            subplotnumber=3,
        )

        axes[2] = _plt.gca()
        axes[2].set_xlabel("x")
        axes[2].set_xticks(
            [0, _D, D_support, 1 - _D, 1], [0, r"$D$", r"D^*", r"$1-D$", 1]
        )
        axes[2].set_yticks([0], [0])

        _plt.title("Residual in the Cone")

        _plt.savefig(f"{prefix}/rayleigh-benchmark.png")
        _plt.close()

    data_bifurcation_v = get_datapoints(bifurcation.perturbation["v"], points)
    data_bifurcation_β = get_datapoints(bifurcation.perturbation["β"], points)
    data_stability_v = get_datapoints(stability.perturbation["v"], points)
    data_stability_β = get_datapoints(stability.perturbation["β"], points)
    data_stability_residual_w = get_datapoints(stability.residual["w"], points)
    data_stability_residual_ζ = get_datapoints(stability.residual["ζ"], points)

    mode_shapes_data["time_steps"].append(0)
    mode_shapes_data["point_values"]["x_values"] = data_stability[0]

    _R_vector = rayleigh_ratio(
        (bifurcation.perturbation["v"], bifurcation.perturbation["β"]), parameters
    )
    _R_cone = rayleigh_ratio(
        (stability.perturbation["v"], stability.perturbation["β"]), parameters
    )

    for mode in range(1, num_modes + 1):
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
        mode_shapes_data["mesh"] = data_stability_β[0][:, 0]

        mode_shapes_data["global_values"]["R_vector"] = _R_vector
        mode_shapes_data["global_values"]["R_cone"] = _R_cone
        mode_shapes_data["global_values"]["D_theory"] = _D
        mode_shapes_data["global_values"]["D_support"] = D_support

    print(mode_shapes_data["global_values"])
    np.savez(f"{prefix}/mode_shapes_data.npz", **mode_shapes_data)

    return None, None, None


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

    parameters["model"] = {}
    parameters["model"]["model_dimension"] = 1
    parameters["model"]["model_type"] = "1D"
    parameters["model"].update({"a": 0.75, "b": 2, "c": -2})
    # _numerical_parameters = eig.book_of_the_numbers()
    # parameters["model"].update(_numerical_parameters)

    parameters["geometry"]["geom_type"] = "infinite-dimensional-unit-test"
    # Get mesh parameters

    parameters["geometry"]["geom_type"] = "traction-bar"
    parameters["geometry"]["mesh_size_factor"] = 4
    parameters["geometry"]["N"] = ndofs

    parameters["stability"]["inactiveset_gatol"] = 1e-1

    parameters["stability"]["cone"]["cone_max_it"] = 400000
    parameters["stability"]["cone"]["cone_atol"] = 1e-10
    parameters["stability"]["cone"]["cone_rtol"] = 1e-10
    parameters["stability"]["cone"]["scaling"] = 1e-3

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    return parameters, signature


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process evolution.")
    parser.add_argument("-N", help="The number of dofs.", type=int, default=50)
    args = parser.parse_args()
    parameters, signature = load_parameters("parameters.yml", ndofs=args.N)
    pretty_parameters = json.dumps(parameters, indent=2)

    _storage = f"output/rayleigh-benchmark/MPI-{MPI.COMM_WORLD.Get_size()}/{signature}"
    ColorPrint.print_bold(f"===================-{_storage}-=================")

    with dolfinx.common.Timer(f"~Computation Experiment") as timer:
        history_data, stability_data, state = test_rayleigh(parameters, _storage)
