import os
import sys
sys.path.append("../")
import test_binarydataio as bio
# from test_extend import test_extend_vector
# from test_cone_project import _cone_project_restricted
from algorithms.so import BifurcationSolver, StabilitySolver
from utils import _logger
import dolfinx
import ufl
import numpy as np
from dolfinx.io import XDMFFile
import random
from dolfinx.fem import locate_dofs_geometrical, dirichletbc
import yaml
from petsc4py import PETSc
from mpi4py import MPI
import pickle 
import logging
import argparse
from utils import ColorPrint
import json
from solvers.function import vec_to_functions
import pyvista
from pyvista.utilities import xvfb
from utils.viz import plot_profile
from pathlib import Path
import matplotlib.pyplot as plt

_logger.setLevel(logging.CRITICAL)

def rayleigh(parameters, storage=None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

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
        with open(f"{prefix}/parameters.yaml", 'w') as file:
            yaml.dump(parameters, file)

        with open(f"{prefix}/signature.md5", 'w') as f:
            f.write(signature)

    _a = parameters['model']['a']
    _b = parameters['model']['b']
    _c = parameters['model']['c']

    a = dolfinx.fem.Constant(mesh, PETSc.ScalarType(_a))
    b = dolfinx.fem.Constant(mesh, PETSc.ScalarType(_b))
    c = dolfinx.fem.Constant(mesh, PETSc.ScalarType(_c))

    
    # (size of the) support of the cone-eigenfunction - if any.
    # 
    _D = (np.pi**2 * _a/(_b*_c**2) )**(1/3)

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
        zero.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
            mode=PETSc.ScatterMode.FORWARD)

    one_alpha.interpolate(lambda x: np.zeros_like(x[0]))
    one_alpha.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
            mode=PETSc.ScatterMode.FORWARD)

    dx = ufl.Measure("dx", alpha.function_space.mesh)

    G = 1/2 * (a * alpha.dx(0)**2 + b * (u.dx(0) - c * alpha)**2) \
            * dx

    F_ = [
        ufl.derivative(
            G, u, ufl.TestFunction(u.ufl_function_space())
        ),
        ufl.derivative(
            G, alpha, ufl.TestFunction(alpha.ufl_function_space()),
        ),
    ]
    F = dolfinx.fem.form(F_)
    
    print(F)

    dofs_alpha_left = locate_dofs_geometrical(
        V_alpha, lambda x: np.isclose(x[0], 0.))
    dofs_alpha_right = locate_dofs_geometrical(
        V_alpha, lambda x: np.isclose(x[0], 1))

    dofs_u_left = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], 0.))
    dofs_u_right = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], 1))
    
    bc_u_left = dirichletbc(
        np.array(0, dtype=PETSc.ScalarType), dofs_u_left, V_u)

    bc_u_right = dirichletbc(
        zero, dofs_u_right)
    
    bcs_u = [bc_u_left, bc_u_right]

    bcs = {"bcs_u": bcs_u, "bcs_alpha": []}

    # Perturbations
    β = dolfinx.fem.Function(V_alpha, name="DamagePerturbation")
    v = dolfinx.fem.Function(V_u, name="DisplacementPerturbation")
    perturbation = {"v": v, "beta": β}
    
    # Pack state
    state = {"u": u, "alpha": alpha}

    mode_shapes_data = {
        'time_steps': [],
        'point_values': {
            'x_values': [],
        }
    }
    num_modes = 1
    
    _logger.setLevel(level=logging.INFO)

    bifurcation = BifurcationSolver(
        G, state, bcs,
        bifurcation_parameters=parameters.get("stability")
    )

    stability = StabilitySolver(
        G, state, bcs,
        cone_parameters=parameters.get("stability")
    )

    is_unique = bifurcation.solve(zero_alpha)
    inertia = bifurcation.get_inertia()
    stable = stability.solve(zero_alpha, eig0=bifurcation.spectrum, inertia = (1, 0, 10))
    
    _logger.setLevel(level=logging.INFO)
    
    if bifurcation.spectrum:
        vec_to_functions(bifurcation.spectrum[0]['xk'], [v, β])
        
        tol = 1e-3
        xs = np.linspace(0 + tol, 1 - tol, 101)
        points = np.zeros((3, 101))
        points[0] = xs
        
        plotter = pyvista.Plotter(
            title="Perturbation profile",
            window_size=[800, 600],
            shape=(1, 2),
        )
        fig, axes = plt.subplots(nrows=1, ncols=2)

        _plt, data_bifurcation = plot_profile(
            β,
            points,
            plotter,
            subplot=(1, 2),
            fig = fig,
            ax = axes[0],
            lineproperties={
                "c": "k",
                "label": f"$\\beta$"
            },
            subplotnumber=1
        )
        axes[0] = _plt.gca()
        axes[0].set_xlabel('x')
        axes[0].set_yticks([-1, 0, 1])
        axes[0].set_ylabel('$\\beta$')
        
        _plt.legend()
        _plt.fill_between(data_bifurcation[0], data_bifurcation[1].reshape(len(data_bifurcation[1])))
        _plt.title("Perurbation in Vector Space")

        _plt, data_stability = plot_profile(
            stability.perturbation['beta'],
            points,
            plotter,
            fig = fig,
            ax = axes[1],
            subplot=(1, 2),
            lineproperties={
                "c": "k",
                "label": f"$\\beta$"
            },
            subplotnumber=2,
        )

        axes[1] = _plt.gca()
        axes[1].set_xlabel('x')
        axes[1].set_xticks([0, _D, 1], [0, r"$D$", 1])
        axes[1].set_yticks([0, 1], [0, 1])
        axes[1].set_ylabel('$\\beta$')
        _plt.legend()
        _plt.fill_between(data_stability[0], data_stability[1].reshape(len(data_stability[1])))
        _plt.title("Perurbation in the Cone")
        _plt.savefig(f"{prefix}/rayleigh-benchmark.png")
        _plt.close()

    mode_shapes_data['time_steps'].append(0)
    mode_shapes_data['point_values']['x_values'] = data_stability[0]
                    
    for mode in range(1, num_modes + 1):
        bifurcation_values_mode = data_bifurcation[1].flatten()  # Replace with actual values
        stability_values_mode = data_stability[1].flatten()  # Replace with actual values
        # Append mode-specific fields to the data structure
        mode_key = f'mode_{mode}'
        mode_shapes_data['point_values'][mode_key] = {
            'bifurcation': mode_shapes_data['point_values'].get(mode_key, {}).get('bifurcation', []),
            'stability': mode_shapes_data['point_values'].get(mode_key, {}).get('stability', []),
        }
        mode_shapes_data['point_values'][mode_key]['bifurcation'].append(bifurcation_values_mode)
        mode_shapes_data['point_values'][mode_key]['stability'].append(stability_values_mode)

    np.savez(f'{prefix}/mode_shapes_data.npz', **mode_shapes_data)

    return None, None, None

def load_parameters(file_path, ndofs, model='at1'):
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
    parameters["model"]["model_type"] = '1D'
    parameters["model"].update({'a': 1,
                                'b': 1,
                                'c': 8})

    parameters["geometry"]["geom_type"] = "infinite-dimensional-unit-test"
    # Get mesh parameters

    parameters["geometry"]["geom_type"] = "traction-bar"
    parameters["geometry"]["mesh_size_factor"] = 4
    parameters["geometry"]["N"] = ndofs

    parameters["stability"]["inactiveset_gatol"] = 1e-1
    
    parameters["stability"]["cone"]["cone_max_it"] = 400000
    parameters["stability"]["cone"]["cone_atol"] = 1e-8
    parameters["stability"]["cone"]["cone_rtol"] = 1e-8
    parameters["stability"]["cone"]["scaling"] = 1e-2
    
    signature = hashlib.md5(str(parameters).encode('utf-8')).hexdigest()

    return parameters, signature

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process evolution.')
    parser.add_argument("-N", help="The number of dofs.", type=int, default=10)
    args = parser.parse_args()
    parameters, signature = load_parameters("parameters.yml", ndofs=args.N)
    pretty_parameters = json.dumps(parameters, indent=2)


    _storage = f"output/rayleigh-benchmark/MPI-{MPI.COMM_WORLD.Get_size()}/{signature}"
    ColorPrint.print_bold(f"===================-{_storage}-=================")

    with dolfinx.common.Timer(f"~Computation Experiment") as timer:
        history_data, stability_data, state = rayleigh(parameters, _storage)
