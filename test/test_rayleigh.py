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

_logger.setLevel(logging.CRITICAL)

def rayleigh(parameters, storage=None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    with XDMFFile(comm, "data/input_data.xdmf", "r") as file: 
        mesh = file.read_mesh(name='mesh')

    a = dolfinx.fem.Constant(mesh, PETSc.ScalarType(parameters['model']['a']))
    b = dolfinx.fem.Constant(mesh, PETSc.ScalarType(parameters['model']['b']))
    c = dolfinx.fem.Constant(mesh, PETSc.ScalarType(parameters['model']['c']))
    
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

    # Pack state
    state = {"u": u, "alpha": alpha}

    bifurcation = BifurcationSolver(
        G, state, bcs,
        bifurcation_parameters=parameters.get("stability")
    )

    stability = StabilitySolver(
        G, state, bcs,
        cone_parameters=parameters.get("stability")
    )

    is_unique = bifurcation.solve(zero_alpha)
    __import__('pdb').set_trace()


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
                                'c': 1})

    parameters["geometry"]["geom_type"] = "infinite-dimensional-unit-test"
    # Get mesh parameters

    parameters["geometry"]["geom_type"] = "traction-bar"
    parameters["geometry"]["mesh_size_factor"] = 4
    parameters["geometry"]["N"] = ndofs

    parameters["stability"]["inactiveset_gatol"] = 1e-1
    
    parameters["stability"]["cone"]["cone_max_it"] = 400000
    parameters["stability"]["cone"]["cone_atol"] = 1e-6
    parameters["stability"]["cone"]["cone_rtol"] = 1e-6
    parameters["stability"]["cone"]["scaling"] = 1e-2
    
    signature = hashlib.md5(str(parameters).encode('utf-8')).hexdigest()

    return parameters, signature

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process evolution.')
    parser.add_argument("-N", help="The number of dofs.", type=int, default=10)
    args = parser.parse_args()
    parameters, signature = load_parameters("parameters.yml", ndofs=args.N)
    pretty_parameters = json.dumps(parameters, indent=2)


    _storage = f"output/one-dimensional-bar/MPI-{MPI.COMM_WORLD.Get_size()}/{args.N}/{signature}"
    ColorPrint.print_bold(f"===================-{_storage}-=================")


    with dolfinx.common.Timer(f"~Computation Experiment") as timer:
        history_data, stability_data, state = rayleigh(parameters, _storage)

    