# # Import statements

# import logging
# import json
# import yaml
# from pathlib import Path
# from dolfinx import FunctionSpace, Function
# from dolfinx.fem import locate_dofs_geometrical, dirichletbc, assemble_scalar, form
# from dolfinx.mesh import CellType
# from dolfinx.io import XDMFFile, gmshio
# from dolfinx.plot import plot
# from dolfinx.mesh import create_mesh
# from dolfinx.cpp.mesh import to_type
# import numpy as np
# from mpi4py import MPI
# import petsc4py
# from petsc4py import PETSc
# import ufl
# import sys
# import os


#!/usr/bin/env python3
import pdb
import sys
import os
import yaml
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sympy import derive_by_array
import ufl
import logging

import petsc4py
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.plot
from dolfinx import log
from dolfinx.common import Timer, list_timings, TimingType
from dolfinx.fem import (
    Constant,
    Function,
    FunctionSpace,
    assemble_scalar,
    dirichletbc,
    form,
    locate_dofs_geometrical,
    set_bc,
)
from dolfinx.fem.petsc import set_bc
from dolfinx.io import XDMFFile, gmshio
from dolfinx.mesh import CellType
import dolfinx.mesh

sys.path.append("../")

# from algorithms.am import AlternateMinimisation, HybridSolver
from algorithms.so import BifurcationSolver, StabilitySolver
from meshes.primitives import mesh_bar_gmshapi
from irrevolutions.utils import ColorPrint
from utils.plots import plot_energies
from irrevolutions.utils import norm_H1, norm_L2


# Configuration handling (load parameters from YAML)


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
    parameters["stability"]["cone"]["scaling"] = 0.3

    parameters["model"]["model_dimension"] = 2
    parameters["model"]["model_type"] = "1D"
    parameters["model"]["w1"] = 1
    parameters["model"]["ell"] = 0.1
    parameters["model"]["k_res"] = 0.0
    parameters["loading"]["min"] = 0.8
    parameters["loading"]["max"] = 10.5
    parameters["loading"]["steps"] = 10

    parameters["geometry"]["geom_type"] = "traction-bar"
    parameters["geometry"]["ell_lc"] = 5
    # Get mesh parameters
    Lx = parameters["geometry"]["Lx"]
    Ly = parameters["geometry"]["Ly"]
    tdim = parameters["geometry"]["geometric_dimension"]

    _nameExp = parameters["geometry"]["geom_type"]
    ell_ = parameters["model"]["ell"]

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    return parameters, signature


# Mesh creation function


def create_mesh(parameters):
    """
    Create a mesh based on the specified parameters.

    Args:
        parameters (dict): Simulation parameters.

    Returns:
        dolfinx.Mesh: Generated mesh.
    """
    # Extract mesh parameters from parameters dictionary
    from meshes.primitives import mesh_bar_gmshapi

    Lx = parameters["geometry"]["Lx"]
    Ly = parameters["geometry"]["Ly"]
    lc = parameters["model"]["ell"] / parameters["geometry"]["ell_lc"]
    tdim = parameters["geometry"]["geometric_dimension"]
    geom_type = parameters["geometry"]["geom_type"]
    comm = MPI.COMM_WORLD
    model_rank = 0

    # Create mesh

    gmsh_model, tdim = mesh_bar_gmshapi(geom_type, Lx, Ly, lc, tdim)

    # Get mesh and meshtags
    mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

    return mesh


# Function space creation function


def create_function_space(mesh):
    """
    Create function spaces for displacement and damage fields.

    Args:
        mesh (dolfinx.Mesh): Mesh for the simulation.

    Returns:
        dolfinx.FunctionSpace: Function space for displacement.
        dolfinx.FunctionSpace: Function space for damage.
    """
    element_u = ufl.VectorElement("Lagrange", mesh.ufl_cell(), degree=1, dim=2)
    V_u = FunctionSpace(mesh, element_u)

    element_alpha = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)
    V_alpha = FunctionSpace(mesh, element_alpha)

    return V_u, V_alpha


def init_state(V_u, V_alpha):
    """
    Create the state variables u and alpha.

    Args:
        V_u (FunctionSpace): Function space for displacement u.
        V_alpha (FunctionSpace): Function space for damage variable alpha.

    Returns:
        u (Function): Displacement function.
        alpha (Function): Damage variable function.
    """
    u = Function(V_u, name="Displacement")
    alpha = Function(V_alpha, name="Damage")
    state = {"u": u, "alpha": alpha}

    return state


# Boundary conditions setup function


def setup_boundary_conditions(V_u, V_alpha, Lx):
    """
    Set up boundary conditions for displacement and damage fields.

    Args:
        V_u (dolfinx.FunctionSpace): Function space for displacement.
        V_alpha (dolfinx.FunctionSpace): Function space for damage.
        Lx (float): Length of the specimen.

    Returns:
        list of dolfinx.DirichletBC: List of boundary conditions.
    """
    dofs_alpha_left = locate_dofs_geometrical(V_alpha, lambda x: np.isclose(x[0], 0.0))
    dofs_alpha_right = locate_dofs_geometrical(V_alpha, lambda x: np.isclose(x[0], Lx))

    dofs_u_left = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], 0.0))
    dofs_u_right = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], Lx))

    zero_u = Function(V_u)
    u_ = Function(V_u, name="Boundary Displacement")

    zero_alpha = Function(V_alpha)

    bc_u_left = dirichletbc(zero_u, dofs_u_left)
    bc_u_right = dirichletbc(u_, dofs_u_right)

    bcs_u = [bc_u_left, bc_u_right]
    bcs_alpha = []
    bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}

    return bcs


# Model initialization function


def initialise_model(parameters):
    """
    Initialise the material model based on simulation parameters.

    Args:
        parameters (dict): Simulation parameters.

    Returns:
        BrittleAT2: Initialised material model.
    """
    # Extract model parameters from parameters dictionary
    model_parameters = parameters["model"]

    from models import DamageElasticityModel as Brittle

    class BrittleAT2(Brittle):
        """Brittle AT_2 model, without an elastic phase. For fun only."""

        def w(self, alpha):
            """
            Return the dissipated energy function as a function of the state
            (only depends on damage).
            """
            # Return w(alpha) function
            return self.w1 * alpha**2

    # Initialise material model
    # model = BrittleAT2(model_parameters)
    model = Brittle(model_parameters)

    return model


# Energy functional definition function


def define_energy_functional(state, model):
    """
    Define the energy functional for the simulation.

    Args:
        V_u (dolfinx.FunctionSpace): Function space for displacement.
        V_alpha (dolfinx.FunctionSpace): Function space for damage.
        u (dolfinx.Function): Displacement field.
        alpha (dolfinx.Function): Damage field.
        model: Initialised material model.

    Returns:
        ufl.form.Form: Energy functional.
    """
    u = state["u"]
    dx = ufl.Measure("dx", domain=u.function_space.mesh)
    ds = ufl.Measure("ds", domain=u.function_space.mesh)

    # state = {"u": u, "alpha": alpha}
    # Define the external load
    f = Constant(mesh, np.array([0, 0], dtype=PETSc.ScalarType))

    # Define the energy density
    elastic_energy_density = model.elastic_energy_density(state)
    damage_energy_density = model.damage_energy_density(state)

    # Define the total energy functional
    external_work = ufl.dot(f, u) * dx
    total_energy = (elastic_energy_density + damage_energy_density) * dx - external_work

    return total_energy


# Solver initialization functions


def initialise_solver(total_energy, state, bcs, parameters):
    """
    Initialise the solver for the simulation.

    Args:
        total_energy (ufl.form.Form): Energy functional.
        V_alpha (dolfinx.FunctionSpace): Function space for damage.
        bcs_alpha (list of dolfinx.DirichletBC): List of damage boundary conditions.
        parameters (dict): Solver parameters.

    Returns:
        AlternateMinimisation: Initialised solver.
    """

    # V_u, V_alpha, u, alpha

    from algorithms.am import AlternateMinimisation

    # alpha = Function(V_alpha, name="Damage")
    V_alpha = state["alpha"].function_space

    alpha_lb = Function(V_alpha, name="Lower bound")
    alpha_ub = Function(V_alpha, name="Upper bound")

    for f in [alpha_lb, alpha_ub]:
        f.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

    set_bc(alpha_ub.vector, bcs["bcs_alpha"])
    alpha_ub.vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )

    # Initialise solver
    solver = AlternateMinimisation(
        total_energy,
        state,
        bcs,
        solver_parameters=parameters,
        bounds=(alpha_lb, alpha_ub),
    )

    return solver


# Logging setup function


def setup_logging():
    """
    Set up logging for the simulation.
    """
    logging.basicConfig(level=logging.INFO)


# Results storage functions/classes


class ResultsStorage:
    """
    Class for storing and saving simulation results.
    """

    def __init__(self, comm, prefix):
        self.comm = comm
        self.prefix = prefix

    def store_results(self, parameters, history_data, state):
        """
        Store simulation results in XDMF and JSON formats.

        Args:
            history_data (dict): Dictionary containing simulation data.
        """
        t = history_data["load"][-1]

        u = state["u"]
        alpha = state["alpha"]

        if self.comm.rank == 0:
            with open(f"{self.prefix}/parameters.yaml", "w") as file:
                yaml.dump(parameters, file)

        with XDMFFile(
            self.comm,
            f"{self.prefix}/simulation_results.xdmf",
            "w",
            encoding=XDMFFile.Encoding.HDF5,
        ) as file:
            # for t, data in history_data.items():
            # file.write_scalar(data, t)
            file.write_mesh(u.function_space.mesh)

            file.write_function(u, t)
            file.write_function(alpha, t)

        if self.comm.rank == 0:
            with open(f"{self.prefix}/time_data.json", "w") as file:
                json.dump(history_data, file)


# Visualization functions/classes


class Visualization:
    """
    Class for visualizing simulation results.
    """

    def __init__(self, prefix):
        self.prefix = prefix

    def visualise_results(self, history_data):
        """
        Visualise simulation results using appropriate visualization libraries.

        Args:
            history_data (dict): Dictionary containing simulation data.
        """
        # Implement visualization code here

    def save_table(self, data, name):
        """
        Save pandas table results using json.

        Args:
            data (dict): Pandas table containing simulation data.
            name (str): Filename.
        """

        if MPI.COMM_WORLD.rank == 0:
            a_file = open(f"{self.prefix}/{name}.json", "w")
            json.dump(data.to_json(), a_file)
            a_file.close()


# Time loop function


def run_time_loop(parameters, solver, model, bcs):
    """
    Main time loop for the simulation.

    Args:
        parameters (dict): Simulation parameters.
        solver: Initialised solver.
        model: Initialised material model.
        V_u (dolfinx.FunctionSpace): Function space for displacement.
        V_alpha (dolfinx.FunctionSpace): Function space for damage.
        bcs_u (list of dolfinx.DirichletBC): List of displacement boundary conditions.
        bcs_alpha (list of dolfinx.DirichletBC): List of damage boundary conditions.

    Returns:
        dict: Dictionary containing simulation data.
    """
    comm = MPI.COMM_WORLD
    dx = ufl.Measure("dx", domain=state["u"].function_space.mesh)

    loads = np.linspace(
        parameters["loading"]["min"],
        parameters["loading"]["max"],
        parameters["loading"]["steps"],
    )

    history_data = {
        "load": [],
        "elastic_energy": [],
        "fracture_energy": [],
        "total_energy": [],
        # Add other simulation data fields here
    }
    mesh = solver.state["u"].function_space.mesh
    map = mesh.topology.index_map(mesh.topology.dim)
    cells = np.arange(map.size_local + map.num_ghosts, dtype=np.int32)

    from dolfinx import cpp as _cpp

    _x = _cpp.fem.interpolation_coords(V_u.element, mesh, cells)

    alpha = state["alpha"]
    u = state["u"]

    # Main time loop
    for i_t, t in enumerate(loads):
        # Update boundary conditions or external loads if necessary
        datum = lambda x: (t * np.ones_like(x[0]), np.zeros_like(x[1]))
        bcs["bcs_u"][1].g.interpolate(datum(_x), cells)
        bcs["bcs_u"][1].g.x.scatter_forward()

        logging.critical(f"\n\n-- {i_t}/{len(loads)}: Solving for t = {t:3.2f} --\n")

        # Implement any necessary updates here
        # update the lower bound

        alpha.vector.copy(solver.alpha_lb.vector)
        solver.alpha_lb.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        # Solve for the current time step
        solver.solve()

        # Compute and store simulation data
        fracture_energy = comm.allreduce(
            assemble_scalar(form(model.damage_energy_density(state) * dx)),
            op=MPI.SUM,
        )
        elastic_energy = comm.allreduce(
            assemble_scalar(form(model.elastic_energy_density(state) * dx)),
            op=MPI.SUM,
        )

        # Add other simulation data calculations here

        history_data["load"].append(t)
        history_data["elastic_energy"].append(elastic_energy)
        history_data["fracture_energy"].append(fracture_energy)
        history_data["total_energy"].append(elastic_energy + fracture_energy)
        # Add other simulation data to history_data

    return history_data


if __name__ == "__main__":
    # Main script execution
    # Load parameters from YAML file
    parameters, signature = load_parameters("../test/parameters.yml")

    # Create mesh
    mesh = create_mesh(parameters)

    # Create function spaces for displacement and damage
    V_u, V_alpha = create_function_space(mesh)

    state = init_state(V_u, V_alpha)

    # Set up boundary conditions
    bcs = setup_boundary_conditions(V_u, V_alpha, parameters["geometry"]["Lx"])

    # Initialise material model
    model = initialise_model(parameters)

    # Define the energy functional
    total_energy = define_energy_functional(state, model)

    # Initialise the solver
    solver = initialise_solver(total_energy, state, bcs, parameters.get("solvers"))

    # Set up logging
    setup_logging()

    def postprocess(history_data, state):
        """Postprocess simulation data at each step."""

        return

    # Run the time loop and store results
    history_data = run_time_loop(parameters, solver, model, bcs, postprocess)

    # Store and visualise results
    storage = ResultsStorage(MPI.COMM_WORLD, f"output/traction_AT2_cone/{signature}")
    storage.store_results(parameters, history_data, state)

    visualization = Visualization(f"output/traction_AT2_cone/{signature}")
    visualization.visualise_results(history_data)

    pdb.set_trace()
    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

    ColorPrint.print_bold(f"===================-{signature}-=================")
