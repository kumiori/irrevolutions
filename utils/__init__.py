import json
import yaml

import ufl
from dolfinx.fem import assemble_scalar, form
import numpy as np
import mpi4py
import sys
from petsc4py import PETSc
import logging
from mpi4py import MPI
from typing import List

comm = MPI.COMM_WORLD

class ColorPrint:
    """
    Colored printing functions for strings that use universal ANSI escape
    sequences.
        - fail: bold red
        - pass: bold green,
        - warn: bold yellow,
        - info: bold blue
        - color: bold cyan
        - bold: bold white
    """

    @staticmethod
    def print_fail(message, end="\n"):
        if comm.rank == 0:
            message = str(message)
            sys.stderr.write("\x1b[1;31m" + message.strip() + "\x1b[0m" + end)
            sys.stderr.flush()

    @staticmethod
    def print_pass(message, end="\n"):
        if comm.rank == 0:
            message = str(message)
            sys.stdout.write("\x1b[1;32m" + message.strip() + "\x1b[0m" + end)
            sys.stdout.flush()

    @staticmethod
    def print_warn(message, end="\n"):
        if comm.rank == 0:
            message = str(message)
            sys.stderr.write("\x1b[1;33m" + message.strip() + "\x1b[0m" + end)
            sys.stderr.flush()

    @staticmethod
    def print_info(message, end="\n"):
        if comm.rank == 0:
            message = str(message)
            sys.stdout.write("\x1b[1;34m" + message.strip() + "\x1b[0m" + end)
            sys.stdout.flush()

    @staticmethod
    def print_color(message, end="\n"):
        if comm.rank == 0:
            message = str(message)
            sys.stdout.write("\x1b[1;36m" + message.strip() + "\x1b[0m" + end)
            sys.stdout.flush()

    @staticmethod
    def print_bold(message, end="\n"):
        if comm.rank == 0:
            message = str(message)
            sys.stdout.write("\x1b[1;37m" + message.strip() + "\x1b[0m" + end)
            sys.stdout.flush()

def setup_logger_mpi(root_priority: int = logging.INFO):
    from mpi4py import MPI
    import dolfinx
    class MPIFormatter(logging.Formatter):
        def format(self, record):
            record.rank = MPI.COMM_WORLD.Get_rank()
            record.size = MPI.COMM_WORLD.Get_size()
            return super(MPIFormatter, self).format(record)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Desired log level for the root process (rank 0)
    root_process_log_level = logging.INFO  # Adjust as needed

    # logger = logging.getLogger('E•volver')
    logger = logging.getLogger()
    logger.setLevel(root_process_log_level if rank == 0 else logging.WARNING)
    # Disable propagation to root logger for your logger
    logger.propagate = False
    # StreamHandler to log messages to the console
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('evolution.log')

    # formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
    formatter = MPIFormatter('%(asctime)s  [Rank %(rank)d, Size %(size)d]  - %(name)s - [%(levelname)s] - %(message)s')

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # file_handler.setLevel(logging.INFO)
    file_handler.setLevel(root_process_log_level if rank == 0 else logging.CRITICAL)
    console_handler.setLevel(root_process_log_level if rank == 0 else logging.CRITICAL)


    # Disable propagation to root logger for both handlers
    console_handler.propagate = False
    file_handler.propagate = False
    
    
    # logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Log messages, and only the root process will log.
    logger.info("The root process spawning an evolution computation (rank 0)")
    logger.info(
    f"DOLFINx version: {dolfinx.__version__} based on GIT commit: {dolfinx.git_commit_hash} of https://github.com/FEniCS/dolfinx/")

    return logger

_logger = setup_logger_mpi()

import subprocess

# Get the current Git branch
branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode("utf-8")

# Get the current Git commit hash
commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")

code_info = {
    "branch": branch,
    "commit_hash": commit_hash,
}

from slepc4py import __version__ as slepc_version
from dolfinx import __version__ as dolfinx_version
from petsc4py import __version__ as petsc_version

library_info = {
    "dolfinx_version": dolfinx_version,
    "petsc4py_version": petsc_version,
    "slepc4py_version": slepc_version,
}

simulation_info = {
    **library_info,
    **code_info,
}

def norm_L2(u):
    """
    Returns the L2 norm of the function u
    """
    comm = u.function_space.mesh.comm
    dx = ufl.Measure("dx", u.function_space.mesh)
    norm_form = form(ufl.inner(u, u) * dx)
    norm = np.sqrt(comm.allreduce(
        assemble_scalar(norm_form), op=mpi4py.MPI.SUM))
    return norm

def norm_H1(u):
    """
    Returns the H1 norm of the function u
    """
    comm = u.function_space.mesh.comm
    dx = ufl.Measure("dx", u.function_space.mesh)
    norm_form = form(
        (ufl.inner(u, u) + ufl.inner(ufl.grad(u), ufl.grad(u))) * dx)
    norm = np.sqrt(comm.allreduce(
        assemble_scalar(norm_form), op=mpi4py.MPI.SUM))
    return norm

def seminorm_H1(u):
    """
    Returns the H1 norm of the function u
    """
    comm = u.function_space.mesh.comm
    dx = ufl.Measure("dx", u.function_space.mesh)
    seminorm = form((ufl.inner(ufl.grad(u), ufl.grad(u))) * dx)
    seminorm = np.sqrt(comm.allreduce(
        assemble_scalar(seminorm), op=mpi4py.MPI.SUM))
    return seminorm

def set_vector_to_constant(x, value):
    with x.localForm() as local:
        local.set(value)
    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

def table_timing_data():
    import pandas as pd
    from dolfinx.common import timing

    timing_data = []
    tasks = ["~First Order: AltMin solver",
        "~First Order: AltMin-Damage solver",
        "~First Order: AltMin-Elastic solver",
        "~First Order: Hybrid solver",
        "~Second Order: Bifurcation",
        "~Second Order: Cone Project",
        "~Second Order: Stability",
        "~Postprocessing and Vis",
        "~Computation Experiment"
        ]

    for task in tasks:
        timing_data.append(timing(task))
    
    df = pd.DataFrame(timing_data, columns=["reps", "wall tot", "usr", "sys"], index=tasks)

    return df

from dolfinx.io import XDMFFile

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
            with open(f"{self.prefix}/parameters.yaml", 'w') as file:
                yaml.dump(parameters, file)

        with XDMFFile(self.comm, f"{self.prefix}/simulation_results.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as file:
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

    def visualise_results(self, df, drop=[]):
        """
        Visualise simulation results using appropriate visualization libraries.

        Args:
            df (dict): Pandas dataframe containing simulation data.
        """
        # Implement visualization code here
        print(df.drop(drop, axis=1))

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

history_data = {
    "load": [],
    "elastic_energy": [],
    "fracture_energy": [],
    "total_energy": [],
    "solver_data": [],
    "cone_data": [],
    "eigs-ball": [],
    "eigs-cone": [],
    "stable": [],
    "unique": [],
    "inertia": [],
}

def _write_history_data(equilibrium, bifurcation, stability, history_data, t, inertia, stable, energies: List):
    
    elastic_energy = energies[0]
    fracture_energy = energies[1]
    unique = True if inertia[0] == 0 and inertia[1] == 0 else False
    
    history_data["load"].append(t)
    history_data["fracture_energy"].append(fracture_energy)
    history_data["elastic_energy"].append(elastic_energy)
    history_data["total_energy"].append(elastic_energy+fracture_energy)
    history_data["solver_data"].append(equilibrium.data)
    history_data["cone_data"].append(stability.data)
    history_data["unique"].append(unique)
    history_data["stable"].append(stable)
    history_data["eigs-ball"].append(bifurcation.data["eigs"])
    history_data["eigs-cone"].append(stability.solution["lambda_t"])
    history_data["inertia"].append(inertia)

    return 
