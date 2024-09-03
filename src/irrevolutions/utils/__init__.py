from dolfinx.io import XDMFFile
from slepc4py import __version__ as slepc_version
from petsc4py import __version__ as petsc_version
from dolfinx import __version__ as dolfinx_version
import json
import logging
import os
import pickle
import subprocess
import sys
from typing import List

import mpi4py
import numpy as np
import ufl
import yaml
from dolfinx.fem import assemble_scalar, form
from mpi4py import MPI
from petsc4py import PETSc

comm = MPI.COMM_WORLD

error_codes = {
    "PETSC_SUCCESS": 0,
    "PETSC_ERR_BOOLEAN_MACRO_FAILURE": 1,
    "PETSC_ERR_MIN_VALUE": 54,
    "PETSC_ERR_MEM": 55,
    "PETSC_ERR_SUP": 56,
    "PETSC_ERR_SUP_SYS": 57,
    "PETSC_ERR_ORDER": 58,
    "PETSC_ERR_SIG": 59,
    "PETSC_ERR_FP": 72,
    "PETSC_ERR_COR": 74,
    "PETSC_ERR_LIB": 76,
    "PETSC_ERR_PLIB": 77,
    "PETSC_ERR_MEMC": 78,
    "PETSC_ERR_CONV_FAILED": 82,
    "PETSC_ERR_USER": 83,
    "PETSC_ERR_SYS": 88,
    "PETSC_ERR_POINTER": 70,
    "PETSC_ERR_MPI_LIB_INCOMP": 87,
    "PETSC_ERR_ARG_SIZ": 60,
    "PETSC_ERR_ARG_IDN": 61,
    "PETSC_ERR_ARG_WRONG": 62,
    "PETSC_ERR_ARG_CORRUPT": 64,
    "PETSC_ERR_ARG_OUTOFRANGE": 63,
    "PETSC_ERR_ARG_BADPTR": 68,
    "PETSC_ERR_ARG_NOTSAMETYPE": 69,
    "PETSC_ERR_ARG_NOTSAMECOMM": 80,
    "PETSC_ERR_ARG_WRONGSTATE": 73,
    "PETSC_ERR_ARG_TYPENOTSET": 89,
    "PETSC_ERR_ARG_INCOMP": 75,
    "PETSC_ERR_ARG_NULL": 85,
    "PETSC_ERR_ARG_UNKNOWN_TYPE": 86,
    "PETSC_ERR_FILE_OPEN": 65,
    "PETSC_ERR_FILE_READ": 66,
    "PETSC_ERR_FILE_WRITE": 67,
    "PETSC_ERR_FILE_UNEXPECTED": 79,
    "PETSC_ERR_MAT_LU_ZRPVT": 71,
    "PETSC_ERR_MAT_CH_ZRPVT": 81,
    "PETSC_ERR_INT_OVERFLOW": 84,
    "PETSC_ERR_FLOP_COUNT": 90,
    "PETSC_ERR_NOT_CONVERGED": 91,
    "PETSC_ERR_MISSING_FACTOR": 92,
    "PETSC_ERR_OPT_OVERWRITE": 93,
    "PETSC_ERR_WRONG_MPI_SIZE": 94,
    "PETSC_ERR_USER_INPUT": 95,
    "PETSC_ERR_GPU_RESOURCE": 96,
    "PETSC_ERR_GPU": 97,
    "PETSC_ERR_MPI": 98,
    "PETSC_ERR_RETURN": 99,
    "PETSC_ERR_MEM_LEAK": 100,
    "PETSC_ERR_MAX_VALUE": 101,
    "PETSC_ERR_MIN_SIGNED_BOUND_DO_NOT_USE": "INT_MIN",
    "PETSC_ERR_MAX_SIGNED_BOUND_DO_NOT_USE": "INT_MAX",
}


# Reverse the dictionary to create an inverse mapping
translatePETScERROR = {v: k for k, v in error_codes.items()}

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
    import dolfinx
    from mpi4py import MPI

    class MPIFormatter(logging.Formatter):
        def format(self, record):
            record.rank = MPI.COMM_WORLD.Get_rank()
            record.size = MPI.COMM_WORLD.Get_size()
            return super(MPIFormatter, self).format(record)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm.Get_size()

    # Desired log level for the root process (rank 0)
    root_process_log_level = logging.INFO  # Adjust as needed

    # logger = logging.getLogger('E•volver')
    logger = logging.getLogger()
    logger.setLevel(root_process_log_level if rank == 0 else logging.WARNING)
    # Disable propagation to root logger for your logger
    logger.propagate = False
    # StreamHandler to log messages to the console
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("evolution.log")

    # formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
    formatter = MPIFormatter(
        "%(asctime)s  [Rank %(rank)d, Size %(size)d]  - %(name)s - [%(levelname)s] - %(message)s"
    )

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
        f"DOLFINx version: {dolfinx.__version__} based on GIT commit: {dolfinx.git_commit_hash} of https://github.com/FEniCS/dolfinx/"
    )

    return logger


_logger = setup_logger_mpi()


# Get the current Git branch
def get_branch_details():
    # Try to get branch name and commit hash using GitHub Actions environment variables
    try:
        # Retrieve branch name
        github_ref = os.getenv("GITHUB_REF")
        if github_ref:
            # Split the reference by '/' and get the last element, which is the branch name
            parts = github_ref.split("/")
            branch_name = parts[-1]
        else:
            print("GITHUB_REF environment variable not found.")
            branch_name = None

        # Retrieve commit hash
        github_sha = os.getenv("GITHUB_SHA")
        if github_sha:
            commit_hash = github_sha[:7]  # Take the first 7 characters of the SHA
        else:
            print("GITHUB_SHA environment variable not found.")
            commit_hash = None

        return branch_name, commit_hash

    except Exception as e:
        print(
            f"Failed to retrieve branch name and commit hash from GitHub Actions environment: {e}"
        )
        branch_name = None
        commit_hash = None

    # If GitHub Actions environment variables are not available, try to get branch name and commit hash locally
    try:
        # Get the current Git branch
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.PIPE
            )
            .strip()
            .decode("utf-8")
        )
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
        return branch, commit_hash

    except Exception as e:
        print(f"Failed to retrieve branch name and commit hash locally: {e}")
        return "unknown", "unknown"


# Get the current branch

branch, commit_hash = get_branch_details()

# Get the current Git commit hash

code_info = {
    "branch": branch,
    "commit_hash": commit_hash,
}


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
    norm = np.sqrt(comm.allreduce(assemble_scalar(norm_form), op=mpi4py.MPI.SUM))
    return norm


def norm_H1(u):
    """
    Returns the H1 norm of the function u
    """
    comm = u.function_space.mesh.comm
    dx = ufl.Measure("dx", u.function_space.mesh)
    norm_form = form((ufl.inner(u, u) + ufl.inner(ufl.grad(u), ufl.grad(u))) * dx)
    norm = np.sqrt(comm.allreduce(assemble_scalar(norm_form), op=mpi4py.MPI.SUM))
    return norm


def seminorm_H1(u):
    """
    Returns the H1 norm of the function u
    """
    comm = u.function_space.mesh.comm
    dx = ufl.Measure("dx", u.function_space.mesh)
    seminorm = form((ufl.inner(ufl.grad(u), ufl.grad(u))) * dx)
    seminorm = np.sqrt(comm.allreduce(assemble_scalar(seminorm), op=mpi4py.MPI.SUM))
    return seminorm


def set_vector_to_constant(x, value):
    with x.localForm() as local:
        local.set(value)
    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


def table_timing_data():
    import pandas as pd
    from dolfinx.common import timing

    timing_data = []
    tasks = [
        "~First Order: AltMin solver",
        "~First Order: AltMin-Damage solver",
        "~First Order: AltMin-Elastic solver",
        "~First Order: Hybrid solver",
        "~Second Order: Bifurcation",
        "~Second Order: Cone Project",
        "~Second Order: Stability",
        "~Postprocessing and Vis",
        "~Computation Experiment",
    ]

    for task in tasks:
        timing_data.append(timing(task))

    df = pd.DataFrame(
        timing_data, columns=["reps", "wall tot", "usr", "sys"], index=tasks
    )

    return df


def find_offending_columns_lengths(data):
    lengths = {}
    for key, value in data.items():
        try:
            lengths[key] = len(value)
        except TypeError:
            lengths[key] = "Non-iterable"
    return lengths




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
    "equilibrium_data": [],
    "cone_data": [],
    "eigs_ball": [],
    "eigs_cone": [],
    "stable": [],
    "unique": [],
    "inertia": [],
}


def _write_history_data(
    equilibrium,
    bifurcation,
    stability,
    history_data,
    t,
    inertia,
    stable,
    energies: List,
):
    elastic_energy = energies[0]
    fracture_energy = energies[1]
    unique = True if inertia[0] == 0 and inertia[1] == 0 else False

    history_data["load"].append(t)
    history_data["fracture_energy"].append(fracture_energy)
    history_data["elastic_energy"].append(elastic_energy)
    history_data["total_energy"].append(elastic_energy + fracture_energy)
    history_data["equilibrium_data"].append(equilibrium.data)
    history_data["inertia"].append(inertia)
    history_data["unique"].append(unique)
    history_data["stable"].append(stable)
    history_data["eigs_ball"].append(bifurcation.data["eigs"])
    history_data["cone_data"].append(stability.data)
    history_data["eigs_cone"].append(stability.solution["lambda_t"])

    return


def indicator_function(v):
    import dolfinx

    # Create the indicator function
    w = dolfinx.fem.Function(v.function_space)
    with w.vector.localForm() as w_loc, v.vector.localForm() as v_loc:
        w_loc[:] = np.where(v_loc[:] > 0, 1.0, 0.0)

    w.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    return w


def save_binary_data(filename, data):
    viewer = PETSc.Viewer().createBinary(filename, "w")

    if isinstance(data, list):
        for item in data:
            item.view(viewer)
    elif isinstance(data, PETSc.Mat):
        data.view(viewer)
    elif isinstance(data, PEtest_binarydataioTSc.Vec):
        data.view(viewer)
    else:
        raise ValueError("Unsupported data type for saving")


def load_binary_data(filename):
    viewer = PETSc.Viewer().createBinary(filename, "r")
    data = []
    vectors = []

    i = 0
    while True:
        try:
            vec = PETSc.Vec().load(viewer)

            vectors.append(vec)
            i += 1
        except PETSc.Error as e:
            # __import__('pdb').set_trace()
            # if e.ierr == PETSc.Error.S_ARG_WRONG:
            print(f"Error {e.ierr}: {translatePETScERROR.get(e.ierr, 'Unknown error')}")
            break
            # else:
            # raise

    return data


def load_binary_vector(filename):
    """
    Load a binary file containing a PETSc vector.

    Args:
        filename (str): Path to the binary file.

    Returns:
        PETSc.Vec: Loaded PETSc vector.
    """
    try:
        # Create a PETSc viewer for reading
        viewer = PETSc.Viewer().createBinary(filename, "r")

        # Load the vector from the viewer
        vector = PETSc.Vec().load(viewer)

        # Close the viewer
        viewer.destroy()

        return vector

    except PETSc.Error as e:
        print(f"Error: {e}")
        return None


def load_binary_matrix(filename):
    """
    Load a binary file containing a PETSc Matrix.

    Args:
        filename (str): Path to the binary file.

    Returns:
        PETSc.Mat: Loaded PETSc Matrix.
    """
    try:
        # Create a PETSc viewer for reading
        viewer = PETSc.Viewer().createBinary(filename, "r")

        # Load the vector from the viewer
        vector = PETSc.Mat().load(viewer)

        # Close the viewer
        viewer.destroy()

        return vector

    except PETSc.Error as e:
        print(f"Error: {e}")
        return None


def save_minimal_constraints(obj, filename):
    minimal_constraints = {
        "bglobal_dofs_mat": obj.bglobal_dofs_mat,
        "bglobal_dofs_mat_stacked": obj.bglobal_dofs_mat_stacked,
        "bglobal_dofs_vec": obj.bglobal_dofs_vec,
        "bglobal_dofs_vec_stacked": obj.bglobal_dofs_vec_stacked,
        "blocal_dofs": obj.blocal_dofs,
        "boffsets_mat": obj.boffsets_mat,
        "boffsets_vec": obj.boffsets_vec,
    }

    with open(filename, "wb") as file:
        pickle.dump(minimal_constraints, file)


def load_minimal_constraints(filename):
    import irrevolutions.solvers.restriction as restriction

    with open(filename, "rb") as file:
        minimal_constraints = pickle.load(file)

    # Assuming you have a constructor for your class
    # Modify this accordingly based on your actual class structure
    reconstructed_obj = restriction.Restriction()
    for key, value in minimal_constraints.items():
        setattr(reconstructed_obj, key, value)

    return reconstructed_obj


def sample_data(N, positive=True):
    import random

    import dolfinx
    from dolfinx.cpp.la.petsc import get_local_vectors, scatter_local_vectors

    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, N - 1)
    comm = MPI.COMM_WORLD
    comm.Get_rank()

    element_u = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)
    element_alpha = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)

    V_u = dolfinx.fem.FunctionSpace(mesh, element_u)
    V_alpha = dolfinx.fem.FunctionSpace(mesh, element_alpha)
    u = dolfinx.fem.Function(V_u, name="Displacement")
    alpha = dolfinx.fem.Function(V_alpha, name="Damage")
    dx = ufl.Measure("dx", alpha.function_space.mesh)

    energy = (1 - alpha) ** 2 * ufl.inner(u, u) * dx

    F_ = [
        ufl.derivative(energy, u, ufl.TestFunction(u.ufl_function_space())),
        ufl.derivative(
            energy,
            alpha,
            ufl.TestFunction(alpha.ufl_function_space()),
        ),
    ]
    F = dolfinx.fem.form(F_)

    v = dolfinx.fem.petsc.create_vector_block(F)
    v.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    if positive:
        v.array = [
            np.around(random.uniform(0.1, 1.5), decimals=1) for r in range(v.local_size)
        ]
    else:
        v.array = [
            np.around(random.uniform(-1.5, 1.5), decimals=1)
            for r in range(v.local_size)
        ]

    maps = [(V.dofmap.index_map, V.dofmap.index_map_bs) for V in [V_u, V_alpha]]
    u, alpha = get_local_vectors(v, maps)
    # for visibility
    u *= 100
    scatter_local_vectors(v, [u, alpha], maps)
    v.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    return F, v
