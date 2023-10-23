import ufl
from dolfinx.fem import assemble_scalar, form
import numpy as np
import mpi4py
import sys
from petsc4py import PETSc
import logging
import slepc4py
import petsc4py
import dolfinx
import subprocess

comm = mpi4py.MPI.COMM_WORLD



# Get the current Git branch
branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode("utf-8")

# Get the current Git commit hash
commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")

code_info = {
    "branch": branch,
    "commit_hash": commit_hash,
}


library_info = {
    "dolfinx_version": dolfinx.__version__,
    "petsc4py_version": petsc4py.__version__,
    "slepc4py_version": slepc4py.__version__,
}

simulation_info = {
    **library_info,
    **code_info,
}

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
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Desired log level for the root process (rank 0)
    root_process_log_level = logging.INFO  # Adjust as needed

    logger = logging.getLogger('E•volver')
    logger.setLevel(root_process_log_level if rank == 0 else logging.CRITICAL)

    # StreamHandler to log messages to the console (or you can use other handlers)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('evolution.log')

    # file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    file_handler.setLevel(root_process_log_level if rank == 0 else logging.CRITICAL)
    console_handler.setLevel(root_process_log_level if rank == 0 else logging.CRITICAL)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Log messages, and only the root process will log.
    logger.info("The root process spawning an evolution computation (rank 0)")
    logger.info(f"This is process {rank} reporting")
    logger.critical(
    f"DOLFINx version: {dolfinx.__version__} based on GIT commit: {dolfinx.git_commit_hash} of https://github.com/FEniCS/dolfinx/")

    return logger


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