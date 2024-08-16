"""Import necessary modules and print DOLFINx version information.

Summary:
    This script imports necessary modules from DOLFINx and other libraries,
    and prints DOLFINx version information.

"""

import logging

import dolfinx
import dolfinx.io
import dolfinx.plot
from mpi4py import MPI

logging.basicConfig(level=logging.INFO)
logging.critical(
    f"DOLFINx version: {dolfinx.__version__} based on GIT commit: \
        {dolfinx.git_commit_hash} of https://github.com/FEniCS/dolfinx/"
)

comm = MPI.COMM_WORLD
