import yaml
from dolfinx.fem import Function
from dolfinx.fem import form
from mpi4py import MPI
import ufl
import dolfinx
import sys

from petsc4py import PETSc
import petsc4py
import numpy as np
import logging
import dolfinx.common
from dolfinx.io import XDMFFile

petsc4py.init(sys.argv)

from .am import AlternateMinimisation
