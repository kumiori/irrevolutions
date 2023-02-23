#!/usr/bin/env python3
from re import A
import numpy as np
import yaml
import json
from pathlib import Path
import sys
import os
from mpi4py import MPI
import petsc4py
from petsc4py import PETSc
import dolfinx
import dolfinx.plot
from dolfinx import log
import ufl
import numpy as np
sys.path.append("../")
from utils.plots import plot_energies
from utils import ColorPrint
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from models import DamageElasticityModel as Brittle
from algorithms.am import AlternateMinimisation

from meshes.pacman import mesh_pacman
from dolfinx.common import Timer, list_timings, TimingType

from ufl import Circumradius, FacetNormal, SpatialCoordinate

import logging

logging.basicConfig(level=logging.DEBUG)

import dolfinx
import dolfinx.plot
from dolfinx.io import XDMFFile, gmshio
from dolfinx.fem import (
    Constant,
    Expression,
    # UserExpression,
    Function,
    FunctionSpace,
    assemble_scalar,
    dirichletbc,
    form,
    locate_dofs_geometrical,
    set_bc,
)
import dolfinx.mesh
from dolfinx.mesh import CellType, locate_entities_boundary

import ufl

from mpi4py import MPI
import petsc4py
from petsc4py import PETSc
import sys
import yaml

sys.path.append("../")
from solvers import SNESSolver
from algorithms.so import StabilitySolver

# ///////////


petsc4py.init(sys.argv)
comm = MPI.COMM_WORLD

# Mesh on node model_rank and then distribute
model_rank = 0

with open("output/test_notch/parameters.yml") as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)


# Get mesh parameters
_r = parameters["geometry"]["r"]
_omega = parameters["geometry"]["omega"]
tdim = parameters["geometry"]["geometric_dimension"]
_nameExp = parameters["geometry"]["geom_type"]
ell_ = parameters["model"]["ell"]
lc = ell_ / 3.0

parameters["geometry"]["lc"] = lc


# Get geometry model
geom_type = parameters["geometry"]["geom_type"]

# Create the mesh of the specimen with given dimensions

gmsh_model, tdim = mesh_pacman(geom_type, parameters["geometry"], tdim)

# Get mesh and meshtags
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

from dolfinx.mesh import CellType, create_unit_square
mesh = create_unit_square(MPI.COMM_WORLD, 3, 3, CellType.triangle)


outdir = "output"
prefix = os.path.join(outdir, "test_notch")
if comm.rank == 0:
    Path(prefix).mkdir(parents=True, exist_ok=True)

with XDMFFile(comm, f"{prefix}/{_nameExp}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as file:
    file.write_mesh(mesh)

def plot_mesh(mesh, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.set_aspect("equal")
    points = mesh.geometry.x
    cells = mesh.geometry.dofmap.array.reshape((-1, mesh.topology.dim + 1))
    tria = tri.Triangulation(points[:, 0], points[:, 1], cells)
    ax.triplot(tria, color="k")
    return ax

if comm.rank == 0:
    plt.figure()
    ax = plot_mesh(mesh)
    fig = ax.get_figure()
    fig.savefig(f"{prefix}/mesh.png")


# Function spaces
element_u = ufl.VectorElement("Lagrange", mesh.ufl_cell(), degree=1, dim=tdim)
V_u = FunctionSpace(mesh, element_u)

# Define the state
u = Function(V_u, name="Displacement")
u_ = Function(V_u, name="Boundary Displacement")
zero_u = Function(V_u, name="   Boundary Displacement")

# Data

uD = Function(V_u)

class Asymptotic():
    def __init__(self, omega, **kwargs):
        self.omega = omega

    def value_shape(self):
        return (tdim,)
    
    def eval(self, value, x):
        self.theta = ufl.atan_2(x[1], x[0])
        # print(self.theta)
        value[0] = x[0]
        value[1] = x[1]
        

class MyExpr:
    def __init__(self, a, **kwargs):
        self.a = a

    def eval(self, value, x):
        value[0] = x[0] + self.a

class MyVExpr:
    def __init__(self, a, **kwargs):
        self.a = a

    def value_shape(self):
        return (2,)
    
    def eval(self, x):
        theta = np.arctan2(x[1], x[0])
        # e_n = 
        # e_t = 
        return (x[0], x[1])
        
f = MyExpr(1, domain=mesh)
fv = MyVExpr(1, domain=mesh)
n = FacetNormal(mesh)
t = ufl.as_vector([n[1], -n[0]])

bd_facets = locate_entities_boundary(
    mesh, dim=1, marker=lambda x: np.greater((x[0]**2 + x[1]**2), _r**2)
    )

bd_cells = locate_entities_boundary(mesh, dim=1, marker=lambda x: np.greater(x[0], 0.5))

bd_facets2 = locate_entities_boundary(
    mesh, dim=1, marker=lambda x: np.greater(x[0], 0.)
    )

u_expr = n + t

boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
cells0 = dolfinx.mesh.locate_entities(mesh, 2, lambda x: x[0] <= 0.5)

uD.interpolate(fv.eval)

_asym = Asymptotic(omega = parameters["geometry"]["omega"])
__import__('pdb').set_trace()

uD.interpolate(_asym)


uD.interpolate(lambda x: [np.zeros_like(x[0]), np.zeros_like(x[1])])
uD.interpolate(_asym)

