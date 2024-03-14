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
from irrevolutions.utils import ColorPrint
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
    locate_dofs_topological,
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
from algorithms.so import BifurcationSolver

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
lc = ell_ / 1.5

parameters["geometry"]["lc"] = lc


# Get geometry model
geom_type = parameters["geometry"]["geom_type"]

# Create the mesh of the specimen with given dimensions

gmsh_model, tdim = mesh_pacman(geom_type, parameters["geometry"], tdim)

# Get mesh and meshtags
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

from dolfinx.mesh import CellType, create_unit_square
# mesh = create_unit_square(MPI.COMM_WORLD, 3, 3, CellType.triangle)


outdir = os.path.join(os.path.dirname(__file__), "output")
prefix = os.path.join(outdir, "test_notch")
if comm.rank == 0:
    Path(prefix).mkdir(parents=True, exist_ok=True)

with XDMFFile(comm, f"{prefix}/{_nameExp}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as file:
    file.write_mesh(mesh)

# Function spaces
element_u = ufl.VectorElement("Lagrange", mesh.ufl_cell(), degree=1, dim=tdim)
V_u = FunctionSpace(mesh, element_u)

# Define the state
u = Function(V_u, name="Displacement")
u_ = Function(V_u, name="Boundary Displacement")
zero_u = Function(V_u, name="   Boundary Displacement")

# Data

uD = Function(V_u)

def singularity_exp(omega):
    """Exponent of singularity, λ\in [1/2, 1]
    lmbda : = sin(2*lmbda*(pi - omega)) + lmbda*sin(2(pi-lmbda)) = 0"""
    from sympy import nsolve, pi, sin, symbols

    x = symbols('x')

    return nsolve(
        sin(2*x*(pi - omega)) + x*sin(2*(pi-omega)), 
        x, .5)

import sympy as sp

bd_facets = locate_entities_boundary(
    mesh, dim=1, marker=lambda x: np.greater((x[0]**2 + x[1]**2), _r**2)
    )

bd_facets = locate_entities_boundary(
    mesh, dim=1, marker=lambda x: np.greater((x[0]**2 + x[1]**2), _r**2)
    )
  
bd_cells = locate_entities_boundary(mesh, dim=1, marker = lambda x: np.greater(x[0], 0.5))

bd_facets2 = locate_entities_boundary(
    mesh, dim=1, marker=lambda x: np.greater(x[0], 0.)
    )


bd_facets3 = locate_entities_boundary(mesh, dim=1, marker = lambda x: np.full(x.shape[1], True, dtype=bool))
boundary_dofs = locate_dofs_topological(V_u, mesh.topology.dim - 1, bd_facets3)

boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

assert((boundary_facets == bd_facets3).all())

def _local_notch_asymptotic(x, ω=np.deg2rad(_omega / 2.), t=1., par = parameters["material"]):
    from sympy import nsolve, pi, sin, cos, pi, symbols
    λ = singularity_exp(ω)
    Θ = symbols('Θ')
    _E = par['E']
    ν = par['ν']
    Θv = np.arctan2(x[1], x[0])
        
    coeff = ( (1+λ) * sin( (1+λ) * (pi - ω) ) ) / ( (1-λ) * sin( (1-λ) * (pi - ω) ) )

    _f = (2*np.pi)**(λ - 1) * ( cos( (1+λ) * Θ) - coeff * cos((1-λ) * Θ) ) / (1-coeff)

    f = sp.lambdify(Θ, _f, "numpy")
    fp = sp.lambdify(Θ, sp.diff(_f, Θ, 1), "numpy")
    fpp = sp.lambdify(Θ, sp.diff(_f, Θ, 2), "numpy")
    fppp = sp.lambdify(Θ, sp.diff(_f, Θ, 3), "numpy")

    # print("F(0)", f(0))
    # print("F(pi - ω)", f(np.float16(pi.n() - ω)))
    # print("F(-pi + ω)", f(np.float16(-pi.n() + ω)))

    assert(np.isclose(f(np.float16(pi.n() - ω)), 0.))
    assert(np.isclose(f(np.float16(-pi.n() + ω)), 0.))

    r = np.sqrt(x[0]**2. + x[1]**2.)
    _c1 = (λ+1)*(1- ν*λ - ν**2.*(λ+1))
    _c2 = 1-ν**2.
    _c3 = 2.*(1+ν)*λ**2. + _c1
    _c4 = _c2
    _c5 = λ**2. * (1-λ**2.)

    ur = t * ( r**λ / _E * (_c1*f(Θv) + _c2*fpp(Θv)) ) / _c5
    uΘ = t * ( r**λ / _E * (_c3*fp(Θv) + _c4*fppp(Θv)) ) / _c5

    values = np.zeros((tdim, x.shape[1]))
    values[0] = ur * np.cos(Θv) - uΘ * np.sin(Θv)
    values[1] = ur * np.sin(Θv) + uΘ * np.cos(Θv)
    return values

uD.interpolate(_local_notch_asymptotic)

with XDMFFile(comm, f"{prefix}/{_nameExp}.xdmf", "a", encoding=XDMFFile.Encoding.HDF5) as file:
    file.write_function(uD, 0)

from utils.viz import (
    plot_mesh,
    plot_profile,
    plot_scalar,
    plot_vector
)
import pyvista
from pyvista.utilities import xvfb

xvfb.start_xvfb(wait=0.05)
pyvista.OFF_SCREEN = True

plotter = pyvista.Plotter(
    title="Test Viz M1 Asymptotic Displacement",
    window_size=[600, 600],
    shape=(1, 1),
)

plt.figure()
ax = plot_mesh(mesh)
fig = ax.get_figure()
fig.savefig(f"{prefix}/test_mesh.png")

_plt = plot_vector(uD, plotter, scale=.1)
logging.critical('plotted vector')
_plt.screenshot(f"{prefix}/test_vector.png")

dirichletbc(value=uD, dofs=boundary_dofs)

