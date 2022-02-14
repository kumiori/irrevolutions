import sys
sys.path.append('../')
import numpy as np

import meshes
from meshes import primitives

# visualisation
from utils import viz
import matplotlib.pyplot as plt
from utils.viz import plot_mesh
import yaml

from mpi4py import MPI

import petsc4py
from petsc4py import PETSc

import dolfinx
import dolfinx.plot
from dolfinx import log
import ufl


from dolfinx.io import XDMFFile

import logging

logging.basicConfig(level=logging.INFO)

import dolfinx
import dolfinx.plot
import dolfinx.io
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
import matplotlib.pyplot as plt
import pyvista 
from pyvista.utilities import xvfb

# Mesh
Lx = 100
Ly = 400
s=2
L0=30
seedDist=10

geom_type = "bar"


gmsh_model, tdim = primitives.mesh_ep_gmshapi(geom_type,
                                    Lx, 
                                    Ly,
                                    L0, 
                                    s,   
                                    seedDist, 
                                    sep=0.1,
                                    tdim=2)

mesh, mts = meshes.gmsh_model_to_mesh(gmsh_model,
                               cell_data=False,
                               facet_data=True,
                               gdim=2, 
                               exportMesh=True, 
                               fileName="twoCrack_ep2.unv")

plt.figure()
ax = plot_mesh(mesh)
fig = ax.get_figure()
fig.savefig(f"mesh.png")


with open("./test/parameters.yml") as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)


# Part to get boundaries 
boundaries = [(1, lambda x: np.isclose(x[0], 0)),
              (2, lambda x: np.isclose(x[0], Lx)),
              (3, lambda x: np.isclose(x[1], 0)),
              (4, lambda x: np.isclose(x[1], Ly))]

facet_indices, facet_markers = [], []
fdim = mesh.topology.dim - 1
for (marker, locator) in boundaries:
    facets = dolfinx.mesh.locate_entities(mesh, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full(len(facets), marker))
facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = dolfinx.mesh.MeshTags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

# Functional setting

element_u = ufl.VectorElement("Lagrange", mesh.ufl_cell(),
                              degree=1, dim=2)
V_u = dolfinx.fem.FunctionSpace(mesh, element_u) 

u = dolfinx.fem.Function(V_u, name="Displacement")
g = dolfinx.fem.Function(V_u, name="Body pressure")

u_ = dolfinx.fem.Function(V_u, name="Boundary Displacement")

# Integral measure
dx = ufl.Measure("dx", domain=mesh)
ds = ufl.Measure("ds", domain=mesh)
dS = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)
x = ufl.SpatialCoordinate(mesh)

# Data
zero = Function(V_u)
# works in parallel!
with zero.vector.localForm() as loc:
    loc.set(0.0)

one = Function(V_u)
# works in parallel!
with one.vector.localForm() as loc:
    loc.set(1.0)

g = Function(V_u)
# works in parallel!
with g.vector.localForm() as loc:
    loc.set(1.0)

# boundary conditions

def left(x):
  return np.isclose(x[0], 0.)

def right(x):
  return np.isclose(x[0], Lx)

def bottom(x):
  return np.isclose(x[1], 0.)

def top(x):
  return np.isclose(x[1], Ly)

# left side

left_facets = dolfinx.mesh.locate_entities_boundary(mesh, 1, left)
left_dofs = dolfinx.fem.locate_dofs_topological(V_u, mesh.topology.dim - 1,
                                                left_facets)


# right side

right_facets = dolfinx.mesh.locate_entities_boundary(mesh, 1, right)
right_dofs = dolfinx.fem.locate_dofs_topological(V_u, mesh.topology.dim - 1,
                                                right_facets)


top_facets = dolfinx.mesh.locate_entities_boundary(mesh, 1, top)
top_dofs = dolfinx.fem.locate_dofs_topological(V_u, mesh.topology.dim - 1,
                                                top_facets)

bottom_facets = dolfinx.mesh.locate_entities_boundary(mesh, 1, bottom)
bottom_dofs = dolfinx.fem.locate_dofs_topological(V_u, mesh.topology.dim - 1,
                                                bottom_facets)

# energy
mu = parameters["model"]["mu"]
lmbda = parameters["model"]["lmbda"]

def _e(u):
  return ufl.sym(ufl.grad(u))

en_density = 1/2 * (2*mu* ufl.inner(_e(u),_e(u))) + lmbda*ufl.tr(_e(u))**2
energy = en_density * dx + ufl.inner(u, g)*dS(4)

#bcs = [dirichletbc(zero, bottom_dofs), dirichletbc(one, top_dofs)]
bcs = [dirichletbc(zero, bottom_dofs)]

# solving
from solvers import SNESSolver
D_energy_u = ufl.derivative(energy, u, ufl.TestFunction(V_u))

problem = SNESSolver(
    D_energy_u,
    u,
    bcs,
    bounds=None,
    petsc_options=parameters.get("solvers").get("snes"),
    prefix="elast",
)


uh = problem.solve()