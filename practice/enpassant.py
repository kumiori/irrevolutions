# library include
from models import DamageElasticityModel as Brittle
import sys
import pyvista
sys.path.append('./')

from utils.viz import plot_mesh, plot_vector, plot_scalar
from meshes import primitives

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
from pyvista.utilities import xvfb
import matplotlib.pyplot as plt
import dolfinx.io
import numpy as np
import yaml
import json

import os
from pathlib import Path

from mpi4py import MPI

import petsc4py
from petsc4py import PETSc

import dolfinx
import dolfinx.plot
from dolfinx import log
import ufl


from dolfinx.io import XDMFFile

import logging



from utils import viz
import meshes
logging.basicConfig(level=logging.INFO)

with open("./parameters.yml") as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)

# meshes

# Mesh
Lx = parameters["geometry"]["Lx"]
Ly = parameters["geometry"]["Ly"]
geom_type = parameters["geometry"]["geom_type"]


gmsh_model, tdim = meshes.enpassant.mesh_ep_gmshapi(geom_type,
                                              Lx,
                                              Ly,
                                              1,
                                              0.5,
                                              0.3,
                                              tdim=2)

mesh, mts = meshes.gmsh_model_to_mesh(gmsh_model,
                                      cell_data=False,
                                      facet_data=True,
                                      gdim=2,
                                      exportMesh=True,
                                      fileName="epTestMesh.msh")
# visualisation


plt.figure()
ax = plot_mesh(mesh)
fig = ax.get_figure()
fig.savefig(f"mesh.png")


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
facet_tag = dolfinx.mesh.MeshTags(
    mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])


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


# boundary conditions
g.interpolate(lambda x: (np.zeros_like(x[0]), np.ones_like(x[1])))
g.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                     mode=PETSc.ScatterMode.FORWARD)


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


model = Brittle(parameters.get('model'))
state = {'u': u, 'alpha': alpha}
