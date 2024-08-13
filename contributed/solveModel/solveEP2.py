# library include


import pyvista
from solvers import SNESSolver
from utils.viz import plot_mesh, plot_vector, plot_scalar
from irrevolutions.utils import viz
from meshes import primitives
import meshes
from pyvista.utilities import xvfb
import matplotlib.pyplot as plt
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
import dolfinx.io
import numpy as np
import yaml
import json
import sys
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

logging.basicConfig(level=logging.INFO)


sys.path.append("./")

# meshes

# visualisation


def plot_vector(u, plotter, subplot=None):
    if subplot:
        plotter.subplot(subplot[0], subplot[1])
    V = u.function_space
    mesh = V.mesh
    topology, cell_types = dolfinx.plot.create_vtk_topology(mesh, mesh.topology.dim)
    num_dofs_local = u.function_space.dofmap.index_map.size_local
    geometry = u.function_space.tabulate_dof_coordinates()[:num_dofs_local]
    values = np.zeros((V.dofmap.index_map.size_local, 3), dtype=np.float64)
    values[:, : mesh.geometry.dim] = u.vector.array.real.reshape(
        V.dofmap.index_map.size_local, V.dofmap.index_map_bs
    )
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    grid["vectors"] = values
    grid.set_active_vectors("vectors")
    # geom = pyvista.Arrow()
    # glyphs = grid.glyph(orient="vectors", factor=1, geom=geom)
    glyphs = grid.glyph(orient="vectors", factor=1.0)
    plotter.add_mesh(glyphs)
    plotter.add_mesh(
        grid, show_edges=True, color="black", style="wireframe", opacity=0.3
    )
    plotter.view_xy()
    return plotter


def plot_scalar(alpha, plotter, subplot=None, lineproperties={}):
    if subplot:
        plotter.subplot(subplot[0], subplot[1])
    V = alpha.function_space
    mesh = V.mesh
    topology, cell_types, _ = dolfinx.plot.create_vtk_mesh(mesh, mesh.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, mesh.geometry.x)

    plotter.subplot(0, 0)
    grid.point_data["alpha"] = alpha.compute_point_values().real
    grid.set_active_scalars("alpha")
    plotter.add_mesh(grid, **lineproperties)
    plotter.view_xy()
    return plotter


# Parameters


parameters = {
    "loading": {"min": 0, "max": 1},
    "geometry": {"geom_type": "bar", "Lx": 5.0, "Ly": 15},
    "model": {"mu": 1.0, "lmbda": 0.0},
    "solvers": {
        "snes": {
            "snes_type": "newtontr",
            "snes_stol": 1e-8,
            "snes_atol": 1e-8,
            "snes_rtol": 1e-8,
            "snes_max_it": 100,
            "snes_monitor": "",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
    },
}

# parameters.get('loading')
with open("./solveModel/parametersSolve.yml") as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)

# Mesh
Lx = parameters["geometry"]["Lx"]
Ly = parameters["geometry"]["Ly"]
geom_type = parameters["geometry"]["geom_type"]


gmsh_model, tdim = primitives.mesh_ep_gmshapi(geom_type, Lx, Ly, 1, 0.5, 0.3, tdim=2)

mesh, mts = meshes.gmsh_model_to_mesh(
    gmsh_model,
    cell_data=False,
    facet_data=True,
    gdim=2,
    exportMesh=True,
    fileName="epTestMesh.msh",
)

# TODO: Plot mesh


plt.figure()
ax = plot_mesh(mesh)
fig = ax.get_figure()
fig.savefig(f"mesh.png")

boundaries = [
    (1, lambda x: np.isclose(x[0], 0)),
    (2, lambda x: np.isclose(x[0], Lx)),
    (3, lambda x: np.isclose(x[1], 0)),
    (4, lambda x: np.isclose(x[1], Ly)),
]

facet_indices, facet_markers = [], []
fdim = mesh.topology.dim - 1
for marker, locator in boundaries:
    facets = dolfinx.mesh.locate_entities(mesh, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full(len(facets), marker))
facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = dolfinx.mesh.MeshTags(
    mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets]
)

# Functional setting

element_u = ufl.VectorElement("Lagrange", mesh.ufl_cell(), degree=1, dim=2)
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
"""
with g.vector.localForm() as loc:
    loc.set(1.0/100.)
"""

# x = ufl.SpatialCoordinate(mesh)
# g = dolfinx.Expression ('4 *x[1]')

# boundary conditions
g.interpolate(lambda x: (np.zeros_like(x[0]), np.ones_like(x[1])))
g.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


def left(x):
    return np.isclose(x[0], 0.0)


def right(x):
    return np.isclose(x[0], Lx)


def bottom(x):
    return np.isclose(x[1], 0.0)


def top(x):
    return np.isclose(x[1], Ly)


# left side


left_facets = dolfinx.mesh.locate_entities_boundary(mesh, 1, left)
left_dofs = dolfinx.fem.locate_dofs_topological(V_u, mesh.topology.dim - 1, left_facets)


# right side

right_facets = dolfinx.mesh.locate_entities_boundary(mesh, 1, right)
right_dofs = dolfinx.fem.locate_dofs_topological(
    V_u, mesh.topology.dim - 1, right_facets
)


top_facets = dolfinx.mesh.locate_entities_boundary(mesh, 1, top)
top_dofs = dolfinx.fem.locate_dofs_topological(V_u, mesh.topology.dim - 1, top_facets)

bottom_facets = dolfinx.mesh.locate_entities_boundary(mesh, 1, bottom)
bottom_dofs = dolfinx.fem.locate_dofs_topological(
    V_u, mesh.topology.dim - 1, bottom_facets
)

# energy
mu = parameters["model"]["mu"]
lmbda = parameters["model"]["lmbda"]


def _e(u):
    return ufl.sym(ufl.grad(u))


en_density = 1 / 2 * (2 * mu * ufl.inner(_e(u), _e(u))) + lmbda * ufl.tr(_e(u)) ** 2
energy = en_density * dx - ufl.inner(u, g) * dS(4)

# bcs = [dirichletbc(zero, bottom_dofs), dirichletbc(one, top_dofs)]
bcs = [dirichletbc(zero, bottom_dofs)]

# solving
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
print(u)

# plt.figure()
# ax = plot_mesh(mesh)
# fig = ax.get_figure()
# fig.savefig(f"mesh.png")

# postprocessing

xvfb.start_xvfb(wait=0.05)
pyvista.OFF_SCREEN = True

plotter = pyvista.Plotter(
    title="Displacement",
    window_size=[1600, 600],
    shape=(1, 2),
)

# _plt = plot_scalar(u_.sub(0), plotter, subplot=(0, 0))
_plt = plot_vector(u, plotter, subplot=(0, 1))
_plt.screenshot(f"displacement_MPI.png")
