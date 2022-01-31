#Numpy -> numerical library for Python. We'll use it for all array operations.
#It's written in C and it's faster (than traditional Python)
import numpy as np

#Yaml (Yet another markup language) -> We'll use it to pass, read and structure
#light text data in .yml files.
import yaml

#Json -> Another form to work with data. It comes from JavaScript. Similar functions
#that Yaml. Used speacily with API request, when we need data "fetch".
import json

#Communication with the machine:
#Sys -> allows to acess the system and launch commandes.
#Os - > allows to acess the operation system.
import sys
import os
from pathlib import Path

#Mpi4py -> Interface that allows parallel interoperability. MPI stands for' Message
#Passager Interface' and will be used to communicate computer nodes when lauching code
#in a parallel way

from mpi4py import MPI
#Petcs4py -> we use this library to handle with the data. Given acesses to solvers
import petsc4py
from petsc4py import PETSc

#Dolfinx
import dolfinx
import dolfinx.plot
from dolfinx import log
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

#UFL (Unified Format Language) -> we'll be used to represent abstract way to 
#represent the language in a quadratic form
import ufl

#XDMFF -> format used for the output binary data
from dolfinx.io import XDMFFile

#Install 'gmsh' library -> we'll be used for the mesh.
#!{sys.executable}: to use the current kernel to make the installation 
import gmsh

import matplotlib.pyplot as plt

# meshes
import meshes
from meshes import primitives

# visualisation
from utils import viz
import matplotlib.pyplot as plt
from utils.viz import plot_mesh, plot_vector, plot_scalar

###################################################################


parameters = {
    #In case of evolution (nonlinear) problems, it's necessary to define a max
    #and a min. For the elastic solution, just one value in needed.
    'loading': {
        'min': 0,
        'max': 1
    },
    'geometry': {
        'geom_type': 'bar',
        'Lx': 1.,
        'Ly': 0.1
    },
    'material': {
        'E': 2e11,
        'poisson': 0.3
    },
    'solvers': {
        'snes': {
            'snes_type': 'newtontr',
            'snes_stol': 1e-8,
            'snes_atol': 1e-8,
            'snes_rtol': 1e-8,
            'snes_max_it': 100,
            'snes_monitor': "",
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps'
        }
    }
}

###################################################################

def mesh(
a,
h,
L,
gamma,
de,
de2,
key=0,
show=False,
):
    """
    Create a 2D mesh of a notched three-point flexure specimen using GMSH.
    a = height of the notch
    h = height of the specimen
    L = width of the specimen
    gamma = notch angle
    de = density of elements at specimen
    de2 = density of elements at the notch and crack
    key = 0 -> create model for Fenicxs (default)
          1 -> create model for Cast3M
    show = False -> doesn't open Gmsh to vizualise the mesh (default)
           True -> open Gmsh to vizualise the mesh
    """
    gmsh.initialize()
    #gmsh.option.setNumber("General.Terminal",1)
    #gmsh.option.setNumber("Mesh.Algorithm",5)
    model = gmsh.model()
    hopen = a*np.tan((gamma/2.0)*np.pi/180)
    c0 = h/10
    tdim = 2 
    #Generating the points of the geometrie
    p0 = model.geo.addPoint(0.0, a, 0.0, de2)
    p1 = model.geo.addPoint(hopen, 0.0, 0.0, de)
    p2 = model.geo.addPoint(L/2, 0.0, 0.0, de)
    p3 = model.geo.addPoint(L/2, h, 0.0, de)
    p4 = model.geo.addPoint(0.0, h, 0.0, de)
    if key == 0:
        p5 = model.geo.addPoint(-L/2, h, 0.0, de)
        p6 = model.geo.addPoint(-L/2, 0.0, 0.0, de)
        p7 = model.geo.addPoint(-hopen, 0.0, 0.0, de)
    elif key == 1:
        p20 = model.geo.addPoint(0, a+c0, 0, de2)
    #Creating the lines by connecting the points
    notch_right = model.geo.addLine(p0, p1) 
    bot_right = model.geo.addLine(p1, p2)
    right = model.geo.addLine(p2, p3)
    top_right = model.geo.addLine(p3, p4)
    if key == 0:
        top_left = model.geo.addLine(p4, p5)
        left = model.geo.addLine(p5, p6)
        bot_left = model.geo.addLine(p6, p7)
        notch_left = model.geo.addLine(p7, p0)
    elif key == 1:
        sym_plan = model.geo.addLine(p4, p20)
        fissure = model.geo.addLine(p20, p0)
    #Creating the surface using the lines created
    if key == 0:
        perimeter = model.geo.addCurveLoop([notch_right, bot_right, right, top_right, top_left, left, bot_left, notch_left])
    elif key == 1:
        perimeter = model.geo.addCurveLoop([notch_right, bot_right, right, top_right, sym_plan, fissure])
    surface = model.geo.addPlaneSurface([perimeter])
    model.geo.addSurfaceLoop([surface,16])
    
    model.geo.synchronize()

    #Creating Physical Groups to extract data from the geometrie
    if key == 0:
        gmsh.model.addPhysicalGroup(tdim-1, [left], tag = 101)
        gmsh.model.setPhysicalName(tdim-1, 101,'Left')

        gmsh.model.addPhysicalGroup(tdim-1, [right], tag=102)
        gmsh.model.setPhysicalName(tdim-1, 102,'Right')

        gmsh.model.addPhysicalGroup(tdim-2, [p2], tag=103)
        gmsh.model.setPhysicalName(tdim-2, 103,'Left_point')

        gmsh.model.addPhysicalGroup(tdim-2, [p6], tag=104)
        gmsh.model.setPhysicalName(tdim-2, 104,'Right_point')

        gmsh.model.addPhysicalGroup(tdim-2, [p4], tag=105)
        gmsh.model.setPhysicalName(tdim-2, 105, 'Load_point')

        gmsh.model.addPhysicalGroup(tdim, [surface],tag=110)
        gmsh.model.setPhysicalName(tdim, 110, 'mesh_surface')

    #Generating the mesh
    model.mesh.generate(tdim)
    gmsh.write('mesh.msh')
    if show:
        gmsh.fltk.run()
    return gmsh.model

a=0.00533
h=0.0178
L=0.0762
gamma = 90
de = a/5
de2 = a/10
gmsh_model = mesh(a, h, L, gamma, de, de2)
mesh = meshes.gmsh_model_to_mesh(gmsh_model, gdim=2)

#Plot mesh
plt.figure()
ax = plot_mesh(mesh)
fig = ax.get_figure()
fig.savefig(f"mesh.png")

###################################################################

element_u = ufl.VectorElement("Lagrange", mesh.ufl_cell(),
                              degree=1, dim=2)

V_u = dolfinx.fem.FunctionSpace(mesh, element_u) 

u = dolfinx.fem.Function(V_u, name="Displacement")

g = dolfinx.fem.Function(V_u, name="Body pressure")
with g.vector.localForm() as loc:
  loc.set(0.0)

u_ = dolfinx.fem.Function(V_u, name="Boundary Displacement")

zero = Function(V_u)
# works in parallel!
with zero.vector.localForm() as loc:
    loc.set(0.0)

one = Function(V_u)
# works in parallel!
with one.vector.localForm() as loc:
    loc.set(1.0)

dx = ufl.Measure("dx", domain=mesh) #-> volume measure
ds = ufl.Measure("ds", domain=mesh) #-> surface measure

E = parameters["material"]["E"]
poisson = parameters["material"]["poisson"]
mu = E/(2*(1+poisson))
lmbda = E*poisson/((1+poisson)*(1-2*poisson)) #carefull, lambda is a reserved word in Py

#Defining a function that returns a way to calculate the deformation, which is 
#symmetric part of the gradient of the displacement.
def _e(u):
  return ufl.sym(ufl.grad(u))

en_density = 1/2 * (2*mu* ufl.inner(_e(u),_e(u))) + lmbda*ufl.tr(_e(u))**2
energy = en_density * dx - ufl.dot(g, u) * dx

###################################################################

# boundary conditions

def left(x):
  return np.isclose(x[0], -L/2)

def right(x):
  return np.isclose(x[0], L/2)

left_facets = dolfinx.mesh.locate_entities_boundary(mesh, 0, left)
left_dofs = dolfinx.fem.locate_dofs_topological(V_u, mesh.topology.dim - 1,
                                                left_facets)

right_facets = dolfinx.mesh.locate_entities_boundary(mesh, 0, right)
right_dofs = dolfinx.fem.locate_dofs_topological(V_u, mesh.topology.dim - 1,
                                                right_facets)

bcs = [dirichletbc(zero, left_dofs), dirichletbc(zero, right_dofs)]


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


problem.solve()

def plot_vector(u, plotter, subplot=None):
    if subplot:
        plotter.subplot(subplot[0], subplot[1])
    V = u.function_space
    mesh = V.mesh
    topology, cell_types, _ = dolfinx.plot.create_vtk_mesh(mesh, mesh.topology.dim)
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



# plt.figure()
# ax = plot_mesh(mesh)
# fig = ax.get_figure()
# fig.savefig(f"mesh.png")

# postprocessing
plotter = pyvista.Plotter(
        title="Displacement",
        window_size=[1600, 600],
        shape=(1, 2),
    )

# _plt = plot_scalar(u_.sub(0), plotter, subplot=(0, 0))
_plt = plot_vector(u, plotter, subplot=(0, 1))
_plt.screenshot(f"displacement_MPI.png")