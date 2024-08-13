# library include
import sys

sys.path.append("../")
from algorithms import am
import algorithms
import pyvista
from utils.viz import plot_mesh, plot_vector, plot_scalar
from models import DamageElasticityModel as Brittle
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
import gmsh
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
parameters = {
    "loading": {"min": 0.0, "max": 1.0, "steps": 10},
    "geometry": {
        "geom_type": "beleza",
    },
    "model": {"tdim": 2, "E": 1, "nu": 0.3, "w1": 1.0, "ell": 0.1, "k_res": 1.0e-8},
    "solvers": {
        "elasticity": {
            "snes": {
                "snes_type": "newtontr",
                "snes_stol": 1e-8,
                "snes_atol": 1e-8,
                "snes_rtol": 1e-8,
                "snes_max_it": 250,
                "snes_monitor": "",
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
        },
        "damage": {
            "snes": {
                "snes_type": "vinewtonrsls",
                "snes_stol": 1e-5,
                "snes_atol": 1e-5,
                "snes_rtol": 1e-8,
                "snes_max_it": 100,
                "snes_monitor": "",
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        },
        "damage_elasticity": {
            "max_it": 2000,
            "alpha_rtol": 1.0e-4,
            "criterion": "alpha_H1",
        },
    },
}


def mesh_V(
    a,
    h,
    L,
    gamma,
    lc,
    key=0,
    show=False,
    filename="mesh.unv",
    order=1,
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
    filename = name and format of the output file for key = 1
    order = order of the function of form
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    hopen = a * np.tan((gamma / 2.0) * np.pi / 180)
    c0 = h / 40
    load_len = min(h / 40, L / 80)
    tdim = 2

    model = gmsh.model()
    model.add("TPB")
    model.setCurrent("TPB")
    # Generating the points of the geometrie
    p0 = model.geo.addPoint(0.0, a, 0.0, lc, tag=0)
    p1 = model.geo.addPoint(hopen, 0.0, 0.0, lc, tag=1)
    p2 = model.geo.addPoint(L / 2, 0.0, 0.0, lc, tag=2)
    p3 = model.geo.addPoint(L / 2, h, 0.0, lc, tag=3)
    p4 = model.geo.addPoint(0.0, h, 0.0, lc, tag=4)
    p5 = model.geo.addPoint(-L / 2, h, 0.0, lc, tag=5)
    p6 = model.geo.addPoint(-L / 2, 0.0, 0.0, lc, tag=6)
    p7 = model.geo.addPoint(-hopen, 0.0, 0.0, lc, tag=7)
    # Load facet
    p21 = model.geo.addPoint(load_len, h, 0.0, lc, tag=30)
    p22 = model.geo.addPoint(-load_len, h, 0.0, lc, tag=31)
    # Creating the lines by connecting the points
    notch_right = model.geo.addLine(p0, p1, tag=8)
    bot_right = model.geo.addLine(p1, p2, tag=9)
    right = model.geo.addLine(p2, p3, tag=10)
    # top_right = model.geo.addLine(p3, p4, tag=11)
    top_right = model.geo.addLine(p3, p21, tag=11)
    top_left = model.geo.addLine(p22, p5, tag=12)
    left = model.geo.addLine(p5, p6, tag=13)
    bot_left = model.geo.addLine(p6, p7, tag=14)
    notch_left = model.geo.addLine(p7, p0, tag=15)
    # Load facet
    load_right = model.geo.addLine(p21, p4, tag=32)
    load_left = model.geo.addLine(p4, p22, tag=33)

    # Creating the surface using the lines created
    perimeter = model.geo.addCurveLoop(
        [
            notch_right,
            bot_right,
            right,
            top_right,
            load_right,
            load_left,
            top_left,
            left,
            bot_left,
            notch_left,
        ]
    )
    surface = model.geo.addPlaneSurface([perimeter])
    # model.geo.addSurfaceLoop([surface,16])
    model.mesh.setOrder(order)

    # Creating Physical Groups to extract data from the geometrie
    gmsh.model.addPhysicalGroup(tdim - 1, [left], tag=101)
    gmsh.model.setPhysicalName(tdim - 1, 101, "Left")

    gmsh.model.addPhysicalGroup(tdim - 1, [right], tag=102)
    gmsh.model.setPhysicalName(tdim - 1, 102, "Right")

    gmsh.model.addPhysicalGroup(tdim - 2, [p6], tag=103)
    gmsh.model.setPhysicalName(tdim - 2, 103, "Left_point")

    gmsh.model.addPhysicalGroup(tdim - 2, [p2], tag=104)
    gmsh.model.setPhysicalName(tdim - 2, 104, "Right_point")

    gmsh.model.addPhysicalGroup(tdim - 2, [p4], tag=105)
    gmsh.model.setPhysicalName(tdim - 2, 105, "Load_point")

    gmsh.model.addPhysicalGroup(tdim - 2, [p0], tag=106)
    gmsh.model.setPhysicalName(tdim - 2, 106, "Notch_point")

    gmsh.model.addPhysicalGroup(tdim - 1, [load_right], tag=107)
    gmsh.model.setPhysicalName(tdim - 1, 107, "load_right")

    gmsh.model.addPhysicalGroup(tdim - 1, [load_left], tag=108)
    gmsh.model.setPhysicalName(tdim - 1, 108, "load_left")

    gmsh.model.addPhysicalGroup(tdim, [surface], tag=110)
    gmsh.model.setPhysicalName(tdim, 110, "mesh_surface")

    # Cast3M can't read Physical Groups of points (dim = 0). Instead, we check the number in the mesh and input in manually in the code.
    # The number of a node doesn't change if it's in a point of the geometry

    model.geo.synchronize()
    model.mesh.generate(tdim)
    if show:
        gmsh.fltk.run()
    if key == 1:
        gmsh.write(filename)
    return gmsh.model


geo_parameters = {
    "a": 0.15,
    "h": 0.5,
    "L": 1,
    "gamma": 90,
    "lc": 0.1,
}

import pdb

parameters.get("geometry").update(geo_parameters)
gmsh_model = mesh_V(**geo_parameters)

mesh, facet_tags = meshes.gmsh_model_to_mesh(
    gmsh_model, cell_data=False, facet_data=True, gdim=2
)

plt.figure()
ax = plot_mesh(mesh)
fig = ax.get_figure()
fig.savefig(f"output/Vnotch_mesh.png")

# pdb.set_trace()
# Functional setting


element_u = ufl.VectorElement("Lagrange", mesh.ufl_cell(), degree=1, dim=2)
element_alpha = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)
V_u = dolfinx.fem.FunctionSpace(mesh, element_u)
V_alpha = dolfinx.fem.FunctionSpace(mesh, element_alpha)

u = dolfinx.fem.Function(V_u, name="Displacement")
# the displacement
u_ = dolfinx.fem.Function(V_u, name="Boundary_Displacement")
u_corner = dolfinx.fem.Function(V_u, name="Corner_Displacement")
alpha = dolfinx.fem.Function(V_alpha, name="Damage")
# Bounds -> the values of alpha must be max([0,1],[alpha(t-1),1])
force = dolfinx.fem.Function(V_u, name="Contact_force")

alpha_ub = dolfinx.fem.Function(V_alpha, name="UpperBoundDamage")
alpha_lb = dolfinx.fem.Function(V_alpha, name="LowerBoundDamage")
alpha_lb.interpolate(lambda x: np.zeros_like(x[0]))
alpha_ub.interpolate(lambda x: np.ones_like(x[0]))

dx = ufl.Measure("dx", domain=mesh)  # -> volume measure
# We include here the subdomain data generated at the gmsh file.
ds = ufl.Measure("ds", subdomain_data=facet_tags, domain=mesh)

model = Brittle(parameters.get("model"))
state = {"u": u, "alpha": alpha}

total_energy = model.total_energy_density(state) * dx

force.interpolate(
    lambda x: (np.zeros_like(x[0]), parameters["loading"]["max"] * np.ones_like(x[1]))
)

u_corner.interpolate(lambda x: (np.zeros_like(x[0]), np.zeros_like(x[1])))

for u in (u_corner,):
    u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

# total_energy = model.total_energy_density(
#     state) * dx - ufl.dot(force, u)*ds(107) - ufl.dot(force, u)*ds(108)
_h = parameters.get("geometry").get("h")
_L = parameters.get("geometry").get("L")


def _small_set(x):
    _eta = 1e-2
    _lower_bound = -_eta
    _upper_bound = +_eta
    return np.logical_and(
        np.isclose(x[1], _h),
        np.logical_and(
            np.greater_equal(x[0], _lower_bound), np.less_equal(x[0], _upper_bound)
        ),
    )


def _corners(x):
    return np.logical_and(
        np.logical_or(np.isclose(x[0], -_L / 2), np.isclose(x[0], _L / 2)),
        np.isclose(x[1], 0),
    )


# _smallset_entities = dolfinx.mesh.locate_entities_boundary(mesh, 0, _small_set)
# _smallset_dofs = dolfinx.fem.locate_dofs_topological(
#     V_u, mesh.topology.dim - 1, _smallset_entities)

# _corners_entities = dolfinx.mesh.locate_entities_boundary(
#     mesh,
#     mesh.topology.dim - 1,
#     _corners)
# _corners_dofs = dolfinx.fem.locate_dofs_topological(
#     V_u, mesh.topology.dim - 1, _corners_entities)

dofs_u_smallset = locate_dofs_geometrical(V_u, _small_set)
dofs_u_corners = locate_dofs_geometrical(V_u, _corners)

# Bcs
bcs_alpha = []
bcs_u = [dirichletbc(u_, dofs_u_smallset), dirichletbc(u_corner, dofs_u_corners)]

bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}
# Update the bounds
set_bc(alpha_ub.vector, bcs_alpha)
set_bc(alpha_lb.vector, bcs_alpha)

model = Brittle(parameters["model"])

total_energy = model.total_energy_density(state) * dx

solver = am.AlternateMinimisation(
    total_energy, state, bcs, parameters.get("solvers"), bounds=(alpha_lb, alpha_ub)
)
# visualisation
loads = np.linspace(
    parameters.get("loading").get("min"),
    parameters.get("loading").get("max"),
    parameters.get("loading").get("steps"),
)

data = {"elastic": [], "surface": [], "total": [], "load": []}
xvfb.start_xvfb(wait=0.05)
pyvista.OFF_SCREEN = True
plotter = pyvista.Plotter(
    title="Displacement",
    window_size=[1600, 600],
    shape=(1, 2),
)

for i_t, t in enumerate(loads):
    # update boundary conditions

    u_.interpolate(lambda x: (np.zeros_like(x[0]), t * np.ones_like(x[1])))
    u_.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # update lower bound for damage
    alpha.vector.copy(alpha_lb.vector)
    alpha.vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )

    # solve for current load step
    print(f"Solving timestep {i_t}, load: {t}")
    solver.solve()

    # global postprocessing
    surface_energy = assemble_scalar(
        dolfinx.fem.form(model.damage_energy_density(state) * dx)
    )

    elastic_energy = assemble_scalar(
        dolfinx.fem.form(model.elastic_energy_density(state) * dx)
    )

    data.get("elastic").append(elastic_energy)
    data.get("surface").append(surface_energy)
    data.get("total").append(surface_energy + elastic_energy)
    data.get("load").append(t)

    print(f"Solved timestep {i_t}, load: {t}")
    print(f"Elastic Energy {elastic_energy:.3g}, Surface energy: {surface_energy:.3g}")
    print("\n\n")

plotter = pyvista.Plotter(
    title="Displacement",
    window_size=[1600, 600],
    shape=(1, 2),
)

_plt = plot_scalar(alpha, plotter, subplot=(0, 0))
_plt = plot_vector(u, plotter, subplot=(0, 1))
_plt.screenshot(f"./output/vnotch_fields.png")
