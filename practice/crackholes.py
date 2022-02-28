
import hashlib
import matplotlib.pyplot as plt
import sys

sys.path.append("../")
##!/usr/bin/env python3

import pyvista

import ufl
from petsc4py import PETSc
import petsc4py
from mpi4py import MPI
import os
from pathlib import Path
import json
import yaml
import numpy as np
# import pandas as pd

import dolfinx
from dolfinx import log
import dolfinx.plot
from dolfinx.io import XDMFFile
from dolfinx.common import list_timings
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
import dolfinx.mesh
from dolfinx.mesh import CellType

import logging

from solvers import SNESSolver
from models import DamageElasticityModel as Brittle
from utils.viz import plot_mesh, plot_vector, plot_scalar
from pyvista.utilities import xvfb
from meshes import gmsh_model_to_mesh
import algorithms
from algorithms import am
# from meshes.crackholes import mesh_crackholes as mesh_function


# from damage.utils import ColorPrint


logging.basicConfig(level=logging.INFO)


sys.path.append("../")


petsc4py.init(sys.argv)
log.set_log_level(log.LogLevel.WARNING)


comm = MPI.COMM_WORLD

# Parameters
parameters = {
    'loading': {
        'min': 0.,
        'max': 3.,
        'steps': 10
    },
    'geometry': {
        'geom_type': 'bar',
        'Lx': 1.0,
        'Ly': 0.5,
        'rhoc': 0.05,
        'deltac': 0.2,
    },
    'model': {
        'tdim': 2,
        'E': 1,
        'nu': .0,
        'w1': 1.,
        'ell': 0.1,
        'k_res': 1.e-8
    },
    'solvers': {
        'elasticity': {
            'snes': {
                'snes_type': 'newtontr',
                'snes_stol': 1e-8,
                'snes_atol': 1e-8,
                'snes_rtol': 1e-8,
                'snes_max_it': 250,
                'snes_monitor': "",
                'ksp_type': 'preonly',
                'pc_type': 'lu',
                'pc_factor_mat_solver_type': 'mumps'
            }
        },
        'damage': {
            'snes': {
                'snes_type': 'vinewtonrsls',
                'snes_stol': 1e-5,
                'snes_atol': 1e-5,
                'snes_rtol': 1e-8,
                'snes_max_it': 100,
                'snes_monitor': "",
                'ksp_type': 'preonly',
                'pc_type': 'lu',
                'pc_factor_mat_solver_type': 'mumps'
            },
        },
        'damage_elasticity': {
            "max_it": 2000,
            "alpha_rtol": 1.0e-4,
            "criterion": "alpha_H1"
        }
    }
}


# Get mesh parameters
Lx = parameters["geometry"]["Lx"]
Ly = parameters["geometry"]["Ly"]
ell_ = parameters["model"]["ell"]
lc = ell_ / 3.0

geo_parameters = {
    'geom_type': 'crackhole',
    "Lx": 1.,
    "Ly": .5,
    "l0":  .6,
    "a":  .5,
    "b":  .2,
    "lc":  .05,
    "xc": .1,
    "c": 0.275,
    "l1": 0.001,
    "d": 0.35,
    "e": 0.40,
    "f": 0.25,
    "g": 0.225,
    "deltac": .1,
    "rhoc": .05,
    "offset": 0,
    "tdim":  2,
    "order":  0
}

# update default parameters
parameters.get("geometry").update(geo_parameters)

# Get geometry model
geom_type = parameters["geometry"]["geom_type"]

# Create the mesh of the specimen with given dimensions
Lx = parameters.get("geometry").get("Lx")
Ly = parameters.get("geometry").get("Ly")
a = parameters.get("geometry").get("a")
l0 = parameters.get("geometry").get("l0")
#b=parameters.get("geometry").get("b")
geom_type = parameters.get("geometry").get("geom_type")
tdim = parameters.get("model").get("tdim")


# ---------------------

import pdb

def mesh_crackholes(geom_type,
                     Lx,
                     Ly,
                     a,
                     b,
                     ax,
                     bx,
                     lc,
                     xc,
                     c,
                     l1,
                     d,
                     e,
                     f,
                     g,
                     deltac,
                     rhoc,
                     offset=0,
                     tdim=2,
                     order=1,
                     msh_file=None,
                     comm=MPI.COMM_WORLD):
    """
    Create mesh of 2d tensile test specimen -
         Lx: 
         Ly: 
         a: centre ellipse
         b: centre ellipse
         ax: horiz axis ellipse 
         bx: vert axis ellipse
         lc: 
         xc:
         c:
         l1:
         d:
         e:
         f:
         g:
         deltac: 
         rhoc: 
         offset (defaults 0): offset between pins 
         refine ratio:
    """
    # Perform Gmsh work only on rank = 0

    if comm.rank == 0:

        import gmsh

        # Initialise gmsh and set options
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)

        gmsh.option.setNumber("Mesh.Algorithm", 6)
        model = gmsh.model()
        model.add("Rectangle")
        model.setCurrent("Rectangle")
         # points = [p1, p2, p3, p4, p5, p6, p7, p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,p26]
        p1 = model.geo.addPoint(0.0, 0.0, 0, lc, tag=1)
        p2 = model.geo.addPoint(Lx, 0.0, 0, lc, tag=2)
        p3 = model.geo.addPoint(Lx, Ly, 0.0, lc, tag=3)
        p4 = model.geo.addPoint(0, Ly, 0, lc, tag=4)
        p5 = model.geo.addPoint(0, c, 0.0, lc, tag=5)
        p6 = model.geo.addPoint(Lx/4, c, 0, lc/2, tag=6)
        p7 = model.geo.addPoint(d, (Ly/2)+l1, 0.0, lc/3, tag=7)
        p8 = model.geo.addPoint(e,f, 0, lc/3 , tag=8)
        p9 = model.geo.addPoint(d, (Ly/2)-l1, 0, lc/3, tag=9)
        p10 = model.geo.addPoint(Lx/4,g, 0, lc/2, tag=10)
        p11 = model.geo.addPoint(0,g, 0, lc, tag=11)

        # xc, deltac, rhoc, offset=0

        p12 = model.geo.addPoint(xc-offset, Ly/2+deltac, 0, lc, tag=12) #cercle1
        p13 = model.geo.addPoint(xc-offset-rhoc, Ly/2+deltac, 0, lc, tag=13)
        p14 = model.geo.addPoint(xc-offset, Ly/2+deltac+rhoc, 0, lc, tag=14)
        p15 = model.geo.addPoint(xc-offset+rhoc, Ly/2+deltac, 0, lc, tag=15)
        p16 = model.geo.addPoint(xc-offset, Ly/2+deltac-rhoc, 0, lc, tag=16)

        p17 = model.geo.addPoint(xc+offset, Ly/2-deltac, 0, lc, tag=17) #cercle2
        p18 = model.geo.addPoint(xc+offset-rhoc, Ly/2-deltac, 0, lc, tag=18)
        p19 = model.geo.addPoint(xc+offset, Ly/2-deltac+rhoc, 0, lc, tag=19)
        p20 = model.geo.addPoint(xc+offset+rhoc, Ly/2-deltac, 0, lc, tag=20)
        p21 = model.geo.addPoint(xc+offset, Ly/2-deltac-rhoc, 0, lc, tag=21)

        p22 = model.geo.addPoint(a,b, 0, lc, tag=22) #ellipse
        p23 = model.geo.addPoint(a,b+bx, 0, lc/3, tag=23)
        p24 = model.geo.addPoint(a+ax,b, 0, lc/3, tag=24)
        p25 = model.geo.addPoint(a,b-bx, 0, lc/3, tag=25)
        p26 = model.geo.addPoint(a-ax,b, 0, lc/3, tag=26)

                # Lines = [L1, L2, L3, L4, L5, L6, L7, L8]
        bottom = model.geo.addLine(p1, p2, tag=1)
        right= model.geo.addLine(p2, p3, tag=2)
        top = model.geo.addLine(p3, p4, tag=3)
        left1= model.geo.addLine(p4, p5, tag=4)
        halftop= model.geo.addLine(p5, p6, tag=5)
        inclined1= model.geo.addLine(p6, p7, tag=6)
        liptop= model.geo.addLine(p7, p8, tag=7)
        lipbot = model.geo.addLine(p8, p9, tag=8)
        inclined2= model.geo.addLine(p9, p10, tag=9)
        halfbottom= model.geo.addLine(p10, p11, tag=10)
        left2= model.geo.addLine(p11, p1, tag=11)
        cloop1 = model.geo.addCurveLoop([bottom, right, top, left1, halftop,inclined1,liptop,lipbot,inclined2,halfbottom,left2]) 
        c1 = gmsh.model.geo.addCircleArc(p13, p12, p14)
        c2 = gmsh.model.geo.addCircleArc(p14, p12, p15)
        c3 = gmsh.model.geo.addCircleArc(p15, p12, p16)
        c4 = gmsh.model.geo.addCircleArc(p16, p12, p13)
        circle1 = model.geo.addCurveLoop([c1, c2, c3, c4])
        c5 = gmsh.model.geo.addCircleArc(p18, p17, p19)
        c6 = gmsh.model.geo.addCircleArc(p19, p17, p20)
        c7 = gmsh.model.geo.addCircleArc(p20, p17, p21)
        c8 = gmsh.model.geo.addCircleArc(p21, p17, p18)
        circle2 = model.geo.addCurveLoop([c5, c6, c7, c8])
        e1= gmsh.model.geo.addEllipseArc(p26, p22, p24,p25)
        e2= gmsh.model.geo.addEllipseArc(p24, p22, p26,p25)
        e3= gmsh.model.geo.addEllipseArc(p24, p22, p26,p23)
        e4= gmsh.model.geo.addEllipseArc(p26, p22, p24,p23)
        Ellipse = model.geo.addCurveLoop([e1, -e2, e3, -e4])
        
        # surface_1 =
        model.geo.addPlaneSurface([cloop1,circle1,circle2,Ellipse])

        gmsh.model.geo.synchronize()
        surface_entities = [model[1] for model in model.getEntities(tdim)]

        gmsh.model.addPhysicalGroup(tdim, surface_entities, tag=1)
        gmsh.model.setPhysicalName(tdim, 1, "Rectangle surface")

        # Set mesh size via points
        # gmsh.model.mesh.setSize(points, lc)  # heuristic

        # gmsh.model.mesh.optimize("Netgen")

        # Set geometric order of mesh cells
        gmsh.model.mesh.setOrder(order)

        # Define physical groups for subdomains (! target tag > 0)
        # domain = 1
        # gmsh.model.addPhysicalGroup(tdim, [v[1] for v in volumes], domain)
        # gmsh.model.setPhysicalName(tdim, domain, 'domain')
              
        # export pins as physical
        gmsh.model.addPhysicalGroup(tdim - 1, [circle1], tag=99)
        gmsh.model.setPhysicalName(tdim - 1, 99, "topPin")

        gmsh.model.addPhysicalGroup(tdim - 1, [circle2], tag=66)
        gmsh.model.setPhysicalName(tdim - 1, 66, "botPin")
        
        # this is class
        gmsh.model.addPhysicalGroup(tdim - 1, [5], tag=9)
        gmsh.model.setPhysicalName(tdim - 1, 9, "botfissure1")
        gmsh.model.addPhysicalGroup(tdim - 1, [6], tag=10)
        gmsh.model.setPhysicalName(tdim - 1, 10, "botfissure2")
        gmsh.model.addPhysicalGroup(tdim - 1, [3], tag=11)
        gmsh.model.setPhysicalName(tdim - 1, 11, "top")
        gmsh.model.addPhysicalGroup(tdim - 1, [1], tag=12)
        gmsh.model.setPhysicalName(tdim - 1, 12, "bottom")
        gmsh.model.addPhysicalGroup(tdim - 1, [7], tag=13)
        gmsh.model.setPhysicalName(tdim - 1, 13, "topfissure1")
        gmsh.model.addPhysicalGroup(tdim - 1, [8], tag=14)
        gmsh.model.setPhysicalName(tdim - 1, 14, "topfissure2")

        gmsh.model.mesh.generate(tdim)

    return gmsh.model if comm.rank == 0 else None



# --------------------

# "l0":  .6,
geo_parameters = {
    'geom_type': 'crackhole',
    "Lx": 1.,
    "Ly": .5,
    "a":  .6,
    "b":  .2,
    "ax": .06,
    "bx": .01,
    "lc":  .05,
    "xc": .1,
    "c": 0.275,
    "l1": 0.001,
    "d": 0.35,
    "e": 0.40,
    "f": 0.25,
    "g": 0.225,
    "deltac": .1,
    "rhoc": .05,
    "offset": 0,
    "tdim":  2,
    "order":  0
}

gmsh_model = mesh_crackholes(**geo_parameters)

geo_signature = hashlib.md5(str(geo_parameters).encode('utf-8')).hexdigest()

mesh, facets = gmsh_model_to_mesh(
    gmsh_model, cell_data=True, facet_data=False, gdim=tdim)

outdir = "output"
if comm.rank == 0:
    Path(outdir).mkdir(parents=True, exist_ok=True)
prefix = os.path.join(outdir, "crackhole")
# Viz the mesh

Path("mec647/practice").mkdir(parents=True, exist_ok=True)
# Path("mec647/excercise").mkdir(parents=True, exist_ok=True)
plt.figure()
ax = plot_mesh(mesh)
fig = ax.get_figure()
plt.title(f"Crackhole (fixed mesh), dimension {tdim}")
fig.savefig(os.path.join(outdir,f"crackhole-mesh-{geo_signature}.png"))

with XDMFFile(comm, f"{prefix}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as file:
    file.write_mesh(mesh)

# Functional Setting

element_u = ufl.VectorElement("Lagrange", mesh.ufl_cell(),
                              degree=1, dim=2)

element_alpha = ufl.FiniteElement("Lagrange", mesh.ufl_cell(),
                                  degree=1)

V_u = dolfinx.fem.FunctionSpace(mesh, element_u)
V_alpha = dolfinx.fem.FunctionSpace(mesh, element_alpha)

u = dolfinx.fem.Function(V_u, name="Displacement")
u_top = dolfinx.fem.Function(V_u, name="BoundaryDisplacement")
u_bot = dolfinx.fem.Function(V_u, name="BoundaryDisplacement")

alpha = dolfinx.fem.Function(V_alpha, name="Damage")

# Pack state
state = {"u": u, "alpha": alpha}

# Bounds
alpha_ub = dolfinx.fem.Function(V_alpha, name="UpperBoundDamage")
alpha_lb = dolfinx.fem.Function(V_alpha, name="LowerBoundDamage")

dx = ufl.Measure("dx", domain=mesh)
# ds = ufl.Measure("ds", domain = mesh)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=facets)

# Boundary sets
dofs_alpha_left = locate_dofs_geometrical(
    V_alpha, lambda x: np.isclose(x[0], 0))
dofs_alpha_right = locate_dofs_geometrical(
    V_alpha, lambda x: np.isclose(x[0], Lx))
dofs_alpha_bottom = locate_dofs_geometrical(
    V_alpha, lambda x: np.isclose(x[1], 0.))
dofs_alpha_top = locate_dofs_geometrical(
    V_alpha, lambda x: np.isclose(x[1], Ly))

dofs_u_left = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], 0.))
dofs_u_right = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], Lx))
dofs_u_top = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[1], Ly))
dofs_u_bottom = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[1], 0))

# Bounds (nontrivial)
alpha_ub.interpolate(lambda x: np.ones_like(x[0]))
alpha_lb.interpolate(lambda x: np.zeros_like(x[0]))
# Boundary data
u_top.interpolate(lambda x: (np.zeros_like(x[0]), np.ones_like(x[1])))
u_bot.interpolate(lambda x: (np.zeros_like(x[0]), -1 * np.ones_like(x[1])))

_radius = parameters.get("geometry").get("rhoc")
# dofs_u_circle_top = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], 0.))
_cx = parameters.get("geometry").get("xc"),
_cy = parameters.get("geometry").get("Ly")/2 + \
    parameters.get("geometry").get("deltac")
_radius = parameters.get("geometry").get("rhoc")

# works only if symmetric
_delta = parameters.get("geometry").get("deltac")


# unsymmetric case: TODO

def _geom_pin(x, pin, radius):
  return np.isclose((x[0]-pin[0])**2. + (x[1]-pin[1])**2 - radius**2, 0)



dofs_u_pin_top = locate_dofs_geometrical(
    V_u, lambda x: _geom_pin(x, [_cx, _cy], _radius))
dofs_u_pin_bot = locate_dofs_geometrical(
    V_u, lambda x: _geom_pin(x, [_cx, _cy-2*_delta], _radius))

dofs_alpha_pin_top = locate_dofs_geometrical(
    V_alpha, lambda x: _geom_pin(x, [_cx, _cy], _radius))
dofs_alpha_pin_bot = locate_dofs_geometrical(
    V_alpha, lambda x: _geom_pin(x, [_cx, _cy-2*_delta], _radius))

assert(len(dofs_u_pin_top) == len(dofs_u_pin_bot))
assert(len(dofs_alpha_pin_top) == len(dofs_alpha_pin_bot))

# Boundary conditions

bcs_u = [
    dirichletbc(u_top,
                dofs_u_pin_top),
    dirichletbc(u_bot,
                dofs_u_pin_bot),
]


bcs_alpha = [
    dirichletbc(np.array(0., dtype=PETSc.ScalarType),
                np.concatenate(
        [dofs_alpha_pin_top, dofs_alpha_pin_bot]),
        V_alpha)
]

bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}

# Define the model

model = Brittle(parameters["model"])

total_energy = model.total_energy_density(state) * dx

solver = am.AlternateMinimisation(total_energy,
                                  state,
                                  bcs,
                                  parameters.get("solvers"),
                                  bounds=(alpha_lb, alpha_ub)
    )

# Loop for evolution
loads = np.linspace(parameters.get("loading").get("min"),
                    parameters.get("loading").get("max"),
                    parameters.get("loading").get("steps"))

history_data = {
    "load": [],
    "elastic_energy": [],
    "total_energy": [],
    "dissipated_energy": [],
    "solver_data": [],
}

data = {
    'elastic': [],
    'surface': [],
    'total': [],
    'load': []
}
xvfb.start_xvfb(wait=0.05)
pyvista.OFF_SCREEN = True
plotter = pyvista.Plotter(
        title="Displacement",
        window_size=[1600, 600],
        shape=(1, 2),
    )
generateStepwiseOutput = True

for (i_t, t) in enumerate(loads):
    # update boundary conditions

    u_top.interpolate(lambda x: ( np.zeros_like(x[0]), t*np.ones_like(x[1])))
    u_top.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                        mode=PETSc.ScatterMode.FORWARD)

    u_bot.interpolate(lambda x: ( np.zeros_like(x[0]), -1 * t*np.ones_like(x[1])))
    u_bot.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                        mode=PETSc.ScatterMode.FORWARD)

    # update lower bound for damage
    alpha.vector.copy(alpha_lb.vector)
    alpha.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                        mode=PETSc.ScatterMode.FORWARD)

    # solve for current load step
    print(f"Solving timestep {i_t}, load: {t}")

    solver.solve()

    # postprocessing
    # global

    surface_energy = assemble_scalar(dolfinx.fem.form(
        model.damage_dissipation_density(state) * dx))

    elastic_energy = assemble_scalar(
        dolfinx.fem.form(model.elastic_energy_density(state) * dx))

    data.get('elastic').append(elastic_energy)
    data.get('surface').append(surface_energy)
    data.get('total').append(surface_energy+elastic_energy)
    data.get('load').append(t)

    print(f"Solved timestep {i_t}, load: {t}")
    print(
        f"Elastic Energy {elastic_energy:.3g}, Surface energy: {surface_energy:.3g}")
    print("\n")

    _plt = plot_scalar(alpha, plotter, subplot=(0, 0))
    _plt.screenshot(f"damage-t-{i_t}.png")

    with XDMFFile(comm, f"{prefix}.xdmf", "a", encoding=XDMFFile.Encoding.HDF5) as file:
        file.write_function(u, t)
        file.write_function(alpha, t)

    if comm.rank == 0:
        a_file = open(f"{prefix}-data.json", "w")
        json.dump(history_data, a_file)
        a_file.close()

list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])


# Viz
xvfb.start_xvfb(wait=0.05)
pyvista.OFF_SCREEN = True


plotter = pyvista.Plotter(
    title="Displacement",
    window_size=[1600, 600],
    shape=(1, 2),
)
_plt = plot_scalar(alpha, plotter, subplot=(0, 0))
_plt = plot_vector(u, plotter, subplot=(0, 1))
_plt.screenshot(f"{outdir}/crackhole-state.png")
# if comm.rank == 0:
#     plot_energies(history_data, file=f"{prefix}_energies.pdf")
#     plot_AMit_load(history_data, file=f"{prefix}_it_load.pdf")

plt.figure()
plt.plot(data.get('load'), data.get('surface'), label='surface')
plt.plot(data.get('load'), data.get('elastic'), label='elastic')

plt.title('Traction bar energetics')
plt.legend()
plt.xticks([0, 1], [0, 1])
plt.savefig(os.path.join(outdir, "energetics.png"))

xvfb.start_xvfb(wait=0.05)
pyvista.OFF_SCREEN = True

plotter = pyvista.Plotter(
    title="Displacement",
    window_size=[1600, 600],
    shape=(1, 2),
)

# _plt = plot_scalar(u_.sub(0), plotter, subplot=(0, 0))
_plt = plot_scalar(alpha, plotter, subplot=(0, 0))
_plt = plot_vector(u, plotter, subplot=(0, 1))
# _plt = plot_vector(u, plotter, subplot=(0, 1), factor=.1)
_plt.screenshot(os.path.join(outdir, "fields.png"))


pdb.set_trace()
df = pd.DataFrame(history_data)
print(df)
