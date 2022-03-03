import pdb
import logging
import sys
sys.path.append('../')
from dolfinx.io import XDMFFile
import ufl
from dolfinx import log
import dolfinx.plot
import dolfinx
from petsc4py import PETSc
import petsc4py
from mpi4py import MPI
from pathlib import Path
import os
import json
import yaml
import numpy as np
import dolfinx.io
import gmsh
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
from pyvista.utilities import xvfb
import meshes
# from meshes import primitives
from meshes import thekink
from utils import viz
from models import DamageElasticityModel as Brittle
from utils.viz import plot_mesh, plot_vector, plot_scalar
import pyvista
import algorithms
from algorithms import am


logging.basicConfig(level=logging.INFO)


sys.path.append('./')
# Parameters
parameters = {
    'loading': {
        'min': 0.0,
        'max': 2.,
        'steps': 10
    },
    'geometry': {
        'geom_type': 'Plate with Kink',
        'Lx': .6,
        'Ly': 1,
        'L0': 0.3,
        'theta': 70.5,
        'eta': 1e-2,
    },
    'model': {
        # 'E': 3.e+9,
        'E': 1,
        'nu': 0.4,
        'w1': 1.,
        # 'ell': 0.1,
        'ell': 0.01,
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


with open("../data/matpar.json", "r") as materials:
    materials_db = json.load(materials)
parameters_pmma = materials_db[3]

Lx = parameters.get("geometry").get("Lx")
Ly = parameters.get("geometry").get("Ly")
L0 = parameters.get("geometry").get("L0")
eta = parameters.get("geometry").get("eta")

geom_type = parameters.get("geometry").get("geom_type")

_ell = parameters.get("model").get("ell")
lc = parameters.get("model").get("ell")/2
theta = parameters.get("geometry").get("theta")/180 * np.pi

gmsh_model = thekink.mesh_kink('mesh', Lx, Ly, L0/2, theta, eta, lc, 2, 1)

mesh, mts = meshes.gmsh_model_to_mesh(gmsh_model,
                                      cell_data=False,
                                      facet_data=True,
                                      gdim=2)

element_u = ufl.VectorElement("Lagrange", mesh.ufl_cell(),
                              degree=1, dim=2)

element_alpha = ufl.FiniteElement("Lagrange", mesh.ufl_cell(),
                                  degree=1)

V_u = dolfinx.fem.FunctionSpace(mesh, element_u)
V_alpha = dolfinx.fem.FunctionSpace(mesh, element_alpha)

_tot_dofs = 0

for _V in [V_u, V_alpha]:
    _tot_dofs += _V.dofmap.index_map.size_global*_V.dofmap.index_map_bs

logging.critical(f'Total number of dofs={_tot_dofs:.1e}')

plt.figure()
ax = plot_mesh(mesh)
fig = ax.get_figure()
fig.suptitle(f'TheKink mesh, $ell/h=$ {_ell/lc}, {_tot_dofs:.1e} dofs')
fig.savefig(f"thekink_mesh.png")



u = dolfinx.fem.Function(V_u, name="Displacement")
u_ = dolfinx.fem.Function(V_u, name="BoundaryDisplacement")

alpha = dolfinx.fem.Function(V_alpha, name="Damage")

# Pack state
state = {"u": u, "alpha": alpha}

# Bounds
alpha_ub = dolfinx.fem.Function(V_alpha, name="UpperBoundDamage")
alpha_lb = dolfinx.fem.Function(V_alpha, name="LowerBoundDamage")


dx = ufl.Measure("dx", domain=mesh)
ds = ufl.Measure("ds", domain=mesh)


# Boundary sets
dofs_alpha_left = locate_dofs_geometrical(
    V_alpha, lambda x: np.isclose(x[0], 0.))
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

# Boundary data
u_.interpolate(lambda x: (np.zeros_like(x[0]), np.ones_like(x[1])))

# Bounds (nontrivial)
alpha_lb.interpolate(lambda x: np.zeros_like(x[0]))
alpha_ub.interpolate(lambda x: np.ones_like(x[0]))

# Boundary conditions
bcs_u = [
    dirichletbc(np.array([0., 0.], dtype=PETSc.ScalarType),
                dofs_u_bottom,
                V_u),
    dirichletbc(u_, dofs_u_top)
]

bcs_alpha = [
    dirichletbc(np.array(0., dtype=PETSc.ScalarType),
                np.concatenate([dofs_alpha_bottom, dofs_alpha_top]),
                V_alpha)
]

bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}

model = Brittle(parameters["model"])

total_energy = model.total_energy_density(state) * dx
solver = am.AlternateMinimisation(total_energy,
                                  state,
                                  bcs,
                                  parameters.get("solvers"),
                                  bounds=(alpha_lb, alpha_ub)
                                  )
loads = np.linspace(parameters.get("loading").get("min"),
                    parameters.get("loading").get("max"),
                    parameters.get("loading").get("steps"))

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

for (i_t, t) in enumerate(loads):
  # update boundary conditions

  u_.interpolate(lambda x: (np.zeros_like(x[0]), t*np.ones_like(x[1])))
  u_.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                        mode=PETSc.ScatterMode.FORWARD)

  # update lower bound for damage
  alpha.vector.copy(alpha_lb.vector)
  alpha.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                           mode=PETSc.ScatterMode.FORWARD)

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
  print("\n\n")


if MPI.COMM_WORLD.size==1:
    _plt = plot_scalar(alpha, plotter, subplot=(0, 0))
    _plt = plot_vector(u, plotter, subplot=(0, 1))
    _plt.screenshot(f"thekink.png")
