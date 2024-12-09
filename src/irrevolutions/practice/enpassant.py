"""
Author: Philipp M. Eisenhardt
Course: Complex Fracture
Phase field solver for the en passent crack problem

Based on code by A. León Baldelli, C. Maurini

Reads in geometry data from YAML file, generates mesh, solves Phase Field equations with alternative minimization approach using DolfinX

To change the data, change the geometry files according to presentation
"""

from utils.viz import plot_mesh, plot_scalar, plot_vector
from pyvista.plotting.utilities import xvfb
from petsc4py import PETSc
from models import DamageElasticityModel as Brittle
from meshes import primitives
from dolfinx.fem import assemble_scalar, dirichletbc, locate_dofs_geometrical
from algorithms import am
import ufl
import pyvista
import numpy as np
import meshes
import matplotlib.pyplot as plt
import dolfinx.plot
import dolfinx.io
import dolfinx
import logging
import sys
import basix.ufl

sys.path.append("../")

logging.basicConfig()
# logging.getLogger().setLevel(logging.DEBUG)
# logging.getLogger('foo').debug('bah')
logging.getLogger().setLevel(logging.INFO)
# logging.getLogger('foo').debug('bah')

logging.basicConfig(level=logging.INFO)


# meshes

# visualisation
generateStepwiseOutput = True

# Parameters
parameters = {
    "loading": {"min": 0.0, "max": 50, "steps": 100},
    "geometry": {
        "geom_type": "bar",
        "Lx": 100,
        "Ly": 200,
        "L0": 15,
        "s": 5,
    },
    "model": {"E": 1e-1, "nu": 0.4, "w1": 1.0, "ell": 2.8571, "k_res": 1.0e-8},
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
# Mesh

Lx = parameters["geometry"]["Lx"]
Ly = parameters["geometry"]["Ly"]
L0 = parameters["geometry"]["L0"]
s = parameters["geometry"]["s"]
geom_type = parameters["geometry"]["geom_type"]

gmsh_model, tdim = primitives.mesh_ep_gmshapi(
    geom_type,
    Lx,
    Ly,
    L0,
    s,
    parameters["model"]["ell"] / 5,
    tdim=2,
    sep=3e-1,
    _n=5,
)

"""gmsh_model, tdim = primitives.mesh_bar_gmshapi(geom_type,
                                    Lx, 
                                    Ly, 
                                    parameters.get("model").get("ell")/3, 
                                    tdim=2)"""

mesh, mts = meshes.gmsh_model_to_mesh(
    gmsh_model, cell_data=False, facet_data=True, gdim=2
)


plt.figure()
ax = plot_mesh(mesh)
fig = ax.get_figure()
fig.savefig("mesh.png")

mesh.topology.create_entities(tdim - 1)


def left_corner(x):
    return np.logical_and(x[0] < Lx / 4, x[1] < Ly / 2)


def middle_area(x):
    return np.logical_and(x[1] < Ly / 2 + s, x[1] > Ly / 2 - s)


edges = dolfinx.mesh.locate_entities(mesh, tdim - 1, middle_area)
mesh_refined_local2 = dolfinx.mesh.refine(mesh, edges, redistribute=False)

"""
edges = dolfinx.mesh.locate_entities(mesh_refined_local2, tdim-1, middle_area)
mesh_refined_local3 = dolfinx.mesh.refine(mesh_refined_local2, edges, redistribute=True)
"""

plt.figure()
ax = plot_mesh(mesh_refined_local2)
fig = ax.get_figure()
fig.savefig("mesh_refined_local_bulk.png")

# Enables opportunity to improve mesh at specific places, might lead to deformed elements
mesh = mesh_refined_local2
# Functional Setting

element_u = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1, shape=(2,))

element_alpha = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1)

V_u = dolfinx.fem.functionspace(mesh, element_u)
V_alpha = dolfinx.fem.functionspace(mesh, element_alpha)

u = dolfinx.fem.Function(V_u, name="Displacement")
u_ = dolfinx.fem.Function(V_u, name="BoundaryDisplacement")

_tot_dofs = 0

for _V in [V_u, V_alpha]:
    _tot_dofs += _V.dofmap.index_map.size_global * _V.dofmap.index_map_bs

logging.critical(f"Total number of dofs={_tot_dofs:.1e}")

alpha = dolfinx.fem.Function(V_alpha, name="Damage")

# Pack state
state = {"u": u, "alpha": alpha}

# Bounds
alpha_ub = dolfinx.fem.Function(V_alpha, name="UpperBoundDamage")
alpha_lb = dolfinx.fem.Function(V_alpha, name="LowerBoundDamage")


dx = ufl.Measure("dx", domain=mesh)
ds = ufl.Measure("ds", domain=mesh)


# Boundary sets
dofs_alpha_left = locate_dofs_geometrical(V_alpha, lambda x: np.isclose(x[0], 0.0))
dofs_alpha_right = locate_dofs_geometrical(V_alpha, lambda x: np.isclose(x[0], Lx))
dofs_alpha_bottom = locate_dofs_geometrical(V_alpha, lambda x: np.isclose(x[1], 0.0))
dofs_alpha_top = locate_dofs_geometrical(V_alpha, lambda x: np.isclose(x[1], Lx))


dofs_u_left = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], 0.0))
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
    dirichletbc(np.array([0.0, 0.0], dtype=PETSc.ScalarType), dofs_u_bottom, V_u),
    dirichletbc(u_, dofs_u_top),
]

bcs_alpha = [
    dirichletbc(
        np.array(0.0, dtype=PETSc.ScalarType),
        np.concatenate([dofs_alpha_bottom, dofs_alpha_top]),
        V_alpha,
    )
]

bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}

model = Brittle(parameters["model"])

total_energy = model.total_energy_density(state) * dx


def monitor(snes, its, fgnorm):
    if its % 10 == 0:
        print(f"Iteration {its:d}, error: {fgnorm:2.3e}")


solver = am.AlternateMinimisation(
    total_energy,
    state,
    bcs,
    parameters.get("solvers"),
    bounds=(alpha_lb, alpha_ub),
    #   monitor=monitor
)

# Loop for evolution
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
    window_size=[800, 600],
    shape=(1, 1),
)
logging.getLogger().setLevel(logging.INFO)

for i_t, t in enumerate(loads):
    # update boundary conditions

    u_.interpolate(lambda x: (np.zeros_like(x[0]), t * np.ones_like(x[1])))
    u_.x.petsc_vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )

    # update lower bound for damage
    alpha.x.petsc_vec.copy(alpha_lb.x.petsc_vec)
    alpha.x.petsc_vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )

    # solve for current load step
    logging.info(f"Solving timestep {i_t}, load: {t}")

    solver.solve(outdir="output/")

    # postprocessing
    # global

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

    #   print(f"Solved timestep {i_t}, load: {t}")
    print(f"Elastic Energy {elastic_energy:.3g}, Surface energy: {surface_energy:.3g}")
    print("\n\n")
    if generateStepwiseOutput:
        if surface_energy > 0.1 and i_t % 25 == 0:
            # if(i_t>1050 and i_t<1100):
            _plt = plot_scalar(alpha, plotter)
            _plt.screenshot("./plots/s05_fine/alpha" + str(i_t) + ".png")
        # if i_t>1100:
        #    break
    if i_t > 20 and elastic_energy < 1e-3 and elastic_energy < surface_energy:
        # Brute force approach to determine, whether we have already reached a crack propagation state
        break

    # savings?
plt.figure()
plt.plot(data.get("load"), data.get("surface"), label="surface")
plt.plot(data.get("load"), data.get("elastic"), label="elastic")
plt.plot(
    data.get("load"),
    [1.0 / 2.0 * t**2 * Lx for t in data.get("load")],
    label="anal elast",
    ls=":",
    c="k",
)

plt.title("Traction bar energetics")
plt.legend()
plt.yticks([0, 1 / 20], [0, "$1/2.\sigma_c^2/E_0$"])
plt.xticks([0, 1], [0, 1])
plt.savefig("energetics.png")

# savings?
plt.figure()
plt.plot(data.get("load")[:-40], data.get("surface")[:-40], label="surface")
plt.plot(data.get("load")[:-40], data.get("elastic")[:-40], label="elastic")

plt.title("Last steps")
plt.legend()
plt.yticks([0, 1 / 20], [0, "$1/2.\sigma_c^2/E_0$"])
plt.xticks([0, 1], [0, 1])
plt.savefig("lastSteps.png")


plotter = pyvista.Plotter(
    title="Displacement",
    window_size=[1600, 600],
    shape=(1, 2),
)

# _plt = plot_scalar(u_.sub(0), plotter, subplot=(0, 0))
_plt = plot_vector(u, plotter, subplot=(0, 1))
_plt.screenshot("displacement_MPI.png")


xvfb.start_xvfb(wait=0.05)
pyvista.OFF_SCREEN = True

plotter = pyvista.Plotter(
    title="Displacement",
    window_size=[1600, 600],
    shape=(1, 2),
)

_plt = plot_scalar(alpha, plotter, subplot=(0, 0))
_plt.screenshot("alpha2.png")
