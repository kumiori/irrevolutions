import numpy as np

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
import dolfinx.mesh
from dolfinx.mesh import CellType
import ufl

from mpi4py import MPI
import petsc4py
from petsc4py import PETSc
import sys
import yaml
import pdb
import os
from pathlib import Path
pdb.set_trace()
sys.path.append("../")
from solvers import SNESSolver

petsc4py.init(sys.argv)

import logging

logging.basicConfig(level=logging.INFO)

with open("parameters.yml") as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)

Lx = parameters.get("geometry").get("Lx")
Ly = parameters.get("geometry").get("Ly")
ell = parameters.get("model").get("ell")



mesh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [Lx, Ly]],
                                     [100, 10],
                                     cell_type=CellType.triangle)
V = FunctionSpace(mesh, ("CG", 1))

zero = Function(V)
with zero.vector.localForm() as loc:
    loc.set(0.0)

one = Function(V)
with one.vector.localForm() as loc:
    loc.set(1.0)


def left(x):
    is_close = np.isclose(x[0], 0.0)
    return is_close


def right(x):
    is_close = np.isclose(x[0], Lx)
    return is_close


left_facets = dolfinx.mesh.locate_entities_boundary(mesh,
                                                    mesh.topology.dim - 1,
                                                    left)
left_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1,
                                                left_facets)

right_facets = dolfinx.mesh.locate_entities_boundary(mesh,
                                                     mesh.topology.dim - 1,
                                                     left)
right_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1,
                                                 right_facets)

bcs = [dirichletbc(zero, left_dofs), dirichletbc(one, right_dofs)]

u = Function(V)
energy = (ell * ufl.inner(ufl.grad(u), ufl.grad(u)) + u / ell) * ufl.dx
denergy = ufl.derivative(energy, u, ufl.TestFunction(V))
ddenergy = ufl.derivative(denergy, u, ufl.TrialFunction(V))

problem = SNESSolver(
    denergy,
    u,
    bcs,
    bounds=(zero, one),
    petsc_options=parameters.get("solvers").get("damage").get("snes"),
    prefix="vi",
)

solver_snes = problem.solver
solver_snes.setType("vinewtonrsls")

solver_snes.setTolerances(rtol=1.0e-8, max_it=250)
solver_snes.getKSP().setType("preonly")
solver_snes.getKSP().setTolerances(rtol=1.0e-8)
solver_snes.getKSP().getPC().setType("lu")


def monitor(snes, its, fgnorm):
    print(f"Iteration {its:d}, error: {fgnorm:2.3e}")


solver_snes.setMonitor(monitor)
solver_snes.solve(None, u.vector)
# solver_snes.view()

prefix = os.path.join("output", "test-vi")
if MPI.COMM_WORLD.rank == 0:
    Path(prefix).mkdir(parents=True, exist_ok=True)

prefix = "output/test-vi"
from pathlib import Path
Path(prefix).mkdir(parents=True, exist_ok=True)

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "output/u.xdmf", "w") as f:
    f.write_mesh(mesh)
    f.write_function(u)

import pyvista
from pyvista.utilities import xvfb

import dolfinx.plot

sys.path.append("../../test")
from utils.viz import plot_mesh, plot_vector, plot_scalar, plot_profile

xvfb.start_xvfb(wait=0.05)
pyvista.OFF_SCREEN = True

plotter = pyvista.Plotter(
    title="Test VI",
    window_size=[800, 600],
    shape=(1, 1),
)
_props = {"show_edges":True, "show_scalar_bar": True, "clim":[0, 1]}
_plt = plot_scalar(u, plotter, subplot=(0, 0), lineproperties=_props)

# _plt = plot_vector(u, plotter, subplot=(0, 1))

_plt.screenshot(f"{prefix}/test_vi_MPI{MPI.COMM_WORLD.size}.png")

if not pyvista.OFF_SCREEN:
    plotter.show()

tol = 1e-3
xs = np.linspace(0 + tol, Lx - tol, 101)
points = np.zeros((3, 101))
points[0] = xs

_plt, data = plot_profile(
    u,
    points,
    plotter,
    subplotnumber=1,
    lineproperties={
        "c": "k",
        "label": f"$u_\ell$ with $\ell$ = {ell:.2f}"
    },
)
ax = _plt.gca()
ax.axvline(0.0, c="k")
ax.axvline(2 * ell, c="k", label='D=$2\ell$')
_plt.legend()
_plt.fill_between(data[0], data[1].reshape(len(data[1])))
_plt.title("Variational Inequality")
_plt.savefig(f"{prefix}/test_vi_profile_MPI{MPI.COMM_WORLD.size}-{ell:.3f}.png")



from dolfinx.fem.assemble import assemble_scalar

min_en = assemble_scalar(dolfinx.fem.form(energy))