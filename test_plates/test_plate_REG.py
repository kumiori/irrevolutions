# Solving a plate problem with mixed formulation
# and Regge elements (ie. tensors with n-n continuity
# across facets)
import pyvista
from pyvista.utilities import xvfb
import sys
sys.path.append("../")

import logging

from utils.viz import plot_mesh, plot_vector, plot_scalar
import numpy as np
import yaml
import json
from pathlib import Path
import os
from mpi4py import MPI
import petsc4py
from petsc4py import PETSc
import dolfinx
import dolfinx.plot
from dolfinx import log, plot
import ufl
from ufl import (CellDiameter, FacetNormal, SpatialCoordinate, TestFunction,
                 TrialFunction, avg, div, ds, dS, dx, grad, inner, jump)
import pdb

from dolfinx.io import XDMFFile, gmshio
# from damage.utils import ColorPrint
from models import ElasticityModel
from solvers import SNESSolver as ElasticitySolver

from meshes.primitives import mesh_bar_gmshapi
from meshes import mesh_bounding_box

import numpy as np
logging.basicConfig(level=logging.INFO)


import dolfinx.plot
import dolfinx.io
from dolfinx.fem import (
    Constant,
    Function,
    FunctionSpace,
    dirichletbc,
    form,
)

from dolfinx.fem.petsc import (
    assemble_matrix,
    apply_lifting,
    set_bc,
    assemble_vector
    )

import dolfinx.mesh
from dolfinx.mesh import CellType
import ufl

from mpi4py import MPI
import petsc4py
from petsc4py import PETSc
import sys
import yaml
from ufl import (CellDiameter, FacetNormal, avg, div, dS, dx, grad, inner,
                 jump, pi, sin)
sys.path.append("../")
from solvers import SNESSolver


petsc4py.init(sys.argv)
log.set_log_level(log.LogLevel.WARNING)

comm = MPI.COMM_WORLD

outdir = 'output'
if comm.rank == 0:
    Path(outdir).mkdir(parents=True, exist_ok=True)

with open(f"parameters.yml") as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)


# Get mesh parameters
Lx = parameters["geometry"]["Lx"]
Ly = parameters["geometry"]["Ly"]
tdim = parameters["geometry"]["geometric_dimension"]
lc = parameters["geometry"]["lc"]

Lx = 1
Ly = 1
tdim = 2
lc = 0.1

# Get geometry model
geom_type = parameters["geometry"]["geom_type"]

# Create the mesh of the specimen with given dimensions
gmsh_model, tdim = mesh_bar_gmshapi(geom_type, Lx, Ly, lc, tdim)

# Get mesh and meshtags
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, 0, tdim)

with XDMFFile(comm, f"{outdir}/mesh.xdmf", "w",
              encoding=XDMFFile.Encoding.HDF5) as file:
    file.write_mesh(mesh)

# prefix = os.path.join(outdir, "plate")

with XDMFFile(comm, f"{outdir}/output.xdmf", "w",
              encoding=XDMFFile.Encoding.HDF5) as file:
    file.write_mesh(mesh)

# Function spaces
r=1
# r=2
# r=3

x_extents = mesh_bounding_box(mesh, 0)
# y_extents = mesh_bounding_box(mesh, 1)

# Measures
dx = ufl.Measure("dx", domain=mesh)
ds = ufl.Measure("ds", domain=mesh)

element = ufl.MixedElement([ufl.FiniteElement("Regge", ufl.triangle, r),
                            ufl.FiniteElement("Lagrange", ufl.triangle, r+1)])

V = FunctionSpace(mesh, element)


V_1 = V.sub(1).collapse()[0]

sigma, u = ufl.TrialFunctions(V)
tau, v = ufl.TestFunctions(V)

def S(tau):
    return tau - ufl.Identity(2) * ufl.tr(tau)

# Discrete duality inner product 
# cf. eq. 4.5 Lizao Li's PhD thesis

def b(tau_S, v):
    n = FacetNormal(mesh)
    return inner(tau_S, grad(grad(v))) * dx \
        - ufl.dot(ufl.dot(tau_S('+'), n('+')), n('+')) * jump(grad(v), n) * dS \
        - ufl.dot(ufl.dot(tau_S, n), n) * ufl.dot(grad(v), n) * ds

sigma_S = S(sigma)
tau_S = S(tau)
f_exact = Constant(mesh, np.array(100., dtype=PETSc.ScalarType))

x = ufl.SpatialCoordinate(mesh)
f_exact = 4.0 * pi**4 * sin(pi * x[0]) * sin(pi * x[1])

# Non-symmetric formulation
a = form(ufl.inner(sigma_S, tau_S) * dx - b(tau_S, u) + b(sigma_S, v))
L = form(ufl.inner(f_exact, v) * dx)

zero_u = Function(V_1)
zero_u.x.array[:] = 0.0

boundary_facets = dolfinx.mesh.locate_entities_boundary(
    mesh, mesh.topology.dim - 1, lambda x: np.full(x.shape[1], True, dtype=bool))
boundary_dofs = dolfinx.fem.locate_dofs_topological(
    (V.sub(1), V_1), mesh.topology.dim - 1, boundary_facets)

bcs = [dirichletbc(zero_u, boundary_dofs, V.sub(1))]

A = assemble_matrix(a, bcs=bcs)
A.assemble()
b = assemble_vector(L)
apply_lifting(b, [a], bcs=[bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, bcs)

# Solve
solver = PETSc.KSP().create(MPI.COMM_WORLD)
PETSc.Options()["ksp_type"] = "preonly"
PETSc.Options()["pc_type"] = "lu"
PETSc.Options()["pc_factor_mat_solver_type"] = "mumps"
solver.setFromOptions()
solver.setOperators(A)

x_h = Function(V)
solver.solve(b, x_h.vector)
x_h.x.scatter_forward()

sigma_h = S(ufl.as_tensor([[x_h[0], x_h[1]], [x_h[2], x_h[3]]]))

(_S, _w) = x_h.split()


_V = FunctionSpace(mesh, ufl.FiniteElement("Lagrange", ufl.triangle, 1))
v = Function(_V)
v.interpolate(_w)

with XDMFFile(comm, f"{outdir}/output.xdmf", "a",
                encoding=XDMFFile.Encoding.HDF5) as file:
    file.write_function(v)

# Viz

xvfb.start_xvfb(wait=0.05)
pyvista.OFF_SCREEN = True

plotter = pyvista.Plotter(
    title="Displacement",
    window_size=[1600, 600],
    shape=(1, 1),
)
# _plt = plot_scalar(_w, plotter, subplot=(0, 0))
_plt = plot_scalar(v, plotter, subplot=(0, 0))
# _plt = plot_vector(u, plotter, subplot=(0, 1))
_plt.screenshot(f"{outdir}/plate_regge_displacement.png")

if not pyvista.OFF_SCREEN:
    plotter.show()

dofs_u_left = dolfinx.fem.locate_dofs_geometrical(
    (V.sub(1), V_1), lambda x: np.isclose(x[0], 0.0))
dofs_u_right = dolfinx.fem.locate_dofs_geometrical(
    (V.sub(1), V_1), lambda x: np.isclose(x[0], Lx))



V1 = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
u1 = dolfinx.fem.Function(V1)
u1.interpolate(v)



try:
    import pyvista
    pyvista.OFF_SCREEN = True
    cells, types, x = plot.vtk_mesh(V1)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = u1.x.array.real
    grid.set_active_scalars("u")
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    warped = grid.warp_by_scalar()
    plotter.add_mesh(warped, show_edges=True)
    
    if pyvista.OFF_SCREEN:
        pyvista.start_xvfb(wait=0.1)
        plotter.screenshot(f"{outdir}/plate.png")
    else:
        plotter.show()
except ModuleNotFoundError:
    print("'pyvista' is required to visualise the solution")
    print("Install 'pyvista' with pip: 'python3 -m pip install pyvista'")
