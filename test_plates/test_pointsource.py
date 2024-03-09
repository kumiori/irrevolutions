# Create a point source for Poisson problem
# Author: Jørgen S. Dokken
# SPDX-License-Identifier: MIT
import sys

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.fem.petsc
from dolfinx.io import XDMFFile, gmshio
import numpy as np
import ufl

sys.path.append("../")
from meshes.primitives import mesh_circle_gmshapi



def compute_cell_contributions(V, points):
    # Initialize empty arrays to store cell indices and basis values
    all_cells = []
    all_basis_values = []

    for point in points:
        # Compute cell contributions for the current point
        cells, basis_values = compute_cell_contribution_point(V, point)

        # Append the results to the arrays
        all_cells.append(cells)
        all_basis_values.append(basis_values)

    # Concatenate the lists to create NumPy arrays
    all_cells = np.concatenate(all_cells)
    all_basis_values = np.concatenate(all_basis_values)

    return all_cells, all_basis_values

def compute_cell_contribution_point(V, points):
    # Determine what process owns a point and what cells it lies within
    mesh = V.mesh
    _, _, owning_points, cells = dolfinx.cpp.geometry.determine_point_ownership(
        mesh._cpp_object, points, 1e-6)
    owning_points = np.asarray(owning_points).reshape(-1, 3)

    # Pull owning points back to reference cell
    mesh_nodes = mesh.geometry.x
    cmap = mesh.geometry.cmaps[0]
    ref_x = np.zeros((len(cells), mesh.geometry.dim),
                     dtype=mesh.geometry.x.dtype)
    for i, (point, cell) in enumerate(zip(owning_points, cells)):
        geom_dofs = mesh.geometry.dofmap[cell]
        ref_x[i] = cmap.pull_back(point.reshape(-1, 3), mesh_nodes[geom_dofs])

    # Create expression evaluating a trial function (i.e. just the basis function)
    u = ufl.TrialFunction(V)
    num_dofs = V.dofmap.dof_layout.num_dofs * V.dofmap.bs
    if len(cells) > 0:
        # NOTE: Expression lives on only this communicator rank
        expr = dolfinx.fem.Expression(u, ref_x, comm=MPI.COMM_SELF)
        values = expr.eval(mesh, np.asarray(cells, dtype=np.int32))

        # Strip out basis function values per cell
        basis_values = values[:num_dofs:num_dofs*len(cells)]
    else:
        basis_values = np.zeros(
            (0, num_dofs), dtype=dolfinx.default_scalar_type)
    return cells, basis_values


N = 30
domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
# domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)

# Create the mesh of the specimen with given dimensions
geom_type = 'circle'
tdim = 2
D = 1
gmsh_model, tdim = mesh_circle_gmshapi(geom_type, D/2, 1/N, tdim)

# Get mesh and meshtags
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, MPI.COMM_WORLD, 0, tdim)

mesh.name = "mesh"
mesh.topology.create_connectivity(1, 2)
# domain.name = "mesh"
# domain.topology.create_connectivity(1, 2)
domain = mesh

V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))

facets = dolfinx.mesh.exterior_facet_indices(domain.topology)

if domain.comm.rank == 0:
    point = np.array([[0.68, 0.36, 0]], dtype=domain.geometry.x.dtype)
    
    points = [np.array([[-0.2, 0.1, 0]], dtype=domain.geometry.x.dtype),
              np.array([[0.2, -0.1, 0]], dtype=domain.geometry.x.dtype)]
    signs = [-1, 1]
else:
    point = np.zeros((0, 3), dtype=domain.geometry.x.dtype)
    points = [np.zeros((0, 3), dtype=domain.geometry.x.dtype),
              np.zeros((0, 3), dtype=domain.geometry.x.dtype)]


u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
a_compiled = dolfinx.fem.form(a)


dofs = dolfinx.fem.locate_dofs_topological(V, 1, facets)
u_bc = dolfinx.fem.Constant(domain, 0.)
bc = dolfinx.fem.dirichletbc(u_bc, dofs, V)

b = dolfinx.fem.Function(V)
b.x.array[:] = 0
# cells, basis_values = compute_cell_contribution_point(V, point)
_cells, _basis_values = compute_cell_contributions(V, points)

for cell, basis_value, sign in zip(_cells, _basis_values, signs):
    dofs = V.dofmap.cell_dofs(cell)
    b.x.array[dofs] += sign * basis_value
dolfinx.fem.petsc.apply_lifting(b.vector, [a_compiled], [[bc]])
b.x.scatter_reverse(dolfinx.la.InsertMode.add)
dolfinx.fem.petsc.set_bc(b.vector, [bc])
b.x.scatter_forward()

A = dolfinx.fem.petsc.assemble_matrix(a_compiled, bcs=[bc])
A.assemble()

ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.PREONLY)
ksp.getPC().setType(PETSc.PC.Type.LU)
ksp.getPC().setFactorSolverType("mumps")


uh = dolfinx.fem.Function(V)
ksp.solve(b.vector, uh.vector)
uh.x.scatter_forward()

with dolfinx.io.VTXWriter(domain.comm, "uh.bp", [uh], engine="BP4") as bp:
    bp.write(0.0)
    

from dolfinx import plot
outdir = 'output'

try:
    import pyvista
    pyvista.OFF_SCREEN = True
    cells, types, x = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = uh.x.array.real
    grid.set_active_scalars("u")
    plotter = pyvista.Plotter()
    # plotter.add_mesh(grid, show_edges=True)
    warped = grid.warp_by_scalar()
    plotter.add_mesh(warped, show_edges=True)
    
    if pyvista.OFF_SCREEN:
        pyvista.start_xvfb(wait=0.1)
        plotter.screenshot(f"{outdir}/point_source.png")
    else:
        plotter.show()
except ModuleNotFoundError:
    print("'pyvista' is required to visualise the solution")
    print("Install 'pyvista' with pip: 'python3 -m pip install pyvista'")
