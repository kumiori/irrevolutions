import ufl
from pathlib import Path
import sys
sys.path.append("../")
import dolfinx.plot
import meshes
from meshes.primitives import mesh_bar_gmshapi
import numpy as np
from datetime import date

today = date.today()


import dolfinx

from petsc4py import PETSc
import petsc4py

petsc4py.init(sys.argv)

opts = PETSc.Options()
# opts.setValue("help", 1)
import logging

logging.basicConfig(level=logging.INFO)


from mpi4py import MPI

comm = MPI.COMM_WORLD
# import pdb
import pyvista
from pyvista.utilities import xvfb



def test_viz():
    Lx = 1.0
    Ly = 0.1
    _nel = 30
    _outdir = './output/test_viz'
    Path(_outdir).mkdir(parents=True, exist_ok=True)

    gmsh_model, tdim = mesh_bar_gmshapi("bar", Lx, Ly, Lx/_nel, 2)
    mesh, mts = meshes.gmsh_model_to_mesh(gmsh_model,
                                cell_data=False,
                                facet_data=True,
                                gdim=2)
    element_u = ufl.VectorElement("Lagrange", mesh.ufl_cell(), degree=1, dim=2)
    V_u = dolfinx.fem.FunctionSpace(mesh, element_u)

    element_alpha = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)
    V_alpha = dolfinx.fem.FunctionSpace(mesh, element_alpha)

    def scal_2D(x):
        return np.sin(np.pi * x[0] * 3.0)

    def vect_2D(x):
        vals = np.zeros((2, x.shape[1]))
        vals[0] = np.sin(x[1])
        vals[1] = 0.1 * x[0]
        return vals

    # Define the state
    u = dolfinx.fem.Function(V_u, name="Displacement")
    u.interpolate(vect_2D)

    alpha = dolfinx.fem.Function(V_alpha, name="Damage")
    alpha.interpolate(scal_2D)

    xvfb.start_xvfb(wait=0.05)
    pyvista.OFF_SCREEN = True

    plotter = pyvista.Plotter(
        title="Test Viz",
        window_size=[1600, 600],
        shape=(1, 2),
    )

    _plt = plot_scalar(alpha, plotter, subplot=(0, 0))

    _plt = plot_vector(u, plotter, subplot=(0, 1))

    _plt.screenshot(f"./output/test_viz/test_viz_MPI{comm.size}-.png")

    if not pyvista.OFF_SCREEN:
        plotter.show()


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
    # figure = plotter.screenshot(f"./output/test_viz/test_viz_MPI{comm.size}-.png")


def plot_scalar(alpha, plotter, subplot=None, lineproperties={}):
    if subplot:
        plotter.subplot(subplot[0], subplot[1])
    V = alpha.function_space
    mesh = V.mesh
    topology, cell_types = dolfinx.plot.create_vtk_topology(mesh, mesh.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, mesh.geometry.x)

    plotter.subplot(0, 0)
    grid.point_data["alpha"] = alpha.compute_point_values().real
    grid.set_active_scalars("alpha")
    plotter.add_mesh(grid, **lineproperties)
    plotter.view_xy()
    return plotter


def plot_profile(u, points, plotter, subplot=None, lineproperties={}):
    import matplotlib.pyplot as plt
    import dolfinx.geometry
    if subplot:
        plotter.subplot(subplot[0], subplot[1])
    mesh = u.function_space.mesh
    bb_tree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.topology.dim)

    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = dolfinx.geometry.compute_collisions(bb_tree, points.T)
    # Choose one of the cells that contains the point
    colliding_cells = dolfinx.geometry.compute_colliding_cells(
        mesh, cell_candidates, points.T
    )
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])

    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    u_values = u.eval(points_on_proc, cells)

    fig = plt.figure()

    if subplot:
        plt.subplot(subplot[0] + 1, subplot[1] + 1, 1)
    # plt.plot(points_on_proc[:, 0], u_values, "k", ls="-", linewidth=1, label="")
    plt.plot(points_on_proc[:, 0], u_values, **lineproperties)
    plt.legend()
    return plt, (points_on_proc[:, 0], u_values)


if __name__ == "__main__":
    test_viz()
