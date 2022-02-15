import numpy as np
import sys
from datetime import date

today = date.today()

sys.path.append("../")

import dolfinx
import logging

logging.basicConfig(level=logging.INFO)

from mpi4py import MPI

comm = MPI.COMM_WORLD
# import pdb
import pyvista
from pyvista.utilities import xvfb

import dolfinx.plot

import matplotlib
import matplotlib.collections
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt

def plot_vector(u, plotter, subplot=None):
    if subplot:
        plotter.subplot(subplot[0], subplot[1])
    V = u.function_space
    mesh = V.mesh
    # topology, cell_types = dolfinx.plot.create_vtk_mesh(mesh, mesh.topology.dim)
    topology, cell_types = dolfinx.plot.create_vtk_topology(
        mesh, mesh.topology.dim)
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
    # topology, cell_types = dolfinx.plot.create_vtk_mesh(mesh, mesh.topology.dim)
    topology, cell_types = dolfinx.plot.create_vtk_topology(
        mesh, mesh.topology.dim)
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


def plot_mesh(mesh, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.set_aspect("equal")
    points = mesh.geometry.x
    cells = mesh.geometry.dofmap.array.reshape((-1, mesh.topology.dim + 1))
    tria = tri.Triangulation(points[:, 0], points[:, 1], cells)
    ax.triplot(tria, color="k")
    return ax
