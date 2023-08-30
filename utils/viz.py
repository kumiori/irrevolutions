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

xvfb.start_xvfb(wait=0.05)

import dolfinx.plot

import matplotlib
import matplotlib.collections
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt

try:
    from dolfinx.plot import create_vtk_mesh as compute_topology
except ImportError:
    from dolfinx.plot import create_vtk_topology as compute_topology


def plot_vector(u, plotter, subplot=None, scale=1.):
    if subplot:
        plotter.subplot(subplot[0], subplot[1])
    V = u.function_space
    mesh = V.mesh
    ret = compute_topology(mesh, mesh.topology.dim)
    if len(ret) == 2:
        topology, cell_types = ret
    else:
        topology, cell_types, _ = ret
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
    glyphs = grid.glyph(orient="vectors", factor=scale)
    plotter.add_mesh(glyphs)
    plotter.add_mesh(
        grid, show_edges=True, color="black", style="wireframe", opacity=0.3
    )
    plotter.view_xy()
    plotter.set_background('white')
    return plotter
    # figure = plotter.screenshot(f"./output/test_viz/test_viz_MPI{comm.size}-.png")


def plot_scalar(u, plotter, subplot=None, lineproperties={}):
    """Plots a scalar function using pyvista

    Args:
        u: Scalar field
        plotter plotter: The plotter object
        subplot plotter: Optional selection of subplot slot
        lineproperties: Optional line properties (dictionary)

    Returns:
        plotter: Updated plotter object
   """
    if subplot:
        plotter.subplot(subplot[0], subplot[1])
    V = u.function_space
    mesh = V.mesh
    
    ret = compute_topology(mesh, mesh.topology.dim)
    if len(ret) == 2:
        topology, cell_types = ret
    else: 
        topology, cell_types, _ = ret
    grid = pyvista.UnstructuredGrid(topology, cell_types, mesh.geometry.x)

    plotter.subplot(0, 0)
    values = u.vector.array.real.reshape(
        V.dofmap.index_map.size_local, V.dofmap.index_map_bs)
    grid.point_data["u"] = values
    grid.set_active_scalars("u")
    plotter.add_mesh(grid, **lineproperties)
    plotter.view_xy()
    plotter.set_background('white')
    return plotter


def plot_profile(u, points, plotter, subplot=None, lineproperties={}, fig=None, ax=None):
    import matplotlib.pyplot as plt
    import dolfinx.geometry
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

    if fig is None:
        fig = plt.figure()

    # if subplot:
    #     plotter.subplot(subplot[0], subplot[1])
    # if subplot:
        # plt.subplot(subplot[0] + 1, subplot[1] + 1, 1)
    # plt.plot(points_on_proc[:, 0], u_values, "k", ls="-", linewidth=1, label="")

    if ax is not None:
        ax.plot(points_on_proc[:, 0], u_values, **lineproperties)
        # ax = plt.gca()
        ax.legend()

    else:
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

import scipy

def plot_matrix(M):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    indptr, indices, data = M.getValuesCSR()
    _M = scipy.sparse.csr_matrix((data, indices, indptr), shape=M.sizes[0])
    ax.matshow(_M.todense(), cmap=plt.cm.Blues)

    for i in range(_M.shape[0]):
        for j in range(_M.shape[0]):
            c = _M[j,i]
            ax.text(i, j, f"{c:.3f}", va='center', ha='center')

    return fig
