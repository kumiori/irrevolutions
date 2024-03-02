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

# try:
#     from dolfinx.plot import create_vtk_mesh as compute_topology
# except ImportError:
#     from dolfinx.plot import create_vtk_topology as compute_topology

from dolfinx.plot import vtk_mesh as compute_topology

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
    # grid.point_data["u"] = values
    grid.point_data["u"] = u.x.array
    grid.set_active_scalars("u")
    plotter.add_mesh(grid, **lineproperties)
    plotter.view_xy()
    plotter.set_background('white')
    return plotter

def plot_profile(u, points, plotter, subplot=None, lineproperties={}, fig=None, ax=None, subplotnumber = 1):
    import matplotlib.pyplot as plt
    import dolfinx.geometry
    mesh = u.function_space.mesh
    bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)

    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, points.T)
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

    if subplot:
        # plotter.subplot(subplot[0], subplot[1])
    # if subplot:
        plt.subplot(subplot[0], subplot[1], subplotnumber)
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
    cells = mesh.geometry.dofmap.reshape((-1, mesh.topology.dim + 1))
    tria = tri.Triangulation(points[:, 0], points[:, 1], cells)
    ax.triplot(tria, color="k")
    return ax


def plot_perturbations(comm, Lx, prefix, β, v, bifurcation, stability, i_t):
    from solvers.function import vec_to_functions
    
    vec_to_functions(bifurcation._spectrum[0]['xk'], [v, β])
    if comm.Get_size() == 1:
        tol = 1e-3
        xs = np.linspace(0 + tol, Lx - tol, 101)
        points = np.zeros((3, 101))
        points[0] = xs
                
        plotter = pyvista.Plotter(
                    title="Perturbation profile",
                    window_size=[800, 600],
                    shape=(1, 1),
                )
        _plt, data = plot_profile(
                    β,
                    points,
                    plotter,
                    subplot=(0, 0),
                    lineproperties={
                        "c": "k",
                        "label": f"$\\beta$"
                    },
                )
        ax = _plt.gca()
        _plt.legend()
        _plt.fill_between(data[0], data[1].reshape(len(data[1])))
        _plt.title("Perurbation")
        _plt.savefig(f"{prefix}/perturbation-profile-{i_t}.png")
        _plt.close()


        plotter = pyvista.Plotter(
                    title="Cone-Perturbation profile",
                    window_size=[800, 600],
                    shape=(1, 1),
                )

        _plt, data = plot_profile(
                    stability.perturbation['beta'],
                    points,
                    plotter,
                    subplot=(0, 0),
                    lineproperties={
                        "c": "k",
                        "label": f"$\\beta$"
                    },
                )
        ax = _plt.gca()
        _plt.legend()
        _plt.fill_between(data[0], data[1].reshape(len(data[1])))
        _plt.title("Perurbation from the Cone")
        _plt.savefig(f"{prefix}/perturbation-profile-cone-{i_t}.png")
        _plt.close()
        
    return plotter

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
