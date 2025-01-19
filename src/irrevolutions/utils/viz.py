import scipy
from dolfinx.plot import vtk_mesh as compute_topology
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from pyvista.plotting.utilities import xvfb
import pyvista
from mpi4py import MPI
import logging
import sys
from datetime import date
import numpy as np

# Set current date
today = date.today()

# Add path to the module search path
sys.path.append("../")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize MPI
comm = MPI.COMM_WORLD

# Start Xvfb for PyVista (for offscreen rendering)
# xvfb.start_xvfb(wait=0.05)


def plot_vector(u, plotter, subplot=None, scale=1.0, lineproperties={}):
    """
    Plots a vector field using PyVista with glyph representation.

    Args:
        u: Vector field to plot.
        plotter: The PyVista plotter object.
        subplot: Optional tuple to specify subplot coordinates.
        scale: Scale factor for the vector field.

    Returns:
        plotter: The updated PyVista plotter object.
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
    num_dofs_local = u.function_space.dofmap.index_map.size_local
    geometry = u.function_space.tabulate_dof_coordinates()[:num_dofs_local]
    values = np.zeros((V.dofmap.index_map.size_local, 3), dtype=np.float64)
    values[:, : mesh.geometry.dim] = u.x.petsc_vec.array.real.reshape(
        V.dofmap.index_map.size_local, V.dofmap.index_map_bs
    )
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    grid["vectors"] = values
    grid.set_active_vectors("vectors")
    glyphs = grid.glyph(orient="vectors", factor=scale)
    plotter.add_mesh(glyphs, **lineproperties)
    plotter.add_mesh(
        grid, show_edges=True, color="black", style="wireframe", opacity=0.3
    )
    plotter.view_xy()
    plotter.set_background("white")
    return plotter, grid


def plot_scalar(u, plotter, subplot=None, lineproperties={}):
    """
    Plots a scalar field using PyVista.

    Args:
        u: Scalar field to plot.
        plotter: The PyVista plotter object.
        subplot: Optional tuple to specify subplot coordinates.
        lineproperties: Dictionary to specify plot line properties.

    Returns:
        plotter: The updated PyVista plotter object.
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
    values = u.x.petsc_vec.array.real.reshape(
        V.dofmap.index_map.size_local, V.dofmap.index_map_bs
    )
    grid.point_data["u"] = values
    grid.set_active_scalars("u")
    plotter.add_mesh(grid, **lineproperties)
    plotter.view_xy()
    plotter.set_background("white")
    return plotter, grid


def plot_profile(
    u,
    points,
    plotter,
    subplot=None,
    lineproperties={},
    fig=None,
    ax=None,
    subplotnumber=1,
):
    """
    Plots a profile of the solution at given points.

    Args:
        u: Scalar or vector field to be evaluated.
        points: Points at which the field should be evaluated.
        plotter: PyVista plotter object.
        subplot: Optional tuple to specify subplot coordinates.
        lineproperties: Dictionary specifying line properties.
        fig: Matplotlib figure object, optional.
        ax: Matplotlib axis object, optional.
        subplotnumber: Subplot number for Matplotlib.

    Returns:
        tuple: Matplotlib plot object and profile data (x, y).
    """
    points_on_proc, u_values = get_datapoints(u, points)

    if fig is None:
        fig = plt.figure()

    if subplot:
        plt.subplot(subplot[0], subplot[1], subplotnumber)

    if ax is not None:
        ax.plot(points_on_proc[:, 0], u_values, **lineproperties)
        ax.legend()
    else:
        plt.plot(points_on_proc[:, 0], u_values, **lineproperties)
    plt.legend()
    return plt, (points_on_proc[:, 0], u_values)


def plot_mesh(mesh, ax=None):
    """
    Plots the mesh using Matplotlib.

    Args:
        mesh: Mesh object to plot.
        ax: Matplotlib axis object, optional.

    Returns:
        ax: Updated Matplotlib axis object.
    """
    if ax is None:
        ax = plt.gca()
    ax.set_aspect("equal")
    points = mesh.geometry.x
    cells = mesh.geometry.dofmap.reshape((-1, mesh.topology.dim + 1))
    tria = tri.Triangulation(points[:, 0], points[:, 1], cells)
    ax.triplot(tria, color="k")
    return ax


def get_datapoints(u, points):
    """
    Retrieves data points for a given solution field evaluated at specific points.

    Args:
        u: Scalar or vector field.
        points: Points at which the field should be evaluated.

    Returns:
        tuple: Points on processor and evaluated values of u at the points.
    """
    import dolfinx.geometry

    mesh = u.function_space.mesh
    cells = []
    bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    points_on_proc = []
    cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(
        mesh, cell_candidates, points.T
    )
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])

    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    u_values = u.eval(points_on_proc, cells)

    return points_on_proc, u_values


def plot_perturbations(comm, Lx, prefix, β, v, bifurcation, stability, i_t):
    """
    Plots the perturbations using PyVista.

    Args:
        comm: MPI communicator.
        Lx: Length of the domain in the x-direction.
        prefix: Directory prefix to save the plots.
        β: Perturbation field.
        v: Solution field.
        bifurcation: Bifurcation data object.
        stability: Stability object.
        i_t: Time step index.

    Returns:
        plotter: Updated PyVista plotter object.
    """
    from solvers.function import vec_to_functions

    vec_to_functions(bifurcation._spectrum[0]["xk"], [v, β])
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
            lineproperties={"c": "k", "label": "$\\beta$"},
        )
        _plt.gca()
        _plt.legend()
        _plt.fill_between(data[0], data[1].reshape(len(data[1])))
        _plt.title("Perturbation")
        _plt.savefig(f"{prefix}/perturbation-profile-{i_t}.png")
        _plt.close()

        plotter = pyvista.Plotter(
            title="Cone-Perturbation profile",
            window_size=[800, 600],
            shape=(1, 1),
        )

        _plt, data = plot_profile(
            stability.perturbation["beta"],
            points,
            plotter,
            subplot=(0, 0),
            lineproperties={"c": "k", "label": "$\\beta$"},
        )
        _plt.gca()
        _plt.legend()
        _plt.fill_between(data[0], data[1].reshape(len(data[1])))
        _plt.title("Perturbation from the Cone")
        _plt.savefig(f"{prefix}/perturbation-profile-cone-{i_t}.png")
        _plt.close()

    return plotter


def plot_matrix(M):
    """
    Plots the matrix as a heatmap using Matplotlib.

    Args:
        M: PETSc matrix to plot.

    Returns:
        fig: Matplotlib figure object with the matrix plot.
    """
    fig, ax = plt.subplots()
    indptr, indices, data = M.getValuesCSR()
    _M = scipy.sparse.csr_matrix((data, indices, indptr), shape=M.sizes[0])
    ax.matshow(_M.todense(), cmap=plt.cm.Blues)

    for i in range(_M.shape[0]):
        for j in range(_M.shape[0]):
            c = _M[j, i]
            ax.text(i, j, f"{c:.3f}", va="center", ha="center")

    return fig
