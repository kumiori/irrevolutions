
import numpy as np
import sys
from datetime import date

today = date.today()

sys.path.append("../")

import dolfinx
import ufl

from petsc4py import PETSc
import petsc4py

petsc4py.init(sys.argv)

opts = PETSc.Options()
# opts.setValue("help", 1)
import logging

from meshes.primitives import mesh_bar_gmshapi
# from meshes import gmsh_model_to_mesh

logging.basicConfig(level=logging.INFO)

from mpi4py import MPI

comm = MPI.COMM_WORLD
# import pdb
import pyvista
from pyvista.utilities import xvfb

import dolfinx.plot
from dolfinx.fem import Function


from utils.viz import (
    plot_scalar,
    plot_vector
)


def test_viz():
    Lx = 1.0
    Ly = 0.1
    _nel = 30
    gmsh_model, tdim = mesh_bar_gmshapi("bar", Lx, Ly, 1/_nel, 2)

    mesh, mts = gmsh_model_to_mesh(
        gmsh_model, cell_data=False, facet_data=True, gdim=2)

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
    u = Function(V_u, name="Displacement")
    u.interpolate(vect_2D)

    alpha = Function(V_alpha, name="Damage")
    alpha.interpolate(scal_2D)

    xvfb.start_xvfb(wait=0.05)
    pyvista.OFF_SCREEN = True

    plotter = pyvista.Plotter(
        title="Test Viz",
        window_size=[1600, 600],
        shape=(1, 2),
    )

    _plt = plot_scalar(alpha, plotter, subplot=(0, 0))
    logging.critical('plotted scalar')
    _plt = plot_vector(u, plotter, subplot=(0, 1))
    logging.critical('plotted vector')

    _plt.screenshot(f"./output/test_viz/test_viz_MPI{comm.size}-.png")

    if not pyvista.OFF_SCREEN:
        plotter.show()


if __name__ == "__main__":
    test_viz()
