#!/usr/bin/env python3
from utils.viz import plot_mesh
import matplotlib.pyplot as plt
from meshes import gmsh_model_to_mesh
from meshes.V_notch_2D import mesh_V_notch
import numpy as np
import sys
from mpi4py import MPI
import petsc4py
from dolfinx import log
import ufl

sys.path.append("../")

# sys.path.append("../../damage")
# from mesh import gmsh_to_dolfin, merge_meshtags

petsc4py.init(sys.argv)
log.set_log_level(log.LogLevel.WARNING)


comm = MPI.COMM_WORLD


# Get mesh parameters
L = 1.0
L_crack = L / 2.0
theta = np.pi / 4.0
ell_ = 0.1
lc = ell_ / 3.0
geom_type = "notch"


# Create the mesh of the specimen with given dimensions
gmsh_model, tdim, tag_names = mesh_V_notch(
    geom_type, L, L_crack, theta, lc, msh_file="output/v-notch2d.msh"
)

# Get mesh and meshtags
mesh, cell_tags, facet_tags = gmsh_model_to_mesh(
    gmsh_model, cell_data=True, facet_data=True, gdim=2
)

# domains_keys = tag_names["cells"]

# Set meshtags index
# interfaces, interfaces_keys = merge_meshtags(mts, tdim - 1)
dx = ufl.Measure("dx", subdomain_data=cell_tags, domain=mesh)
ds = ufl.Measure("ds", subdomain_data=facet_tags, domain=mesh)

plt.figure()
ax = plot_mesh(mesh)
fig = ax.get_figure()
fig.savefig(f"output/v-notch-mesh.png")
