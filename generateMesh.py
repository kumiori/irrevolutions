from random import seed
import numpy as np
import gmsh
import matplotlib.pyplot as plt
import meshes
from meshes import primitives
from utils.viz import plot_mesh#, plot_vector, plot_scalar

# Mesh
Lx = 100
Ly = 400
s=2
L0=30
seedDist=1

geom_type = "bar"

gmsh_model, tdim = primitives.mesh_ep_gmshapi(geom_type,
                                    Lx, 
                                    Ly,
                                    L0, 
                                    s,   
                                    seedDist, 
                                    sep=0.1,
                                    tdim=2)
"""
# Only needed when we are generating a mesh to validate the calculations
gmsh_model, tdim = primitives.mesh_rightCrack_gmshapi(geom_type,
                                    Lx, 
                                    Ly,
                                    L0, 
                                    s,   
                                    seedDist, 
                                    sep=0.1,
                                    tdim=2)
"""
mesh, mts = meshes.gmsh_model_to_mesh(gmsh_model,
                               cell_data=False,
                               facet_data=True,
                               gdim=2, 
                               exportMesh=True, 
                               fileName="twoCrack_ep2.unv")

plt.figure()
ax = plot_mesh(mesh)
fig = ax.get_figure()
fig.savefig(f"mesh.png")

