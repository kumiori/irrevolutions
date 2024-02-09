import gmsh
import sys

gmsh.initialize()

import warnings
warnings.filterwarnings("ignore")
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.option.setNumber("Mesh.Algorithm", 6)

model = gmsh.model
occ = model.occ

gmsh.model.add("test")

# Create a rectangle
rectangle = occ.addRectangle(0, 0, 0, 1, 1, 0)
disk = occ.addDisk(0.5, 0.5, 0, 0.2, 0.2)
# triangle = gmsh.model.occ.addWedge(x = 0, y = 0, z = 0, dx = 1, dy = .1, dz = 0, tag = -1)
gmsh.model.occ.addWedge(0, 0, 0, 0.4, 0.2, 0.1)

boolean = gmsh.model.occ.cut([(2, rectangle)], [(2, disk)], tag=30)

# Generate mesh
model.occ.synchronize()

gmsh.model.mesh.generate(2)

gmsh.write("mesh.msh")
gmsh.finalize()

