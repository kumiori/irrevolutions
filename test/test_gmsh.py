# ------------------------------------------------------------------------------
#
#  Gmsh Python tutorial 19
#
#  Thrusections, fillets, pipes, mesh size from curvature
#
# ------------------------------------------------------------------------------

# The OpenCASCADE geometry kernel supports several useful features for solid
# modelling.

import gmsh
import math
import os
import sys

gmsh.initialize()

gmsh.model.add("t19")
gmsh.logger.start()

# gmsh.model.occ.addCircle(0, 0, 0, 1, tag=10)
# gmsh.model.occ.add_ellipse(0, 0, 0, 1., .5, tag=11)
# gmsh.model.occ.addCurveLoop([11], 11)
# gmsh.model.occ.synchronize()

gmsh.model.occ.addDisk(0, 0, 0, 1.0, 1.0, tag=1)
gmsh.model.occ.addDisk(0.0, -0.3, 0.0, 1.3, 1.0, tag=2)
gmsh.model.occ.cut([(2, 1)], [(2, 2)], 3)
gmsh.model.occ.synchronize()

# We can activate the calculation of mesh element sizes based on curvature
# (here with a target of 20 elements per 2*Pi radians):
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)

# We can constraint the min and max element sizes to stay within reasonnable
# values (see `t10.py' for more details):
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.01)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.02)

gmsh.model.mesh.generate(2)
gmsh.write("t19.msh")
