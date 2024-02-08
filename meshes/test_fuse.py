import gmsh

# initialise gmsh engine
gmsh.initialize()

# assign name to geomtry
gmsh.model.add("simple_2d")

units = 1
rectangle_1 = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, units, 0.5*units, tag=1)
rectangle_2 = gmsh.model.occ.addRectangle(0.0, 0.5*units, 0.0, units, 0.5*units, tag=2)
# link both domains and remove old rectangles
rectangles = gmsh.model.occ.fuse([(2,1)], [(2,2)], tag=3, removeObject=True, removeTool=True)

# sycrhonise geometry with gmsh
gmsh.model.occ.synchronize()

# create groups
gmsh.model.addPhysicalGroup(2, [3], 1)
gmsh.model.setPhysicalName(2, 1, "rectangles")

# gnerate 2D mesh, write mesh and convert to xdmf
gmsh.model.mesh.generate(1)
gmsh.model.mesh.generate(2)
gmsh.write("mesh_2d_minimal.msh")

# finalize gmsh engine
gmsh.finalize()