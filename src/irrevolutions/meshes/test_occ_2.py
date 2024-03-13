import gmsh

gmsh.initialize()

# Create a new model
gmsh.model.add("test")

# Create a rectangle
rectangle = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)

# Create a disk
disk = gmsh.model.occ.addDisk(0.5, 0.5, 0, 0.2, 0.2)

# Create a triangle
point1 = gmsh.model.occ.addPoint(0.2, 0.2, 0)
point2 = gmsh.model.occ.addPoint(0.4, 0.2, 0)
point3 = gmsh.model.occ.addPoint(0.4, 0.4, 0)
line1 = gmsh.model.occ.addLine(point1, point2)
line2 = gmsh.model.occ.addLine(point2, point3)
line3 = gmsh.model.occ.addLine(point3, point1)

triangle_loop = gmsh.model.occ.addCurveLoop([line1, line2, line3])
triangle = gmsh.model.occ.addPlaneSurface([triangle_loop])
# Perform boolean operations
boolean = gmsh.model.occ.cut([(2, rectangle)], [(2, disk)])
boolean = gmsh.model.occ.cut([(2, rectangle)], [(2, triangle)])
# boolean_with_triangle = gmsh.model.occ.cut([2, boolean], [(2, triangle)], tag=4)
# boolean_with_triangle = gmsh.model.occ.cut([(2, rectangle)], [(2, triangle)])
# boolean_with_triangle = gmsh.model.occ.cut([(2, boolean_with_triangle)], [(2, triangle)])

# Generate mesh
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)

# Write the mesh file
gmsh.write("mesh.msh")

# Finalize Gmsh
gmsh.finalize()
