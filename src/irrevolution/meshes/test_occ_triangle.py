import gmsh
import numpy as np


def create_triangle_with_angle(opening_deg, rotation = 0):

    # Create a new model
    gmsh.model.add("triangle_with_angle")

    # Convert angle to radians
    angle_radians = opening_deg * (np.pi / 180.0)
    p0 = [0, 0, 0]
    # Define the base side of the triangle
    base_point1 = gmsh.model.occ.addPoint(*p0)
    base_point2 = gmsh.model.occ.addPoint(1, 0, 0)
    base_line = gmsh.model.occ.addLine(base_point1, base_point2)

    adjacent_length = 1

    # Define the vertex of the triangle
    vertex_point = gmsh.model.occ.addPoint(adjacent_length * np.cos(angle_radians), 
                                           adjacent_length * np.sin(angle_radians), 0)

    # define the lines of the triangle
    
    side_line_1 = gmsh.model.occ.addLine(base_point1, vertex_point)
    side_line_2 = gmsh.model.occ.addLine(vertex_point, base_point2)
    
    # Define the triangle loop
    triangle_loop = gmsh.model.occ.addCurveLoop([base_line, side_line_1, side_line_2])

    # Define the triangle surface
    triangle_surface = gmsh.model.occ.addPlaneSurface([triangle_loop])
    _rotation_axis = [0, 0, 1]
    domain = gmsh.model.occ.rotate([(2, triangle_surface)], *p0, *_rotation_axis, angle_radians)
    
    # gmsh.model.occ.rotate([(2, triangle_surface)], base_point1, 0, 0, 1, angle_radians)

    # Generate mesh
    # gmsh.model.occ.synchronize()
    # gmsh.model.mesh.generate(2)
    __import__('pdb').set_trace()
    # Write the mesh file
    # gmsh.write("triangle_with_angle.msh")

    # Finalize Gmsh
    # gmsh.finalize()
    return domain

def create_pacman(opening_deg, rotation = 0):
    gmsh.initialize()

    # Create a new model
    gmsh.model.add("triangle_with_angle")
    occ = gmsh.model.occ
    _r1 = 1.
    _r2 = 1.3
    # Convert angle to radians
    angle_radians = opening_deg * (np.pi / 180.0)
    p0 = [0, 0, 0]
    # Define the base side of the triangle
    base_point1 = gmsh.model.occ.addPoint(*p0)
    base_point2 = gmsh.model.occ.addPoint(1, 0, 0)
    base_line = gmsh.model.occ.addLine(base_point1, base_point2)

    adjacent_length = 1

    # Define the vertex of the triangle
    vertex_point = gmsh.model.occ.addPoint(adjacent_length * np.cos(angle_radians), 
                                           adjacent_length * np.sin(angle_radians), 0)

    # define the lines of the triangle
    
    side_line_1 = gmsh.model.occ.addLine(base_point1, vertex_point)
    side_line_2 = gmsh.model.occ.addLine(vertex_point, base_point2)
    
    # Define the triangle loop
    triangle_loop = gmsh.model.occ.addCurveLoop([base_line, side_line_1, side_line_2])

    # Define the triangle surface
    triangle_surface = gmsh.model.occ.addPlaneSurface([triangle_loop])
    _rotation_axis = [0, 0, 1]
    domain = gmsh.model.occ.rotate([(2, triangle_surface)], *p0, *_rotation_axis, angle_radians)
    
    disk = occ.addDisk(*p0, 0.2, 0.2)
    __import__('pdb').set_trace()
    
    # gmsh.model.occ.rotate([(2, triangle_surface)], base_point1, 0, 0, 1, angle_radians)
    # Generate mesh
    # gmsh.model.occ.synchronize()
    # gmsh.model.mesh.generate(2)
    # Write the mesh file
    # gmsh.write("triangle_with_angle.msh")

    # Finalize Gmsh
    return domain

# Example: Create a triangle with a 45-degree angle

if __name__ == "__main__":
    gmsh.initialize()
    
    # domain = create_triangle_with_angle(opening_deg = 30, rotation = 0)
    domain = create_pacman(opening_deg = 30, rotation = 0)
    
    
    gmsh.finalize()
    
    pass