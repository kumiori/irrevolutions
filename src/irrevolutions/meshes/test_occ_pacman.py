import gmsh
import numpy as np
import os
import pytest
import sys

def create_triangle_with_angle(opening_deg, rotation = 0):

    gmsh.initialize(sys.argv)

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
    
    gmsh.model.mesh.generate(2)
    output_file = os.path.join(os.path.dirname(__file__), "output", "triangle_with_angle.msh")
    gmsh.write(output_file)

    gmsh.finalize()
    return output_file
    # gmsh.model.occ.rotate([(2, triangle_surface)], base_point1, 0, 0, 1, angle_radians)

    # Generate mesh
    # gmsh.model.occ.synchronize()
    # gmsh.model.mesh.generate(2)
    # Write the mesh file
    # gmsh.write("triangle_with_angle.msh")

    # Finalize Gmsh
    # gmsh.finalize()
    return domain

def create_pacman(opening_deg, rotation = 0):
    gmsh.initialize()
    _entities = []
    # Create a new model
    gmsh.model.add("triangle_with_angle")
    occ = gmsh.model.occ
    _r1 = 1.
    _r2 = 1.3
    adjacent_length = 2
    # Convert angle to radians
    angle_radians = opening_deg * (np.pi / 180.0)
    p0 = [0, 0, 0]
    # Define the base side of the triangle
    base_point1 = gmsh.model.occ.addPoint(*p0)
    base_point2 = gmsh.model.occ.addPoint(adjacent_length, 0, 0)
    base_line = gmsh.model.occ.addLine(base_point1, base_point2)

    # Define the vertex of the triangle
    vertex_point = gmsh.model.occ.addPoint(adjacent_length * np.cos(angle_radians), 
                                           adjacent_length * np.sin(angle_radians), 0)

    # define the lines of the triangle
    
    side_line_1 = gmsh.model.occ.addLine(base_point1, vertex_point)
    side_line_2 = gmsh.model.occ.addLine(vertex_point, base_point2)
    
    # Define the triangle loop
    triangle_loop = gmsh.model.occ.addCurveLoop([base_line, side_line_1, side_line_2])

    # Define the triangle surface
    triangle_surface = gmsh.model.occ.addPlaneSurface([triangle_loop]); _entities.append(triangle_surface)
    _rotation_axis = [0, 0, 1]
    occ.rotate([(2, triangle_surface)], *p0, *_rotation_axis, -angle_radians/2)

    disk = occ.addDisk(*p0, _r1, _r1); _entities.append(disk)
    disk_ext = occ.addDisk(*p0, _r2, _r2); _entities.append(disk_ext)
    
    _machine_tag = 300
    _machine = gmsh.model.occ.cut([(2, disk_ext)], [(2, disk)], tag=_machine_tag, removeObject = True, removeTool = False)
    machine = gmsh.model.occ.cut([(2, _machine_tag)], [(2, triangle_surface)], removeObject = True, removeTool = False)

    pacman = occ.cut([(2, disk)], [(2, triangle_surface)])


    occ.synchronize()
    entities = occ.getEntities()
    
    
    domain = gmsh.model.addPhysicalGroup(2, [2])
    outside = gmsh.model.addPhysicalGroup(2, [300])
    
    gmsh.model.setPhysicalName(2, domain, "Pacman")
    gmsh.model.setPhysicalName(2, outside, "TractionMachine")
    
    # remove()
    # triangle_surface
    
    

    
    # boolean = gmsh.model.occ.intersect([(2, disk_ext)], [(2, disk)], tag=300)
    # boolean = gmsh.model.occ.cut([(2, disk_ext)], [(2, triangle_surface)], tag=30)
    
    # gmsh.model.occ.rotate([(2, triangle_surface)], base_point1, 0, 0, 1, angle_radians)
    # Generate mesh
    occ.synchronize()
    
    gmsh.model.mesh.generate(2)
    output_file = os.path.join(os.path.dirname(__file__), "output", "pacman.msh")
    gmsh.write(output_file)

    gmsh.finalize()
    return output_file


def test_create_triangle_with_angle():
    output_file = create_triangle_with_angle(opening_deg=30, rotation=0)
    assert os.path.isfile(output_file)

def test_create_pacman():
    output_file = create_pacman(opening_deg=30, rotation=180-30/2)
    assert os.path.isfile(output_file)

if __name__ == "__main__":
    pytest.main(args=[__file__])