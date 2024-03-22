import gmsh
import sys
import os
import pytest

def generate_square_with_cracks():

    gmsh.initialize(sys.argv)

    gmsh.model.add("square with cracks")

    surf1 = 1
    gmsh.model.occ.addRectangle(0, 0, 0, 1, 1, surf1)

    pt1 = gmsh.model.occ.addPoint(0.2, 0.2, 0)
    pt2 = gmsh.model.occ.addPoint(0.4, 0.4, 0)
    line1 = gmsh.model.occ.addLine(pt1, pt2)
    pt3 = gmsh.model.occ.addPoint(0.4, 0.4, 0)
    pt4 = gmsh.model.occ.addPoint(0.4, 0.9, 0)
    line2 = gmsh.model.occ.addLine(pt3, pt4)

    o, m = gmsh.model.occ.fragment([(2, surf1)], [(1, line1), (1, line2)])
    gmsh.model.occ.synchronize()

    # m contains, for each input entity (surf1, line1 and line2), the child entities
    # (if any) after the fragmentation, as lists of tuples. To apply the crack
    # plugin we group all the intersecting lines in a physical group

    new_surf = m[0][0][1]
    new_lines = [item[1] for sublist in m[1:] for item in sublist]

    gmsh.model.addPhysicalGroup(2, [new_surf], 100)
    gmsh.model.addPhysicalGroup(1, new_lines, 101)

    gmsh.model.mesh.generate(2)

    gmsh.plugin.setNumber("Crack", "Dimension", 1)
    gmsh.plugin.setNumber("Crack", "PhysicalGroup", 101)
    gmsh.plugin.setNumber("Crack", "DebugView", 1)
    gmsh.plugin.run("Crack")

    # save all the elements in the mesh (even those that do not belong to any
    # physical group):
    gmsh.option.setNumber("Mesh.SaveAll", 1)
    output_file = os.path.join(os.path.dirname(__file__), "output", "crack.msh")

    gmsh.finalize()

    return output_file

def test_generate_square_with_cracks():
    output_file = generate_square_with_cracks()
    assert os.path.isfile(output_file)
    
if __name__ == "__main__":
    pytest.main(args=[__file__])