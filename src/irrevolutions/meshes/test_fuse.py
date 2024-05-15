import os
import gmsh
import pytest

# Define the test parameters
TEST_CASES = [
    {"name": "minimal_2d", "units": 1, "expected_mesh_size": 2}
    # Add more test cases if needed
]

@pytest.mark.parametrize("test_case", TEST_CASES)
def test_generate_2d_mesh(test_case):
    # Run the script
    gmsh.initialize()
    gmsh.model.add("simple_2d")
    units = test_case["units"]
    rectangle_1 = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, units, 0.5*units, tag=1)
    rectangle_2 = gmsh.model.occ.addRectangle(0.0, 0.5*units, 0.0, units, 0.5*units, tag=2)
    rectangles = gmsh.model.occ.fuse([(2,1)], [(2,2)], tag=3, removeObject=True, removeTool=True)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [3], 1)
    gmsh.model.setPhysicalName(2, 1, "rectangles")
    gmsh.model.mesh.generate(1)
    gmsh.model.mesh.generate(2)

    # Verify the generated mesh
    mesh_file_path = os.path.join(os.path.dirname(__file__), "output", f"{test_case['name']}_mesh.msh")
    gmsh.write(mesh_file_path)

    assert os.path.isfile(mesh_file_path)

    # Clean up
    gmsh.finalize()
