from irrevolutions.meshes.boolean import create_disk_with_hole
from mpi4py import MPI
from dolfinx.io import XDMFFile, gmshio
import gmsh
import numpy as np
from dolfinx import mesh, fem
from mpi4py import MPI
import dolfinx.io


def create_arc_ring_mesh(inner_radius, outer_radius, angle=180, lc=0.1):
    """
    Create a 2D arc-shaped ring (annular sector) using Gmsh.

    Args:
        inner_radius (float): Inner radius of the arc.
        outer_radius (float): Outer radius of the arc.
        angle (float): Arc angle in degrees (e.g., 90 for a quarter ring).
        lc (float): Mesh characteristic length.

    Returns:
        Dolfinx mesh object.
    """
    gmsh.initialize()
    gmsh.model.add("ArcRing")

    # Convert angle to radians
    theta = np.radians(angle)

    # Create arc points
    _p0 = gmsh.model.occ.addPoint(0, 0, 0, lc)
    p0 = gmsh.model.occ.addPoint(inner_radius, 0, 0, lc)
    p1 = gmsh.model.occ.addPoint(outer_radius, 0, 0, lc)
    p2 = gmsh.model.occ.addPoint(
        inner_radius * np.cos(theta), inner_radius * np.sin(theta), 0, lc
    )
    p3 = gmsh.model.occ.addPoint(
        outer_radius * np.cos(theta), outer_radius * np.sin(theta), 0, lc
    )
    tdim = 2  # 2D mesh
    # Create arcs
    arc_inner = gmsh.model.occ.addCircleArc(p0, _p0, p2)
    arc_outer = gmsh.model.occ.addCircleArc(p1, _p0, p3)
    line1 = gmsh.model.occ.addLine(p0, p1)
    line2 = gmsh.model.occ.addLine(p2, p3)

    # Create loop & surface
    loop = gmsh.model.occ.addCurveLoop([arc_inner, line2, -arc_outer, -line1])
    surface = gmsh.model.occ.addPlaneSurface([loop])
    # Synchronize & generate mesh
    gmsh.model.occ.synchronize()
    model = gmsh.model
    surface_entities = [model[1] for model in model.getEntities(tdim)]

    domain = model.addPhysicalGroup(tdim, [surface], name="Domain")
    model.setPhysicalName(tdim, domain, "Surface")

    outer_boundary = gmsh.model.getBoundary([(2, surface)], oriented=False)
    fixed_edges = []
    for edge in outer_boundary:
        edge_id = edge[1]
        com = gmsh.model.occ.getCenterOfMass(1, edge_id)
        if np.isclose(com[1], 0.0, atol=1e-6):  # Select the bottom edge
            fixed_edges.append(edge_id)

    # **Define physical group for the bottom boundary**
    fixed_tag = gmsh.model.addPhysicalGroup(1, fixed_edges, name="Fixed_Boundary")
    gmsh.model.setPhysicalName(1, fixed_tag, "Fixed_Boundary")

    gmsh.model.mesh.generate(tdim)
    # Convert to Dolfinx mesh
    msh_file = "arc_ring.msh"
    gmsh.write(msh_file)
    # gmsh.finalize()
    tdim = 2  # 2D mesh

    # return dolfinx.io.gmshio.read_from_msh(msh_file, MPI.COMM_WORLD, 0, gdim=2)
    return gmsh.model if comm.rank == 0 else None, tdim


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    geom_params = {
        "R_outer": 1.0,  # Outer disk radius
        "R_inner": 0.3,  # Inner hole radius (set to 0.0 for no hole)
        "lc": 0.05,  # Mesh element size
        "a": 0.1,  # Half-width of the refined region (-a < x < a)
    }
    # create_disk_with_hole(comm, geom_params)
    # gmsh_model, tdim = create_disk_with_hole(comm, geom_params)
    # mesh, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh_model, comm, 0, gdim=2)
    # __import__("pdb").set_trace()

    model, _ = create_arc_ring_mesh(0.5, 1.0, angle=180, lc=0.05)
    mesh, cell_tags, facet_tags = gmshio.model_to_mesh(model, comm, 0, gdim=2)
    __import__("pdb").set_trace()
