#!/usr/bin/env python3

from mpi4py import MPI


def mesh_bar_gmshapi(
    name, Lx, Ly, lc, tdim, order=1, msh_file=None, comm=MPI.COMM_WORLD
):
    """
    Create mesh of 3d tensile test specimen according to ISO 6892-1:2019 using the Python API of Gmsh.
    """
    # Perform Gmsh work only on rank = 0

    if comm.rank == 0:
        import gmsh

        # Initialise gmsh and set options
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)

        # gmsh.option.setNumber("Mesh.Algorithm", 6)

        gmsh.option.setNumber("Mesh.Algorithm", 5)
        gmsh.model.mesh.optimize("Netgen")
        model = gmsh.model()
        model.add("Rectangle")
        model.setCurrent("Rectangle")
        p0 = model.geo.addPoint(0.0, 0.0, 0, lc, tag=0)
        p1 = model.geo.addPoint(Lx, 0.0, 0, lc, tag=1)
        p2 = model.geo.addPoint(Lx, Ly, 0.0, lc, tag=2)
        p3 = model.geo.addPoint(0, Ly, 0, lc, tag=3)
        # points = [p0, p1, p2, p3]
        bottom = model.geo.addLine(p0, p1)
        right = model.geo.addLine(p1, p2)
        top = model.geo.addLine(p2, p3)
        left = model.geo.addLine(p3, p0)
        cloop1 = model.geo.addCurveLoop([bottom, right, top, left])
        # surface_1 =
        model.geo.addPlaneSurface([cloop1])

        model.geo.synchronize()
        surface_entities = [model[1] for model in model.getEntities(tdim)]
        model.addPhysicalGroup(tdim, surface_entities, tag=5)
        model.setPhysicalName(tdim, 5, "Rectangle surface")

        # Set geometric order of mesh cells
        gmsh.model.mesh.setOrder(order)
        gmsh.model.addPhysicalGroup(tdim - 1, [3, 0], tag=6)
        gmsh.model.setPhysicalName(tdim - 1, 6, "left")
        gmsh.model.addPhysicalGroup(tdim - 1, [1, 2], tag=7)
        gmsh.model.setPhysicalName(tdim - 1, 7, "right")
        gmsh.model.addPhysicalGroup(tdim - 1, [2, 3], tag=8)
        gmsh.model.setPhysicalName(tdim - 1, 8, "top")
        gmsh.model.addPhysicalGroup(tdim - 1, [0, 1], tag=9)
        gmsh.model.setPhysicalName(tdim - 1, 9, "bottom")

        model.mesh.generate(tdim)

        # Optional: Write msh file
        if msh_file is not None:
            os.makedirs(os.path.dirname(msh_file), exist_ok=True)
            gmsh.write(msh_file)

    return gmsh.model if comm.rank == 0 else None, tdim


if __name__ == "__main__":
    import os
    import sys

    sys.path.append("../../damage")
    import dolfinx.plot
    from dolfinx.io import XDMFFile
    from gmsh_mesh import gmsh_model_to_mesh

    # from mesh import gmsh_to_dolfin
    # , merge_meshtags, locate_dofs_topological
    from mpi4py import MPI

    gmsh_model, tdim = mesh_bar_gmshapi(
        "bar", 1, 0.1, 0.01, 2, msh_file="output/bar.msh"
    )
    mesh, cell_tags, facet_tags = gmsh_model_to_mesh(
        gmsh_model, cell_data=True, facet_data=True, gdim=2
    )

    xdmf_file = "output/bar.xdmf"
    os.makedirs(os.path.dirname(xdmf_file), exist_ok=True)

    with XDMFFile(MPI.COMM_WORLD, xdmf_file, "w") as ofile:
        ofile.write_mesh(mesh)
        ofile.write_meshtags(cell_tags)
        ofile.write_meshtags(facet_tags)

    import pyvista
    from pyvista.utilities import xvfb

    xvfb.start_xvfb(wait=0.05)
    pyvista.OFF_SCREEN = True
    plotter = pyvista.Plotter(title="Bar mesh")
    topology, cell_types = dolfinx.plot.create_vtk_topology(mesh, mesh.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, mesh.geometry.x)
    # plotter.subplot(0, 0)
    actor_1 = plotter.add_mesh(grid, show_edges=True)

    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        plotter.show()
    figure = plotter.screenshot("output/bar.png")
