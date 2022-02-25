from mpi4py import MPI
import numpy as np


def mesh_kink(name,
                Lx,
                Ly,
                # a,
                # b,
               rad,
               theta,
              # theta
                eta,
                lc,
                tdim,
                order=1,
                msh_file=None,
                comm=MPI.COMM_WORLD):
    """
    Create mesh, with a kink
    """
    # Perform Gmsh work only on rank = 0
    
    if comm.rank == 0:

        import gmsh

        # Initialise gmsh and set options
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)

        gmsh.option.setNumber("Mesh.Algorithm", 6)
        model = gmsh.model()
        model.add("Rectangle")
        model.setCurrent("Rectangle")
        # points = [p1, p2, p3, p4, p5, p6, p7, p8,p9,p10,p11]
        p1 = model.geo.addPoint(0.0, 0.0, 0, lc, tag=1)
        p2 = model.geo.addPoint(Lx, 0.0, 0, lc, tag=2)
        p3 = model.geo.addPoint(Lx, Ly, 0.0, lc, tag=3)
        p4 = model.geo.addPoint(0, Ly, 0, lc, tag=4)

        p5 = model.geo.addPoint(Lx/2 + rad*np.cos(theta), Ly/2 + rad*np.cos(theta), 0.0, lc/3, tag=5)
        p7 = model.geo.addPoint(Lx/2 - rad*np.cos(theta), Ly/2 - rad*np.cos(theta), 0.0, lc/3, tag=7)

        # p5 = model.geo.addPoint(Lx/2 +a, Ly/2 +b,0.0, lc/1.5, tag=5)
        p6 = model.geo.addPoint(Lx/2, Ly/2 + eta,0, lc, tag=6)
        # p7 = model.geo.addPoint(Lx/2-a,Ly/2-b,0,lc/1.5, tag=7)
        p8 = model.geo.addPoint(Lx/2,Ly/2 - eta,0, lc, tag=8)

        # Lines = [L1, L2, L3, L4, L5, L6, L7, L8]
        L1 = model.geo.addLine(p1, p2, tag=1)
        L2 = model.geo.addLine(p2, p3, tag=2)
        L3 = model.geo.addLine(p3, p4, tag=3)
        L4 = model.geo.addLine(p4, p1, tag=4)
        L5 = model.geo.addLine(p5, p6, tag=5)
        L6 = model.geo.addLine(p6, p7, tag=6)
        L7 = model.geo.addLine(p7, p8, tag=7)
        L8 = model.geo.addLine(p8, p5, tag=8)


        
        cloop1 = model.geo.addCurveLoop([L1, L2, L3, L4])
        cloop2 = model.geo.addCurveLoop([L5, L6, L7, L8])
        
        # surface_1 =
        model.geo.addPlaneSurface([cloop1, cloop2])

        model.geo.synchronize()
        surface_entities = [model[1] for model in model.getEntities(tdim)]

        model.addPhysicalGroup(tdim, surface_entities, tag=1)
        model.setPhysicalName(tdim, 1, "Rectangle surface")

        # Set mesh size via points
        # gmsh.model.mesh.setSize(points, lc)  # heuristic

        # gmsh.model.mesh.optimize("Netgen")

        # Set geometric order of mesh cells
        gmsh.model.mesh.setOrder(order)

        # Define physical groups for subdomains (! target tag > 0)
        # domain = 1
        # gmsh.model.addPhysicalGroup(tdim, [v[1] for v in volumes], domain)
        # gmsh.model.setPhysicalName(tdim, domain, 'domain')
              

        gmsh.model.addPhysicalGroup(tdim - 1, [1], tag=9)
        gmsh.model.setPhysicalName(tdim - 1, 9, "Bottom  of the Plate")
        gmsh.model.addPhysicalGroup(tdim - 1, [4], tag=10)
        gmsh.model.setPhysicalName(tdim - 1, 10, "Top  of the Plate")
        gmsh.model.addPhysicalGroup(tdim - 1, [5], tag=11)
        gmsh.model.setPhysicalName(tdim - 1, 11, "1/2 Top of the TipCrack")
        gmsh.model.addPhysicalGroup(tdim - 1, [6], tag=12)
        gmsh.model.setPhysicalName(tdim - 1, 12, "1/2 Top of the TipCrack")
        gmsh.model.addPhysicalGroup(tdim - 1, [7], tag=13)
        gmsh.model.setPhysicalName(tdim - 1, 13, "1/2 bottom of the TipCrack")
        gmsh.model.addPhysicalGroup(tdim - 1, [8], tag=14)
        gmsh.model.setPhysicalName(tdim - 1, 14, "1/2 bottom of the TipCrack")

        model.mesh.generate(tdim)


        #Generating the mesh
    model.geo.synchronize()
    model.mesh.generate(tdim)

    return gmsh.model


if __name__ == "__main__":
    import sys

    sys.path.append("../")
    from dolfinx.io import XDMFFile
    from meshes import gmsh_model_to_mesh
    
    # , merge_meshtags, locate_dofs_topological
    from mpi4py import MPI
    from pathlib import Path
    import dolfinx.plot

    import pyvista
    from pyvista.utilities import xvfb
    import numpy as np

    
    parameters = {
        'geometry': {
            'geom_type': 'Plate with Kink',
            'Lx': .6,
            'Ly': 1,
            'L0': 0.3,
            'theta': 30,
            'eta': 1e-2,
        },
    }
    
    Lx = parameters.get("geometry").get("Lx")
    Ly = parameters.get("geometry").get("Ly")
    L0 = parameters.get("geometry").get("L0")
    eta = parameters.get("geometry").get("eta")

    geom_type = parameters.get("geometry").get("geom_type")

    lc = 0.05
    theta = 45/180 * np.pi

    gmsh_model = mesh_kink('mesh', Lx, Ly, L0/2, theta, eta, lc, 2, 1)

    mesh, mts = gmsh_model_to_mesh(gmsh_model,
                                        cell_data=False,
                                        facet_data=True,
                                        gdim=2)



    xvfb.start_xvfb(wait=0.05)
    pyvista.OFF_SCREEN = True
    plotter = pyvista.Plotter(title="Bar mesh")
    topology, cell_types = dolfinx.plot.create_vtk_topology(
        mesh, mesh.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, mesh.geometry.x)
    # plotter.subplot(0, 0)
    actor_1 = plotter.add_mesh(grid, show_edges=True)

    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        plotter.show()
    figure = plotter.screenshot("output/thekink.png")
