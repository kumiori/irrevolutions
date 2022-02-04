#!/usr/bin/env python3

from mpi4py import MPI


def mesh_bar_gmshapi(name,
                     Lx,
                     Ly,
                     lc,
                     tdim,
                     order=1,
                     msh_file=None,
                     comm=MPI.COMM_WORLD):
    """
    Create mesh of 3d tensile test specimen according to ISO 6892-1:2019 using the Python API of Gmsh.
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
        p0 = model.geo.addPoint(0.0, 0.0, 0, lc, tag=0)
        p1 = model.geo.addPoint(Lx, 0.0, 0, lc, tag=1)
        p2 = model.geo.addPoint(Lx, Ly, 0.0, lc, tag=2)
        p3 = model.geo.addPoint(0, Ly, 0, lc, tag=3)
        # points = [p0, p1, p2, p3]
        bottom = model.geo.addLine(p0, p1, tag=0)
        right = model.geo.addLine(p1, p2, tag=1)
        top = model.geo.addLine(p2, p3, tag=2)
        left = model.geo.addLine(p3, p0, tag=3)
        cloop1 = model.geo.addCurveLoop([bottom, right, top, left])
        # surface_1 =
        model.geo.addPlaneSurface([cloop1])

        model.geo.synchronize()
        surface_entities = [model[1] for model in model.getEntities(tdim)]
        model.addPhysicalGroup(tdim, surface_entities, tag=5)
        model.setPhysicalName(tdim, 5, "Rectangle surface")

        # Set mesh size via points
        # gmsh.model.mesh.setSize(points, lc)  # heuristic

        # gmsh.model.mesh.optimize("Netgen")

        # Set geometric order of mesh cells
        gmsh.model.mesh.setOrder(order)

        # Define physical groups for subdomains (! target tag > 0)
        # domain = 1
        # gmsh.model.addPhysicalGroup(tdim, [v[1] for v in volumes], domain)
        # gmsh.model.setPhysicalName(tdim, domain, 'domain')
        gmsh.model.addPhysicalGroup(tdim - 1, [3], tag=6)
        gmsh.model.setPhysicalName(tdim - 1, 6, "left")
        gmsh.model.addPhysicalGroup(tdim - 1, [1], tag=7)
        gmsh.model.setPhysicalName(tdim - 1, 7, "right")
        gmsh.model.addPhysicalGroup(tdim - 1, [2], tag=8)
        gmsh.model.setPhysicalName(tdim - 1, 8, "top")
        gmsh.model.addPhysicalGroup(tdim - 1, [0], tag=9)
        gmsh.model.setPhysicalName(tdim - 1, 9, "bottom")

        model.mesh.generate(tdim)

        # Define physical groups for interfaces (! target tag > 0)
        # surface = 1
        # gmsh.model.addPhysicalGroup(tdim - 1, [s[1] for s in surfaces], surface)
        # gmsh.model.setPhysicalName(tdim - 1, surface, 'surface')
        """surface_grip_left = 2
        gmsh.model.addPhysicalGroup(tdim - 1, [s0], surface_grip_left)
        gmsh.model.setPhysicalName(tdim - 1, surface_grip_left, 'surface_grip_left')
        surface_grip_right = 3
        gmsh.model.addPhysicalGroup(tdim - 1, [s1], surface_grip_right)
        gmsh.model.setPhysicalName(tdim - 1, surface_grip_right, 'surface_grip_right')
        surface_plane_left = 4
        gmsh.model.addPhysicalGroup(tdim - 1, [s2], surface_plane_left)
        gmsh.model.setPhysicalName(tdim - 1, surface_plane_left, 'surface_plane_left')
        surface_plane_right = 5
        gmsh.model.addPhysicalGroup(tdim - 1, [s3], surface_plane_right)
        gmsh.model.setPhysicalName(tdim - 1, surface_plane_right, 'surface_plane_right')"""

        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)
            # gmsh.write(name + ".step")

    return gmsh.model if comm.rank == 0 else None, tdim


def mesh_circle_gmshapi(name,
                        R,
                        lc,
                        tdim,
                        order=1,
                        msh_file=None,
                        comm=MPI.COMM_WORLD):
    """
    Create 2d circle mesh using the Python API of Gmsh.
    """
    # Perform Gmsh work only on rank = 0

    if comm.rank == 0:
        import gmsh

        # Initialise gmsh and set options
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)

        gmsh.option.setNumber("Mesh.Algorithm", 6)
        model = gmsh.model()
        model.add("Circle")
        model.setCurrent("Circle")
        p0 = model.geo.addPoint(0.0, 0.0, 0, lc, tag=0)
        p1 = model.geo.addPoint(R, 0.0, 0, lc, tag=1)
        p2 = model.geo.addPoint(0.0, R, 0.0, lc, tag=2)
        p3 = model.geo.addPoint(-R, 0, 0, lc, tag=3)
        p4 = model.geo.addPoint(0, -R, 0, lc, tag=4)
        # points = [p0, p1, p2, p3]
        c1 = gmsh.model.geo.addCircleArc(p1, p0, p2)
        c2 = gmsh.model.geo.addCircleArc(p2, p0, p3)
        c3 = gmsh.model.geo.addCircleArc(p3, p0, p4)
        c4 = gmsh.model.geo.addCircleArc(p4, p0, p1)

        circle = model.geo.addCurveLoop([c1, c2, c3, c4])
        model.geo.addPlaneSurface([circle])

        model.geo.synchronize()
        surface_entities = [model[1] for model in model.getEntities(tdim)]
        model.addPhysicalGroup(tdim, surface_entities, tag=5)
        model.setPhysicalName(tdim, 5, "Film surface")

        gmsh.model.mesh.setOrder(order)

        model.mesh.generate(tdim)

        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)

    return gmsh.model if comm.rank == 0 else None, tdim


if __name__ == "__main__":
    import sys

    sys.path.append("../../damage")
    from xdmf import XDMFFile
    from mesh import gmsh_to_dolfin

    # , merge_meshtags, locate_dofs_topological
    from mpi4py import MPI
    from pathlib import Path
    import dolfinx.plot

    gmsh_model, tdim = mesh_bar_gmshapi("bar",
                                        1,
                                        0.1,
                                        0.01,
                                        2,
                                        msh_file="output/bar.msh")
    mesh, mts = gmsh_to_dolfin(gmsh_model, tdim, prune_z=True)
    Path("output").mkdir(parents=True, exist_ok=True)
    with XDMFFile(MPI.COMM_WORLD, "output/bar.xdmf", "w") as ofile:
        ofile.write_mesh_meshtags(mesh, mts)

    import pyvista
    from pyvista.utilities import xvfb

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
    figure = plotter.screenshot("output/bar.png")

def mesh_V(
a,
h,
L,
gamma,
de,
de2,
key=0,
show=False,
filename='mesh.unv',
order = 1,
):
    """
    Create a 2D mesh of a notched three-point flexure specimen using GMSH.
    a = height of the notch
    h = height of the specimen
    L = width of the specimen
    gamma = notch angle
    de = density of elements at specimen
    de2 = density of elements at the notch and crack
    key = 0 -> create model for Fenicxs (default)
          1 -> create model for Cast3M
    show = False -> doesn't open Gmsh to vizualise the mesh (default)
           True -> open Gmsh to vizualise the mesh
    filename = name and format of the output file for key = 1 
    order = order of the function of form
    """
    gmsh.initialize()
    #gmsh.option.setNumber("General.Terminal",1)
    #gmsh.option.setNumber("Mesh.Algorithm",5)
    hopen = a*np.tan((gamma/2.0)*np.pi/180)
    c0 = h/10
    tdim = 2 
    
    model = gmsh.model()
    model.add('TPB')
    model.setCurrent('TPB')
    #Generating the points of the geometrie
    p0 = model.geo.addPoint(0.0, a, 0.0, de2, tag=0)
    p1 = model.geo.addPoint(hopen, 0.0, 0.0, de, tag=1)
    p2 = model.geo.addPoint(L/2, 0.0, 0.0, de, tag=2)
    p3 = model.geo.addPoint(L/2, h, 0.0, de, tag=3)
    p4 = model.geo.addPoint(0.0, h, 0.0, de, tag=4)
    if key == 0:
        p5 = model.geo.addPoint(-L/2, h, 0.0, de, tag=5)
        p6 = model.geo.addPoint(-L/2, 0.0, 0.0, de, tag=6)
        p7 = model.geo.addPoint(-hopen, 0.0, 0.0, de, tag=7)
    elif key == 1:
        p20 = model.geo.addPoint(0, a+c0, 0, de2, tag=20)
    #Creating the lines by connecting the points
    notch_right = model.geo.addLine(p0, p1, tag=8) 
    bot_right = model.geo.addLine(p1, p2, tag=9)
    right = model.geo.addLine(p2, p3, tag=10)
    top_right = model.geo.addLine(p3, p4, tag=11)
    if key == 0:
        top_left = model.geo.addLine(p4, p5, tag=12)
        left = model.geo.addLine(p5, p6, tag=13)
        bot_left = model.geo.addLine(p6, p7, tag=14)
        notch_left = model.geo.addLine(p7, p0, tag=15)
    elif key == 1:
        sym_plan = model.geo.addLine(p4, p20, tag=21)
        fissure = model.geo.addLine(p20, p0, tag=22)
    #Creating the surface using the lines created
    if key == 0:
        perimeter = model.geo.addCurveLoop([notch_right, bot_right, right, top_right, top_left, left, bot_left, notch_left])
    elif key == 1:
        perimeter = model.geo.addCurveLoop([notch_right, bot_right, right, top_right, sym_plan, fissure])
    surface = model.geo.addPlaneSurface([perimeter])
    #model.geo.addSurfaceLoop([surface,16])
    model.mesh.setOrder(order)
    
    #Creating Physical Groups to extract data from the geometrie
    if key == 0:
        gmsh.model.addPhysicalGroup(tdim-1, [left], tag = 101)
        gmsh.model.setPhysicalName(tdim-1, 101,'Left')

        gmsh.model.addPhysicalGroup(tdim-1, [right], tag=102)
        gmsh.model.setPhysicalName(tdim-1, 102,'Right')

        gmsh.model.addPhysicalGroup(tdim-2, [p6], tag=103)
        gmsh.model.setPhysicalName(tdim-2, 103,'Left_point')

        gmsh.model.addPhysicalGroup(tdim-2, [p2], tag=104)
        gmsh.model.setPhysicalName(tdim-2, 104,'Right_point')

        gmsh.model.addPhysicalGroup(tdim-2, [p4], tag=105)
        gmsh.model.setPhysicalName(tdim-2, 105, 'Load_point')

        gmsh.model.addPhysicalGroup(tdim-2, [p0], tag=106)
        gmsh.model.setPhysicalName(tdim-2, 106, 'Notch_point')

        gmsh.model.addPhysicalGroup(tdim, [surface],tag=110)
        gmsh.model.setPhysicalName(tdim, 110, 'mesh_surface')
    
    if key == 1:
        gmsh.model.addPhysicalGroup(tdim, [surface],tag=110)
        gmsh.model.setPhysicalName(tdim, 110, 'mesh_surface')

        gmsh.model.addPhysicalGroup(tdim-1, [fissure], tag=111)
        gmsh.model.setPhysicalName(tdim-1, 111, 'fissure')

        gmsh.model.addPhysicalGroup(tdim-1, [sym_plan], tag=112)
        gmsh.model.setPhysicalName(tdim-1, 112, 'sym_plan')

        gmsh.model.addPhysicalGroup(tdim-2, [p20], tag=113)
        gmsh.model.setPhysicalName(tdim-2, 113, 'Crack_tip')

        gmsh.model.addPhysicalGroup(tdim-2, [p4], tag=114)
        gmsh.model.setPhysicalName(tdim-2, 114, 'Load_point')

        gmsh.model.addPhysicalGroup(tdim-2, [p2], tag=115)
        gmsh.model.setPhysicalName(tdim-2, 115,'Right_point')   
    #Generating the mesh
    model.geo.synchronize()
    model.mesh.generate(tdim)
    if show:
        gmsh.fltk.run()
    if key == 1:
        gmsh.write(filename)
    return gmsh.model