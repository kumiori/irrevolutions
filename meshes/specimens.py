from mpi4py import MPI


def mesh_kink(name,
              Lx,
              Ly,
              lc,
              tdim,
              order=1,
              msh_file=None,
              comm=MPI.COMM_WORLD):
    """
    Create mesh of ...
    TODO: 
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
        # points = [p1, p2, p3, p4, p5, p6, p7, p8]
        p1 = model.geo.addPoint(0.0, 0.0, 0, lc, tag=1)
        p2 = model.geo.addPoint(Lx, 0.0, 0, lc, tag=2)
        p3 = model.geo.addPoint(Lx, Ly, 0.0, lc, tag=3)
        p4 = model.geo.addPoint(0, Ly, 0, lc, tag=4)
        p5 = model.geo.addPoint(11.2, 18.8, 0.0, lc, tag=5)
        p6 = model.geo.addPoint(7.5, 12.6, 0, lc, tag=6)
        p7 = model.geo.addPoint(3.75, 6.25, 0, lc, tag=7)
        p8 = model.geo.addPoint(7.5, 12.4, 0, lc, tag=8)
        #Les lignes = [L1, L2, L3, L4, L5, L6, L7, L8]
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
        model.setPhysicalName(tdim, 1, "Bottom  of the Plate")
        model.addPhysicalGroup(tdim, surface_entities, tag=4)
        model.setPhysicalName(tdim, 4, " Top  of the Plate")
        model.addPhysicalGroup(tdim, surface_entities, tag=5)
        model.setPhysicalName(tdim, 5, " 1/2 Top of the TipCrack")
        model.addPhysicalGroup(tdim, surface_entities, tag=6)
        model.setPhysicalName(tdim, 6, " 1/2 Top of the TipCrack")
        model.addPhysicalGroup(tdim, surface_entities, tag=7)
        model.setPhysicalName(tdim, 7, "1/2 bottom of the TipCrack")
        model.addPhysicalGroup(tdim, surface_entities, tag=8)
        model.setPhysicalName(tdim, 8, "1/2 bottom of the TipCrack")

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

# mesh=mesh_kink('mesh', 15, 25, 1, 2, 1,)
