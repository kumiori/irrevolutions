#!/usr/bin/env python3

from mpi4py import MPI
import numpy as np


def mesh_pacman(
    name,
    geom_parameters,
    tdim=2,
    order=1,
    msh_file=None,
    comm=MPI.COMM_WORLD,
):
    """
    Create mesh of 2d pacman specimen according to ... using the Python API of Gmsh.
    """
    # Perform Gmsh work only on rank = 0

    if comm.rank == 0:
        import numpy as np
        import gmsh
        import warnings
        warnings.filterwarnings("ignore")
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        # model = gmsh.model()
        gmsh.model.add("pacman")
        # geom_parameters = {'omega': np.pi/4, 'r': 1, 'lc': 0.1}

        omega = np.deg2rad(geom_parameters.get("omega"))
        radius = geom_parameters.get("r")
        lc = geom_parameters.get("lc")
        elltomesh = geom_parameters.get("elltomesh")

        refinement = geom_parameters.get("refinement")

        # refinement = geom_parameters.get("refinement")
        # R1=1.; R2=2.3; R3=1.; ex=0.; ey=-.3
        model = gmsh.model
        # model.occ.addDisk(0, 0, 0, R, R, tag=10)


        # print("Model name: " + gmsh.model.getCurrent())

        # get all elementary entities in the model
        entities = gmsh.model.occ.getEntities()


        # for e in entities:
        #     print("Entity " + str(e) + " of type " + gmsh.model.getType(e[0], e[1]))
        #     # get the mesh nodes for each elementary entity
        #     nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes(e[0], e[1])
        #     # get the mesh elements for each elementary entity
        #     elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(e[0], e[1])
        #     # count number of elements
        #     numElem = sum(len(i) for i in elemTags)
        #     print(" - mesh has " + str(len(nodeTags)) + " nodes and " + str(numElem) +
        #         " elements")
        #     boundary = gmsh.model.occ.getBoundary([e])
        #     print(" - boundary entities " + str(boundary))
        #     partitions = gmsh.model.occ.getPartitions(e[0], e[1])


        # print(entities)
        p0 = model.geo.addPoint(0, 0, 0, lc/refinement, tag=0)
        p1 = model.geo.addPoint( - radius*np.cos(omega / 2), radius*np.sin(omega / 2), 0.0, lc, tag=1)
        p2 = model.geo.addPoint( - radius*np.cos(omega / 2), - radius*np.sin(omega / 2), 0.0, lc, tag=2)
        p3 = model.geo.addPoint(radius, 0, 0.0, lc/refinement, tag=12)

        top = model.geo.addLine(p1, p0, tag=3)
        bot = model.geo.addLine(p0, p2, tag=4)
        arc1 = model.geo.addCircleArc(2, 0, 12, tag=5)
        arc2 = model.geo.addCircleArc(12, 0, 1, tag=6)
        cloop = model.geo.addCurveLoop([top, bot, arc1, arc2])


        s = model.geo.addPlaneSurface([cloop])
        model.geo.addSurfaceLoop([s, 1000])
        model.geo.synchronize()

        _n = 10
        refinement_pts = [model.geo.addPoint(radius * i/(_n), 0, 0.0,
            lc/refinement, 
            tag=111+i) for i in range(1,_n)]

        gmsh.model.geo.synchronize()

        # gmsh.model.mesh.embed(0, [refinement_pt], 2, s)
        gmsh.model.mesh.embed(0, refinement_pts, 2, s)

        surface_entities = [model[1] for model in model.getEntities(tdim)]
        domain = model.addPhysicalGroup(tdim, surface_entities)
        model.setPhysicalName(tdim, domain, "Surface")
        gmsh.model.mesh.setOrder(order)

        gmsh.model.addPhysicalGroup(tdim - 1, [5], tag=20)
        gmsh.model.setPhysicalName(tdim - 1, 20, "dirichlet_boundary")


        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)

        # We can constrain resolution
        # values (see `t10.py' for more details):
        gmsh.option.setNumber("Mesh.MeshSizeMin", lc/3.)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 2*lc)

        gmsh.model.mesh.generate(tdim)
        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)

    return gmsh.model if comm.rank == 0 else None, tdim


def mesh_embedded_pacman(
    name,
    geom_parameters,
    tdim=2,
    order=1,
    msh_file=None,
    comm=MPI.COMM_WORLD,
):
    """
    Create mesh of 2d pacman specimen according to ... using the Python API of Gmsh.
    with embedded crack (for potential computations ;)
    """
    # Perform Gmsh work only on rank = 0

    if comm.rank == 0:
        import numpy as np
        import gmsh
        import warnings
        warnings.filterwarnings("ignore")
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        # model = gmsh.model()
        gmsh.model.add("pacman")
        # geom_parameters = {'omega': np.pi/4, 'r': 1, 'lc': 0.1}

        omega = np.deg2rad(geom_parameters.get("omega"))
        radius = geom_parameters.get("r")
        lc = geom_parameters.get("lc")
        elltomesh = geom_parameters.get("elltomesh")

        refinement = geom_parameters.get("refinement")
        l0 = geom_parameters.get("l0")

        # refinement = geom_parameters.get("refinement")
        # R1=1.; R2=2.3; R3=1.; ex=0.; ey=-.3
        model = gmsh.model
        # model.occ.addDisk(0, 0, 0, R, R, tag=10)


        # print("Model name: " + gmsh.model.getCurrent())

        # get all elementary entities in the model
        entities = gmsh.model.occ.getEntities()


        # for e in entities:
        #     print("Entity " + str(e) + " of type " + gmsh.model.getType(e[0], e[1]))
        #     # get the mesh nodes for each elementary entity
        #     nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes(e[0], e[1])
        #     # get the mesh elements for each elementary entity
        #     elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(e[0], e[1])
        #     # count number of elements
        #     numElem = sum(len(i) for i in elemTags)
        #     print(" - mesh has " + str(len(nodeTags)) + " nodes and " + str(numElem) +
        #         " elements")
        #     boundary = gmsh.model.occ.getBoundary([e])
        #     print(" - boundary entities " + str(boundary))
        #     partitions = gmsh.model.occ.getPartitions(e[0], e[1])


        # print(entities)
        p0 = model.geo.addPoint(0, 0, 0, lc/refinement, tag=0)
        p1 = model.geo.addPoint( - radius*np.cos(omega / 2), radius*np.sin(omega / 2), 0.0, lc, tag=1)
        p2 = model.geo.addPoint( - radius*np.cos(omega / 2), - radius*np.sin(omega / 2), 0.0, lc, tag=2)
        p3 = model.geo.addPoint(radius, 0, 0.0, lc/refinement, tag=12)

        top = model.geo.addLine(p1, p0, tag=3)
        bot = model.geo.addLine(p0, p2, tag=4)
        arc1 = model.geo.addCircleArc(2, 0, 12, tag=5)
        arc2 = model.geo.addCircleArc(12, 0, 1, tag=6)
        cloop = model.geo.addCurveLoop([top, bot, arc1, arc2])


        s = model.geo.addPlaneSurface([cloop])
        model.geo.addSurfaceLoop([s, 1000])
        # model.geo.synchronize()

        _cracktip = model.geo.addPoint(l0, 0, 0.0, lc/refinement, tag=111)
        crack = model.geo.addLine(0, _cracktip, tag=30)

        # synchronize
        model.geo.synchronize()

        # embed "crack" on plate
        model.mesh.embed(1, [crack], 2, s)

        surface_entities = [model[1] for model in model.getEntities(tdim)]
        domain = model.addPhysicalGroup(tdim, surface_entities)
        model.setPhysicalName(tdim, domain, "Surface")
        gmsh.model.mesh.setOrder(order)

        gmsh.model.addPhysicalGroup(tdim - 1, [5], tag=20)
        gmsh.model.setPhysicalName(tdim - 1, 20, "dirichlet_boundary")


        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)

        # We can constrain resolution
        # values (see `t10.py' for more details):
        gmsh.option.setNumber("Mesh.MeshSizeMin", lc/3.)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 2*lc)

        gmsh.model.mesh.generate(tdim)
        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)

    return gmsh.model if comm.rank == 0 else None, tdim


def mesh_cut_pacman(
    name,
    geom_parameters,
    tdim=2,
    order=1,
    msh_file=None,
    comm=MPI.COMM_WORLD,
):
    """
    Create mesh of 2d pacman specimen according to ... using the Python API of Gmsh.
    with embedded crack (for potential computations ;)
    """
    # Perform Gmsh work only on rank = 0

    if comm.rank == 0:
        import numpy as np
        import gmsh
        import warnings
        warnings.filterwarnings("ignore")
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        # model = gmsh.model()
        gmsh.model.add("pacman")
        # geom_parameters = {'omega': np.pi/4, 'r': 1, 'lc': 0.1}

        omega = np.deg2rad(geom_parameters.get("omega"))
        radius = geom_parameters.get("r")
        lc = geom_parameters.get("lc")
        elltomesh = geom_parameters.get("elltomesh")

        refinement = geom_parameters.get("refinement")
        l0 = geom_parameters.get("l0")

        # refinement = geom_parameters.get("refinement")
        # R1=1.; R2=2.3; R3=1.; ex=0.; ey=-.3
        model = gmsh.model
        # model.occ.addDisk(0, 0, 0, R, R, tag=10)


        # print("Model name: " + gmsh.model.getCurrent())

        # get all elementary entities in the model
        entities = gmsh.model.occ.getEntities()


        # for e in entities:
        #     print("Entity " + str(e) + " of type " + gmsh.model.getType(e[0], e[1]))
        #     # get the mesh nodes for each elementary entity
        #     nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes(e[0], e[1])
        #     # get the mesh elements for each elementary entity
        #     elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(e[0], e[1])
        #     # count number of elements
        #     numElem = sum(len(i) for i in elemTags)
        #     print(" - mesh has " + str(len(nodeTags)) + " nodes and " + str(numElem) +
        #         " elements")
        #     boundary = gmsh.model.occ.getBoundary([e])
        #     print(" - boundary entities " + str(boundary))
        #     partitions = gmsh.model.occ.getPartitions(e[0], e[1])


        # print(entities)
        p0 = model.geo.addPoint(0, 0, 0, lc/refinement, tag=0)
        p1 = model.geo.addPoint( - radius*np.cos(omega / 2), radius*np.sin(omega / 2), 0.0, lc, tag=1)
        p2 = model.geo.addPoint( - radius*np.cos(omega / 2), - radius*np.sin(omega / 2), 0.0, lc, tag=2)
        p3 = model.geo.addPoint(radius, 0, 0.0, lc/refinement, tag=12)

        top = model.geo.addLine(p1, p0, tag=3)
        bot = model.geo.addLine(p0, p2, tag=4)
        arc1 = model.geo.addCircleArc(2, 0, 12, tag=5)
        arc2 = model.geo.addCircleArc(12, 0, 1, tag=6)
        cloop = model.geo.addCurveLoop([top, bot, arc1, arc2])


        s = model.geo.addPlaneSurface([cloop])
        model.geo.addSurfaceLoop([s, 1000])
        # model.geo.synchronize()

        _cracktip = model.geo.addPoint(l0, 0, 0.0, lc/refinement, tag=111)
        crack = model.geo.addLine(0, _cracktip, tag=30)

        # synchronize
        model.geo.synchronize()

        # embed "crack" on plate
        model.mesh.embed(1, [crack], 2, s)

        surface_entities = [model[1] for model in model.getEntities(tdim)]
        domain = model.addPhysicalGroup(tdim, surface_entities)
        model.setPhysicalName(tdim, domain, "Surface")
        gmsh.model.mesh.setOrder(order)

        gmsh.model.addPhysicalGroup(tdim - 1, [5], tag=20)
        gmsh.model.setPhysicalName(tdim - 1, 20, "dirichlet_boundary")


        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)

        # We can constrain resolution
        # values (see `t10.py' for more details):
        gmsh.option.setNumber("Mesh.MeshSizeMin", lc/3.)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 2*lc)

        gmsh.model.mesh.generate(tdim)
        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)

    return gmsh.model if comm.rank == 0 else None, tdim


# mesh_pacman('pacman',
#     geom_parameters,
#     tdim=2,
#     order=1,
#     msh_file='pacman.msh')
