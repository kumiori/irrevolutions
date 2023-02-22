from mpi4py import MPI


def mesh_bar_gmshapi(name,
                     msh_file=None,
                     comm=MPI.COMM_WORLD):
    """
    Create mesh.
    """

    # http://jsdokken.com/converted_files/tutorial_gmsh.html
    # https://moorejustinmusic.com/trending/how-do-i-use-gmsh-in-python/
    # https://stackoverflow.com/questions/54951925/questions-about-the-gmsh-python-api

    # Perform Gmsh work only on rank = 0

    if comm.rank == 0:

        import gmsh
        import warnings
        warnings.filterwarnings("ignore")
        # Initialise gmsh and set options
        gmsh.initialize()
        # gmsh.option.setNumber("General.Terminal", 1)
        # gmsh.option.setNumber("Mesh.Algorithm", 6)
        # model = gmsh.model()
        gmsh.model.add("Bool 2D")
        L, H, r = 1., 1., .1
        hole = gmsh.model.occ.addCircle(L / 2, L / 2, 0., r, tag=1)
        domain = gmsh.model.occ.addRectangle(0, 0, 0., L, H, tag=2, roundedRadius=.1)
        boolean = gmsh.model.occ.cut([(2, hole)], [(2, domain)], tag=3)

def mesh_moonslice_gmshapi(
    name,
    geom_parameters,
    lc,
    tdim=2,
    order=1,
    msh_file='bool',
    comm=MPI.COMM_WORLD,
):
    """
    Create mesh.
    """

    # http://jsdokken.com/converted_files/tutorial_gmsh.html
    # https://moorejustinmusic.com/trending/how-do-i-use-gmsh-in-python/
    # https://stackoverflow.com/questions/54951925/questions-about-the-gmsh-python-api

    # Perform Gmsh work only on rank = 0
    if comm.rank == 0:
        import gmsh
        import warnings
        warnings.filterwarnings("ignore")
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        # model = gmsh.model()
        gmsh.model.add("moonslice 2D")

        ex = geom_parameters.get("ex")
        ey = geom_parameters.get("ey")
        R1 = geom_parameters.get("R1")
        R2 = geom_parameters.get("R2")
        R3 = geom_parameters.get("R3")
        # R1=1.; R2=2.3; R3=1.; ex=0.; ey=-.3

        gmsh.model.occ.addDisk(0, 0, 0, R1, R1, tag=1)
        gmsh.model.occ.addDisk(ex, ey, 0., R2, R3, tag=2)
        gmsh.model.occ.cut([(tdim, 1)], [(tdim, 2)], 3)
        gmsh.model.occ.synchronize()
        model = gmsh.model

        surface_entities = [model[1] for model in model.getEntities(tdim)]
        domain = model.addPhysicalGroup(tdim, surface_entities)
        model.setPhysicalName(tdim, domain, "Surface")
        gmsh.model.mesh.setOrder(order)


        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)

        # We can constraint the min and max element sizes to stay within reasonnable
        # values (see `t10.py' for more details):
        gmsh.option.setNumber("Mesh.MeshSizeMin", lc/2.)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 2*lc)

        gmsh.model.mesh.generate(tdim)

        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)

    return gmsh.model if comm.rank == 0 else None, tdim

