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
