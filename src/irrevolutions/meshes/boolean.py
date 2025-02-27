from mpi4py import MPI


def mesh_bar_gmshapi(name, msh_file=None, comm=MPI.COMM_WORLD):
    """
    Create mesh.
    """

    # http://jsdokken.com/converted_files/tutorial_gmsh.html
    # https://moorejustinmusic.com/trending/how-do-i-use-gmsh-in-python/
    # https://stackoverflow.com/questions/54951925/questions-about-the-gmsh-python-api

    # Perform Gmsh work only on rank = 0

    if comm.rank == 0:
        import warnings

        import gmsh

        warnings.filterwarnings("ignore")
        # Initialise gmsh and set options
        gmsh.initialize()
        # gmsh.option.setNumber("General.Terminal", 1)
        # gmsh.option.setNumber("Mesh.Algorithm", 6)
        # model = gmsh.model()
        gmsh.model.add("Bool 2D")
        L, H, r = 1.0, 1.0, 0.1
        hole = gmsh.model.occ.addCircle(L / 2, L / 2, 0.0, r, tag=1)
        domain = gmsh.model.occ.addRectangle(0, 0, 0.0, L, H, tag=2, roundedRadius=0.1)
        gmsh.model.occ.cut([(2, hole)], [(2, domain)], tag=3)


def mesh_moonslice_gmshapi(
    name,
    geom_parameters,
    lc,
    tdim=2,
    order=1,
    msh_file="bool",
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
        import warnings

        import gmsh

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
        gmsh.model.occ.addDisk(ex, ey, 0.0, R2, R3, tag=2)
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
        gmsh.option.setNumber("Mesh.MeshSizeMin", lc / 2.0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 2 * lc)

        gmsh.model.mesh.generate(tdim)

        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)

    return gmsh.model if comm.rank == 0 else None, tdim


def create_disk_with_hole(comm=MPI.COMM_WORLD, geom_parameters=None):
    """
    Creates a 2D disk with an optional central hole using Gmsh.

    Args:
        comm (MPI.Comm): MPI communicator.
        geom_parameters (dict): Dictionary containing geometric parameters:
            - 'R_outer': Outer disk radius.
            - 'R_inner': Inner hole radius (set to 0 for no hole).
            - 'lc': Mesh element size.
            - 'a': Half-width of the refined symmetric region (default 7 * lc).

    Returns:
        None (Gmsh model is created and can be meshed/exported).
    """
    if geom_parameters is None:
        geom_parameters = {
            "R_outer": 1.0,  # Outer disk radius
            "R_inner": 0.0,  # Inner hole radius (0 means no hole)
            "lc": 0.05,  # Mesh element size
            "a": None,  # Half-width of the refined region (-a < x < a)
        }

    R_outer = geom_parameters["R_outer"]
    R_inner = geom_parameters["R_inner"]
    lc = geom_parameters["lc"]
    a = geom_parameters["a"] if geom_parameters["a"] is not None else 7 * lc

    if comm.rank == 0:
        import warnings
        import gmsh

        warnings.filterwarnings("ignore")

        # Initialize gmsh
        gmsh.initialize()
        gmsh.model.add("DiskWithHole")

        # Create outer circle (disk boundary)
        outer_circle = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, R_outer, R_outer, tag=1)

        # Create inner hole if R_inner > 0
        if R_inner > 0:
            inner_hole = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, R_inner, R_inner, tag=2)
            cut_entities, _ = gmsh.model.occ.cut([(2, outer_circle)], [(2, inner_hole)])
            surface_tag = cut_entities[0][1]  # Extract tag from result
        else:
            gmsh.model.occ.synchronize()
            surface_tag = outer_circle

        # Synchronize before meshing
        gmsh.model.occ.synchronize()

        # Define physical groups
        gmsh.model.addPhysicalGroup(2, [surface_tag], tag=1)
        gmsh.model.setPhysicalName(2, 1, "DiskDomain")

        # Mesh settings
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)

        refinement_field = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(
            refinement_field, "VIn", lc / 3
        )  # Finer mesh inside
        gmsh.model.mesh.field.setNumber(
            refinement_field, "VOut", lc
        )  # Coarser mesh outside
        gmsh.model.mesh.field.setNumber(refinement_field, "XMin", -a)
        gmsh.model.mesh.field.setNumber(refinement_field, "XMax", a)
        gmsh.model.mesh.field.setNumber(refinement_field, "YMin", -R_outer)
        gmsh.model.mesh.field.setNumber(refinement_field, "YMax", R_outer)
        gmsh.model.mesh.field.setAsBackgroundMesh(refinement_field)

        gmsh.model.mesh.generate(2)

        # Save mesh to file
        gmsh.write("disc_with_hole.msh")

        print("Mesh created and saved as 'disc_with_hole.msh'")

        # Finalize gmsh
        gmsh.finalize()

        tdim = 2

    return gmsh.model if comm.rank == 0 else None, tdim
