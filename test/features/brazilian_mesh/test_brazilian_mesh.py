from irrevolutions.meshes.boolean import create_disk_with_hole
from mpi4py import MPI
from dolfinx.io import XDMFFile, gmshio


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    geom_params = {
        "R_outer": 1.0,  # Outer disk radius
        "R_inner": 0.3,  # Inner hole radius (set to 0.0 for no hole)
        "lc": 0.05,  # Mesh element size
        "a": 0.1,  # Half-width of the refined region (-a < x < a)
    }
    # create_disk_with_hole(comm, geom_params)
    gmsh_model, tdim = create_disk_with_hole(comm, geom_params)
    mesh, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh_model, comm, 0, gdim=2)
    __import__("pdb").set_trace()
