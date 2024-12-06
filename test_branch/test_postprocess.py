import dolfinx
import numpy as np
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import os
import h5py


def read_timestep_data(h5file, function, function_name, timestep):
    """
    Read data for a specific timestep from the HDF5 file and update the function vector.

    Parameters:
    h5file (h5py.File): The HDF5 file object.
    function (dolfinx.fem.Function): The function to update.
    function_name (str): The name of the function in the HDF5 file.
    timestep (str): The timestep to read.
    """
    dataset_path = f"Function/{function_name}/{timestep}"
    with h5py.File(h5_file_path, "r") as h5file:
        if dataset_path in h5file:
            data = h5file[dataset_path][:]

            local_range = function.vector.getOwnershipRange()
            local_data = data[local_range[0]:local_range[1]]
            function.vector.setArray(local_data.flatten())
            function.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            # function.vector.setArray(data.flatten())
            function.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        else:
            print(f"Timestep {timestep} not found for function {function_name}.")

def get_timesteps(h5file, function_name):
    """
    Retrieve timesteps for a specific function from the HDF5 file.

    Parameters:
    h5file (h5py.File): The HDF5 file object.
    function_name (str): The name of the function in the HDF5 file.

    Returns:
    list: A list of timesteps available for the specified function.
    """
    timesteps = []
    function_group_path = f"Function/{function_name}"
    if function_group_path in h5file:
        function_group = h5file[function_group_path]
        timesteps = list(function_group.keys())
    else:
        print(f"Function {function_name} not found in the HDF5 file.")
    
    return timesteps


# Initialize MPI communicator
comm = MPI.COMM_WORLD

# Define the path to the xdmf file
file_path = os.path.join(os.path.dirname(__file__), 'data/1d-bar')
xdmf_file_path = file_path + '.xdmf'
h5_file_path = file_path + '.h5'

if __name__ == "__main__":

    # Load the mesh and function spaces
    with XDMFFile(comm, xdmf_file_path, "r") as xdmf_file:
        mesh = xdmf_file.read_mesh(name="mesh")

    # Define the function spaces
    element_u = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)
    V_u = dolfinx.fem.FunctionSpace(mesh, element_u)

    element_alpha = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)
    V_alpha = dolfinx.fem.FunctionSpace(mesh, element_alpha)

    # Read function data from HDF5
    with h5py.File(h5_file_path, "r") as h5file:
        u_data = h5file["Function/Displacement"]
        alpha_data = h5file["Function/Damage"]

        def print_attrs(name, obj):
            print(f"{name}: {obj}")

        h5file.visititems(print_attrs)
        
    with h5py.File(h5_file_path, "r") as h5file:
        displacement_timesteps = get_timesteps(h5file, "Displacement")
        damage_timesteps = get_timesteps(h5file, "Damage")

        if displacement_timesteps == damage_timesteps:
            timesteps = displacement_timesteps
        
    # Assign the read data to the functions
    u = dolfinx.fem.Function(V_u, name="Displacement")
    alpha = dolfinx.fem.Function(V_alpha, name="Damage")

    # Example usage: Read data for a specific timestep
    timestep = "0_31034482758620691"  # Adjust this to the desired timestep
    read_timestep_data(h5_file_path, u, "Displacement", timestep)
    read_timestep_data(h5_file_path, alpha, "Damage", timestep)

    # Example postprocessing: Compute norms of the fields
    u_norm = u.vector.norm()
    alpha_norm = alpha.vector.norm()

    print(f"Norm of displacement field u: {u_norm:.4e}")
    print(f"Norm of damage field alpha: {alpha_norm:.4e}")

    # Additional postprocessing can be done here, such as visualization or further analysis.