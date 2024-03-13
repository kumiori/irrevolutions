from petsc4py import PETSc
import numpy as np

def display_vector_info(v):
    # Ensure the vector is assembled
    v.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    v.assemble()

    # Access the local data
    local_data = v.array

    # Print information about the vector
    print("Size of the vector:", v.getSize())
    print("Local data of the vector:", local_data)
    print("Nonzero entries in the local data:", len(local_data.nonzero()[0]))
    print("Global indices of nonzero entries:", v.getOwnershipRange())
    print("Global indices of nonzero entries:", v.getOwnershipRanges())

# Assuming you have a vector v

def restricted_cone_project(x_local, subset_dofs):
    x_local.array[subset_dofs] = np.maximum(x_local.array[subset_dofs], 0)

if __name__ == "__main__":
    # Example usage
    comm = PETSc.COMM_WORLD
    # size = comm.Get_size()
    size = len([1.0, -2.0, 3.0, -4.0])

    # Create a sequential Petsc vector
    x_local = PETSc.Vec().createSeq(size)
    x_local.setArray([1.0, -2.0, 3.0, -4.0])

    # Define a subset of degrees of freedom
    subset_dofs = [1, 3]
    x_local.view()

    # Truncate negative values in the subset using NumPy
    restricted_cone_project(x_local, subset_dofs)

    # Display the modified vector
    x_local.view()
    display_vector_info(x_local)
