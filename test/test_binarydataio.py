from petsc4py import PETSc
import pickle


def save_binary_data(filename, data):
    viewer = PETSc.Viewer().createBinary(filename, "w")
    
    if isinstance(data, list):
        for item in data:
            item.view(viewer)
    elif isinstance(data, PETSc.Mat):
        data.view(viewer)
    elif isinstance(data, PETSc.Vec):
        data.view(viewer)
    else:
        raise ValueError("Unsupported data type for saving")

from test_errorcodes import translatePETScERROR

def load_binary_data(filename):
    viewer = PETSc.Viewer().createBinary(filename, "r")
    data = []
    vectors = []
    
    i = 0
    while True:
        try:
            vec = PETSc.Vec().load(viewer)
            
            vectors.append(vec)
            i += 1
        except PETSc.Error as e:
            # __import__('pdb').set_trace()
            # if e.ierr == PETSc.Error.S_ARG_WRONG:
            print(f"Error {e.ierr}: {translatePETScERROR.get(e.ierr, 'Unknown error')}")
            break
            # else:
                # raise

    return data


def load_binary_vector(filename):
    """
    Load a binary file containing a PETSc vector.

    Args:
        filename (str): Path to the binary file.

    Returns:
        PETSc.Vec: Loaded PETSc vector.
    """
    try:
        # Create a PETSc viewer for reading
        viewer = PETSc.Viewer().createBinary(filename, 'r')
        
        # Load the vector from the viewer
        vector = PETSc.Vec().load(viewer)

        # Close the viewer
        viewer.destroy()

        return vector

    except PETSc.Error as e:
        print(f"Error: {e}")
        return None


def load_binary_matrix(filename):
    """
    Load a binary file containing a PETSc Matrix.

    Args:
        filename (str): Path to the binary file.

    Returns:
        PETSc.Mat: Loaded PETSc Matrix.
    """
    try:
        # Create a PETSc viewer for reading
        viewer = PETSc.Viewer().createBinary(filename, 'r')
        
        # Load the vector from the viewer
        vector = PETSc.Mat().load(viewer)

        # Close the viewer
        viewer.destroy()

        return vector

    except PETSc.Error as e:
        print(f"Error: {e}")
        return None

def save_minimal_constraints(obj, filename):
    minimal_constraints = {
        'bglobal_dofs_mat': obj.bglobal_dofs_mat,
        'bglobal_dofs_mat_stacked': obj.bglobal_dofs_mat_stacked,
        'bglobal_dofs_vec': obj.bglobal_dofs_vec,
        'bglobal_dofs_vec_stacked': obj.bglobal_dofs_vec_stacked,
        'blocal_dofs': obj.blocal_dofs,
        'boffsets_mat': obj.boffsets_mat,
        'boffsets_vec': obj.boffsets_vec,
    }

    with open(filename, 'wb') as file:
        pickle.dump(minimal_constraints, file)
    
    
def load_minimal_constraints(filename):
    with open(filename, 'rb') as file:
        minimal_constraints = pickle.load(file)

    # Assuming you have a constructor for your class
    # Modify this accordingly based on your actual class structure
    reconstructed_obj = Restriction()
    for key, value in minimal_constraints.items():
        setattr(reconstructed_obj, key, value)

    return reconstructed_obj

if __name__ == "__main__":

    m, n  = 16, 32

    # Example usage
    matrix = PETSc.Mat().create(PETSc.COMM_WORLD)
    # matrix.setValue(0, 0, 1.0)
    # matrix.setValue(1, 1, 2.0)
    # matrix.setValue(2, 2, 3.0)

    matrix.setSizes([n*n, n*n])
    matrix.setFromOptions()
    matrix.setUp()

    Istart, Iend = matrix.getOwnershipRange()
    for I in range(Istart, Iend):
        matrix[I,I] = 4
        i = I//n
        if i>0  : J = I-n; matrix[I,J] = -1
        if i<m-1: J = I+n; matrix[I,J] = -1
        j = I-i*n
        if j>0  : J = I-1; matrix[I,J] = -1
        if j<n-1: J = I+1; matrix[I,J] = -1

    matrix.assemblyBegin()
    matrix.assemblyEnd()


    x, y = matrix.createVecs()
    x.set(1)
    matrix.mult(x,y)

    # save
    viewer = PETSc.Viewer().createBinary('matrix-A.dat', 'w')
    viewer(matrix)
    viewer = PETSc.Viewer().createBinary('vector-x.dat', 'w')
    viewer(x)
    viewer = PETSc.Viewer().createBinary('vector-y.dat', 'w')
    viewer(y)


    # load
    viewer = PETSc.Viewer().createBinary('matrix-A.dat', 'r')
    B = PETSc.Mat().load(viewer)
    viewer = PETSc.Viewer().createBinary('vector-x.dat', 'r')
    u = PETSc.Vec().load(viewer)
    viewer = PETSc.Viewer().createBinary('vector-y.dat', 'r')
    v = PETSc.Vec().load(viewer)

    u = load_binary_vector('vector-x.dat')
    save_binary_data("x.vec", u)
    save_binary_data("matrix.mat", matrix)
    
    __import__('pdb').set_trace()
    B = load_binary_vector('vector-x.dat')
    
    
    # vector = PETSc.Vec().createWithArray([1.0, 2.0, 3.0])
    # # from test_savebinarydata import save_binary_data
    # # Save list of vectors and matrices
    # save_binary_data("output_data.bin", [matrix, vector])
    save_binary_data("output_data.bin", matrix)


    # Example usage
    loaded_data = load_binary_vector("output_data.bin")
    __import__('pdb').set_trace()

    # Perform operations on the loaded data
    if len(loaded_data) > 0:
        for i, obj in enumerate(loaded_data):
            print(f"Object {i}:")
            print(obj.getArray())
            print()
    else:
        print("No data loaded.")
