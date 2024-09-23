import typing
import dolfinx
import numpy
from petsc4py import PETSc


class Restriction:
    """
    Class for restricting a problem to a subset of degree-of-freedom (DOF) indices.

    This class enables restriction of matrices and vectors to a subset of DOFs, allowing 
    for the efficient manipulation of smaller systems. It works by restricting the 
    problem to block-local DOF indices and manages the associated offsets.

    Attributes:
    ----------
    function_spaces : list
        List of dolfinx.fem.FunctionSpace objects for the problem.
    blocal_dofs : list
        List of block-local DOF indices for each function space.
    comm : MPI communicator
        MPI communicator from the first function space.
    bglobal_dofs_vec : list
        Global DOF indices for vectors.
    bglobal_dofs_mat : list
        Global DOF indices for matrices.
    boffsets_vec : list
        Offset values for vector DOFs.
    boffsets_mat : list
        Offset values for matrix DOFs.
    bglobal_dofs_vec_stacked : ndarray
        Stacked global DOF indices for vectors.
    bglobal_dofs_mat_stacked : ndarray
        Stacked global DOF indices for matrices.
    """

    def __init__(
        self,
        function_spaces: typing.List[dolfinx.fem.FunctionSpace],
        blocal_dofs: typing.List[numpy.ndarray],
    ):
        """
        Initialize the Restriction class for a problem with block-local DOF indices.

        Parameters:
        ----------
        function_spaces : list
            List of dolfinx.fem.FunctionSpace objects for the problem.
        blocal_dofs : list
            Block-local DOF indices for each function space.

        Note:
        ----
        Currently, the restriction of a matrix and vector is sub-optimal since it assumes 
        a different parallel layout every time restriction is called.
        """
        self.function_spaces = function_spaces
        self.blocal_dofs = blocal_dofs
        self.comm = self.function_spaces[0].mesh.comm

        self.bglobal_dofs_vec = []
        self.bglobal_dofs_mat = []

        self.boffsets_mat = [0]
        self.boffsets_vec = [0]
        offset_mat = 0
        offset_vec = 0

        for i, space in enumerate(function_spaces):
            bs = space.dofmap.index_map_bs

            size_local = space.dofmap.index_map.size_local
            num_ghosts = space.dofmap.index_map.num_ghosts

            self.boffsets_mat.append(
                self.boffsets_mat[-1] + bs * (size_local + num_ghosts)
            )
            offset_mat += self.boffsets_mat[-1]

            self.boffsets_vec.append(self.boffsets_vec[-1] + bs * size_local)
            offset_vec += self.boffsets_vec[-1]

            dofs = self.blocal_dofs[i].copy()
            # Remove any ghost DOFs
            dofs = dofs[dofs < bs * size_local]
            dofs += self.boffsets_mat[i]
            self.bglobal_dofs_mat.append(dofs)

            dofs = self.blocal_dofs[i].copy()
            dofs = dofs[dofs < bs * size_local]
            dofs += self.boffsets_vec[i]
            self.bglobal_dofs_vec.append(dofs)

        self.bglobal_dofs_vec_stacked = numpy.hstack(self.bglobal_dofs_vec)
        self.bglobal_dofs_mat_stacked = numpy.hstack(self.bglobal_dofs_mat)

    def restrict_matrix(self, A: PETSc.Mat) -> PETSc.Mat:
        """
        Restrict a matrix to a subset of DOF indices.

        Parameters:
        ----------
        A : PETSc.Mat
            PETSc matrix to be restricted.

        Returns:
        -------
        PETSc.Mat
            Restricted matrix.
        """
        # Fetching IS only for owned DOFs
        local_isrow = PETSc.IS(self.comm).createGeneral(self.bglobal_dofs_mat_stacked)
        global_isrow = A.getLGMap()[0].applyIS(local_isrow)

        subA = A.createSubMatrix(isrow=global_isrow, iscol=global_isrow)
        subA.assemble()

        return subA

    def restrict_vector(self, x: PETSc.Vec) -> PETSc.Vec:
        """
        Restrict a vector to a subset of DOF indices.

        Parameters:
        ----------
        x : PETSc.Vec
            PETSc vector to be restricted.

        Returns:
        -------
        PETSc.Vec
            Restricted vector.
        """
        arr = x.array[self.bglobal_dofs_vec_stacked]
        subx = PETSc.Vec().createWithArray(arr)

        return subx

    def update_functions(self, f: typing.List[dolfinx.fem.Function], rx: PETSc.Vec):
        """
        Update functions using restricted DOF indices.

        This method updates the function objects based on the restricted vector data.

        Parameters:
        ----------
        f : list
            List of dolfinx.fem.Function objects to update.
        rx : PETSc.Vec
            Restricted vector containing the new function values.
        """
        rdof_offset = 0
        for i, fi in enumerate(f):
            num_rdofs = self.bglobal_dofs_vec[i].shape[0]

            fi.vector.array[self.bglobal_dofs_vec[i] - self.boffsets_vec[i]] = (
                rx.array_r[rdof_offset : (rdof_offset + num_rdofs)]
            )

            fi.vector.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )
            rdof_offset += num_rdofs