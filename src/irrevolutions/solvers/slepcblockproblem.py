import logging
import typing

import dolfinx
import ufl
from petsc4py import PETSc
from slepc4py import SLEPc

from .function import vec_to_functions

# -----------------------------------------------------------------------------
#  License
#  The code is based on dolfiny, by Michal Habera and Andreas Zilian.
#  dolfiny is free software: you can redistribute it and/or modify it under the terms of
#  the GNU Lesser General Public License as published by the Free Software Foundation, 
#  either version 3 of the License, or (at your option) any later version.
#
#  dolfiny is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
#  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
#  See the GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License along with 
#  dolfiny. If not, see http://www.gnu.org/licenses/.
#
#  The original code can be found at:
#  https://github.com/fenics-dolfiny/dolfiny
#
#  Original Authors:
#    Michal Habera, Rafinex, Luxembourg.
#    Andreas Zilian, University of Luxembourg, Luxembourg.
#
#  Modifications and contributions:
#    Andrés A León Baldelli, CNRS.
# -----------------------------------------------------------------------------

class SLEPcBlockProblem:
    def __init__(
        self,
        F_form: typing.List,
        u: typing.List,
        lmbda: dolfinx.fem.Function,
        bcs=[],
        A_form=None,
        B_form=None,
        prefix=None,
    ):
        """
        Initialize the SLEPcBlockProblem for solving a generalised eigenvalue problem.

        Wrapper for a generalised eigenvalue problem obtained from UFL residual forms.
        
        Parameters
        ----------
        F_form : List
            Residual forms.
        u : List
            Current solution vectors.
        lmbda : dolfinx.fem.Function
            Eigenvalue function. Residual forms must be linear in lambda for linear eigenvalue problems.
        bcs : List, optional
            List of boundary conditions.
        A_form : optional
            Override automatically derived A matrix.
        B_form : optional
            Override automatically derived B matrix.
        prefix : str, optional
            Prefix for SLEPc options.

        Note
        ----
        In general, eigenvalue problems have the form T(lambda) * x = 0,
        where T(lambda) is a matrix-valued function.
        Linear eigenvalue problems have T(lambda) = A + lambda * B, and if B is not identity matrix
        then this problem is called a generalized (linear) eigenvalue problem.
        """
        self.F_form = F_form
        self.u = u
        self.lmbda = lmbda
        self.bcs = bcs
        self.comm = u[0].function_space.mesh.mpi_comm()

        self.ur = []
        self.ui = []
        for func in u:
            self.ur.append(dolfinx.Function(func.function_space, name=func.name))
            self.ui.append(dolfinx.Function(func.function_space, name=func.name))

        # Prepare tangent form M0 which has terms involving lambda
        self.M0 = [[None for i in range(len(self.u))] for j in range(len(self.u))]
        for i in range(len(self.u)):
            for j in range(len(self.u)):
                self.M0[i][j] = ufl.algorithms.expand_derivatives(
                    ufl.derivative(
                        F_form[i],
                        self.u[j],
                        ufl.TrialFunction(self.u[j].function_space),
                    )
                )

        if B_form is None:
            B0 = [[None for i in range(len(self.u))] for j in range(len(self.u))]

            for i in range(len(self.u)):
                for j in range(len(self.u)):
                    # Differentiate wrt. lambda and replace all remaining lambda with Zero
                    B0[i][j] = ufl.algorithms.expand_derivatives(
                        ufl.diff(self.M0[i][j], lmbda)
                    )
                    B0[i][j] = ufl.replace(B0[i][j], {lmbda: ufl.zero()})

                    if B0[i][j].empty():
                        B0[i][j] = None

            self.B_form = B0
        else:
            self.B_form = B_form

        if A_form is None:
            A0 = [[None for i in range(len(self.u))] for j in range(len(self.u))]

            for i in range(len(self.u)):
                for j in range(len(self.u)):
                    A0[i][j] = ufl.replace(self.M0[i][j], {lmbda: ufl.zero()})

                    if A0[i][j].empty():
                        A0[i][j] = None
                        continue
            self.A_form = A0
        else:
            self.A_form = A_form

        self.eps = SLEPc.EPS().create(self.comm)
        self.eps.setOptionsPrefix(prefix)
        self.eps.setFromOptions()
        self.A = dolfinx.fem.create_matrix_block(self.A_form)
        self.B = None

        if not self.empty_B():
            self.B = dolfinx.fem.create_matrix_block(self.B_form)

    def solve(self):
        """
        Solve the generalized eigenvalue problem.

        Assemble the matrices A and B, and solve the eigenvalue problem using SLEPc.
        """
        self.A.zeroEntries()
        dolfinx.fem.assemble_matrix_block(self.A, self.A_form, self.bcs)
        self.A.assemble()

        if not self.empty_B():
            self.B.zeroEntries()
            dolfinx.fem.assemble_matrix_block(self.B, self.B_form, self.bcs)
            self.B.assemble()

        self.eps.setOperators(self.A, self.B)
        self.eps.solve()

    def getEigenpair(self, i):
        """
        Get the eigenvalue and eigenvector pair.

        Parameters
        ----------
        i : int
            Index of the eigenvalue/eigenvector to retrieve.

        Returns
        -------
        eigval : float
            The eigenvalue corresponding to the i-th eigenpair.
        ur : List[dolfinx.fem.Function]
            The real part of the eigenvector.
        ui : List[dolfinx.fem.Function]
            The imaginary part of the eigenvector.
        """
        xr, xi = self.A.getVecs()
        eigval = self.eps.getEigenpair(i, xr, xi)

        vec_to_functions(xr, self.ur)
        vec_to_functions(xi, self.ui)

        return (eigval, self.ur, self.ui)

    def empty_B(self):
        """
        Check if the B matrix is empty.

        Returns
        -------
        bool
            True if B matrix is empty, False otherwise.
        """
        for i in range(len(self.B_form)):
            for j in range(len(self.B_form[i])):
                if self.B_form[i][j] is not None:
                    return False

        return True

class SLEPcBlockProblemRestricted:
    def __init__(
        self,
        F_form: typing.List,
        u: typing.List,
        lmbda: dolfinx.fem.Function,
        bcs=[],
        restriction=None,
        A_form=None,
        B_form=None,
        prefix=None,
    ):
        """
        Initialize the SLEPcBlockProblemRestricted class for solving a restricted eigenvalue problem.

        Wrapper for a generalized eigenvalue problem obtained from UFL residual forms with restrictions.

        Parameters
        ----------
        F_form : List
            Residual forms.
        u : List
            Current solution vectors.
        lmbda : dolfinx.fem.Function
            Eigenvalue function. Residual forms must be linear in lmbda for linear eigenvalue problems.
        bcs : List, optional
            List of boundary conditions.
        restriction : optional
            `Restriction` class used to provide information about the degree-of-freedom indices 
            for which this solver should solve.
        A_form : optional
            Override automatically derived A matrix.
        B_form : optional
            Override automatically derived B matrix.
        prefix : str, optional
            Prefix for SLEPc options.

        Note
        ----
        In general, eigenvalue problems have the form T(lmbda) * x = 0, where T(lmbda) is a matrix-valued function.
        Linear eigenvalue problems have T(lmbda) = A + lmbda * B, and if B is not the identity matrix,
        this problem is called a generalized (linear) eigenvalue problem.
        """
        self.F_form = F_form
        self.u = u
        self.lmbda = lmbda
        self.comm = u[0].function_space.mesh.comm
        self.restriction = restriction
        self.bcs = bcs

        self.ur = []
        self.ui = []
        for func in u:
            self.ur.append(dolfinx.fem.Function(func.function_space, name=func.name))
            self.ui.append(dolfinx.fem.Function(func.function_space, name=func.name))

        # Prepare tangent form M0 which has terms involving lambda
        self.M0 = [[None for i in range(len(self.u))] for j in range(len(self.u))]
        for i in range(len(self.u)):
            for j in range(len(self.u)):
                self.M0[i][j] = ufl.algorithms.expand_derivatives(
                    ufl.derivative(
                        F_form[i],
                        self.u[j],
                        ufl.TrialFunction(self.u[j].function_space),
                    )
                )

        if B_form is None:
            B0 = [[None for i in range(len(self.u))] for j in range(len(self.u))]

            for i in range(len(self.u)):
                for j in range(len(self.u)):
                    # Differentiate with respect to lambda and replace all remaining lambda with zero
                    B0[i][j] = ufl.algorithms.expand_derivatives(
                        ufl.diff(self.M0[i][j], lmbda)
                    )
                    B0[i][j] = ufl.replace(B0[i][j], {lmbda: ufl.zero()})

                    if B0[i][j].empty():
                        B0[i][j] = None

            self.B_form_ = B0
        else:
            self.B_form_ = B_form

        if A_form is None:
            A0 = [[None for i in range(len(self.u))] for j in range(len(self.u))]

            for i in range(len(self.u)):
                for j in range(len(self.u)):
                    A0[i][j] = ufl.replace(self.M0[i][j], {lmbda: ufl.zero()})

                    if A0[i][j].empty():
                        A0[i][j] = None
                        continue
            self.A_form_ = A0
        else:
            self.A_form_ = A_form

        self.A_form = dolfinx.fem.form(self.A_form_)
        self.B_form = dolfinx.fem.form(self.B_form_)
        self.eps = SLEPc.EPS().create(self.comm)
        self.eps.setOptionsPrefix(prefix)
        self.eps.setFromOptions()

        self.A = dolfinx.fem.petsc.create_matrix_block(self.A_form)
        self.B = None

        if not self.empty_B():
            self.B = dolfinx.fem.petsc.create_matrix_block(self.B_form)

        if self.restriction is not None:
            _A = dolfinx.fem.petsc.create_matrix_block(self.A_form)
            _A.assemble()
            self.rA = self.restriction.restrict_matrix(_A)
            self.rB = None
            self.rA.assemble()

            if not self.empty_B():
                _B = dolfinx.fem.petsc.create_matrix_block(self.B_form)
                _B.assemble()
                self.rB = self.restriction.restrict_matrix(_B)
                self.rB.assemble()

    def solve(self):
        """
        Solve the generalized eigenvalue problem with restrictions if provided.

        Assemble the matrices A and B, apply restrictions if necessary, and solve the eigenvalue problem using SLEPc.
        """
        if self.restriction is not None:
            self.A.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix_block(self.A, self.A_form, self.bcs)
            self.A.assemble()

            self.restriction.restrict_matrix(self.A).copy(self.rA)
            self.rA.assemble()
        else:
            self.A.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix_block(self.A, self.A_form, self.bcs)
            self.A.assemble()

        if not self.empty_B():
            self.B.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix_block(self.B, self.B_form, self.bcs)
            self.B.assemble()

            if self.restriction is not None:
                self.B.zeroEntries()
                dolfinx.fem.petsc.assemble_matrix_block(self.B, self.B_form, self.bcs)
                self.restriction.restrict_matrix(self.B).copy(self.rB)
                self.B.assemble()

                self.restriction.restrict_matrix(self.B).copy(self.rB)
                self.rB.assemble()

        if self.restriction is not None:
            self.eps.setOperators(self.rA, self.rB)
        else:
            self.eps.setOperators(self.A, self.B)

        self.eps.solve()

    def getEigenpair(self, i):
        """
        Get the eigenvalue and eigenvector pair for the restricted eigenvalue problem.

        Parameters
        ----------
        i : int
            Index of the eigenvalue/eigenvector to retrieve.

        Returns
        -------
        eigval : float
            The eigenvalue corresponding to the i-th eigenpair.
        ur : List[dolfinx.fem.Function]
            The real part of the eigenvector.
        ui : List[dolfinx.fem.Function]
            The imaginary part of the eigenvector.
        """
        if self.restriction is not None:
            xr, xi = self.rA.getVecs()
        else:
            xr, xi = self.A.getVecs()

        eigval = self.eps.getEigenpair(i, xr, xi)

        if self.restriction is not None:
            self.restriction.update_functions(self.ur, xr)
            self.restriction.update_functions(self.ui, xi)
        else:
            vec_to_functions(xr, self.ur)
            vec_to_functions(xi, self.ui)

        return eigval, self.ur, self.ui

    def empty_B(self):
        """
        Check if the B matrix is empty.

        Returns
        -------
        bool
            True if the B matrix is empty, False otherwise.
        """
        for i in range(len(self.B_form)):
            for j in range(len(self.B_form[i])):
                if self.B_form[i][j] is not None:
                    return False
        return True