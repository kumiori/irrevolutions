import typing

# from yaml.tokens import BlockSequenceStartToken

import dolfinx
import ufl
from .function import vec_to_functions
from slepc4py import SLEPc
from irrevolutions.utils.viz import plot_matrix

# plot_matrix

from petsc4py import PETSc
import logging
import pdb


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
        """SLEPc problem and solver wrapper.

        Wrapper for a generalised eigenvalue problem obtained from UFL residual forms.

        Parameters
        ----------
        F_form
            Residual forms
        u
            Current solution vectors
        lmbda
            Eigenvalue function. Residual forms must be linear in lmbda for
            linear eigenvalue problems.
        bcs
            List of boundary conditions.
        A_form, optional
            Override automatically derived A
        B_form, optional
            Override automatically derived B

        Note
        ----
        In general, eigenvalue problems have form T(lmbda) * x = 0,
        where T(lmbda) is a matrix-valued function.
        Linear eigenvalue problems have T(lmbda) = A + lmbda * B, and if B is not identity matrix
        then this problem is called generalized (linear) eigenvalue problem.

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
        xr, xi = self.A.getVecs()
        eigval = self.eps.getEigenpair(i, xr, xi)

        vec_to_functions(xr, self.ur)
        vec_to_functions(xi, self.ui)

        return (eigval, self.ur, self.ui)

    def empty_B(self):
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
        """SLEPc problem and solver wrapper.

        Wrapper for a generalised eigenvalue problem obtained from UFL residual forms.

        Parameters
        ----------
        F_form
            Residual forms
        u
            Current solution vectors
        lmbda
            Eigenvalue function. Residual forms must be linear in lmbda for
            linear eigenvalue problems.
        bcs
            List of boundary conditions.
        restriction: optional
            ``Restriction`` class used to provide information about degree-of-freedom
            indices for which this solver should solve.
        A_form, optional
            Override automatically derived A
        B_form, optional
            Override automatically derived B

        Note
        ----
        In general, eigenvalue problems have form T(lmbda) * x = 0,
        where T(lmbda) is a matrix-valued function.
        Linear eigenvalue problems have T(lmbda) = A + lmbda * B, and if B is not identity matrix
        then this problem is called generalized (linear) eigenvalue problem.

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
                    # Differentiate wrt. lambda and replace all remaining lambda with Zero
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
        # self.eps.view()
        self.A = dolfinx.fem.petsc.create_matrix_block(self.A_form)
        self.B = None
        if not self.empty_B():
            self.B = dolfinx.fem.petsc.create_matrix_block(self.B_form)

        if self.restriction is not None:
            _A = dolfinx.fem.petsc.create_matrix_block(self.A_form)
            # dolfinx.fem.assemble_matrix_block(A, self.A_form, [])
            _A.assemble()
            self.rA = self.restriction.restrict_matrix(_A)
            self.rB = None
            self.rA.assemble()

            if not self.empty_B():
                _B = dolfinx.fem.create_matrix_block(self.B_form)
                _B.assemble()
                self.rB = self.restriction.restrict_matrix(_B)
                self.rB.assemble()

    def solve(self):
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

        # logging.debug(f"mat rA-{self.eps.getOptionsPrefix()[0:-1]}")
        # logging.debug(f"mat rA sizes {self.rA.sizes}")
        # logging.debug(f"mat  A sizes {self.A.sizes}")

        if logging.getLevelName(logging.getLogger().getEffectiveLevel()) == "DEBUG":
            viewer = PETSc.Viewer().createASCII(
                f"rA-{self.eps.getOptionsPrefix()[0:-1]}.txt"
            )
            self.rA.view(viewer)

            # viewer = open(f"rA-{self.eps.getOptionsPrefix()[0:-1]}.txt", "r")

            # for line in viewer.readlines():
            #     logging.debug(line)

            # logging.critical(f"rB-{self.eps.getOptionsPrefix()[0:-1]}")
            # logging.critical(f"mat rB sizes {self.rB.sizes}")
            # logging.critical(f"mat  B sizes {self.B.sizes}")
            # logging.critical(f"mat  B norm {self.B.norm()}")

        if not self.empty_B():
            # pdb.set_trace()

            viewer = PETSc.Viewer().createASCII(
                f"rB-{self.eps.getOptionsPrefix()[0:-1]}.txt"
            )
            self.rB.view(viewer)
            # viewer = open(f"rB-{self.eps.getOptionsPrefix()[0:-1]}.txt", "r")
            # for line in viewer.readlines():
            #     logging.debug(line)

            # data_rA = analyse_matrix(self.rA, prefix="rA")
            # data_rB = analyse_matrix(self.rB, prefix="rB")

        if self.restriction is not None:
            self.eps.setOperators(self.rA, self.rB)
        else:
            self.eps.setOperators(self.A, self.B)

        self.eps.solve()

    def getEigenpair(self, i):
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

        return (eigval, self.ur, self.ui)

    def empty_B(self):
        for i in range(len(self.B_form)):
            for j in range(len(self.B_form[i])):
                if self.B_form[i][j] is not None:
                    return False

        return True
