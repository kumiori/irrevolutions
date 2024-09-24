import logging
import typing

# from yaml.tokens import BlockSequenceStartToken
import dolfinx
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from .function import functions_to_vec, vec_to_functions


class SNESBlockProblem:
    def __init__(
        self,
        F_form: typing.List,
        u: typing.List,
        bcs=[],
        J_form=None,
        nest=False,
        bounds=None,
        restriction=None,
        prefix=None,
    ):
        """
        SNES problem and solver wrapper.

        Wrapper for solving systems of nonlinear equations using the 
        Scalable Nonlinear Equations Solver (SNES) from PETSc.

        Parameters
        ----------
        F_form : list of ufl.Form
            Residual forms of the system.
        u : list of dolfinx.fem.Function
            Current solution functions.
        bcs : list, optional
            List of boundary conditions (default is []).
        J_form : list of ufl.Form, optional
            Jacobian forms. If not provided, they will be automatically derived (default is None).
        nest : bool, optional
            Whether to use a matrix nesting for the solver (default is False).
        bounds : tuple, optional
            Bounds on the solution for optimization problems [lb, ub] (default is None).
        restriction : Restriction class, optional
            Used to restrict the solver to a subset of DOF indices (default is None).
        prefix : str, optional
            Prefix for the solver options (default is None).
        """
        self.F_form = F_form
        self.u = u
        self.bounds = bounds

        # Validation checks
        if not len(self.F_form) > 0:
            raise RuntimeError("List of provided residual forms is empty!")
        if not len(self.u) > 0:
            raise RuntimeError("List of provided solution functions is empty!")
        if not isinstance(self.u[0], dolfinx.fem.Function):
            raise RuntimeError(
                "Provided solution function not of type dolfinx.fem.Function!"
            )

        self.comm = self.u[0].function_space.mesh.comm

        if J_form is None:
            # Automatically derive Jacobian forms if not provided
            self.J_form = [
                [None for i in range(len(self.u))] for j in range(len(self.u))
            ]
            for i in range(len(self.u)):
                for j in range(len(self.u)):
                    self.J_form[i][j] = ufl.algorithms.expand_derivatives(
                        ufl.derivative(
                            F_form[i],
                            self.u[j],
                            ufl.TrialFunction(self.u[j].function_space),
                        )
                    )
                    if self.J_form[i][j].empty():
                        self.J_form[i][j] = None
        else:
            self.J_form = J_form

        # Create forms
        self.F_form = dolfinx.fem.form(self.F_form)
        self.J_form = dolfinx.fem.form(self.J_form)

        self.bcs = bcs
        self.restriction = restriction

        self.solution = []

        # Prepare empty functions on the corresponding sub-spaces to store solution sub-functions
        for i, ui in enumerate(self.u):
            u = dolfinx.fem.Function(self.u[i].function_space, name=self.u[i].name)
            self.solution.append(u)

        self.norm_r = {}
        self.norm_dx = {}
        self.norm_x = {}

        # Initialize SNES solver
        self.snes = PETSc.SNES().create(self.comm)

        if bounds:
            self.snes.setVariableBounds(self.lb.vector, self.ub.vector)

        # Set up solver for nested or block matrix form
        if nest:
            if restriction is not None:
                raise RuntimeError("Restriction for MATNEST not yet supported.")

            self.J = dolfinx.fem.petsc.create_matrix_nest(self.J_form)
            self.F = dolfinx.fem.petsc.create_vector_nest(self.F_form)
            self.x = self.F.copy()

            self.snes.setFunction(self._F_nest, self.F)
            self.snes.setJacobian(self._J_nest, self.J)
            self.snes.setMonitor(self._monitor_nest)

        else:
            self.J = dolfinx.fem.petsc.create_matrix_block(self.J_form)
            self.F = dolfinx.fem.petsc.create_vector_block(self.F_form)
            self.x = self.F.copy()

            if restriction is not None:
                self._J = dolfinx.fem.petsc.create_matrix_block(self.J_form)
                self._J.assemble()

                self._x = self.x.copy()

                self.rJ = restriction.restrict_matrix(self._J)
                self.rF = restriction.restrict_vector(self.F)
                self.rx = restriction.restrict_vector(self._x)

                self.snes.setFunction(self._F_block, self.rF)
                self.snes.setJacobian(self._J_block, self.rJ)
            else:
                self.snes.setFunction(self._F_block, self.F)
                self.snes.setJacobian(self._J_block, self.J)

            self.snes.setMonitor(self._monitor_block)

        self.snes.setConvergenceTest(self._converged)
        self.snes.setOptionsPrefix(prefix)
        self.snes.setFromOptions()

    def update_functions(self, x):
        """
        Update solution functions based on the provided vector x.

        Parameters
        ----------
        x : PETSc.Vec
            Vector representing the solution.
        """
        if self.restriction is not None:
            self.restriction.update_functions(self.u, x)
            functions_to_vec(self.u, self.x)
        else:
            vec_to_functions(x, self.u)
            x.copy(self.x)
            self.x.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )

    def _F_block(self, snes, x, F):
        """
        Computes the residual for block matrices.
        """
        with self.F.localForm() as f_local:
            f_local.set(0.0)

        self.update_functions(x)

        dolfinx.fem.petsc.assemble_vector_block(
            self.F, self.F_form, self.J_form, self.bcs, x0=self.x, scale=-1.0
        )

        if self.restriction is not None:
            self.restriction.restrict_vector(self.F).copy(self.rF)
            self.rF.copy(F)
        else:
            self.F.copy(F)

    def _F_nest(self, snes, x, F):
        """
        Computes the residual for nested matrices.
        """
        vec_to_functions(x, self.u)
        x = x.getNestSubVecs()

        bcs1 = dolfinx.cpp.fem.bcs_cols(
            dolfinx.fem.assemble._create_cpp_form(self.J_form), self.bcs
        )
        for L, F_sub, a, bc in zip(self.F_form, F.getNestSubVecs(), self.J_form, bcs1):
            with F_sub.localForm() as F_sub_local:
                F_sub_local.set(0.0)
            dolfinx.fem.assemble_vector(F_sub, L)
            dolfinx.fem.apply_lifting(F_sub, a, bc, x0=x, scale=-1.0)
            F_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        F.assemble()

    def _J_block(self, snes, x, J, P):
        """
        Assembles the Jacobian for block matrices.
        """
        self.J.zeroEntries()
        self.update_functions(x)
        dolfinx.fem.petsc.assemble_matrix_block(
            self.J, self.J_form, self.bcs, diagonal=1.0
        )
        self.J.assemble()

        if self.restriction is not None:
            self.restriction.restrict_matrix(self.J).copy(self.rJ)

    def _J_nest(self, snes, u, J, P):
        """
        Assembles the Jacobian for nested matrices.
        """
        self.J.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix_nest(
            self.J, self.J_form, self.bcs, diagonal=1.0
        )
        self.J.assemble()

    def _converged(self, snes, it, norms):
        """
        Convergence test for SNES solver.

        Parameters
        ----------
        snes : PETSc.SNES
            SNES solver.
        it : int
            Current iteration number.
        norms : dict
            Dictionary storing the norms of residuals, solution, and updates.
        """
        it = snes.getIterationNumber()

        atol_x = []
        rtol_x = []
        atol_dx = []
        rtol_dx = []
        atol_r = []
        rtol_r = []

        for i, ui in enumerate(self.u):
            atol_x.append(self.norm_x[it][i] < snes.atol)
            atol_dx.append(self.norm_dx[it][i] < snes.atol)
            atol_r.append(self.norm_r[it][i] < snes.atol)

            rtol_r0 = self.norm_r[0][i]
            if np.isclose(rtol_r0, 0.0):
                rtol_r0 = 1.0

            rtol_x.append(self.norm_x[it][i] < self.norm_x[0][i] * snes.rtol)
            rtol_dx.append(self.norm_dx[it][i] < self.norm_dx[0][i] * snes.rtol)
            rtol_r.append(self.norm_r[it][i] < rtol_r0 * snes.rtol)

        if it > snes.max_it:
            return -5
        elif all(atol_r) and it > 0:
            return 2
        elif all(rtol_r):
            return 3
        elif all(rtol_dx):
            return 4

    def _monitor_block(self, snes, it, norm):
        """
        Monitor for block matrices.

        Parameters
        ----------
        snes : PETSc.SNES
            SNES solver.
        it : int
            Current iteration number.
        norm : float
            Norm of the solution update.
        """
        self.compute_norms_block(snes)
        self.print_norms(it)

    def print_norms(self, it):
        """
        Print the norms of the solution, updates, and residuals for each block.

        Parameters
        ----------
        it : int
            Current iteration number.
        """
        logging.info("\n### SNES iteration {}".format(it))
        for i, ui in enumerate(self.u):
            logging.info(
                "# sub {:2d} |x|={:1.3e} |dx|={:1.3e} |r|={:1.3e} ({})".format(
                    i,
                    self.norm_x[it][i],
                    self.norm_dx[it][i],
                    self.norm_r[it][i],
                    ui.name,
                )
            )
        logging.info(
            "# all    |x|={:1.3e} |dx|={:1.3e} |r|={:1.3e}".format(
                np.linalg.norm(np.asarray(self.norm_x[it])),
                np.linalg.norm(np.asarray(self.norm_dx[it])),
                np.linalg.norm(np.asarray(self.norm_r[it])),
            )
        )

    def compute_norms_block(self, snes):
        """
        Compute the norms of residuals, solution, and updates for block matrices.
        """
        r = snes.getFunction()[0].getArray(readonly=True)
        dx = snes.getSolutionUpdate().getArray(readonly=True)
        x = snes.getSolution().getArray(readonly=True)

        ei_r = []
        ei_dx = []
        ei_x = []

        offset = 0
        for i, ui in enumerate(self.u):
            if self.restriction is not None:
                size_local = self.restriction.bglobal_dofs_vec[i].shape[0]
            else:
                size_local = ui.vector.getLocalSize()

            subvec_r = r[offset : offset + size_local]
            subvec_dx = dx[offset : offset + size_local]
            subvec_x = x[offset : offset + size_local]

            ei_r.append(
                np.sqrt(self.comm.allreduce(np.linalg.norm(subvec_r) ** 2, op=MPI.SUM))
            )
            ei_dx.append(
                np.sqrt(self.comm.allreduce(np.linalg.norm(subvec_dx) ** 2, op=MPI.SUM))
            )
            ei_x.append(
                np.sqrt(self.comm.allreduce(np.linalg.norm(subvec_x) ** 2, op=MPI.SUM))
            )

            offset += size_local

        it = snes.getIterationNumber()
        self.norm_r[it] = ei_r
        self.norm_dx[it] = ei_dx
        self.norm_x[it] = ei_x

    def compute_norms_nest(self, snes):
        """
        Compute the norms of residuals, solution, and updates for nested matrices.
        """
        r = snes.getFunction()[0].getNestSubVecs()
        dx = snes.getSolutionUpdate().getNestSubVecs()
        x = snes.getSolution().getNestSubVecs()

        ei_r = []
        ei_dx = []
        ei_x = []

        for i in range(len(self.u)):
            ei_r.append(r[i].norm())
            ei_dx.append(dx[i].norm())
            ei_x.append(x[i].norm())

        it = snes.getIterationNumber()
        self.norm_r[it] = ei_r
        self.norm_dx[it] = ei_dx
        self.norm_x[it] = ei_x

    def solve(self, u_init=None):
        """
        Solve the nonlinear system.

        Parameters
        ----------
        u_init : list, optional
            Initial guess for the solution (default is None).

        Returns
        -------
        list
            Solution to the nonlinear system.
        """
        if u_init is not None:
            functions_to_vec(u_init, self.x)

        if self.restriction is not None:
            self.snes.solve(None, self.rx)
            self.restriction.update_functions(self.solution, self.rx)
        else:
            self.snes.solve(None, self.x)
            vec_to_functions(self.x, self.solution)

        return self.solution