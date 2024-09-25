import logging

import dolfinx
import numpy as np
import ufl
from dolfinx.fem import Function, assemble_scalar, form
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc

from irrevolutions.solvers import SNESSolver
from irrevolutions.solvers.function import functions_to_vec
from irrevolutions.solvers.snesblockproblem import SNESBlockProblem
from irrevolutions.utils import (ColorPrint, norm_H1, norm_L2,
                                 set_vector_to_constant)
from dolfinx.fem.petsc import assemble_vector, set_bc

comm = MPI.COMM_WORLD

# Set up basic logging
logging.basicConfig()

class AlternateMinimisation:
    """
    First order Alternate Minimisation solver for elasticity and damage fields.

    The solver seeks equilibia (critical states for the total energy) of a system 
    by alternating between solving for the elasticity and the damage field in an iterative process,
    exploiting the separate convexity of the total energy.
    
    Remark: check the assumptions.

    Parameters:
    ----------
    total_energy : ufl.Form
        Total energy functional of the system to be minimized.
    state : dict
        Dictionary containing the current state of the system with keys 'u' (displacement)
        and 'alpha' (damage).
    bcs : dict
        Dictionary of boundary conditions with keys 'bcs_u' for displacement and 'bcs_alpha' 
        for damage.
    solver_parameters : dict, optional
        Dictionary of solver parameters, default is an empty dictionary.
    bounds : tuple of dolfinx.fem.function.Function, optional
        Tuple containing lower and upper bounds for the damage field (default: no bounds).
    monitor : callable, optional
        Function to monitor progress, if provided.
    """

    def __init__(
        self,
        total_energy,
        state,
        bcs,
        solver_parameters={},
        bounds=(dolfinx.fem.function.Function, dolfinx.fem.function.Function),
        monitor=None,
    ):
        # Initialize state and energy
        self.u = state["u"]
        self.alpha = state["alpha"]
        self.alpha_old = Function(self.alpha.function_space)
        self.alpha.x.petsc_vec.copy(self.alpha_old.x.petsc_vec)
        self.alpha.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        self.total_energy = total_energy
        self.state = state
        self.alpha_lb = bounds[0]
        self.alpha_ub = bounds[1]
        self.solver_parameters = solver_parameters
        self.monitor = monitor

        V_u = self.u.function_space
        V_alpha = self.alpha.function_space

        # Derivatives of the total energy
        energy_u = ufl.derivative(self.total_energy, self.u, ufl.TestFunction(V_u))
        energy_alpha = ufl.derivative(
            self.total_energy, self.alpha, ufl.TestFunction(V_alpha)
        )
        self.F = [energy_u, energy_alpha]

        # SNESSolver for elasticity and damage
        self.elasticity = SNESSolver(
            energy_u,
            self.u,
            bcs.get("bcs_u"),
            petsc_options=self.solver_parameters.get("elasticity").get("snes"),
            prefix=self.solver_parameters.get("elasticity").get("prefix"),
        )

        self.damage = SNESSolver(
            energy_alpha,
            self.alpha,
            bcs.get("bcs_alpha"),
            bounds=(self.alpha_lb, self.alpha_ub),
            petsc_options=self.solver_parameters.get("damage").get("snes"),
            prefix=self.solver_parameters.get("damage").get("prefix"),
        )

    def solve(self, outdir=None):
        """
        Solve the problem using alternating enforcing first order conditions
        for energy minimality.

        Parameters:
        ----------
        outdir : str, optional
            Output directory for saving intermediate results. If provided, the mesh 
            and functions will be written to an XDMF file.
        """

        alpha_diff = dolfinx.fem.Function(self.alpha.function_space)
        self.data = {
            "iteration": [], "error_alpha_L2": [], "error_alpha_H1": [], "F_norm": [],
            "error_alpha_max": [], "error_residual_F": [], "error_residual_u": [],
            "solver_alpha_reason": [], "solver_alpha_it": [], "solver_u_reason": [],
            "solver_u_it": [], "total_energy": []
        }

        # Write the initial mesh
        if outdir:
            with XDMFFile(
                comm,
                f"{outdir}/fields.xdmf",
                "w",
                encoding=XDMFFile.Encoding.HDF5,
            ) as file:
                file.write_mesh(self.u.function_space.mesh)

        for iteration in range(1, self.solver_parameters.get("damage_elasticity").get("max_it")):
            # Elasticity solver step
            with dolfinx.common.Timer("~First Order: AltMin-Elastic solver"):
                (solver_u_it, solver_u_reason) = self.elasticity.solve()
            
            # Damage solver step
            with dolfinx.common.Timer("~First Order: AltMin-Damage solver"):
                (solver_alpha_it, solver_alpha_reason) = self.damage.solve()

            # Compute errors and residuals
            self.alpha.x.petsc_vec.copy(alpha_diff.x.petsc_vec)
            alpha_diff.x.petsc_vec.axpy(-1, self.alpha_old.x.petsc_vec)
            alpha_diff.x.petsc_vec.ghostUpdate(

                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )

            error_alpha_H1 = norm_H1(alpha_diff)
            error_alpha_L2 = norm_L2(alpha_diff)

            Fv = [assemble_vector(form(F)) for F in self.F]
            Fnorm = np.sqrt(np.array([comm.allreduce(Fvi.norm(), op=MPI.SUM) for Fvi in Fv]).sum())

            error_alpha_max = alpha_diff.x.petsc_vec.max()[1]
            total_energy_int = comm.allreduce(assemble_scalar(form(self.total_energy)), op=MPI.SUM)

            residual_u = assemble_vector(self.elasticity.F_form)
            residual_u.ghostUpdate(
                addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
            )
            set_bc(residual_u, self.elasticity.bcs, self.u.x.petsc_vec)
            error_residual_u = ufl.sqrt(residual_u.dot(residual_u))

            self.alpha.x.petsc_vec.copy(self.alpha_old.x.petsc_vec)
            self.alpha_old.x.petsc_vec.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )

            # Store results
            self.data["iteration"].append(iteration)
            self.data["error_alpha_L2"].append(error_alpha_L2)
            self.data["error_alpha_H1"].append(error_alpha_H1)
            self.data["F_norm"].append(Fnorm)
            self.data["error_alpha_max"].append(error_alpha_max)
            self.data["error_residual_F"].append(Fnorm)
            self.data["error_residual_u"].append(error_residual_u)
            self.data["solver_alpha_it"].append(solver_alpha_it)
            self.data["solver_alpha_reason"].append(solver_alpha_reason)
            self.data["solver_u_reason"].append(solver_u_reason)
            self.data["solver_u_it"].append(solver_u_it)
            self.data["total_energy"].append(total_energy_int)

            # Save field data
            if outdir:
                with XDMFFile(
                    comm,
                    f"{outdir}/fields.xdmf",
                    "a",
                    encoding=XDMFFile.Encoding.HDF5,
                ) as file:
                    file.write_function(self.u, iteration)
                    file.write_function(self.alpha, iteration)

            # Monitoring and convergence check
            if self.monitor is not None:
                self.monitor(self)

            # Convergence criteria
            if self.solver_parameters.get("damage_elasticity").get("criterion") == "residual_u":
                logging.debug(f"AM - Iteration: {iteration:3d}, Error: ||Du E||_L2 {error_residual_u:3.4e}, alpha_max: {self.alpha.x.petsc_vec.max()[1]:3.4e}")
                if error_residual_u <= self.solver_parameters.get("damage_elasticity").get("alpha_rtol"):
                    error = error_residual_u
                    break

            if self.solver_parameters.get("damage_elasticity").get("criterion") == "alpha_H1":
                logging.debug(f"AM - Iteration: {iteration:3d}, Error ||Δα_i||_H1: {error_alpha_H1:3.4e}, alpha_max: {self.alpha.x.petsc_vec.max()[1]:3.4e}")
                if error_alpha_H1 <= self.solver_parameters.get("damage_elasticity").get("alpha_rtol"):
                    error = error_alpha_H1
                    break
        else:
            raise RuntimeError(f"Could not converge after {iteration:3d} iterations, error {error_alpha_H1:3.4e}")

        _crit = self.solver_parameters.get("damage_elasticity").get("criterion")
        ColorPrint.print_info(f"ALTMIN - Iterations: {iteration:3d}, Error: {error:3.4e}, {_crit}, alpha_max: {self.alpha.x.petsc_vec.max()[1]:3.4e}")


class HybridSolver(AlternateMinimisation):
    """
    Hybrid (AltMin+Newton) solver for fracture problems.

    This solver combines alternating minimization for elasticity and damage 
    fields with a Newton method to improve convergence and performance. 
    It ensures that the total energy of the system is minimized efficiently, 
    using a combination of the two methods.
    First Order Hybrid (AlternateMinimisation + Newton) solver for fracture problems.
    Combines alternating minimisation to convergence with a coarse tolerance,
    to a Newton step on the full nonlinear functional, for sharp convergence.
    
    Parameters
    ----------
    total_energy : ufl.Form
        The total energy functional for the system.
    state : dict
        Dictionary containing the state of the system, 
        including 'u' (displacement) and 'alpha' (damage) fields.
    bcs : dict
        Boundary conditions dictionary containing 'bcs_u' for displacement and 'bcs_alpha' for damage.
    solver_parameters : dict, optional
        Dictionary containing solver parameters.
    bounds : tuple of dolfinx.fem.Function, optional
        Tuple containing the lower and upper bounds for the displacement and damage fields.
    monitor : callable, optional
        Function to monitor solver progress, if provided.
    """

    def __init__(
        self,
        total_energy,
        state,
        bcs,
        solver_parameters={},
        bounds=(dolfinx.fem.Function, dolfinx.fem.Function),
        monitor=None,
    ):
        super(HybridSolver, self).__init__(
            total_energy, state, bcs, solver_parameters, bounds, monitor
        )

        self.u_lb = dolfinx.fem.Function(
            state["u"].function_space, name="displacement lower bound"
        )
        self.u_ub = dolfinx.fem.Function(
            state["u"].function_space, name="displacement upper bound"
        )
        self.alpha_lb = bounds[0]
        self.alpha_ub = bounds[1]

        set_vector_to_constant(self.u_lb.x.petsc_vec, PETSc.NINFINITY)
        set_vector_to_constant(self.u_ub.x.petsc_vec, PETSc.PINFINITY)
        set_vector_to_constant(self.alpha_lb.x.petsc_vec, 0)
        set_vector_to_constant(self.alpha_ub.x.petsc_vec, 1)

        self.z = [self.u, self.alpha]
        bcs_z = bcs.get("bcs_u") + bcs.get("bcs_alpha")
        self.prefix = "blocknewton"
        nest = False
        self.newton = SNESBlockProblem(
            self.F, self.z, bcs=bcs_z, nest=nest, prefix="block"
        )
        newton_options = self.solver_parameters.get("newton", self.default_options())
        self.set_newton_options(newton_options)
        logging.info(self.newton.snes.getTolerances())

        self.lb = dolfinx.fem.petsc.create_vector_nest(self.newton.F_form)
        self.ub = dolfinx.fem.petsc.create_vector_nest(self.newton.F_form)
        functions_to_vec([self.u_lb, self.alpha_lb], self.lb)
        functions_to_vec([self.u_ub, self.alpha_ub], self.ub)

    def default_options(self):
        """
        Default options for the Newton solver.

        Returns
        -------
        dict
            A dictionary containing the default PETSc options for the Newton solver.
        """
        opts = PETSc.Options(self.prefix)
        opts.setValue("snes_type", "vinewtonrsls")
        opts.setValue("snes_linesearch_type", "basic")
        opts.setValue("snes_rtol", 1.0e-08)
        opts.setValue("snes_atol", 1.0e-08)
        opts.setValue("snes_max_it", 30)
        opts.setValue("snes_monitor", "")
        opts.setValue("linesearch_damping", 0.5)

        nest = False

        if nest:
            opts.setValue("ksp_type", "cg")
            opts.setValue("pc_type", "fieldsplit")
            opts.setValue("fieldsplit_pc_type", "lu")
            opts.setValue("ksp_rtol", 1.0e-10)
        else:
            opts.setValue("ksp_type", "preonly")
            opts.setValue("pc_type", "lu")
            opts.setValue("pc_factor_mat_solver_type", "mumps")

        return opts

    def set_newton_options(self, newton_options):
        """
        Set custom options for the Newton solver.

        Parameters
        ----------
        newton_options : dict
            A dictionary of options to override the default settings for the Newton solver.
        """
        opts = PETSc.Options(self.prefix)
        logging.info(newton_options)

        for k, v in newton_options.items():
            opts[k] = v

        self.newton.snes.setOptionsPrefix(self.prefix)
        self.newton.snes.setFromOptions()

    def compute_bounds(self, v, alpha_lb):
        """
        Compute the bounds for the solution.

        Parameters
        ----------
        v : list of dolfinx.fem.Function
            List of functions for which bounds are being computed.
        alpha_lb : dolfinx.fem.Function
            The lower bound for the damage field.

        Returns
        -------
        tuple
            A tuple containing the lower and upper bounds.
        """
        lb = dolfinx.fem.create_vector_nest(v)
        ub = dolfinx.fem.create_vector_nest(v)

        with lb.getNestSubVecs()[0].localForm() as u_sub:
            u_sub.set(PETSc.NINFINITY)

        with ub.getNestSubVecs()[0].localForm() as u_sub:
            u_sub.set(PETSc.PINFINITY)

        with lb.getNestSubVecs()[1].localForm() as alpha_sub, alpha_lb.vector.localForm() as alpha_lb_loc:
            alpha_lb_loc.copy(result=alpha_sub)

        with ub.getNestSubVecs()[1].localForm() as alpha_sub:
            alpha_sub.set(1.0)

        return lb, ub

    def scaled_rate_norm(self, alpha, parameters):
        """
        Compute the scaled norm of the rate function.

        Parameters
        ----------
        alpha : dolfinx.fem.Function
            The damage field.
        parameters : dict
            Model parameters.

        Returns
        -------
        float
            The scaled norm of the rate function.
        """
        dx = ufl.Measure("dx", alpha.function_space.mesh)
        _form = dolfinx.fem.form(
            (
                ufl.inner(alpha, alpha)
                + parameters["model"]["ell"] ** 2.0 * ufl.inner(ufl.grad(alpha), ufl.grad(alpha))
            ) * dx
        )
        return np.sqrt(comm.allreduce(assemble_scalar(_form), op=MPI.SUM))

    def unscaled_rate_norm(self, alpha):
        """
        Compute the unscaled norm of the rate function.

        Parameters
        ----------
        alpha : dolfinx.fem.Function
            The damage field.

        Returns
        -------
        float
            The unscaled norm of the rate function.
        """
        dx = ufl.Measure("dx", alpha.function_space.mesh)
        _form = dolfinx.fem.form(
            (ufl.inner(alpha, alpha) + ufl.inner(ufl.grad(alpha), ufl.grad(alpha))) * dx
        )
        return np.sqrt(comm.allreduce(assemble_scalar(_form), op=MPI.SUM))

    def getReducedNorm(self):
        """
        Retrieve the reduced residual norm of the system.

        Returns
        -------
        float
            The reduced residual norm.
        """
        self.newton.compute_norms_block(self.newton.snes)
        return self.newton.norm_r

    def monitor(self, its, rnorm):
        """
        Monitor the iteration process by printing the number of iterations and the residual norm.

        Parameters
        ----------
        its : int
            The number of iterations.
        rnorm : float
            The residual norm.
        """
        logging.info("Num it, rnorm:", its, rnorm)

    def solve(self, alpha_lb, outdir=None):
        """
        Solve the fracture problem using the hybrid solver.

        Parameters
        ----------
        alpha_lb : dolfinx.fem.Function
            The lower bound for the damage field.
        outdir : str, optional
            Directory for saving output fields.
        """
        # Perform AM as customary
        with dolfinx.common.Timer("~First Order: AltMin solver"):
            super().solve(outdir)

        self.newton_data = {
            "iteration": [],
            "residual_Fnorm": [],
            "residual_Frxnorm": [],
        }

        # Update bounds and perform Newton step
        with dolfinx.common.Timer("~First Order: Hybrid solver"):
            functions_to_vec([self.u_lb, self.alpha_lb], self.lb)
            self.newton.snes.setVariableBounds(self.lb, self.ub)
            self.newton.solve(u_init=[self.u, self.alpha])

        # Save Newton iteration data
        self.newton_data["iteration"].append(self.newton.snes.getIterationNumber() + 1)
        self.newton_data["residual_Fnorm"].append(self.newton.snes.getFunctionNorm())
        self.newton_data["residual_Frxnorm"].append(self.getReducedNorm())

        self.data.update(self.newton_data)