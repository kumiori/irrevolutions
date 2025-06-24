from dolfinx import fem, nls, la
from petsc4py import PETSc
import ufl
import numpy as np
from dolfinx.fem import Function
from dolfinx.common import Timer
from irrevolutions.utils import setup_logger_mpi
import logging
from mpi4py import MPI
from dolfinx.io import XDMFFile
from pathlib import Path

logger = setup_logger_mpi(logging.INFO)


class JumpSolver:
    def __init__(
        self, energy_form: ufl.form.Form, state: dict, bcs: list, parameters: dict
    ):
        """
        Initialize the JumpSolver for performing a constrained gradient-flow “jump.”

        Parameters
        ----------
        energy_form : ufl.form.Form
            A UFL form defining the total potential energy E(·) whose gradient w.r.t.
            the internal field drives the flow.
        state : dict
            Dictionary of Functions and auxiliary data:
            - state["alpha"]: the damage/internal‐variable Function to evolve
            - state["alpha_old"]: a copy used for convergence checks
            - optionally state["u"], etc.
        bcs : list of dolfinx.fem.DirichletBC
            Boundary conditions to enforce on the evolving field at each step.
        parameters : dict
            Algorithmic parameters, e.g.:
            - "tau": pseudo-time step Δτ
            - "maxit": maximum number of flow iterations
            - "tol": convergence tolerance
            - "outdir": optional directory for XDMF output
            - etc.
        """
        self.comm = state["u"].function_space.mesh.comm

        self.energy_form = energy_form
        self.state = state  # dict with keys "u" and "alpha"
        self.bcs = bcs
        self.tau = parameters.get("tau", 1e-2)
        self.max_steps = parameters.get("max_steps", 200)
        self.rtol = parameters.get("rtol", 1e-6)
        self.verbose = parameters.get("verbose", True)

        self.V_alpha = self.state["alpha"].function_space

        self.alpha = self.state["alpha"]  # pointer to quasistatically-evolving field
        self.u = self.state["u"]

        # self.alpha_old = fem.Function(self.V_alpha)

        # self.alpha.x.scatter_forward()
        # self.alpha.x.petsc_vec.copy(result=self.alpha_old.x.petsc_vec)
        self.alpha_old = self.alpha.copy()
        self.alpha_old.name = "Damage_old"

        self.parameters = parameters
        self.jump_data = {
            "iterations": [],
            "grad_norms": [],
            "alpha_diffs": [],
            "dissipation": [],
            "dissipated_energy": 0.0,
            "converged": False,
        }
        self.outdir = parameters.get("outdir", None)
        self.jump_counter = 0

        if parameters.get("save_state", False):
            if self.outdir is None:
                raise ValueError("Output directory must be specified for saving state.")
            Path(self.outdir).mkdir(parents=True, exist_ok=True)
            self.fname = f"{self.outdir}/jumps.xdmf"

            with XDMFFile(self.comm, self.fname, "w") as file:
                file.write_mesh(self.u.function_space.mesh)

    def solve(self, perturbation: dict = None, h: float = 0.0):
        """
        Perform one jump by running the projected gradient descent on the damage field.

        Parameters
        ----------
        perturbation : dict, >0 pointwise
            A map (e.g. {"u", Function, "alpha": Function}) providing an initial perturbation
            to bias the first step.
            TODO: If None, uses the negative part of the computed gradient.
        h : float, >0
            Scaling factor for the perturbation in the first iterate (default: 0.0).

        Returns
        -------
        dolfinx.Function
            The updated damage/internal-variable Function after convergence or reaching
            the maximum iterations.
        """
        alpha_local_sum = self.comm.allreduce(self.alpha.x.array[:].sum(), op=MPI.SUM)
        logger.info(f"Total alpha sum before gradient flow: {alpha_local_sum}")
        logging.info(
            f"Damage field norm before jump: {self.alpha.x.petsc_vec.norm(PETSc.NormType.NORM_2):.4e}"
        )
        logger.info(f"Jump counter: {self.jump_counter}")

        # Create copies of alpha and u for the jth-jump

        alpha_j = self.alpha.copy()
        u_j = self.u.copy()

        alpha_j.name = f"Damage_jump_{self.jump_counter:03}"
        u_j.name = f"Displacement_jump_{self.jump_counter:03}"

        with Timer("~Jump Solver"):
            beta = perturbation.get("beta", None)

            if beta is None:
                raise ValueError("Perturbation 'beta' must be provided.")
            if not isinstance(beta, fem.Function):
                raise TypeError("Perturbation 'beta' must be a dolfinx.fem.Function.")
            if beta.function_space != self.V_alpha:
                raise ValueError(
                    "Perturbation 'beta' must be defined on the same function space as 'alpha'."
                )

            if h > 0.0:
                with (
                    alpha_j.x.petsc_vec.localForm() as alpha_local,
                    beta.x.petsc_vec.localForm() as beta_local,
                ):
                    # Apply perturbation to alpha
                    # axpy = alpha_local + h * beta_local
                    alpha_local.axpy(h, beta_local)

                alpha_j.x.scatter_forward()
            else:
                raise ValueError(
                    "Perturbation step 'h' must be greater than 0.0 to apply the perturbation."
                )
            dissipation = 0.0

            for i in range(self.max_steps):
                alpha_j.x.scatter_forward()
                alpha_j.x.petsc_vec.copy(result=self.alpha_old.x.petsc_vec)

                # Residual and directional derivative (Jacobian)
                # energy = self.energy_form(u=self.u, alpha=self.alpha)
                energy = self.energy_form

                dE_alpha = fem.petsc.assemble_vector(
                    fem.form(
                        -ufl.derivative(
                            energy, self.alpha, ufl.TestFunction(self.V_alpha)
                        )
                    )
                )
                # Project gradient onto the dual cone (positive part only)
                # drive alpha increase only where gradient is negative

                grad_proj = dE_alpha.copy()
                zero_vec = grad_proj.copy()
                zero_vec.set(0.0)

                with (
                    grad_proj.localForm() as grad_proj_local,
                    zero_vec.localForm() as zero_vec_local,
                ):
                    grad_proj_local.pointwiseMin(grad_proj_local, zero_vec_local)

                grad_proj.ghostUpdate(
                    addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
                )
                __import__("pdb").set_trace()
                if self.verbose:
                    print(f"Step {i}: ||grad_proj||_2 = {grad_proj.norm(2):.4e}")
                    print(f"Step {i}: grad_proj[:5] = {grad_proj.array[:5]}")

                # Gradient descent step
                with (
                    alpha_j.x.petsc_vec.localForm() as alpha_local,
                    grad_proj.localForm() as grad_proj_local,
                    self.alpha_old.x.petsc_vec.localForm() as alpha_old_local,
                ):
                    alpha_local.axpy(-self.tau, grad_proj_local)

                alpha_j.x.scatter_forward()

                if self.bcs["bcs_alpha"]:
                    for bc in self.bcs["bcs_alpha"]:
                        bc.set(alpha_j.x.array)

                    alpha_j.x.scatter_forward()

                    if self.verbose:
                        print(
                            f"Step {i}: alpha_local size = {alpha_local.size}, grad_proj_local size = {grad_proj_local.size}"
                        )
                        print(
                            f"Step {i}: alpha_local array = {alpha_local.array[:5]}, grad_proj_local array = {grad_proj_local.array[:5]}"
                        )
                        print(
                            f"Step {i}: Updated alpha_local array = {alpha_local.array[:5]}"
                        )
                        print(
                            f"Differences of local arrays: {alpha_local.array - alpha_old_local.array}"
                        )

                        # Global (MPI-wide) diagnostics
                        grad_proj_norm = grad_proj.norm(PETSc.NormType.NORM_2)
                        grad_proj_max = grad_proj.max()[1]  # Returns (index, value)
                        grad_proj_min = grad_proj.min()[1]  # Returns (index, value)

                        # if self.comm.rank == 0:
                        print(
                            f"{self.comm.rank}: Step {i}: ||grad_proj||_2 = {grad_proj_norm:.4e}"
                        )
                        print(
                            f"{self.comm.rank}: Step {i}: max(grad_proj) = {grad_proj_max:.4e}"
                        )
                        print(
                            f"{self.comm.rank}: Step {i}: min(grad_proj) = {grad_proj_min:.4e}"
                        )

                # Scatter the updated alpha to all processes
                # self.alpha.x.petsc_vec.ghostUpdate(
                #     addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
                # )

                # Check convergence
                diff_vec = alpha_j.x.petsc_vec.copy()
                diff_vec.axpy(-1.0, self.alpha_old.x.petsc_vec)
                diff = diff_vec.norm(PETSc.NormType.NORM_2) ** 2

                if self.verbose:
                    logger.info(f"Step {i}: ||alpha_j(s) - alpha_old||^2 = {diff:.4e}")

                norm_grad_proj = grad_proj.norm(PETSc.NormType.NORM_2)
                norm_diff = diff_vec.norm(PETSc.NormType.NORM_2)

                dissipation_increment = norm_grad_proj**2 * self.tau
                dissipation += dissipation_increment

                self.jump_data["iterations"].append(i)
                self.jump_data["grad_norms"].append(norm_grad_proj)
                self.jump_data["alpha_diffs"].append(norm_diff)
                self.jump_data["dissipation"].append(dissipation_increment)

                if self.parameters.get("save_state", False):
                    self.save_state(state={"u": u_j, "alpha": alpha_j}, s=self.tau * i)

                if diff < self.rtol:
                    self.jump_data["converged"] = True
                    self.jump_data["dissipated_energy"] = dissipation
                    self.jump_counter += 1

                    alpha_j.x.petsc_vec.copy(result=self.alpha.x.petsc_vec)

                    if self.verbose:
                        logger.critical(f"Jump {self.jump_counter} converged.")
                    break

        return dissipation

    def save_state(self, state, s=None):
        """
        Record or write out the current state during the jump iteration.

        Parameters
        ----------
        state : dict
            The same dictionary passed into __init__, with updated Function values.
        s : float or None, optional
            The pseudo‐time or iteration index at which this snapshot is taken. Used as
            the time tag if writing to XDMF. If None, uses the internal iteration counter.

        Notes
        -----
        - Uses renamed `state["alpha"]` to include the current jump index,
          writes XDMF fields if enabled.
        """
        u = state["u"]
        alpha = state["alpha"]

        with XDMFFile(self.comm, self.fname, "a") as file:
            # file.write_mesh(self.u.function_space.mesh)
            if u is not None:
                file.write_function(u, s)
            if alpha is not None:
                file.write_function(alpha, s)

    def log(self, logger=logger):
        """
        Log the current state of the jump solver.

        Parameters
        ----------
        logger : logging.Logger, optional
            The logger to use for logging. If None, uses the default logger.
        """
        if logger is None:
            logger = logging.getLogger(__name__)

        logger.info(f"Jump {self.jump_counter}:")
        logger.info(f"  Converged: {self.jump_data['converged']}")
        logger.info(f"  Dissipation: {self.jump_data['dissipation'][-1]:.4e}")
        logger.info(f"  Gradient Norm: {self.jump_data['grad_norms'][-1]:.4e}")
        logger.info(f"  Alpha Diff: {self.jump_data['alpha_diffs'][-1]:.4e}")
        logger.info(
            f"  Total Dissipated Energy: {self.jump_data['dissipated_energy']:.4e}"
        )
        logger.info(f"  Total Iterations: {len(self.jump_data['iterations'])}")
