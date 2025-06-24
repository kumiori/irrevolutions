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
        Initialize the gradient flow solver.

        Args:
            energy (ufl.form.Form): The energy functional.
            state (dict): Dictionary containing state variables 'u' and 'alpha'.
            bcs (list): List of boundary conditions.
            parameters (dict): Parameters for the solver.
        """
        self.comm = state["u"].function_space.mesh.comm

        self.energy_form = energy_form
        self.state = state  # e.g. dict with keys "u" and "alpha"
        self.bcs = bcs
        self.tau = parameters.get("tau", 1e-2)
        self.max_steps = parameters.get("max_steps", 200)
        self.rtol = parameters.get("rtol", 1e-6)
        self.verbose = parameters.get("verbose", True)

        self.V_alpha = self.state["alpha"].function_space

        self.alpha = self.state["alpha"]  # pointer to evolving field
        self.u = self.state["u"]

        self.alpha_old = fem.Function(self.V_alpha)

        self.alpha.x.scatter_forward()
        self.alpha.x.petsc_vec.copy(result=self.alpha_old.x.petsc_vec)

        self.jump_data = {
            "iterations": [],
            "grad_norms": [],
            "alpha_diffs": [],
            "dissipation": [],
            "dissipated_energy": 0.0,
            "converged": False,
        }
        self.outdir = parameters.get("outdir", None)

        if parameters.get("save_state", False):
            if self.outdir is None:
                raise ValueError("Output directory must be specified for saving state.")
            Path(self.outdir).mkdir(parents=True, exist_ok=True)
            self.fname = f"{self.outdir}/jumps.xdmf"

            with XDMFFile(self.comm, self.fname, "w") as file:
                file.write_mesh(self.u.function_space.mesh)

    def solve(self, perturbation: dict = None, h: float = 0.0):
        alpha_local_sum = self.comm.allreduce(self.alpha.x.array[:].sum(), op=MPI.SUM)
        logger.critical(f"Total alpha sum before loop: {alpha_local_sum}")

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
                    self.alpha.x.petsc_vec.localForm() as alpha_local,
                    beta.x.petsc_vec.localForm() as beta_local,
                ):
                    # Apply perturbation to alpha
                    alpha_local.axpy(h, beta_local)

                self.alpha.x.scatter_forward()

            dissipation = 0.0

            for i in range(self.max_steps):
                self.alpha.x.scatter_forward()
                self.alpha.x.petsc_vec.copy(result=self.alpha_old.x.petsc_vec)

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
                if self.verbose:
                    print(f"Step {i}: ||grad_proj||_2 = {grad_proj.norm(2):.4e}")
                    print(f"Step {i}: grad_proj[:5] = {grad_proj.array[:5]}")

                # Gradient descent step
                with (
                    self.alpha.x.petsc_vec.localForm() as alpha_local,
                    grad_proj.localForm() as grad_proj_local,
                    self.alpha_old.x.petsc_vec.localForm() as alpha_old_local,
                ):
                    alpha_local.axpy(-self.tau, grad_proj_local)

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
                self.alpha.x.scatter_forward()

                # Check convergence
                diff_vec = self.alpha.x.petsc_vec.copy()
                diff_vec.axpy(-1.0, self.alpha_old.x.petsc_vec)
                diff = diff_vec.norm(PETSc.NormType.NORM_2) ** 2

                if self.verbose:
                    print(f"Step {i}: ||alpha - alpha_old||^2 = {diff:.4e}")

                norm_grad_proj = grad_proj.norm(PETSc.NormType.NORM_2)
                norm_diff = diff_vec.norm(PETSc.NormType.NORM_2)

                dissipation_increment = norm_grad_proj**2 * self.tau
                dissipation += dissipation_increment

                self.jump_data["iterations"].append(i)
                self.jump_data["grad_norms"].append(norm_grad_proj)
                self.jump_data["alpha_diffs"].append(norm_diff)
                self.jump_data["dissipation"].append(dissipation_increment)

                if self.parameters.get("save_state", False):
                    self.save_state(time=self.tau * i)

                if diff < self.rtol:
                    self.jump_data["converged"] = True
                    self.jump_data["dissipated_energy"] = dissipation

                    if self.verbose:
                        print("Converged.")
                    break

        return self.state

    def save_state(self, step=None):
        u = self.state["u"]
        alpha = self.state["alpha"]
        __import__("pdb").set_trace()
        with XDMFFile(self.comm, self.fname, "a") as file:
            # file.write_mesh(self.u.function_space.mesh)
            if u is not None:
                file.write_function(u, f"u_jump_{step}")
            if alpha is not None:
                file.write_function(alpha, f"alpha_jump_{step}")
