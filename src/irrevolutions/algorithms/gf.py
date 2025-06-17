from dolfinx import fem, nls, la
from petsc4py import PETSc
import ufl
import numpy as np
from dolfinx.fem import Function


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
        self.energy_form = energy_form
        self.state = state  # e.g. dict with keys "u" and "alpha"
        # self.perturbation = perturbation  # dict with keys "v", "beta"
        self.bcs = bcs
        self.tau = parameters.get("tau", 1e-2)
        self.max_steps = parameters.get("max_steps", 200)
        self.rtol = parameters.get("rtol", 1e-6)
        self.verbose = parameters.get("verbose", False)

        self.V_alpha = self.state["alpha"].function_space
        self.alpha = fem.Function(self.V_alpha, name="alpha_jump")
        self.alpha.x.array[:] = self.state["alpha"].x.array[:]
        self.alpha_old = fem.Function(self.V_alpha)

        # self.beta = self.perturbation["beta"]

    def solve(self):
        t = 0.0
        for i in range(self.max_steps):
            self.alpha_old.x.array[:] = self.alpha.x.array[:]

            # Residual and directional derivative (Jacobian)
            E = self.energy_form(u=self.state["u"], alpha=self.alpha)
            dE_alpha = fem.form(ufl.derivative(E, self.alpha, self.beta))

            # Gradient descent step
            with self.alpha.vector.localForm() as alpha_local:
                alpha_local.axpy(-self.tau, fem.petsc.assemble_vector(dE_alpha))

            self.alpha.x.scatter_forward()

            # Project alpha back onto admissible set (irreversibility: alpha increases)
            with (
                self.alpha.vector.localForm() as a,
                self.alpha_old.vector.localForm() as a_old,
            ):
                a.setArray(np.maximum(a.array, a_old.array))

            # Check convergence
            diff = fem.assemble_scalar(
                fem.form(
                    ufl.inner(self.alpha - self.alpha_old, self.alpha - self.alpha_old)
                    * ufl.dx
                )
            )
            if self.verbose:
                print(f"Step {i}: ||alpha - alpha_old||^2 = {diff:.4e}")

            if diff < self.rtol:
                if self.verbose:
                    print("Converged.")
                break

        # Update state
        self.state["alpha"].x.array[:] = self.alpha.x.array[:]
        self.state["alpha"].x.scatter_forward()
        return self.state
