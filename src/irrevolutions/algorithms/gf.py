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

    def solve(self, perturbation: dict = None, h: float = 0.0):
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
            alpha_array = self.alpha.x.array
            beta_array = beta.x.array
            self.alpha.x.array[:] = alpha_array + h * beta_array
            self.alpha.x.scatter_forward()

        t = 0.0
        for i in range(self.max_steps):
            self.alpha_old.x.array[:] = self.alpha.x.array[:]

            # Residual and directional derivative (Jacobian)
            E = self.energy_form(u=self.state["u"], alpha=self.alpha)
            # dE_alpha = fem.form(ufl.derivative(E, self.alpha, self.beta))

            dE_alpha = fem.petsc.assemble_vector(
                fem.form(ufl.derivative(E, self.alpha, ufl.TestFunction(self.V_alpha)))
            )

            # Project gradient onto the dual cone (positive part only)
            # drive alpha increase only where gradient is negative
            grad_proj = dE_alpha.copy()
            grad_proj.array[:] = np.minimum(grad_proj.array, 0.0)
            # Gradient descent step
            with (
                self.alpha.x.petsc_vec.localForm() as alpha_local,
                grad_proj.localForm() as grad_proj_local,
            ):
                alpha_local.axpy(-self.tau, grad_proj_local)

            self.alpha.x.scatter_forward()

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
