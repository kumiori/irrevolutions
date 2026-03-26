import ufl
from dolfinx.fem import Constant, form, assemble_scalar
from irrevolutions.utils import setup_logger_mpi
import logging
from numpy import nan

logger = setup_logger_mpi(logging.INFO)


class Brittle1D:
    """
    Base class for 1D models.
    This class provides a unified interface for elastic and damage energy density
    computations in 1D.
    """

    def __init__(self, parameters, eps_0):
        """
        Initialize model parameters.
        """
        self.parameters = parameters
        self.eps_0 = eps_0

    def a(self, alpha):
        """
        Damage degradation function.
        """
        return (1 - alpha) ** 2

    def w(self, alpha):
        """
        Homogeneous damage energy function.
        """
        n = self.parameters["at_number"]
        return alpha**n

    def grad_1d(self, u):
        """
        Gradient computation in 1D.
        """
        return ufl.grad(u)

    def elastic_energy_density(self, state):
        """
        Elastic energy density for the 1D state.
        """
        alpha = state["alpha"]
        u = state["u"]
        u_x = self.grad_1d(u)[0, 0] - self.eps_0
        _mu = self.parameters["E"]

        return _mu / 2.0 * self.a(alpha) * u_x**2

    def damage_energy_density(self, state):
        """
        Damage energy density for the 1D state.
        """
        _w1 = self.parameters["w1"]
        _ell = self.parameters["ell"]

        alpha = state["alpha"]
        grad_alpha = self.grad_1d(alpha)

        return _w1 * self.w(alpha) + _w1 * _ell**2 / 2.0 * grad_alpha**2

    def stress(self, state):
        """
        Compute 1D stress.
        """
        alpha = state["alpha"]
        u = state["u"]
        u_x = self.grad_1d(u)

        return self.parameters["E"] * self.a(alpha) * u_x

    def s(self, alpha):
        """
        Compute the compliance from the state.
        """
        return 1 / self.a(alpha)

    def _a_prime(self, alpha):
        return ufl.diff(self.a(alpha), alpha)

    def _a_dd(self, alpha):
        return ufl.diff(self._a_prime(alpha), alpha)

    def _s_dd(self, alpha):
        return ufl.diff(ufl.diff(self.s(alpha), alpha), alpha)

    def _w_prime(self, alpha):
        return ufl.diff(self.w(alpha), alpha)

    def _w_dd(self, alpha):
        return ufl.diff(self._w_prime(alpha), alpha)

    def rayleigh_coeffs(self, state, perturbation):
        """
        Compute the Rayleigh coefficients for the given state and perturbation.
        """
        u = state["u"]
        alpha = state["alpha"]

        mesh = alpha.function_space.mesh
        dx = ufl.Measure("dx", domain=mesh)
        tol_hom = 1e-6

        alpha_grad = ufl.grad(alpha)
        seminorm_form = form(ufl.inner(alpha_grad, alpha_grad) * dx)
        seminorm = assemble_scalar(seminorm_form)
        if seminorm > tol_hom:
            raise ValueError(
                f"Damage field is not homogeneous: H1 seminorm = {seminorm}"
            )

        a_val = 0
        b_val = 0
        c_val = 0

        rayleigh_coeffs = {"a": a_val, "b": b_val, "c": c_val}

        return rayleigh_coeffs

    def rayleigh_ratio(self, state, perturbation):
        dx = ufl.Measure("dx", state["alpha"].function_space.mesh)
        a = self.a(state["alpha"])
        u = state["u"]
        a_p = self._a_prime(state["alpha"])
        s_dd = self._s_dd(state["alpha"])
        w_dd = self._w_dd(state["alpha"])

        sigma = self.stress(state)

        ell = self.parameters["ell"]

        if perturbation is None:
            logger.info("Computing Rayleigh ratio for state, no perturbation.")
            return {
                "N": nan,
                "D": nan,
            }

        v = perturbation["v"]
        beta = perturbation["β"]

        # ------------------------------------------------------------------
        # Rayleigh numerator N(yε)(z)^2
        # ------------------------------------------------------------------
        # Term (∇v + (a'/a) ∇u β)
        inner_term = ufl.grad(v) + (a_p / a) * ufl.grad(u) * beta
        N_form = (
            a * ufl.inner(inner_term, inner_term) * dx
            + ell**2 * ufl.inner(ufl.grad(beta), ufl.grad(beta)) * dx
        )

        # ------------------------------------------------------------------
        # Rayleigh denominator D(yε)(z)^2
        # ------------------------------------------------------------------
        D_pref = 1 / 2 * s_dd * ufl.inner(sigma, sigma) - w_dd
        D_form = D_pref * beta**2 * dx

        if perturbation is None:
            rayleigh_terms = {"N": form(N_form), "D": form(D_form)}
        else:
            rayleigh_terms = {
                "N": assemble_scalar(form(N_form)),
                "D": assemble_scalar(form(D_form)),
            }
        return rayleigh_terms


class FilmModel1D(Brittle1D):
    """
    Model for thin films in 1D.
    Includes substrate interaction energy density.
    """

    def elastic_energy_density(self, state, u_zero=None):
        """
        Elastic energy density including substrate interaction.
        """
        alpha = state["alpha"]
        u = state["u"]
        u_x = self.grad_1d(u)
        _mu = self.parameters["E"]
        _kappa = self.parameters.get("kappa", 1.0)

        energy_density = _mu / 2.0 * self.a(alpha) * u_x**2

        if u_zero is None:
            u_zero = Constant(u.function_space.mesh, 0.0)

        substrate_density = _kappa / 2.0 * (u - u_zero) ** 2

        return energy_density + substrate_density
