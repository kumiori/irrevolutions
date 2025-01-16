import ufl
from dolfinx.fem import Constant, Function


class Brittle1D:
    """
    Base class for 1D models.
    This class provides a unified interface for elastic and damage energy density
    computations in 1D.
    """

    def __init__(self, parameters, eps_0=1):
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
        n = self.parameters["model"]["at_number"]
        return alpha**n

    def grad_1d(self, u):
        """
        Gradient computation in 1D.
        """
        return ufl.grad(u)[0]

    def elastic_energy_density(self, state):
        """
        Elastic energy density for the 1D state.
        """
        alpha = state["alpha"]
        u = state["u"]
        u_x = self.grad_1d(u) - self.eps_0
        _mu = self.parameters["model"]["E"]

        return _mu / 2.0 * self.a(alpha) * u_x**2

    def damage_energy_density(self, state):
        """
        Damage energy density for the 1D state.
        """
        _w1 = self.parameters["model"]["w1"]
        _ell = self.parameters["model"]["ell"]

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

        return self.parameters["model"]["E"] * self.a(alpha) * u_x


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
        _mu = self.parameters["model"]["E"]
        _kappa = self.parameters["model"].get("kappa", 1.0)

        energy_density = _mu / 2.0 * self.a(alpha) * u_x**2

        if u_zero is None:
            u_zero = Constant(u.function_space.mesh, 0.0)

        substrate_density = _kappa / 2.0 * (u - u_zero) ** 2

        return energy_density + substrate_density
