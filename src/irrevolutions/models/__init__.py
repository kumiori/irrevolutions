from dolfinx.fem.function import Function
import dolfinx
import os

import ufl
import yaml

# import pdb

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(f"{dir_path}/default_parameters.yml") as f:
    default_parameters = yaml.load(f, Loader=yaml.FullLoader)

default_model_parameters = default_parameters["model"]


class ElasticityModel:
    # Basic class for elasticity

    def __init__(self, eps_0=None, model_parameters={}):
        """
        Initializes the sound material parameters.
        * Sound material parameters:
            - model_parameters["E"]: sound Young modulus
            - model_parameters["nu"]: sound Poisson ratio
        """
        self.model_parameters = default_parameters["model"]
        if model_parameters:
            self.model_parameters.update(model_parameters)
        # Sound material paramaters
        self.E = self.model_parameters["E"]
        self.nu = self.model_parameters["nu"]
        self.model_dimension = self.model_parameters["model_dimension"]
        self.model_type = self.model_parameters["model_type"]
        # geometric values
        # self.Ly = geometry_parameters["Ly"]
        # calculating Lame coefficients
        self.lmbda = (
            self.E
            * self.nu
            / ((1 + self.nu) * (1 - (self.model_dimension - 1) * self.nu))
        )
        self.mu = self.E / (2 * (1 + self.nu))

    def eps(self, u):
        if self.model_type == "2D":
            return ufl.sym(ufl.grad(u))
        if self.model_type == "plane-strain":
            return ufl.sym(
                ufl.as_matrix(
                    [
                        [u[0].dx(0), u[0.0].dx(1), 0],
                        [u[1].dx(0), u[1].dx(1), 0],
                        [0, 0, 0],
                    ]
                )
            )

    def elastic_energy_density_strain(self, eps):
        """
        Returns the elastic energy density from the strain variables.
        """
        # Parameters
        lmbda = self.lmbda
        mu = self.mu
        # Elastic energy density
        return 1 / 2 * (2 * mu * ufl.inner(eps, eps) + lmbda * ufl.tr(eps) ** 2)

    def elastic_energy_density(self, state):
        """
        Returns the elastic energy density from the state.
        """
        # Compute the elastic strain tensor
        strain = self.eps(state["u"])
        # Elastic energy density
        return self.elastic_energy_density_strain(strain)

    def total_energy_density(self, state):
        """
        Returns the total energy density calculated from the state.
        """
        # Calculate total energy density
        return self.elastic_energy_density(state)


class DamageElasticityModel(ElasticityModel):
    """
    Base class for elasticity coupled with damage.
    """

    def __init__(self, model_parameters={}):
        """
        Initializes the sound material parameters.
        * Sound material parameters:
            - E_0: sound Young modulus
            - nu_0: sound plasticity ratio
            - sig_d_0: sound damage yield stress
            - ell: internal length
            - k_res: fully damaged stiffness modulation
        """
        # Initialize the elastic parameters
        super().__init__(model_parameters)
        if model_parameters:
            self.model_parameters.update(model_parameters)

        # Initialize the damage parameters
        self.w1 = self.model_parameters["w1"]
        self.ell = self.model_parameters["ell"]
        self.k_res = self.model_parameters["k_res"]

    def a(self, alpha):
        k_res = self.k_res
        return (1 - alpha) ** 2 + k_res

    def w(self, alpha):
        """
        Return the dissipated energy function as a function of the state
        (only depends on damage).
        """
        # Return w(alpha) function
        return alpha

    def elastic_energy_density_strain(self, eps, alpha):
        """
        Returns the elastic energy density from the strain and the damage.
        """
        # Parameters
        lmbda = self.lmbda
        mu = self.mu

        energy_density = (
            self.a(alpha)
            * 1.0
            / 2.0
            * (2 * mu * ufl.inner(eps, eps) + lmbda * ufl.tr(eps) ** 2)
        )
        return energy_density

    def elastic_energy_density(self, state):
        """
        Returns the elastic energy density from the state.
        """
        # Parameters
        alpha = state["alpha"]
        u = state["u"]
        eps = self.eps(u)
        return self.elastic_energy_density_strain(eps, alpha)

    def stress(self, strain, alpha):
        # Differentiate the elastic energy w.r.t. the strain tensor
        eps_ = ufl.variable(strain)
        # Derivative of energy w.r.t. the strain tensor to obtain the stress
        # tensor
        sigma = ufl.diff(self.elastic_energy_density_strain(eps_, alpha), eps_)
        return sigma

    def stress0(self, u):
        strain = self.eps(u)
        lmbda = self.lmbda
        mu = self.mu
        sigma = 2 * mu * strain + lmbda * ufl.tr(strain) * ufl.Identity(
            self.model_dimension
        )
        return sigma

    def damage_energy_density(self, state):
        """
        Return the damage dissipation density from the state.
        """
        # Get the material parameters
        self.E
        # w1 = params["material"]["sigma_D0"] ** 2 / E0
        w1 = self.w1
        ell = self.ell
        # Get the damage
        alpha = state["alpha"]
        # Compute the damage gradient
        grad_alpha = ufl.grad(alpha)
        # Compute the damage dissipation density
        D_d = w1 * self.w(alpha) + w1 * ell**2 * ufl.dot(grad_alpha, grad_alpha)
        return D_d

    def total_energy_density(self, state):
        """
        Return the damage dissipation density from the state.
        """
        # Get the material parameters
        energy = self.elastic_energy_density(state) + self.damage_energy_density(state)
        return energy


class DeviatoricSplit(DamageElasticityModel):
    """Lancioni and Royer-Carfagni, 2009

    Args:
        DamageElasticityModel (...): ...
    """

    def __init__(self, model_parameters={}):
        super().__init__(model_parameters)

    def elastic_energy_density_strain(self, eps, alpha):
        """
        Returns the elastic energy density from the strain and the damage.
        """
        # Parameters
        mu = self.mu
        dim = eps.ufl_shape[0]
        kappa = self.lmbda + 2 * self.mu / dim  # Bulk modulus

        # Deviatoric part of the strain
        eps_dev = ufl.dev(eps)

        energy_density = (
            1.0 / 2.0 * kappa * ufl.tr(eps) ** 2  # Volumetric part
            + self.a(alpha) * mu * ufl.inner(eps_dev, eps_dev)  # Deviatoric part
        )
        return energy_density


class PositiveNegativeSplit(DamageElasticityModel):
    """Amor et al., 2009"""

    def __init__(self, model_parameters={}):
        super().__init__(model_parameters)

    def positive_negative_trace(self, eps):
        """
        Compute the positive and negative parts of the trace of the strain tensor.
        """
        tr_eps = ufl.tr(eps)
        tr_plus = ufl.max_value(tr_eps, 0)
        tr_minus = ufl.min_value(tr_eps, 0)
        return tr_plus, tr_minus

    def elastic_energy_density_strain(self, eps, alpha):
        """
        Returns the elastic energy density from the strain and the damage.
        """
        # Parameters
        lmbda = self.lmbda
        mu = self.mu
        dim = eps.ufl_shape[0]
        kappa = lmbda + 2 / dim * mu

        # Deviatoric part of the strain
        eps_dev = ufl.dev(eps)
        eps_vol = ufl.tr(eps) * ufl.Identity(dim) / dim

        tr_minus, tr_plus = self.positive_negative_trace(eps_vol)

        energy_density = (
            1.0 / 2.0 * kappa * tr_minus** 2  # Negative volumetric part
            + self.a(alpha)
            * (
                1.0 / 2.0 * kappa * tr_plus**2 + mu * ufl.inner(eps_dev, eps_dev)
            )  # Positive volumetric + deviatoric
        )
        return energy_density


class GeometricNonlinearElasticityModel(DamageElasticityModel):
    """Neo-Hookean model for geometrically nonlinear elasticity

    Args:
        DamageElasticityModel (_type_): _description_

    Returns:
        _type_: _description_

    Yields:
        _type_: _description_
    """

    def strain_energy_density(self, u):
        """
        Compute the strain energy density for a given displacement field u.

        Parameters:
            u (ufl.Expr): Displacement field.

        Returns:
            ufl.Expr: Strain energy density.
        """
        mu = self.mu
        Id = ufl.Identity(self.dim)  # Identity tensor
        F = Id + ufl.grad(u)  # Deformation gradient
        C = F.T * F  # Right Cauchy-Green tensor
        # Invariants
        J = ufl.det(F)  # Determinant of F
        Ic = ufl.tr(C)
        # Neo-Hookean strain energy density
        W = (mu / 2) * (ufl.inner(F, F) - 3 - 2 * ufl.ln(J))

        # psi = (mu / 2) * (Ic - 3) - mu * ln(J) + (lmbda / 2) * (ln(J)) ** 2

        return W


class BrittleMembraneOverElasticFoundation(DamageElasticityModel):
    """
    Base class for thin film elasticity coupled with damage.
    """

    def __init__(self, model_parameters={}, eps_0=ufl.Identity(2)):
        """
        Initialie material parameters.
        * Sound material:
            - E_0: sound Young modulus
            - K: elastic foundation modulus
            - nu_0: sound plasticity ratio
            - eps_0: inelastic strain
            - sig_d_0: sound damage yield stress
            - ell: internal length
            - k_res: residual stiffness
        """
        # Initialize elastic parameters
        super().__init__(model_parameters)
        if model_parameters:
            self.model_parameters.update(model_parameters)

        # Initialize the damage parameters
        self.w1 = self.model_parameters["w1"]
        self.ell = self.model_parameters["ell"]
        self.ell_e = self.model_parameters["ell_e"]
        self.k_res = self.model_parameters["k_res"]
        self.eps_0 = eps_0

    def elastic_foundation_density(self, u):
        K = self.ell_e ** (-2.0)
        return 0.5 * K * ufl.inner(u, u)

    def elastic_energy_density(self, state):
        """
        Returns the elastic energy density from the state.
        """
        # Parameters
        alpha = state["alpha"]
        u = state["u"]
        eps = self.eps(u) - self.eps_0
        return self.elastic_energy_density_strain(
            eps, alpha
        ) + self.elastic_foundation_density(u)

    def stress_average(self, strain, alpha):
        from dolfinx.fem import assemble_scalar, form
        from numpy import ndarray

        # Differentiate the elastic energy w.r.t. the strain tensor
        eps_ = ufl.variable(strain)
        # Derivative of energy w.r.t. the strain tensor to obtain the stress
        # tensor
        _sigma = ufl.diff(self.elastic_energy_density_strain(eps_, alpha), eps_)
        dx = ufl.Measure("dx", domain=alpha.function_space.mesh)
        sigma = ndarray(shape=(self.model_dimension, self.model_dimension))

        for i in range(self.model_dimension):
            for j in range(self.model_dimension):
                # compute the average value for the field sigma
                sigma[i, j] = assemble_scalar(form(_sigma[i, j] * dx))

        return _sigma, ufl.as_tensor(sigma)


class VariableThickness:
    def __init__(self, model):
        self.model = model

    def __call__(
        self,
        thickness: Function,
        model_parameters={},
        eps_0=ufl.Identity(2),
    ):
        # Wrap energy density functions
        def wrap_with_thickness(original_function):
            """
            Wrapper to multiply the result of the original function by the thickness.
            """

            def wrapped(*args, **kwargs):
                # Call the original function
                original_result = original_function(*args, **kwargs)
                # Multiply the result by the thickness
                return thickness * original_result

            return wrapped

        # Wrap and replace the energy density functions
        if callable(self.model.elastic_energy_density):
            self.model.elastic_energy_density = wrap_with_thickness(
                self.model.elastic_energy_density
            )

        # if callable(self.model.elastic_foundation_density):
        #     self.model.elastic_foundation_density = wrap_with_thickness(
        #         self.model.elastic_foundation_density
        #     )

        if callable(self.model.damage_energy_density):
            self.model.damage_energy_density = wrap_with_thickness(
                self.model.damage_energy_density
            )

        # Return the instance of the decorated class
        obj = self.model(thickness, model_parameters, eps_0)
        return obj


@VariableThickness
class BanquiseVaryingThickness(BrittleMembraneOverElasticFoundation):
    def __init__(self, thickness: Function, model_parameters={}, eps_0=ufl.Identity(2)):
        super().__init__(model_parameters)
        self.h = thickness
