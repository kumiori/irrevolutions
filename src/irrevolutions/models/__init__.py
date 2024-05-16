import ufl

import yaml
import os

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
        self.lmbda = (self.E * self.nu /
                      ((1 + self.nu) * (1 -
                                        (self.model_dimension - 1) * self.nu)))
        self.mu = self.E / (2 * (1 + self.nu))

    def eps(self, u):
        if self.model_type == "2D":
            return ufl.sym(ufl.grad(u))
        if self.model_type == "plane-strain":
            return ufl.sym(
                ufl.as_matrix([
                    [u[0].dx(0), u[0.].dx(1), 0],
                    [u[1].dx(0), u[1].dx(1), 0],
                    [0, 0, 0],
                ]))

    def elastic_energy_density_strain(self, eps):
        """
        Returns the elastic energy density from the strain variables.
        """
        # Parameters
        lmbda = self.lmbda
        mu = self.mu
        # Elastic energy density
        return 1 / 2 * (2 * mu * ufl.inner(eps, eps) + lmbda * ufl.tr(eps)**2)

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
        return (1 - alpha)**2 + k_res

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
            self.a(alpha) * 1.0 / 2.0 *
            (2 * mu * ufl.inner(eps, eps) + lmbda * ufl.tr(eps)**2))
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
            self.model_dimension)
        return sigma

    def damage_energy_density(self, state):
        """
        Return the damage dissipation density from the state.
        """
        # Get the material parameters
        self.E
        w1 = self.w1
        ell = self.ell
        # Get the damage
        alpha = state["alpha"]
        # Compute the damage gradient
        grad_alpha = ufl.grad(alpha)
        # Compute the damage dissipation density
        D_d = w1 * self.w(alpha) + w1 * ell**2 * ufl.dot(
            grad_alpha, grad_alpha)
        return D_d

    def total_energy_density(self, state):
        """
        Return the damage dissipation density from the state.
        """
        # Get the material parameters
        energy = self.elastic_energy_density(
            state) + self.damage_energy_density(state)
        return energy

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
        K = self.ell_e**(-2.)
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
            eps, alpha) + self.elastic_foundation_density(u)

    def stress(self, strain, alpha):
        from numpy import ndarray
        from dolfinx.fem import assemble_scalar, form
        # Differentiate the elastic energy w.r.t. the strain tensor
        eps_ = ufl.variable(strain)
        # Derivative of energy w.r.t. the strain tensor to obtain the stress
        # tensor
        _sigma = ufl.diff(self.elastic_energy_density_strain(eps_, alpha), eps_)
        dx = ufl.Measure("dx", domain = alpha.function_space.mesh)
        sigma = ndarray(shape=(self.model_dimension, self.model_dimension))

        for i in range(self.model_dimension):
            for j in range(self.model_dimension):
                # compute the average value for the field sigma
                sigma[i, j] = assemble_scalar(form(_sigma[i, j] * dx))
        
        return ufl.as_tensor(sigma)

from dolfinx.fem.function import Function
class VariableThickness:
    #accept the class as argument
    def __init__(self, model):
        self.model = model
    
    #accept the class's __init__ method arguments
    def __call__(self, thickness: Function, model_parameters={}, eps_0=ufl.Identity(2)):

        #replace energy densities with newdisplay
        self.model.elastic_energy_density = thickness * self.model.elastic_energy_density
        self.model.elastic_foundation_density = thickness * self.model.elastic_foundation_density
        self.model.damage_dissipation_density = thickness * self.model.damage_dissipation_density
        
        #return the instance of the class
        obj = self.model(thickness, model_parameters, eps_0)
        return obj


@VariableThickness
class BanquiseVaryingThickness(BrittleMembraneOverElasticFoundation):
    def __init__(self, thickness: Function, model_parameters={}, eps_0=ufl.Identity(2)):
        super().__init__(model_parameters)
        self.h = thickness
