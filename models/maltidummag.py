import ufl

import yaml
import os

from . import DamageElasticityModel

default_parameters = {
    'damage_modes': 2
}


class MultiDamage(DamageElasticityModel):
    """
    Class to implement a multi-variable damage model, as in
        https://doi.org/10.1016/j.cma.2018.03.012
        see also
        https://zenodo.org/record/5764329
    """

    def __init__(self, eps_0=None, model_parameters={}):
        """
        Initializes material parameters.
        * Sound material parameters:
            - model_parameters["E"]: sound Young modulus
            - model_parameters["nu"]: sound Poisson ratio
        * Fracture material parameters:
            - 
        * Numerical parameters
            - k_pen: penalisation, large parameter, or
            - eta:=1/k_pen: penalisation, small parameter
        """
        super().__init__(model_parameters)

        # Penalisation parameter eta = 1/alpha, with alpha real
        self.k_pen = self.model_parameters["k_pen"]
    
    def B_k(self):
        """
        The matrix encoding penalised gradients
        """
        damage_modes = self.model_parameters.get('damage_modes')
        k_pen = self.model_parameters.get('k_pen')
        M = None

        # FIXME: WTF is k_dam?
        # https://github.com/jeanmichelscherer/GRADAM/blob/main/src/gradam/material_models.py#L181
        _Id = ufl.as_tensor(ufl.id(damage_modes))
        B_k = [_Id + k_pen*(_Id - ufl.outer(M[n], M[n])) for n in range(damage_modes)]

        B_sample = [(ufl.dot(self.R.T, ufl.dot(B_k[n], self.R)))
                    for n in range(damage_modes)]

        if (self.model_parameters.get('model_dimension') == 2):
            _B = [ufl.as_tensor([[B_sample[n][0, 0], B_sample[n][0, 1]],
                                 [B_sample[n][1, 0], B_sample[n][1, 1]]]) for n in range(damage_modes)]
        # else?

    def damage_dissipation_density(self, state):
        """
        Return the damage dissipation density from the state.
        """
        # Get the material parameters
        self.E
        w1 = self.w1
        ell = self.ell
        B_ = self.B_
        damage_modes = self.model_parameters.get('damage_modes')

        # Get the damage, a vector of functions, dim = damage_modes
        alpha = state["alpha"]

        B_ = ufl.id(damage_modes)
        B_k = self.B_k()

        # Compute the damage gradient
        grad_alpha = [ufl.grad(alpha[n]) for n in range(damage_modes)]

        # Compute the dissipation density
        D_d=[ w1 * self.w(alpha[n]) + \
              ell**2 *
              ufl.dot(ufl.dot(B_k[n], ufl.grad(alpha[n])), ufl.grad(alpha[n]))
                 for n in range(damage_modes)]
        
        return D_d
