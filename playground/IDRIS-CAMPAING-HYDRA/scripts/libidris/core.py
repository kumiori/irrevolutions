#!/usr/bin/env python3

from typing import Optional
import dolfinx
import ufl

from dolfinx.fem import (Constant, Function, assemble_scalar, dirichletbc,
                         form, locate_dofs_geometrical, set_bc)

def a(alpha, parameters):
    return (1 - alpha)**2

def w(alpha, parameters):
    """
    Return the homogeneous damage energy term,
    as a function of the state
    (only depends on damage).
    """
    # Return w(alpha) function
    n = parameters["model"]["at_number"]
    return alpha**n
    # return alpha

def elastic_energy_density(state,
                           parameters, 
                           ):
    """
    Returns the elastic energy density of the state.
    """
    # Parameters
    alpha = state["alpha"]
    u = state["u"]
    eps = ufl.grad(u)
    _mu = parameters["model"]["E"]
    
    energy_density = _mu / 2.0 * a(alpha, parameters) * ufl.inner(eps, eps)

    return energy_density

def elastic_energy_density_film(state,
                           parameters, 
                           u_zero: Optional[dolfinx.fem.function.Function] = None, 
                           ):
    """
    Returns the elastic energy density of the state.
    """
    # Parameters
    alpha = state["alpha"]
    u = state["u"]
    eps = ufl.grad(u)
    _mu = parameters["model"]["E"]
    _kappa = parameters["model"].get("kappa", 1.0)
    
    energy_density = _mu / 2.0 * a(alpha, parameters) * ufl.inner(eps, eps)
    
    if u_zero is None:
        u_zero = Constant(u.function_space.mesh, 0.0)

    substrate_density = _kappa / 2.0 * ufl.inner(u - u_zero, u - u_zero)

    return energy_density + substrate_density

def damage_energy_density(state, parameters):
    """
    Return the damage energy density of the state.
    """

    _w1 = parameters["model"]["w1"]
    _ell = parameters["model"]["ell"]

    alpha = state["alpha"]
    grad_alpha = ufl.grad(alpha)

    # Compute the damage dissipation density
    damage_density = _w1 * w(alpha, parameters) + \
        _w1 * _ell**2 / 2. * ufl.dot(grad_alpha, grad_alpha)

    return damage_density

def stress(state, parameters):
    """
    Return the one-dimensional stress
    """
    u = state["u"]
    alpha = state["alpha"]
    dx = ufl.Measure("dx", domain=u.function_space.mesh)
    u_x = ufl.grad(u)[0]
    
    return parameters["model"]["E"] * a(alpha, parameters) * u_x * dx


from pathlib import Path
import hashlib
import yaml
import os
from mpi4py import MPI

def setup_output_directory(storage, parameters, outdir):
    if storage is None:
        prefix = os.path.join(outdir, f"1d-{parameters['geometry']['geom_type']}-first-new-hybrid")
    else:
        prefix = storage

    if MPI.COMM_WORLD.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    return prefix

def save_parameters(parameters, prefix):
    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()
    if MPI.COMM_WORLD.rank == 0:
        with open(f"{prefix}/parameters.yaml", "w") as file:
            yaml.dump(parameters, file)
        with open(f"{prefix}/signature.md5", "w") as f:
            f.write(signature)
            
from ufl import FiniteElement, Measure
from dolfinx import mesh, fem

def create_function_spaces_1d(mesh):
    element_u = FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)
    element_alpha = FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)
    V_u = fem.FunctionSpace(mesh, element_u)
    V_alpha = fem.FunctionSpace(mesh, element_alpha)
    return V_u, V_alpha

def initialize_functions(V_u, V_alpha):
    u = fem.Function(V_u, name="Displacement")
    u_ = fem.Function(V_u, name="BoundaryDisplacement")
    alpha = fem.Function(V_alpha, name="Damage")
    beta = fem.Function(V_alpha, name="DamagePerturbation")
    v = fem.Function(V_u, name="DisplacementPerturbation")
    state = {"u": u, "alpha": alpha}

    return u, u_, alpha, beta, v, state