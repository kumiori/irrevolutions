import pytest
import ufl
from dolfinx.fem import Function, functionspace, assemble_scalar, form
from dolfinx.mesh import create_unit_square
from mpi4py import MPI
import numpy as np
import os
import yaml

import basix.ufl
import basix

from irrevolutions.models import DeviatoricSplit


@pytest.fixture
def setup_mesh():
    """Create a simple mesh and function space for testing."""
    mesh = create_unit_square(MPI.COMM_WORLD, 4, 4)
    element_u = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1, shape=(2,))
    element_alpha = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1)
    V_u = functionspace(mesh, element_u)
    V_alpha = functionspace(mesh, element_alpha)

    return mesh, V_u, V_alpha


@pytest.fixture
def load_default_parameters():
    # """Load default model parameters from a YAML file."""
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # yaml_file = os.path.join(dir_path, "default_parameters.yml")
    # with open(yaml_file, "r") as f:
    #     default_parameters = yaml.load(f, Loader=yaml.FullLoader)
    # return default_parameters["model"]
    from irrevolutions.models import default_model_parameters

    return default_model_parameters


def test_deviatoric_split_zero_strain(setup_mesh, load_default_parameters):
    """Test DeviatoricSplit energy density with zero strain."""
    mesh, V_u, V_alpha = setup_mesh
    model_params = load_default_parameters

    model = DeviatoricSplit(model_params)

    eps = ufl.Identity(2) * 0  # Zero strain
    alpha = Function(V_alpha)
    alpha.x.array[:] = 0.0  # No damage
    dx = ufl.Measure("dx", domain=mesh)
    energy = model.elastic_energy_density_strain(eps, alpha)

    assert np.isclose(
        float(assemble_scalar(form(energy * dx))), 0.0
    ), "Energy should be zero for zero strain"


def test_deviatoric_split_pure_shear(setup_mesh, load_default_parameters):
    """Test DeviatoricSplit under pure shear strain."""
    mesh, V_u, V_alpha = setup_mesh
    model_params = load_default_parameters
    model = DeviatoricSplit(model_params)

    gamma = 0.01  # Shear strain magnitude
    eps_shear = ufl.as_tensor([[0, gamma / 2], [gamma / 2, 0]])  # Pure shear strain
    alpha = Function(V_alpha)
    alpha.x.array[:] = 0.0  # No damage
    dx = ufl.Measure("dx", domain=mesh)

    energy = model.elastic_energy_density_strain(eps_shear, alpha)
    assembled_energy = assemble_scalar(form(energy * dx))
    print("shear", assembled_energy)

    assert (
        assembled_energy > 0
    ), f"Energy should be positive for pure shear, got {assembled_energy}"


def test_deviatoric_split_with_damage(setup_mesh, load_default_parameters):
    """Test DeviatoricSplit with damage applied."""
    mesh, V_u, V_alpha = setup_mesh
    model_params = load_default_parameters

    model = DeviatoricSplit(model_params)

    gamma = 0.01  # Shear strain magnitude
    eps_shear = ufl.as_tensor([[0, gamma / 2], [gamma / 2, 0]])  # Pure shear strain
    alpha = Function(V_alpha)
    alpha.x.array[:] = 0.5  # 50% damage across the domain
    dx = ufl.Measure("dx", domain=mesh)

    energy = model.elastic_energy_density_strain(eps_shear, alpha)
    assembled_energy = assemble_scalar(form(energy * dx))
    assert (
        assembled_energy > 0
    ), f"Energy should be positive even with damage, got {assembled_energy}"
    assert assembled_energy < gamma**2, "Energy should be reduced due to damage"


@pytest.mark.parametrize("gamma", [0.0, 0.01, 0.05])
def test_deviatoric_split_parametric_shear(setup_mesh, load_default_parameters, gamma):
    """Test DeviatoricSplit under varying shear strain magnitudes."""
    mesh, V_u, V_alpha = setup_mesh
    model_params = load_default_parameters
    model = DeviatoricSplit(model_params)

    eps_shear = ufl.as_tensor([[0, gamma / 2], [gamma / 2, 0]])  # Shear strain
    alpha = Function(V_alpha)
    alpha.x.array[:] = 0.0  # No damage
    dx = ufl.Measure("dx", domain=mesh)

    energy = model.elastic_energy_density_strain(eps_shear, alpha)
    assembled_energy = assemble_scalar(form(energy * dx))

    if gamma == 0.0:
        assert np.isclose(
            assembled_energy, 0.0
        ), "Energy should be zero for zero strain"
    else:
        assert (
            assembled_energy > 0
        ), f"Energy should be positive for gamma={gamma}, got {assembled_energy}"
