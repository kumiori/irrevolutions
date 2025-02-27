import os
import logging
from mpi4py import MPI
import dolfinx
from irrevolutions.utils import (
    Visualization,
    ColorPrint,
)
from dolfinx.common import list_timings

# from irrevolutions.models import default_model_parameters
import irrevolutions.models as models
import yaml


def run_computation(parameters, storage):
    return None, None, None


def load_parameters(file_path):
    """
    Load parameters from a YAML file.

    Args:
        file_path (str): Path to the YAML parameter file.

    Returns:
        dict: Loaded parameters.
    """
    import hashlib

    with open(file_path) as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    parameters["model"]["model_dimension"] = 2
    parameters["model"]["model_type"] = "2D"

    # parameters["model"]["at_number"] = 1
    parameters["loading"]["min"] = 0.0
    parameters["loading"]["max"] = 1.0
    parameters["loading"]["steps"] = 10

    parameters["geometry"]["geom_type"] = "circle"
    parameters["geometry"]["mesh_size_factor"] = 3
    parameters["geometry"]["R_outer"] = 1.0  # Outer disk radius
    parameters["geometry"]["R_inner"] = 0.3  # Inner hole radius (0.0 for no hole)
    parameters["geometry"]["lc"] = 0.05  # Mesh element size
    parameters["geometry"]["a"] = 0.1  # Half-width of the refined region (-a < x < a)

    parameters["stability"]["cone"]["cone_max_it"] = 400000
    parameters["stability"]["cone"]["cone_atol"] = 1e-6
    parameters["stability"]["cone"]["cone_rtol"] = 1e-6
    parameters["stability"]["cone"]["scaling"] = 1e-3

    parameters["model"]["w1"] = 1
    parameters["model"]["k_res"] = 0.0
    parameters["model"]["mu"] = 1
    parameters["model"]["ell"] = 0.02
    parameters["solvers"]["damage_elasticity"]["max_it"] = 1000

    parameters["solvers"]["damage_elasticity"]["alpha_rtol"] = 1e-3
    parameters["solvers"]["newton"]["snes_atol"] = 1e-8
    parameters["solvers"]["newton"]["snes_rtol"] = 1e-8

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    return parameters, signature


if __name__ == "__main__":
    # Set the logging level
    logging.basicConfig(level=logging.INFO)

    # Load parameters
    parameters, signature = load_parameters("parameters.yaml")

    # Run computation
    _storage = f"./output/MPI-{MPI.COMM_WORLD.Get_size()}/{signature[0:6]}"
    visualization = Visualization(_storage)

    with dolfinx.common.Timer(f"~Computation Experiment") as timer:
        history_data, _, state = run_computation(parameters, _storage)

    from irrevolutions.utils import table_timing_data

    tasks = [
        "~First Order: Equilibrium",
        "~First Order: AltMin-Damage solver",
        "~First Order: AltMin-Elastic solver",
        "~Postprocessing and Vis",
        "~Output and Storage",
        "~Computation Experiment",
    ]

    _timings = table_timing_data()
    visualization.save_table(_timings, "timing_data")
    list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

    ColorPrint.print_bold(f"===================- {signature} -=================")
    ColorPrint.print_bold(f"===================- {_storage} -=================")
