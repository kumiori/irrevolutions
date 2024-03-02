#!/usr/bin/env python3
import pdb
import pandas as pd
import numpy as np
from sympy import derive_by_array
import yaml
import json
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
import hashlib

from dolfinx.fem import locate_dofs_geometrical, dirichletbc
from dolfinx.mesh import CellType
import dolfinx.mesh
from dolfinx.fem import (
    Constant,
    Function,
    FunctionSpace,
    assemble_scalar,
    dirichletbc,
    form,
    locate_dofs_geometrical,
    set_bc,
)
from mpi4py import MPI
import petsc4py
from petsc4py import PETSc
import dolfinx
import dolfinx.plot
from dolfinx import log
import ufl

from dolfinx.fem.petsc import (
    set_bc,
    assemble_vector
    )
from dolfinx.io import XDMFFile, gmshio
import logging
from dolfinx.common import Timer, list_timings, TimingType

sys.path.append("../")
from algorithms.so import BifurcationSolver, StabilitySolver
from algorithms.am import AlternateMinimisation, HybridFractureSolver
from meshes.primitives import mesh_bar_gmshapi
from models import DamageElasticityModel as Brittle
from models import ElasticityModel as Elastic
from meshes.extended_pacman import mesh_extended_pacman as mesh_pacman
from utils import ColorPrint, set_vector_to_constant
from utils.lib import _local_notch_asymptotic
from utils.viz import plot_mesh
from utils import ColorPrint
from utils.plots import plot_energies
from utils import norm_H1, norm_L2
from utils import history_data, _write_history_data
from utils import _logger
from solvers import SNESSolver
from utils import _logger
import pyvista
from pyvista.utilities import xvfb
from dolfinx.mesh import locate_entities_boundary, CellType, create_rectangle
from dolfinx.fem import locate_dofs_topological
# 
from utils.viz import plot_mesh, plot_vector, plot_scalar, plot_profile
from solvers.function import vec_to_functions


xvfb.start_xvfb(wait=0.05)
pyvista.OFF_SCREEN = True



description = """We solve here a basic 2d of a notched specimen.
The caveat here is that we solve on an extended domain to allow cracks at the boundary.

Imagine a dinner a pizza which is missing a slice, and lots of hungry friends
that pull from the sides of the pizza. Will a real pizza will break at the centre?

We solve this problem as an example of localisation with singularity.
"""

petsc4py.init(sys.argv)
comm = MPI.COMM_WORLD

# Mesh on node model_rank and then distribute
model_rank = 0

def main(parameters, storage):
    # Load mesh

    _radius = parameters["geometry"]["r"]
    _omega = parameters["geometry"]["omega"]
    _nameExp = parameters["geometry"]["geom_type"]
    tdim = parameters["geometry"]["geometric_dimension"]
    ell = parameters["model"]["ell"]
    geom_type = parameters["geometry"]["geom_type"]


    _geom_parameters = """
        elltomesh: 1
        geom_type: local_notch
        geometric_dimension: 2
        lc: 0.1
        omega: 45
        r: 1.0
        rho: 1.3
        mesh_size_factor: 2
        refinement: 4
    """
    
    geom_parameters = yaml.load(_geom_parameters, Loader=yaml.FullLoader)
    parameters["geometry"] = geom_parameters

    parameters["geometry"]["meshsize"] = ell / parameters["geometry"]["mesh_size_factor"]

    gmsh_model, tdim = mesh_pacman(geom_type, parameters["geometry"], tdim)

    # Get mesh and meshtags
    mesh, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)


    if comm.rank == 0:
        from dolfinx.plot import create_vtk_mesh

        pyvista.start_xvfb()
        plotter = pyvista.Plotter()
        grid = pyvista.UnstructuredGrid(*create_vtk_mesh(mesh, mesh.topology.dim))
        num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
        grid.cell_data["Marker"] = cell_tags.values[cell_tags.indices < num_local_cells]
        grid.set_active_scalars("Marker")
        actor = plotter.add_mesh(grid, show_edges=True)
        plotter.view_xy()
        
        if not pyvista.OFF_SCREEN:
            plotter.show()
        else:
            cell_tag_fig = plotter.screenshot("cell_tags.png")    
    
    outdir = "output"
    if storage is None:
        prefix = os.path.join(outdir, f"test_boundary_cracks/{_nameExp}")
    else:
        prefix = storage
    
    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)

    with XDMFFile(comm, f"{prefix}/{_nameExp}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as file:
        file.write_mesh(mesh)

    if comm.rank == 0:
        ax = plot_mesh(mesh)
        fig = ax.get_figure()
        fig.savefig(f"{prefix}/mesh.png")
    signature = hashlib.md5(str(parameters).encode('utf-8')).hexdigest()

    # Function spaces
    element_u = ufl.VectorElement("Lagrange", mesh.ufl_cell(), degree=1, dim=2)
    V_u = FunctionSpace(mesh, element_u)

    element_alpha = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)
    V_alpha = FunctionSpace(mesh, element_alpha)

    # Define the state
    u = Function(V_u, name="Displacement")
    alpha = Function(V_alpha, name="Damage")
    alphadot = Function(V_alpha, name="Damage rate")

    # upper/lower bound for the damage field
    alpha_lb = Function(V_alpha, name="Lower bound")
    alpha_ub = Function(V_alpha, name="Upper bound")

    state = {"u": u, "alpha": alpha}

    # Data

    uD = Function(V_u, name="Asymptotic Notch Displacement")

    # Measures
    dx = ufl.Measure("dx", subdomain_data=cell_tags, domain=mesh)
    ds = ufl.Measure("ds", domain=mesh)
    _logger.critical("Checking sanity of the mesh")
    area_1 = assemble_scalar(dolfinx.fem.form(Constant(mesh, 1.)*dx(100)))
    area_2 = assemble_scalar(dolfinx.fem.form(Constant(mesh, 1.)*dx(1)))
    _logger.critical(f"Area 1: {area_1}")
    _logger.critical(f"Area 2: {area_2}")
    
    # pdb.set_trace()
    # Set Bcs Function
    ext_radius = geom_parameters["rho"] * geom_parameters["r"]
    ext_bd_facets = locate_entities_boundary(
        mesh, dim=1, marker=lambda x: np.isclose(x[0]**2. + x[1]**2. - ext_radius**2, 0., atol=1.e-4)
    )

    boundary_dofs_u = locate_dofs_topological(
        V_u, mesh.topology.dim - 1, ext_bd_facets)
    boundary_dofs_alpha = locate_dofs_topological(
        V_alpha, mesh.topology.dim - 1, ext_bd_facets)

    uD.interpolate(lambda x: _local_notch_asymptotic(
        x, ω=np.deg2rad(_omega / 2.), par=parameters["material"]))

    alpha_lb.interpolate(lambda x: np.zeros_like(x[0]))
    alpha_ub.interpolate(lambda x: np.ones_like(x[0]))

    for f in [alpha_lb, alpha_ub]:
        f.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                             mode=PETSc.ScatterMode.FORWARD)

    bcs_u = [dirichletbc(value=uD, dofs=boundary_dofs_u)]

    bcs_alpha = [
        dirichletbc(
            np.array(0, dtype=PETSc.ScalarType),
            boundary_dofs_alpha,
            V_alpha,
        )
    ]
    
    # bcs_alpha = []
    set_bc(alpha_ub.vector, bcs_alpha)
    alpha_ub.vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )

    bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}
    
    # Bounds for Newton solver

    u_lb = Function(V_u, name="displacement lower bound")
    u_ub = Function(V_u, name="displacement upper bound")
    alpha_lb = Function(V_alpha, name="damage lower bound")
    alpha_ub = Function(V_alpha, name="damage upper bound")
    set_vector_to_constant(u_lb.vector, PETSc.NINFINITY)
    set_vector_to_constant(u_ub.vector, PETSc.PINFINITY)
    set_vector_to_constant(alpha_lb.vector, 0)
    set_vector_to_constant(alpha_ub.vector, 1)

    model = Brittle(parameters["model"])
    
    _stiff_elastic_parameters = yaml.load("""
            E: 1000.
            nu: 0.3
            model_dimension: 2
            model_type: "2D"
            """)

    machine = Elastic(_stiff_elastic_parameters)

    # Energy functional
    f = dolfinx.fem.Constant(mesh, np.array([0, 0], dtype=PETSc.ScalarType))
    external_work = ufl.dot(f, state["u"]) * dx(1)
    total_energy = model.total_energy_density(state) * dx(1)    \
        + machine.elastic_energy_density(state) * dx(100)       \
        - external_work

    load_par = parameters["loading"]
    loads = np.linspace(load_par["min"],
                        load_par["max"], load_par["steps"])
    loads = [0., 0.5, 1.01]
    # loads = [0.5]

    equilibrium = HybridFractureSolver(
        total_energy,
        state,
        bcs,
        bounds=(alpha_lb, alpha_ub),
        solver_parameters=parameters.get("solvers"),
    )

    bifurcation = BifurcationSolver(
        total_energy, state, bcs,
        bifurcation_parameters=parameters.get("stability")
    )

    stability = StabilitySolver(
        total_energy, state, bcs,
        cone_parameters=parameters.get("stability")
    )

    
    mode_shapes_data = {
        'time_steps': [],
        'point_values': {
            'x_values': [],
        }
    }
    num_modes = 1

    _logger.setLevel(level=logging.CRITICAL)

    for i_t, t in enumerate(loads):

        uD.interpolate(lambda x: _local_notch_asymptotic(
            x,
            ω=np.deg2rad(_omega / 2.),
            t=t,
            par=parameters["material"]
        ))

        # update the lower bound
        alpha.vector.copy(alpha_lb.vector)
        alpha_lb.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        logging.info(f"-- Solving for t = {t:3.2f} --")
        equilibrium.solve(alpha_lb)

        is_unique = bifurcation.solve(alpha_lb)
        is_elastic = not bifurcation._is_critical(alpha_lb)

        inertia = bifurcation.get_inertia()

        stable = stability.solve(alpha_lb, eig0=bifurcation._spectrum, inertia = inertia)
        
        with dolfinx.common.Timer(f"~Postprocessing and Vis") as timer:

            fracture_energy = comm.allreduce(
                assemble_scalar(form(model.damage_energy_density(state) * dx(1))),
                op=MPI.SUM,
            )
            elastic_energy = comm.allreduce(
                assemble_scalar(form(model.elastic_energy_density(state) * dx(1))),
                op=MPI.SUM,
            )
            
            _write_history_data(
                equilibrium,
                bifurcation,
                stability,
                history_data,
                t,
                inertia,
                stable,
                [fracture_energy, elastic_energy])
            
            with XDMFFile(comm, f"{prefix}/{_nameExp}.xdmf", "a", encoding=XDMFFile.Encoding.HDF5) as file:
                file.write_function(u, t)
                file.write_function(alpha, t)

            if comm.rank == 0:
                a_file = open(f"{prefix}/time_data.json", "w")
                json.dump(history_data, a_file)
                a_file.close()



            xvfb.start_xvfb(wait=0.05)
            pyvista.OFF_SCREEN = True
            plotter = pyvista.Plotter(
                title="State of the System",
                window_size=[1600, 600],
                shape=(1, 2),
            )
            _plt = plot_scalar(alpha, plotter, subplot=(0, 0))
            _plt = plot_vector(u, plotter, subplot=(0, 1))
            if comm.rank == 0:
                Path("output").mkdir(parents=True, exist_ok=True)
            _plt.screenshot(f"{prefix}/{_nameExp}-{comm.size}-{i_t}.png")
            _plt.close()

    from utils.plots import plot_energies, plot_AMit_load
    
    if comm.rank == 0:
        plot_energies(history_data, file=f"{prefix}/{_nameExp}_energies.pdf")
        # plot_AMit_load(history_data, file=f"{prefix}/{_nameExp}_it_load.pdf")

    return history_data, stability.data, state

def load_parameters(file_path, ndofs, model='at1'):
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
    parameters["model"]["model_type"] = '2D'
    # parameters["model"]["mu"] = 1
    parameters["model"]["w1"] = 1

    parameters["geometry"]["geom_type"] = "brittle-damageable"
    # Get mesh parameters

    if model == 'at2':
        parameters["loading"]["min"] = .9
        parameters["loading"]["max"] = .9
        parameters["loading"]["steps"] = 1

    elif model == 'at1':
        parameters["loading"]["min"] = .0
        parameters["loading"]["max"] = 1.5
        parameters["loading"]["steps"] = 20

    parameters["geometry"]["geom_type"] = "local_notch"
    parameters["geometry"]["mesh_size_factor"] = 1
    parameters["geometry"]["refinement"] = 10
    # parameters["geometry"]["N"] = ndofs

    parameters["stability"]["cone"]["cone_max_it"] = 400000
    parameters["stability"]["cone"]["cone_atol"] = 1e-6
    parameters["stability"]["cone"]["cone_rtol"] = 1e-6
    parameters["stability"]["cone"]["scaling"] = 1e-4

    parameters["model"]["model_dimension"] = 2
    parameters["model"]["w1"] = 1
    parameters["model"]["ell"] = .1
    parameters["model"]["k_res"] = 0.

    signature = hashlib.md5(str(parameters).encode('utf-8')).hexdigest()

    return parameters, signature


if __name__ == "__main__":
    import argparse
    from mpi4py import MPI
    
    parser = argparse.ArgumentParser(description='Process evolution.')
    parser.add_argument("-N", help="The number of dofs.", type=int, default=10)
    args = parser.parse_args()
    parameters, signature = load_parameters("data/pacman/parameters.yaml", ndofs=args.N)
    pretty_parameters = json.dumps(parameters, indent=2)
    print(pretty_parameters)

    _storage = f"output/two-dimensional-pizza/MPI-{MPI.COMM_WORLD.Get_size()}/{signature}"
    ColorPrint.print_bold(f"===================-{_storage}-=================")

    with dolfinx.common.Timer(f"~Computation Experiment") as timer:
        history_data, stability_data, state = main(parameters, _storage)

    ColorPrint.print_bold(history_data["eigs-cone"])
    from utils import ResultsStorage, Visualization
    storage = ResultsStorage(MPI.COMM_WORLD, _storage)
    # storage.store_results(parameters, history_data, state)
    visualization = Visualization(_storage)
    # visualization.visualise_results(pd.DataFrame(history_data), drop = ["solver_data", "cone_data"])
    visualization.save_table(pd.DataFrame(history_data), "history_data")
    # visualization.save_table(pd.DataFrame(stability_data), "stability_data")
    pd.DataFrame(stability_data).to_json(f'{_storage}/stability_data.json')
    
    ColorPrint.print_bold(f"===================-{signature}-=================")
    ColorPrint.print_bold(f"===================-{_storage}-=================")
    
    print(pd.DataFrame(history_data))
    ColorPrint.print_bold(f"===================-{signature}-=================")
    print(pd.DataFrame(stability_data))
    
    __import__('pdb').set_trace()
    # list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

    # from utils import table_timing_data
    # _timings = table_timing_data()

    # visualization.save_table(_timings, "timing_data")
