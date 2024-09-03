#!/usr/bin/env python3
import json
import logging
import os
import sys
from pathlib import Path

import dolfinx
import dolfinx.mesh
import dolfinx.plot
import numpy as np
import pandas as pd
import petsc4py
import ufl
import yaml
from dolfinx.common import list_timings
from dolfinx.fem import (Constant, Function, FunctionSpace, assemble_scalar,
                         dirichletbc, form, locate_dofs_geometrical, set_bc)
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI
from petsc4py import PETSc

sys.path.append("../")

from algorithms.am import HybridSolver
from algorithms.so import BifurcationSolver, StabilitySolver
from irrevolutions.utils import ColorPrint
from meshes.primitives import mesh_bar_gmshapi
from models import DamageElasticityModel

logging.getLogger().setLevel(logging.ERROR)


"""Traction damageable bar

0|WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW|========> t


[WWW]: damageable bar, alpha
load: displacement hard-t

"""


petsc4py.init(sys.argv)
comm = MPI.COMM_WORLD

# Mesh on node model_rank and then distribute
model_rank = 0


class DamageElasticityModelATJJ(DamageElasticityModel):
    """Linear softening model, quadratic damage density"""

    def __init__(self, model_parameters={}):
        self.k = model_parameters["k"]

        super(DamageElasticityModelATJJ, self).__init__(model_parameters)

    def a(self, alpha):
        k_res = np.float(self.k_res)
        w = self.w(alpha)
        _k = self.k

        return ((1 - w) + k_res) / (1 + (_k - 1) * w)

    def w(self, alpha):
        """
        Return the dissipated energy function as a function of the state
        (only depends on damage).
        """
        # Return w(alpha) function

        return 1 - (1 - alpha) ** 2


def parameters_vs_ell(parameters=None, ell=0.1):
    if parameters is None:
        with open("../test/atk_parameters.yml") as f:
            parameters = yaml.load(f, Loader=yaml.FullLoader)

    parameters["model"]["ell"] = ell

    parameters["stability"]["cone"]["cone_max_it"] = 400000
    parameters["stability"]["cone"]["cone_atol"] = 3e-6
    parameters["stability"]["cone"]["cone_rtol"] = 3e-6
    parameters["stability"]["cone"]["scaling"] = 0.01

    # parameters["model"]["model_dimension"] = 2
    # parameters["model"]["model_type"] = '1D'
    # parameters["model"]["w1"] = 1
    # parameters["model"]["k_res"] = 0.

    parameters["loading"]["min"] = 0.0
    parameters["loading"]["max"] = parameters["model"]["k"]
    parameters["loading"]["steps"] = 30

    # parameters["geometry"]["geom_type"] = "traction-bar"
    parameters["geometry"]["ell_lc"] = 4

    return parameters


def parameters_vs_SPA_scaling(file=None, s=0.01):
    if file is None:
        with open("../test/parameters.yml") as f:
            parameters = yaml.load(f, Loader=yaml.FullLoader)
    else:
        with open(file) as f:
            parameters = yaml.load(f, Loader=yaml.FullLoader)

    parameters["stability"]["cone"]["scaling"] = s
    parameters["stability"]["cone"]["cone_max_it"] = 400000
    parameters["stability"]["cone"]["cone_atol"] = 1e-6
    parameters["stability"]["cone"]["cone_rtol"] = 1e-5
    parameters["model"]["ell"] = 0.1
    parameters["loading"]["min"] = 0.9
    parameters["loading"]["max"] = parameters["model"]["k"]
    parameters["loading"]["steps"] = 30

    return parameters


def traction_with_parameters(parameters, slug=""):
    # Get mesh parameters
    Lx = parameters["geometry"]["Lx"]
    Ly = parameters["geometry"]["Ly"]
    tdim = parameters["geometry"]["geometric_dimension"]

    _nameExp = parameters["geometry"]["geom_type"]
    ell_ = parameters["model"]["ell"]
    _lc = ell_ / parameters["geometry"]["ell_lc"]

    # Get geometry model
    geom_type = parameters["geometry"]["geom_type"]

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    # Create the mesh of the specimen with given dimensions
    print("ell:", parameters["model"]["ell"])

    outdir = os.path.join("output", slug, signature)

    # prefix = os.path.join(outdir, "traction_parametric_vs_ell")

    prefix = os.path.join(outdir)

    if comm.rank == 0:
        Path(prefix).mkdir(parents=True, exist_ok=True)
    # _lc = Lx/2

    gmsh_model, tdim = mesh_bar_gmshapi(geom_type, Lx, Ly, _lc, tdim)

    # Get mesh and meshtags
    mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

    signature = hashlib.md5(str(parameters).encode("utf-8")).hexdigest()

    if comm.rank == 0:
        with open(f"{prefix}/parameters.yaml", "w") as file:
            yaml.dump(parameters, file)

    with open(f"{prefix}/parameters.yaml") as f:
        _parameters = yaml.load(f, Loader=yaml.FullLoader)

    if comm.rank == 0:
        with open(f"{prefix}/signature.md5", "w") as f:
            f.write(signature)

    with XDMFFile(
        comm, f"{prefix}/{_nameExp}.xdmf", "w", encoding=XDMFFile.Encoding.HDF5
    ) as file:
        file.write_mesh(mesh)

    # Functional Setting

    element_u = ufl.VectorElement("Lagrange", mesh.ufl_cell(), degree=1, dim=tdim)
    V_u = FunctionSpace(mesh, element_u)

    element_alpha = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)
    V_alpha = FunctionSpace(mesh, element_alpha)

    # Define the state
    u = Function(V_u, name="Displacement")
    u_ = Function(V_u, name="Boundary Displacement")
    zero_u = Function(V_u, name="   Boundary Displacement")
    alpha = Function(V_alpha, name="Damage")
    zero_alpha = Function(V_alpha, name="Damage Boundary Field")
    alphadot = dolfinx.fem.Function(V_alpha, name="Damage rate")

    state = {"u": u, "alpha": alpha}

    z = [u, alpha]
    # need upper/lower bound for the damage field
    alpha_lb = Function(V_alpha, name="Lower bound")
    alpha_ub = Function(V_alpha, name="Upper bound")

    # Measures
    dx = ufl.Measure("dx", domain=mesh)
    ufl.Measure("ds", domain=mesh)

    locate_dofs_geometrical(V_alpha, lambda x: np.isclose(x[0], 0.0))
    locate_dofs_geometrical(V_alpha, lambda x: np.isclose(x[0], Lx))

    dofs_u_left = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], 0.0))
    dofs_u_right = locate_dofs_geometrical(V_u, lambda x: np.isclose(x[0], Lx))
    # Set Bcs Function
    zero_u.interpolate(lambda x: (np.zeros_like(x[0]), np.zeros_like(x[1])))
    zero_alpha.interpolate((lambda x: np.zeros_like(x[0])))
    u_.interpolate(lambda x: (np.ones_like(x[0]), 0 * np.ones_like(x[1])))
    alpha_lb.interpolate(lambda x: np.zeros_like(x[0]))
    alpha_ub.interpolate(lambda x: np.ones_like(x[0]))

    for f in [zero_u, zero_alpha, u_, alpha_lb, alpha_ub]:
        f.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

    bc_u_left = dirichletbc(np.array([0, 0], dtype=PETSc.ScalarType), dofs_u_left, V_u)

    bc_u_right = dirichletbc(u_, dofs_u_right)
    bcs_u = [bc_u_left, bc_u_right]

    bcs_alpha = []
    # bcs_alpha = [
    #     dolfinx.fem.dirichletbc(zero_alpha, dofs_alpha_left),
    #     dolfinx.fem.dirichletbc(zero_alpha, dofs_alpha_right),
    # ]

    set_bc(alpha_ub.vector, bcs_alpha)
    alpha_ub.vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )

    bcs = {"bcs_u": bcs_u, "bcs_alpha": bcs_alpha}
    # Define the model

    model = DamageElasticityModelATJJ(parameters["model"])

    # Pack state
    state = {"u": u, "alpha": alpha}

    # Material behaviour

    # Energy functional
    f = Constant(mesh, np.array([0, 0], dtype=PETSc.ScalarType))
    external_work = ufl.dot(f, state["u"]) * dx
    total_energy = model.total_energy_density(state) * dx - external_work

    loads = np.linspace(
        parameters["loading"]["min"],
        parameters["loading"]["max"],
        parameters["loading"]["steps"],
    )

    # solver = AlternateMinimisation(
    #     total_energy, state, bcs, parameters.get("solvers"),
    #     bounds=(alpha_lb, alpha_ub)
    # )

    hybrid = HybridSolver(
        total_energy,
        state,
        bcs,
        bounds=(alpha_lb, alpha_ub),
        solver_parameters=parameters.get("solvers"),
    )

    bifurcation = BifurcationSolver(
        total_energy, state, bcs, stability_parameters=parameters.get("stability")
    )

    cone = StabilitySolver(
        total_energy, state, bcs, cone_parameters=parameters.get("stability")
    )

    history_data = {
        "load": [],
        "elastic_energy": [],
        "fracture_energy": [],
        "total_energy": [],
        "solver_data": [],
        "solver_HY_data": [],
        "solver_KS_data": [],
        # "cone-eig": [],
        "eigs": [],
        "uniqueness": [],
        "inertia": [],
        "F": [],
        "alphadot_norm": [],
        "rate_12_norm": [],
        "unscaled_rate_12_norm": [],
        "cone-stable": [],
    }

    check_stability = []

    # logging.basicConfig(level=logging.INFO)
    # logging.getLogger().setLevel(logging.ERROR)
    # logging.getLogger().setLevel(logging.INFO)
    # logging.getLogger().setLevel(logging.DEBUG)

    for i_t, t in enumerate(loads):
        plotter = None

        # for i_t, t in enumerate([0., .99, 1.0, 1.01]):
        u_.interpolate(lambda x: (t * np.ones_like(x[0]), np.zeros_like(x[1])))
        u_.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        # update the lower bound
        alpha.vector.copy(alpha_lb.vector)
        alpha_lb.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        logging.critical("--  --")
        logging.critical("")
        logging.critical("")
        logging.critical("")

        ColorPrint.print_bold("===================-=========")

        logging.critical(f"-- {i_t}/{len(loads)}: Solving for t = {t:3.2f} --")

        # solver.solve()

        ColorPrint.print_bold("   Solving first order: AM*Hybrid   ")
        ColorPrint.print_bold("===================-=============")

        logging.info(f"-- {i_t}/{len(loads)}: Solving for t = {t:3.2f} --")
        hybrid.solve(alpha_lb)

        # compute the rate
        alpha.vector.copy(alphadot.vector)
        alphadot.vector.axpy(-1, alpha_lb.vector)
        alphadot.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        rate_12_norm = hybrid.scaled_rate_norm(alpha, parameters)
        urate_12_norm = hybrid.unscaled_rate_norm(alpha)

        logging.critical(f"alpha vector norm: {alpha.vector.norm()}")
        logging.critical(f"alpha lb norm: {alpha_lb.vector.norm()}")
        logging.critical(f"alphadot norm: {alphadot.vector.norm()}")
        logging.critical(f"vector norms [u, alpha]: {[zi.vector.norm() for zi in z]}")
        logging.critical(f"scaled rate state_12 norm: {rate_12_norm}")
        logging.critical(f"unscaled scaled rate state_12 norm: {urate_12_norm}")

        ColorPrint.print_bold("   Solving second order: Rate Pb.    ")
        ColorPrint.print_bold("===================-=================")

        # n_eigenvalues = 10
        is_stable = bifurcation.solve(alpha_lb)
        is_elastic = bifurcation.is_elastic()
        inertia = bifurcation.get_inertia()
        # bifurcation.save_eigenvectors(filename=f"{prefix}/{_nameExp}_eigv_{t:3.2f}.xdmf")
        check_stability.append(is_stable)

        logging.critical(f"State is elastic: {is_elastic}")
        logging.critical(f"State's inertia: {inertia}")

        ColorPrint.print_bold("   Solving second order: Cone Pb.    ")
        ColorPrint.print_bold("===================-=================")

        stable = cone.my_solve(alpha_lb, eig0=bifurcation._spectrum)

        fracture_energy = comm.allreduce(
            assemble_scalar(form(model.damage_energy_density(state) * dx)),
            op=MPI.SUM,
        )
        elastic_energy = comm.allreduce(
            assemble_scalar(form(model.elastic_energy_density(state) * dx)),
            op=MPI.SUM,
        )
        _stress = model.stress(model.eps(u), alpha)

        stress = comm.allreduce(
            assemble_scalar(form(_stress[0, 0] * dx)),
            op=MPI.SUM,
        )

        _unique = True if inertia[0] == 0 and inertia[1] == 0 else False

        history_data["load"].append(t)
        history_data["fracture_energy"].append(fracture_energy)
        history_data["elastic_energy"].append(elastic_energy)
        history_data["total_energy"].append(elastic_energy + fracture_energy)
        history_data["solver_data"].append(hybrid.data)
        history_data["solver_HY_data"].append(hybrid.newton_data)
        history_data["solver_KS_data"].append(cone.data)
        history_data["eigs"].append(bifurcation.data["eigs"])
        history_data["F"].append(stress)
        history_data["alphadot_norm"].append(alphadot.vector.norm())
        history_data["rate_12_norm"].append(rate_12_norm)
        history_data["unscaled_rate_12_norm"].append(urate_12_norm)
        history_data["cone-stable"].append(stable)
        # history_data["cone-eig"].append(cone.data["lambda_0"])
        history_data["uniqueness"].append(_unique)
        history_data["inertia"].append(inertia)

        with XDMFFile(
            comm, f"{prefix}/{_nameExp}.xdmf", "a", encoding=XDMFFile.Encoding.HDF5
        ) as file:
            file.write_function(u, t)
            file.write_function(alpha, t)

        if comm.rank == 0:
            a_file = open(f"{prefix}/time_data.json", "w")
            json.dump(history_data, a_file)
            a_file.close()

        ColorPrint.print_bold("   Written timely data.    ")
        print()
        print()
        print()

        if hasattr(cone, "perturbation"):
            plotter, _plt = _plot_perturbations_profile(
                [bifurcation.spectrum[0]["beta"], cone.perturbation["beta"]],
                parameters,
                prefix,
                plotter=plotter,
                label="$\\beta(x) $\lambda$={bifurcation._spectrum[0]['lambda']:.2f}$",
                idx=i_t,
                aux=[bifurcation.spectrum, cone.data["lambda_0"]],
            )
            _plt.savefig(f"{prefix}/test_profile-{i_t}.png")

        if hasattr(bifurcation, "spectrum") and len(bifurcation.spectrum) > 0:
            plotter, _plt = _plot_bif_spectrum_profile(
                bifurcation.spectrum,
                parameters,
                prefix,
                plotter=None,
                label="",
                idx=i_t,
            )
            _plt.savefig(f"{prefix}/test_spectrum-{i_t}.png")

            # plotter, _plt = _plot_bif_spectrum_profile_fullvec(bifurcation._spectrum,
            #     parameters, prefix, plotter=None, label='', idx=i_t)
            # _plt.savefig(f"{prefix}/test_spectrum_full-{i_t}.png")

    _timings = list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])

    # Viz

    from utils.plots import plot_AMit_load, plot_force_displacement
    from utils.viz import plot_profile

    if comm.rank == 0:
        plot_energies(history_data, file=f"{prefix}/{_nameExp}_energies.pdf")
        plot_AMit_load(history_data, file=f"{prefix}/{_nameExp}_it_load.pdf")
        plot_force_displacement(
            history_data, file=f"{prefix}/{_nameExp}_stress-load.pdf"
        )

    import pyvista
    from pyvista.utilities import xvfb

    from utils.viz import plot_scalar, plot_vector

    #
    xvfb.start_xvfb(wait=0.05)
    pyvista.OFF_SCREEN = True

    plotter = pyvista.Plotter(
        title="Traction test",
        window_size=[1600, 600],
        shape=(1, 2),
    )
    _plt = plot_scalar(alpha, plotter, subplot=(0, 0))
    _plt = plot_vector(u, plotter, subplot=(0, 1))
    _plt.screenshot(f"{prefix}/traction-state.png")

    xvfb.start_xvfb(wait=0.05)
    pyvista.OFF_SCREEN = True

    plotter = pyvista.Plotter(
        title="Test Profile",
        window_size=[800, 600],
        shape=(1, 1),
    )

    tol = 1e-3
    xs = np.linspace(0 + tol, parameters["geometry"]["Lx"] - tol, 101)
    points = np.zeros((3, 101))
    points[0] = xs

    _plt, data = plot_profile(
        alpha,
        points,
        plotter,
        subplot=(0, 0),
        lineproperties={
            "c": "k",
            "label": f"$\\alpha$ with $\ell$ = {parameters['model']['ell']:.2f}",
        },
    )
    _plt.gca()
    _plt.legend()
    _plt.fill_between(data[0], data[1].reshape(len(data[1])))
    _plt.title("Damage profile")
    _plt.savefig(f"{prefix}/test_alpha_profile.png")

    return history_data, signature, _timings


def _plot_bif_spectrum_profile(
    spectrum, parameters, prefix, plotter=None, label="", idx=""
):
    """docstring for _plot_bif_spectrum_profile"""

    import matplotlib.pyplot as plt

    from utils.viz import plot_profile

    # __import__('pdb').set_trace()
    # fields = spectrum["perturbations_beta"]
    # fields = spectrum["perturbations_beta"]
    fields = [item.get("beta") for item in spectrum]
    n = len(fields)
    num_cols = 1
    num_rows = (n + num_cols - 1) // num_cols

    if plotter is None:
        import pyvista

        # from pyvista.utilities import xvfb

        plotter = pyvista.Plotter(
            title="Bifurcation Spectrum Profile",
            window_size=[1600, 600],
            shape=(num_rows, num_cols),
        )

    # figure, axes = plt.figure()
    # figure, axes = plt.subplots(1, 1, figsize=(8, 6))
    figure, axes = plt.subplots(num_rows, num_cols, figsize=(6, 18))

    tol = 1e-3
    xs = np.linspace(0 + tol, parameters["geometry"]["Lx"] - tol, 101)
    points = np.zeros((3, 101))
    points[0] = xs

    # if n==1: __import__('pdb').set_trace()

    for i, field in enumerate(fields):
        u = field
        # u = field['xk'][1]
        row = i // num_cols
        col = i % num_cols

        _axes = axes[row] if n > 1 else axes
        # __import__('pdb').set_trace()
        # if label == '':
        label = f"$\lambda_{i}$ = {spectrum[i].get('lambda'):.1e}, |$\\beta$|={u.vector.norm():.2f}"

        _plt, data = plot_profile(
            u,
            points,
            plotter,
            subplot=(row, col),
            lineproperties={
                "c": "k",
                "ls": "-",
                # "label": f"$\\alpha$ with $\ell$ = {parameters['model']['ell']:.2f}"
                "label": label,
            },
            fig=figure,
            ax=_axes,
        )

        _axes.axis("off")
        _axes.axhline("0", lw=3, c="k")
        # _plt = None

        # axes[row].plot(xs, xs*i, label=f'mode {i}')
        # _plt = plt.gcf()

    return plotter, _plt


def _plot_bif_spectrum_profile_fullvec(
    fields, parameters, prefix, plotter=None, label="", idx=""
):
    """docstring for _plot_bif_spectrum_profile"""

    import matplotlib.pyplot as plt

    from utils.viz import plot_profile

    # fields = data["perturbations_beta"]
    # fields = data["perturbations_beta"]
    # fields = [item.get('beta') for item in data]
    n = len(fields)
    num_cols = 1
    num_rows = (n + num_cols - 1) // num_cols

    if plotter is None:
        import pyvista

        # from pyvista.utilities import xvfb

        plotter = pyvista.Plotter(
            title="Bifurcation Spectrum Profile",
            window_size=[1600, 600],
            shape=(num_rows, num_cols),
        )

    # figure, axes = plt.figure()
    # figure, axes = plt.subplots(1, 1, figsize=(8, 6))
    figure, axes = plt.subplots(num_rows, num_cols, figsize=(6, 18))

    tol = 1e-3
    xs = np.linspace(0 + tol, parameters["geometry"]["Lx"] - tol, 101)
    points = np.zeros((3, 101))
    points[0] = xs

    # if n==1: __import__('pdb').set_trace()

    for i, field in enumerate(fields):
        u = field["xk"][1]
        # u = field['xk'][1]
        row = i // num_cols
        col = i % num_cols
        print(i)
        print(i)
        print(i)

        # __import__('pdb').set_trace()
        _axes = axes[row] if n > 1 else axes

        # if label == '':
        label = (
            f"mode {i} $\lambda_{i}$ = {field.get('lambda'):.2e}, ||={u.vector.norm()}"
        )

        print(label)

        _plt, data = plot_profile(
            u,
            points,
            plotter,
            subplot=(row, col),
            lineproperties={
                "c": "k",
                "ls": "-",
                # "label": f"$\\alpha$ with $\ell$ = {parameters['model']['ell']:.2f}"
                "label": label,
            },
            fig=figure,
            ax=_axes,
        )

        _axes.axis("off")
        _axes.axhline("0", lw=3, c="k")
        # _plt = None

        # axes[row].plot(xs, xs*i, label=f'mode {i}')
        # _plt = plt.gcf()

    return plotter, _plt


def _plot_perturbations_profile(
    fields, parameters, prefix, plotter=None, label="", idx="", aux=None
):
    import matplotlib.pyplot as plt

    from utils.viz import plot_profile

    u = fields[0]
    # u = fields[0]['xk'][1]
    v = fields[1]

    figure = plt.figure()

    if plotter is None:
        import pyvista

        # from pyvista.utilities import xvfb

        plotter = pyvista.Plotter(
            title="Test Profile",
            window_size=[800, 600],
            shape=(1, 1),
        )

    tol = 1e-3
    xs = np.linspace(0 + tol, parameters["geometry"]["Lx"] - tol, 101)
    points = np.zeros((3, 101))
    points[0] = xs

    _plt, data = plot_profile(
        u,
        points,
        plotter,
        subplot=(0, 0),
        lineproperties={
            "c": "k",
            "ls": "--",
            # "label": f"$\\alpha$ with $\ell$ = {parameters['model']['ell']:.2f}"
            "label": f'space, $\\lambda_0=${aux[0][0].get("lambda"):.1e}',
        },
        fig=figure,
    )

    _plt, data = plot_profile(
        v,
        points,
        plotter,
        subplot=(0, 0),
        lineproperties={
            "c": "k",
            # "label": f"$\\alpha$ with $\ell$ = {parameters['model']['ell']:.2f}"
            "label": f"cone, $\\lambda_K=${aux[1]:.1e}",
        },
        fig=figure,
    )
    ax = _plt.gca()
    ax.set_xticks([0, 1], ["0", "1"])
    ax.set_yticks([])

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    _plt.legend()
    _plt.fill_between(data[0], data[1].reshape(len(data[1])))

    _plt.title("Profile of perturbation")
    # _plt.savefig(f"{prefix}/test_profile{idx}.png")

    return plotter, _plt



def param_ell():
    # for ell in [0.1, 0.2, 0.3]:

    for ell in [0.1, 0.3]:
        with open("../test/atk_parameters.yml") as f:
            parameters = yaml.load(f, Loader=yaml.FullLoader)

        parameters = parameters_vs_ell(parameters, ell)

        message = f"Running test with ell={ell}"

        pretty_parameters = json.dumps(parameters, indent=2)
        ColorPrint.print_bold(f"   {message}     ")
        ColorPrint.print_bold("===================-===============")

        print(pretty_parameters)

        ColorPrint.print_bold(f"   {message}   ")
        ColorPrint.print_bold("===================-===============")

        history_data, signature, timings = traction_with_parameters(
            parameters, slug="atk_vs_ell"
        )
        df = pd.DataFrame(history_data)
        ColorPrint.print_bold(f"   {message}    ")
        ColorPrint.print_bold("===================-===============")
        ColorPrint.print_bold(f"   signature {signature}    ")
        print(df.drop(["solver_data", "solver_KS_data", "solver_HY_data"], axis=1))


def param_s():
    for s in [0.001, 0.01, 0.05]:
        with open("../test/atk_parameters.yml") as f:
            parameters = yaml.load(f, Loader=yaml.FullLoader)

        message = f"Running SPA test with s={s}"

        parameters = parameters_vs_SPA_scaling(parameters, s)
        pretty_parameters = json.dumps(parameters, indent=2)
        ColorPrint.print_bold(f"   {message}    ")
        ColorPrint.print_bold("===================-===============")

        print(pretty_parameters)

        ColorPrint.print_bold(f"   {message}    ")
        ColorPrint.print_bold("===================-===============")

        history_data, signature, timings = traction_with_parameters(
            parameters, slug="atk_vs_s"
        )
        df = pd.DataFrame(history_data)
        ColorPrint.print_bold(f"   {message}    ")
        ColorPrint.print_bold("===================-===============")
        ColorPrint.print_bold(f"   signature {signature}    ")
        print(df.drop(["solver_data", "solver_KS_data", "solver_HY_data"], axis=1))


if __name__ == "__main__":
    from irrevolutions.utils import ColorPrint

    logging.getLogger().setLevel(logging.ERROR)

    # param_ell()
    # param_s()

    with open("../test/atk_parameters.yml") as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    message = "Running SPA test with parameters"

    pretty_parameters = json.dumps(parameters, indent=2)
    ColorPrint.print_bold(pretty_parameters)

    history_data, signature, timings = traction_with_parameters(
        parameters, slug="atk_traction"
    )
    ColorPrint.print_bold(f"   signature {signature}    ")
    df = pd.DataFrame(history_data)
    print(df.drop(["solver_data", "solver_KS_data", "solver_HY_data"], axis=1))
