from mpi4py import MPI
import numpy as np
import os
import sys

sys.path.append("../")
from meshes import (
    _addPoint as addPoint,
    _addLine as addLine,
    _addCircleArc as addCircleArc,
    _addCurveLoop as addCurveLoop,
    _addPlaneSurface as _addPlaneSurface,
    _addPhysicalSurface as _addPhysicalSurface,
)

from pathlib import Path


def mesh_extended_pacman(
    name,
    geom_parameters,
    tdim=2,
    order=1,
    msh_file="extended_pacman.msh",
    comm=MPI.COMM_WORLD,
):
    """
    Create mesh.
    """

    if comm.rank == 0:
        import gmsh
        import warnings

        warnings.filterwarnings("ignore")
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        # model = gmsh.model()
        omega = np.deg2rad(geom_parameters.get("omega"))
        radius = geom_parameters.get("r")
        rho = geom_parameters.get("rho")
        lc = geom_parameters.get("lc")
        refinement = geom_parameters.get("refinement")

        model = gmsh.model
        model.add("extended_pacman")

        p0 = addPoint(0, 0, 0, lc / refinement, tag=0)
        p1 = addPoint(
            -radius * np.cos(omega / 2), radius * np.sin(omega / 2), 0.0, lc, tag=1
        )
        p2 = addPoint(
            -radius * np.cos(omega / 2), -radius * np.sin(omega / 2), 0.0, lc, tag=2
        )
        p3 = addPoint(radius, 0, 0.0, lc / refinement, tag=12)

        p10 = addPoint(
            -rho * radius * np.cos(omega / 2),
            rho * radius * np.sin(omega / 2),
            0.0,
            lc,
            tag=10,
        )
        p20 = addPoint(
            -rho * radius * np.cos(omega / 2),
            -rho * radius * np.sin(omega / 2),
            0.0,
            lc,
            tag=20,
        )
        p30 = addPoint(rho * radius, 0, 0.0, lc, tag=120)

        top = addLine(p1, p0, tag=3)
        bot = addLine(p0, p2, tag=4)
        arc1 = addCircleArc(2, 0, 12, tag=5)
        arc2 = addCircleArc(12, 0, 1, tag=6)
        cloop = addCurveLoop([top, bot, arc1, arc2], tag=1000)

        top_ext = addLine(p10, p1, tag=30)
        bot_ext = addLine(p2, p20, tag=40)
        arc1_int = addCircleArc(1, 0, 12, tag=50)
        arc2_int = addCircleArc(12, 0, 2, tag=60)
        arc1_ext = addCircleArc(20, 0, 120, tag=51)
        arc2_ext = addCircleArc(120, 0, 10, tag=61)
        cloop_ext = addCurveLoop(
            [top_ext, arc1_int, arc2_int, bot_ext, arc1_ext, arc2_ext], tag=1010
        )

        _addPlaneSurface([cloop], tag=100)
        _addPlaneSurface([cloop_ext], tag=101)

        model.geo.addSurfaceLoop([cloop, cloop_ext, 15])

        model.geo.synchronize()
        entities = model.getEntities(dim=2)

        _addPhysicalSurface(tdim, [entities[0][1]], tag=1)
        model.setPhysicalName(tdim, 1, "Pacman")

        _addPhysicalSurface(tdim, [entities[1][1]], tag=100)
        model.setPhysicalName(tdim, 100, "Extended Domain")
        # model.addPhysicalGroup(tdim, [entities[0][1], entities[1][1]], tag=1000)

        model.geo.synchronize()
        model.mesh.generate(tdim)

        if msh_file is not None:
            Path(os.path.dirname(msh_file)).mkdir(parents=True, exist_ok=True)
            gmsh.write(msh_file)

    return gmsh.model if comm.rank == 0 else None, tdim


if __name__ == "__main__":
    import sys
    import yaml

    # , merge_meshtags, locate_dofs_topological
    from mpi4py import MPI

    _geom_parameters = """
        elltomesh: 1
        geom_type: local_notch
        geometric_dimension: 2
        lc: 0.1
        omega: 45
        r: 1.0
        rho: 1.3
        refinement: 4
    """

    geom_parameters = yaml.load(_geom_parameters, Loader=yaml.FullLoader)

    mesh = mesh_extended_pacman(
        "extended pacman",
        geom_parameters,
        tdim=2,
        order=1,
        msh_file="extended_pacman.msh",
        comm=MPI.COMM_WORLD,
    )
