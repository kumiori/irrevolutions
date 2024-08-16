#!/usr/bin/env python3

import numpy as np
from mpi4py import MPI


def mesh_V_notch(
    name,
    L,
    L_crack,
    theta,
    lc,
    shift=(0, 0),
    tdim=2,
    order=1,
    msh_file=None,
    comm=MPI.COMM_WORLD,
):
    """
    Create mesh of 3d tensile test specimen according to ISO 6892-1:2019 using the Python API of Gmsh.
    """
    # Perform Gmsh work only on rank = 0

    if comm.rank == 0:
        import gmsh

        # Initialise gmsh and set options
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)

        gmsh.option.setNumber("Mesh.Algorithm", 5)
        model = gmsh.model()

        p0 = model.geo.addPoint(shift[0] + 0.0, shift[1] + 0.0, 0, lc, tag=0)
        p1 = model.geo.addPoint(shift[0] + 0.0, shift[1] + L, 0.0, lc, tag=1)
        p2 = model.geo.addPoint(shift[0] + L, shift[1] + L, 0.0, lc, tag=2)
        p3 = model.geo.addPoint(shift[0] + L, shift[1] + 0.0, 0.0, lc, tag=3)
        p4 = model.geo.addPoint(shift[0] + L_crack, shift[1] + L / 2.0, 0.0, lc, tag=4)
        p5 = model.geo.addPoint(
            shift[0] + 0.0,
            shift[1] + L / 2.0 + L_crack * np.tan(theta / 2.0),
            0.0,
            lc,
            tag=5,
        )
        p6 = model.geo.addPoint(
            shift[0] + 0.0,
            shift[1] + L / 2.0 - L_crack * np.tan(theta / 2.0),
            0.0,
            lc,
            tag=6,
        )

        top = model.geo.addLine(p1, p2, tag=7)
        right = model.geo.addLine(p2, p3, tag=8)
        bottom = model.geo.addLine(p3, p0, tag=9)
        left_bottom = model.geo.addLine(p0, p6, tag=10)
        left_top = model.geo.addLine(p5, p1, tag=11)
        notch_top = model.geo.addLine(p6, p4, tag=12)
        notch_bottom = model.geo.addLine(p4, p5, tag=13)

        cell_tag_names = {"Domain": 15}

        cloop = model.geo.addCurveLoop(
            [top, right, bottom, left_bottom, notch_bottom, notch_top, left_top]
        )

        s = model.geo.addPlaneSurface([cloop])
        model.geo.addSurfaceLoop([s, 14])
        model.geo.synchronize()
        surface_entities = [model[1] for model in model.getEntities(tdim)]
        model.addPhysicalGroup(tdim, surface_entities, tag=15)
        model.setPhysicalName(tdim, 15, "Square surface")

        # Define physical groups for subdomains (! target tag > 0)
        # gmsh.model.addPhysicalGroup(tdim - 1, [10, 11], tag=16)
        # gmsh.model.setPhysicalName(tdim - 1, 16, "left")
        # gmsh.model.addPhysicalGroup(tdim - 1, [7], tag=17)
        # gmsh.model.setPhysicalName(tdim - 1, 17, "top")
        # gmsh.model.addPhysicalGroup(tdim - 1, [8], tag=18)
        # gmsh.model.setPhysicalName(tdim - 1, 18, "right")
        # gmsh.model.addPhysicalGroup(tdim - 1, [9], tag=19)
        # gmsh.model.setPhysicalName(tdim - 1, 19, "bottom")
        # gmsh.model.addPhysicalGroup(tdim - 1, [12, 13], tag=20)
        # gmsh.model.setPhysicalName(tdim - 1, 20, "notch")
        gmsh.model.addPhysicalGroup(tdim - 1, [10, 11, 7, 8, 9], tag=100)
        gmsh.model.setPhysicalName(tdim - 1, 100, "extboundary")

        model.mesh.generate(tdim)

        facet_tag_names = {"extboundary": 100}
        tag_names = {"facets": facet_tag_names, "cells": cell_tag_names}

        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)

    return gmsh.model if comm.rank == 0 else None, tdim, tag_names
