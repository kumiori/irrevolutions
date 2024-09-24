#!/usr/bin/env python3

from mpi4py import MPI


def mesh_tdcb(
    name,
    geom_parameters,
    lc,
    tdim=2,
    order=1,
    msh_file=None,
    comm=MPI.COMM_WORLD,
):
    """
    Create mesh of 2d TDCB test specimen according to ... using the Python API of Gmsh.
    """
    # Perform Gmsh work only on rank = 0

    if comm.rank == 0:
        import gmsh

        # Initialise gmsh and set options
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)

        gmsh.option.setNumber("Mesh.Algorithm", 5)
        model = gmsh.model()

        eta = geom_parameters.get("eta")
        Lx = geom_parameters.get("Lx")
        h1 = geom_parameters.get("L1")
        h2 = geom_parameters.get("L2")
        a0 = geom_parameters.get("Lcrack")
        cx = geom_parameters.get("Cx")
        cy = geom_parameters.get("Cy")
        rad = geom_parameters.get("rad")

        # pdb.set_trace()
        p1 = model.geo.addPoint(0, eta, 0, lc, tag=1)
        p100 = model.geo.addPoint(0, -eta, 0, lc, tag=100)
        p2 = model.geo.addPoint(0, h1 / 2.0, 0, lc, tag=2)
        p200 = model.geo.addPoint(0, -h1 / 2.0, 0, lc, tag=200)
        p3 = model.geo.addPoint(Lx, h2 / 2.0, 0, lc, tag=3)
        p300 = model.geo.addPoint(Lx, -h2 / 2.0, 0, lc, tag=300)
        p4 = model.geo.addPoint(a0, eta, 0, lc, tag=4)
        p400 = model.geo.addPoint(a0, -eta, 0, lc, tag=400)

        model.geo.addPoint(cx, cy, 0.0, lc, tag=5)
        model.geo.addPoint(cx, cy - rad, 0.0, lc, tag=500)
        model.geo.addPoint(cx, cy + rad, 0.0, lc, tag=501)
        model.geo.addPoint(cx - rad, cy, 0.0, lc, tag=502)
        model.geo.addPoint(cx + rad, cy, 0.0, lc, tag=503)
        model.geo.addPoint(cx, -cy, 0.0, lc, tag=6)
        model.geo.addPoint(cx, -cy + rad, 0.0, lc, tag=600)
        model.geo.addPoint(cx, -cy - rad, 0.0, lc, tag=601)
        model.geo.addPoint(cx - rad, -cy, 0.0, lc, tag=602)
        model.geo.addPoint(cx + rad, -cy, 0.0, lc, tag=603)

        # left = model.geo.addLine(p200, p2, tag=5)
        left_top = model.geo.addLine(p1, p2, tag=5)
        top = model.geo.addLine(p2, p3, tag=6)
        right = model.geo.addLine(p3, p300, tag=7)
        bottom = model.geo.addLine(p300, p200, tag=8)
        left_bottom = model.geo.addLine(p200, p100, tag=9)
        notch_bottom = model.geo.addLine(p100, p400, tag=10)
        notch_tip = model.geo.addLine(p400, p4, tag=11)
        notch_top = model.geo.addLine(p4, p1, tag=13)

        model.geo.addCircleArc(501, 5, 503, tag=14)
        model.geo.addCircleArc(503, 5, 500, tag=15)
        model.geo.addCircleArc(500, 5, 502, tag=16)
        model.geo.addCircleArc(502, 5, 501, tag=17)
        model.geo.addCircleArc(600, 6, 603, tag=18)
        model.geo.addCircleArc(603, 6, 601, tag=19)
        model.geo.addCircleArc(601, 6, 602, tag=20)
        model.geo.addCircleArc(602, 6, 600, tag=21)

        cloop1 = model.geo.addCurveLoop(
            [
                left_top,
                top,
                right,
                bottom,
                # left,
                left_bottom,
                notch_bottom,
                notch_tip,
                notch_top,
            ]
        )

        cloop2 = model.geo.addCurveLoop([15, 16, 17, 14])
        cloop3 = model.geo.addCurveLoop([18, 19, 20, 21])

        s = model.geo.addPlaneSurface([cloop1, cloop2, cloop3])
        model.geo.addSurfaceLoop([s, 1000])
        model.geo.synchronize()
        surface_entities = [model[1] for model in model.getEntities(tdim)]
        model.addPhysicalGroup(tdim, surface_entities, tag=1)
        model.setPhysicalName(tdim, 1, "TDCB surface")

        # Define physical groups for subdomains (! target tag > 0)

        gmsh.model.addPhysicalGroup(tdim - 1, [15, 16, 17, 14], tag=2)
        gmsh.model.setPhysicalName(tdim - 1, 2, "top_pin")

        gmsh.model.addPhysicalGroup(tdim - 1, [18, 19, 20, 21], tag=3)
        gmsh.model.setPhysicalName(tdim - 1, 3, "bottom_pin")

        gmsh.model.addPhysicalGroup(tdim - 1, [6], tag=4)
        gmsh.model.setPhysicalName(tdim - 1, 4, "top_boundary")

        gmsh.model.addPhysicalGroup(tdim - 1, [8], tag=5)
        gmsh.model.setPhysicalName(tdim - 1, 5, "bottom_boundary")

        model.mesh.generate(tdim)

        cell_tag_names = {"Domain": 1}

        facet_tag_names = {
            "top_pin": 2,
            "bottom_pin": 3,
            "top_boundary": 4,
            "bottom_boundary": 5,
        }

        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)

    tag_names = {"facets": facet_tag_names, "cells": cell_tag_names}

    return gmsh.model if comm.rank == 0 else None, tdim, tag_names
