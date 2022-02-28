#!/usr/bin/env python3
import sys

sys.path.append('../')

from meshes import (
    _addPoint,
    _addLine,
    _addCurveLoop,
    _addPlaneSurface,
    _addPhysicalGroup,
    _addCircleArc
)
from mpi4py import MPI
import pdb
import numpy as np
import matplotlib.pyplot as plt

algorithms = {'Delaunay': 5, 'FrontalDelaunay': 6}


def mesh_crackholes(geom_type,
                    Lx,
                    Ly,
                    a,
                    b,
                    ax,
                    bx,
                    lc,
                    xc,
                    c,
                    l1,
                    d,
                    e,
                    f,
                    g,
                    deltac,
                    rhoc,
                    offset=0,
                    tdim=2,
                    order=1,
                    msh_file=None,
                    comm=MPI.COMM_WORLD):
    """
    Create mesh of 2d tensile test specimen -
         Lx: 
         Ly: 
         a: centre ellipse
         b: centre ellipse
         ax: horiz axis ellipse 
         bx: vert axis ellipse
         lc: 
         xc:
         c:
         l1:
         d:
         e:
         f:
         g:
         deltac: 
         rhoc: 
         offset (defaults 0): offset between pins 
         refine ratio:
    """
    # Perform Gmsh work only on rank = 0

    if comm.rank == 0:

        import gmsh

        # Initialise gmsh and set options
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)

        gmsh.option.setNumber("Mesh.Algorithm", 6)
        model = gmsh.model()
        model.add("Rectangle")
        model.setCurrent("Rectangle")
        p1 = model.geo.addPoint(0.0, 0.0, 0, lc, tag=1)
        p2 = model.geo.addPoint(Lx, 0.0, 0, lc, tag=2)
        p3 = model.geo.addPoint(Lx, Ly, 0.0, lc, tag=3)
        p4 = model.geo.addPoint(0, Ly, 0, lc, tag=4)
        p5 = model.geo.addPoint(0, c, 0.0, lc, tag=5)
        p6 = model.geo.addPoint(Lx/4, c, 0, lc/2, tag=6)
        p7 = model.geo.addPoint(d, (Ly/2)+l1, 0.0, lc/3, tag=7)
        p8 = model.geo.addPoint(e, f, 0, lc/3, tag=8)
        p9 = model.geo.addPoint(d, (Ly/2)-l1, 0, lc/3, tag=9)
        p10 = model.geo.addPoint(Lx/4, g, 0, lc/2, tag=10)
        p11 = model.geo.addPoint(0, g, 0, lc, tag=11)

        # xc, deltac, rhoc, offset=0

        p12 = model.geo.addPoint(
            xc-offset, Ly/2+deltac, 0, lc, tag=12)  # cercle1
        p13 = model.geo.addPoint(xc-offset-rhoc, Ly/2+deltac, 0, lc, tag=13)
        p14 = model.geo.addPoint(xc-offset, Ly/2+deltac+rhoc, 0, lc, tag=14)
        p15 = model.geo.addPoint(xc-offset+rhoc, Ly/2+deltac, 0, lc, tag=15)
        p16 = model.geo.addPoint(xc-offset, Ly/2+deltac-rhoc, 0, lc, tag=16)

        p17 = model.geo.addPoint(
            xc+offset, Ly/2-deltac, 0, lc, tag=17)  # cercle2
        p18 = model.geo.addPoint(xc+offset-rhoc, Ly/2-deltac, 0, lc, tag=18)
        p19 = model.geo.addPoint(xc+offset, Ly/2-deltac+rhoc, 0, lc, tag=19)
        p20 = model.geo.addPoint(xc+offset+rhoc, Ly/2-deltac, 0, lc, tag=20)
        p21 = model.geo.addPoint(xc+offset, Ly/2-deltac-rhoc, 0, lc, tag=21)

        p22 = model.geo.addPoint(a, b, 0, lc, tag=22)  # ellipse
        p23 = model.geo.addPoint(a, b+bx, 0, lc/3, tag=23)
        p24 = model.geo.addPoint(a+ax, b, 0, lc/3, tag=24)
        p25 = model.geo.addPoint(a, b-bx, 0, lc/3, tag=25)
        p26 = model.geo.addPoint(a-ax, b, 0, lc/3, tag=26)

        # Lines = [L1, L2, L3, L4, L5, L6, L7, L8]
        bottom = model.geo.addLine(p1, p2, tag=1)
        right = model.geo.addLine(p2, p3, tag=2)
        top = model.geo.addLine(p3, p4, tag=3)
        left1 = model.geo.addLine(p4, p5, tag=4)
        halftop = model.geo.addLine(p5, p6, tag=5)
        inclined1 = model.geo.addLine(p6, p7, tag=6)
        liptop = model.geo.addLine(p7, p8, tag=7)
        lipbot = model.geo.addLine(p8, p9, tag=8)
        inclined2 = model.geo.addLine(p9, p10, tag=9)
        halfbottom = model.geo.addLine(p10, p11, tag=10)
        left2 = model.geo.addLine(p11, p1, tag=11)
        cloop1 = model.geo.addCurveLoop(
            [bottom, right, top, left1, halftop, inclined1, liptop, lipbot, inclined2, halfbottom, left2])
        c1 = gmsh.model.geo.addCircleArc(p13, p12, p14)
        c2 = gmsh.model.geo.addCircleArc(p14, p12, p15)
        c3 = gmsh.model.geo.addCircleArc(p15, p12, p16)
        c4 = gmsh.model.geo.addCircleArc(p16, p12, p13)
        circle1 = model.geo.addCurveLoop([c1, c2, c3, c4])
        c5 = gmsh.model.geo.addCircleArc(p18, p17, p19)
        c6 = gmsh.model.geo.addCircleArc(p19, p17, p20)
        c7 = gmsh.model.geo.addCircleArc(p20, p17, p21)
        c8 = gmsh.model.geo.addCircleArc(p21, p17, p18)
        circle2 = model.geo.addCurveLoop([c5, c6, c7, c8])
        e1 = gmsh.model.geo.addEllipseArc(p26, p22, p24, p25)
        e2 = gmsh.model.geo.addEllipseArc(p24, p22, p26, p25)
        e3 = gmsh.model.geo.addEllipseArc(p24, p22, p26, p23)
        e4 = gmsh.model.geo.addEllipseArc(p26, p22, p24, p23)
        Ellipse = model.geo.addCurveLoop([e1, -e2, e3, -e4])

        # surface_1 =
        model.geo.addPlaneSurface([cloop1, circle1, circle2, Ellipse])

        gmsh.model.geo.synchronize()
        surface_entities = [model[1] for model in model.getEntities(tdim)]

        gmsh.model.addPhysicalGroup(tdim, surface_entities, tag=1)
        gmsh.model.setPhysicalName(tdim, 1, "Rectangle surface")

        gmsh.model.mesh.setOrder(order)

        # export pins as physical
        gmsh.model.addPhysicalGroup(tdim - 1, [circle1], tag=99)
        gmsh.model.setPhysicalName(tdim - 1, 99, "topPin")

        gmsh.model.addPhysicalGroup(tdim - 1, [circle2], tag=66)
        gmsh.model.setPhysicalName(tdim - 1, 66, "botPin")

        # this is class
        gmsh.model.addPhysicalGroup(tdim - 1, [5], tag=9)
        gmsh.model.setPhysicalName(tdim - 1, 9, "botfissure1")
        gmsh.model.addPhysicalGroup(tdim - 1, [6], tag=10)
        gmsh.model.setPhysicalName(tdim - 1, 10, "botfissure2")
        gmsh.model.addPhysicalGroup(tdim - 1, [3], tag=11)
        gmsh.model.setPhysicalName(tdim - 1, 11, "top")
        gmsh.model.addPhysicalGroup(tdim - 1, [1], tag=12)
        gmsh.model.setPhysicalName(tdim - 1, 12, "bottom")
        gmsh.model.addPhysicalGroup(tdim - 1, [7], tag=13)
        gmsh.model.setPhysicalName(tdim - 1, 13, "topfissure1")
        gmsh.model.addPhysicalGroup(tdim - 1, [8], tag=14)
        gmsh.model.setPhysicalName(tdim - 1, 14, "topfissure2")

        gmsh.model.mesh.generate(tdim)

    return gmsh.model if comm.rank == 0 else None


def print_info(gmsh, model, cy, cell_width, cx, points, circles, px, py, ax, k, circle_arcs, _offset):

    print('centr', 'tag', points[_offset + 0],
          cx, cy+k*cell_width + cell_width/2., 0)
    print('point', 'tag', points[_offset + 1],
          px[0], py[0]+k*cell_width + cell_width/2., 0)
    print('point', 'tag', points[_offset + 2],
          px[1], py[1]+k*cell_width + cell_width/2., 0)
    print('point', 'tag', points[_offset + 3],
          px[2], py[2]+k*cell_width + cell_width/2., 0)
    print('point', 'tag', points[_offset + 4],
          px[3], py[3]+k*cell_width + cell_width/2., 0)
    print()

    print(f'circle arc k={k}:', points[1], points[0], points[2])
    print(f'circle arc k={k}:', points[2], points[0], points[3])
    print(f'circle arc k={k}:', points[3], points[0], points[4])
    print(f'circle arc k={k}:', points[4], points[0], points[1])

    print(k, 'circle_arcs', circle_arcs)
    plt.scatter(cx, cy+k*cell_width + cell_width/2., c=k)
    plt.scatter(px[0], py[0]+k*cell_width + cell_width/2., c=k)
    plt.scatter(px[1], py[1]+k*cell_width + cell_width/2., c=k)
    plt.scatter(px[2], py[2]+k*cell_width + cell_width/2., c=k)
    plt.scatter(px[3], py[3]+k*cell_width + cell_width/2., c=k)


def plot_info(gmsh, model, cy, cell_width, cx, points, circles, px, py, ax, k, circle_arcs, _offset):

    ax.annotate(f'{_offset+k}', xy=(cx, cy+k *
                                    cell_width + cell_width/2.), xycoords='data')
    ax.annotate(
        f'{_offset+k+1}', xy=(px[0], py[0]+k*cell_width + cell_width/2.), xycoords='data')
    ax.annotate(
        f'{_offset+k+2}', xy=(px[1], py[1]+k*cell_width + cell_width/2.), xycoords='data')
    ax.annotate(
        f'{_offset+k+3}', xy=(px[2], py[2]+k*cell_width + cell_width/2.), xycoords='data')
    ax.annotate(
        f'{_offset+k+4}', xy=(px[3], py[3]+k*cell_width + cell_width/2.), xycoords='data')


if __name__ == "__main__":
    import sys

    sys.path.append("../damage")
    from dolfinx.io import XDMFFile
    from meshes import gmsh_model_to_mesh

    # , merge_meshtags, locate_dofs_topological
    from mpi4py import MPI
    from pathlib import Path
    import dolfinx.plot



    # Mesh
    parameters = {
        'geometry': {
            'geom_type': 'crackhole',
            # 'Lx': 1.0,
            # 'Ly': 0.5,
            # 'a': .5,
            # 'b': .2,
            # 'lc': .05,
            "Lx" : 1.,
            "Ly" : .5,
            "a" :  .5,
            "b" :  .2,
            "lc" :  .05,
            "xc": .1,
            "deltac": .1,
            "rhoc": .05,
            "offset": 0,
            "tdim" :  2,
            "order" :  0
        },
    }
    Lx = parameters.get("geometry").get("Lx")
    Ly = parameters.get("geometry").get("Ly")
    a = parameters.get("geometry").get("a")
    #b=parameters.get("geometry").get("b")
    geom_type = parameters.get("geometry").get("geom_type")

    gmsh_model=mesh_crackholes('ellipseHole',
                            Lx,
                            Ly,
                            a = .5,
                            b = .2,
                            lc = .05,
                            xc=.1,
                            deltac=.1,
                            rhoc=.05,
                            offset=0,
                            tdim = 2,
                            order = 0
                            )





    # model, tdim, tag_names = mesh_crackholes('ikea_real',
    #                                     geom_parameters=parameters,
    #                                     lc=.1,
    #                                     tdim=2,
    #                                     order=0,
    #                                     msh_file='ikea_real.msh'
    #                                     )

    # model, tdim, tag_names = mesh_crackholes('ikea_real',
    #                                     geom_parameters=parameters,
    #                                     lc=.1,
    #                                     tdim=2,
    #                                     order=0,
    #                                     msh_file='ikea_real.msh'
    #                                     )

    mesh, mts = gmsh_model_to_mesh(
        gmsh_model, cell_data=True, facet_data=False, gdim=2)

    # Path("output").mkdir(parents=True, exist_ok=True)
    # with XDMFFile(MPI.COMM_WORLD, "output/ikea.xdmf", "w") as ofile:
    #     ofile.write_meshtags(mesh, mts)

    import pyvista
    from pyvista.utilities import xvfb

    xvfb.start_xvfb(wait=0.05)
    pyvista.OFF_SCREEN = True
    plotter = pyvista.Plotter(title="Crack Holes mesh")
    topology, cell_types = dolfinx.plot.create_vtk_topology(
        mesh, mesh.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, mesh.geometry.x)
    # plotter.subplot(0, 0)
    actor_1 = plotter.add_mesh(grid, show_edges=True)

    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        plotter.show()
    figure = plotter.screenshot("output/crackholes.png")

    sys.exit()
