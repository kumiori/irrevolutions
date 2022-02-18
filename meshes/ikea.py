#!/usr/bin/env python3

from mpi4py import MPI
import pdb
import numpy as np
import matplotlib.pyplot as plt
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


def mesh_ikea_real(name,
                    geom_parameters,
                    lc,
                    tdim,
                    order=1,
                    msh_file=None,
                    comm=MPI.COMM_WORLD):
    if comm.rank == 0:
        import gmsh

        # Initialise gmsh and set options
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)


        algorithms = {'Delaunay': 5, 'FrontalDelaunay': 6}

        gmsh.option.setNumber("Mesh.Algorithm",
                              algorithms.get('FrontalDelaunay'))
        model = gmsh.model()

        H = geom_parameters.get('H')
        L = geom_parameters.get('L')
        n = geom_parameters.get('n')
        eps = H/n
        # cx = 3*eps
        h = geom_parameters.get('h')
        cy = geom_parameters.get('cy')
        cell_width = H/n
        rad = .2*cell_width
        cx = .4*cell_width
        _localrefine = 10
        
        boundary_points = []

        points = []
        points.append(_addPoint(0,  H, 0, meshSize=lc, tag=len(points)+1))
        points.append(_addPoint(0, 0, 0,  meshSize=lc, tag=len(points)+1))
        points.append(_addPoint(L, 0, 0,  meshSize=lc, tag=len(points)+1))
        points.append(_addPoint(L, H, 0,  meshSize=lc, tag=len(points)+1))

        circles = []
        
        px = [rad+cx, cx, -rad+cx, cx]
        py = [0,     rad,       0, -rad]

        plt.figure(figsize=(10, 30))
        ax = plt.gca()
        for k in range(n):
            circle_arcs = []

            _offset = len(points)+1
            _offset_arcs = 4
            # centre
            points.append(_addPoint(
                cx, cy+k*cell_width + cell_width/2., 0,
                meshSize=2.*np.pi*rad/_localrefine,
                tag=_offset+k))
            # -
            points.append(_addPoint(
                px[0], py[0]+k*cell_width + cell_width/2., 0,
                meshSize=2.*np.pi*rad/_localrefine,
                tag=_offset+k+1))

            points.append(_addPoint(
                px[1], py[1]+k*cell_width + cell_width/2., 0,
                meshSize=2.*np.pi*rad/_localrefine,
                tag=_offset+k+2))

            points.append(_addPoint(
                px[2], py[2]+k*cell_width + cell_width/2., 0,
                meshSize=2.*np.pi*rad/_localrefine,
                tag=_offset+k+3))

            points.append(_addPoint(
                px[3], py[3]+k*cell_width + cell_width/2., 0,
                meshSize=2.*np.pi*rad/_localrefine,
                tag=_offset+k+4))

            _arc_points = points[-5::]
            # print("_arc_points", _arc_points)

            circle_arcs.append(_addCircleArc(
                _arc_points[1], _arc_points[0], _arc_points[2], tag=4*k + 0 + 1))
            circle_arcs.append(_addCircleArc(
                _arc_points[2], _arc_points[0], _arc_points[3], tag=4*k + 1 + 1))
            circle_arcs.append(_addCircleArc(
                _arc_points[3], _arc_points[0], _arc_points[4], tag=4*k + 2 + 1))
            circle_arcs.append(_addCircleArc(
                _arc_points[4], _arc_points[0], _arc_points[1], tag=4*k + 3 + 1))

            # print('k', k, 'circles', circles)
            # print('k', k, 'circle_arcs', circle_arcs)
            circles.append(_addCurveLoop(circle_arcs, tag=k+1))
            # print_info(gmsh, model, cy, cell_width, cx, points, circles, px, py, ax, k, circle_arcs, _offset)
            plot_info(gmsh, model, cy, cell_width, cx, points,
                      circles, px, py, ax, k, circle_arcs, _offset)
 

        ax.annotate('p1', xy=(0, H),  xycoords='data')
        ax.annotate('p2', xy=(0, 0),  xycoords='data')
        ax.annotate('p3', xy=(L, 0),  xycoords='data')
        ax.annotate('p4', xy=(L, H),  xycoords='data')

        plt.scatter(0,  H, c='r')
        plt.scatter(0, 0, c='r')
        plt.scatter(L, 0, c='r')
        plt.scatter(L, H, c='r')

        plt.xlim(0, cell_width)

        plt.savefig('mesh.pdf')
        boundary = []
        boundary.append(_addLine(points[0], points[1], tag=len(boundary)+100))
        boundary.append(_addLine(points[1], points[2], tag=len(boundary)+100))
        boundary.append(_addLine(points[2], points[3], tag=len(boundary)+100))
        boundary.append(_addLine(points[3], points[0], tag=len(boundary)+100))

        _omega = _addCurveLoop(boundary, tag = 100)
        # _addPlaneSurface([omega], tag=1)
        omega = list([_omega])
        holes = [-c for c in circles]
        omega.extend(holes)
        _addPlaneSurface(omega, tag=10)
        # holes = _addCurveLoop(circles, tag = 1001)
        # _addPlaneSurface([omega, -holes])
        # _addPlaneSurface(circles, tag=2)
        model.geo.synchronize()
        surface_entities = [model[1] for model in model.getEntities(tdim)]
        pdb.set_trace()

        s0 = model.addPhysicalGroup(tdim, [surface_entities[0]], tag=surface_entities[0])
        model.setPhysicalName(tdim, surface_entities[0], "OmegaEps")
        # s0 = model.addPhysicalGroup(tdim, [surface_entities[1]], tag=100)
        # model.setPhysicalName(tdim, 100, "OmegaEps")

        gmsh.model.mesh.setOrder(order)
        gmsh.model.addPhysicalGroup(tdim - 1, [circles], tag=1)
        gmsh.model.setPhysicalName(tdim - 1, 1, "nails")

        facet_tag_names = {
            "nails": 1,
            "ext_boundary": 2,
        }
        cell_tag_names = {
            "nails": 1,
            "omega_eps": 2,
        }

        tag_names = {"facets": facet_tag_names, "cells": cell_tag_names}

        model.mesh.generate(tdim)

        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)

    return gmsh.model if comm.rank == 0 else None, tdim, tag_names


def mesh_ikea_nails(name,
                   geom_parameters,
                   lc,
                   tdim,
                   order=1,
                   msh_file=None,
                   comm=MPI.COMM_WORLD):
    if comm.rank == 0:
        import gmsh

        # Initialise gmsh and set options
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)

        algorithms = {'Delaunay': 5, 'FrontalDelaunay': 6}

        gmsh.option.setNumber("Mesh.Algorithm",
                              algorithms.get('Delaunay'))
        model = gmsh.model()
        model.add("Nails")
        model.setCurrent("Nails")

        H = geom_parameters.get('H')
        L = geom_parameters.get('L')
        n = geom_parameters.get('n')
        eps = H/n
        h = geom_parameters.get('h')
        cy = geom_parameters.get('cy')
        cell_width = H/n
        rad = .2*cell_width
        cx = .4*cell_width
        _localrefine = 3

        i = 5
        points = []
        circles = []

        px = [rad+cx, cx, -rad+cx, cx]
        py = [0,     rad,       0, -rad]

        plt.figure(figsize=(10, 30))
        ax = plt.gca()

        for k in range(n):
            circle_arcs = []

            _offset = len(points)
            _offset_arcs = 4
            # centre
            # geo_decorate(model.geo.addPoint, args, kwargs)
            points.append(_addPoint(
                cx, cy+k*cell_width + cell_width/2., 0, 
                meshSize=lc/_localrefine,
                tag=_offset+i+k))
            # -
            points.append(_addPoint(
                px[0], py[0]+k*cell_width + cell_width/2., 0, 
                meshSize=lc/_localrefine, 
                tag=_offset+i+k+1))

            points.append(_addPoint(
                px[1], py[1]+k*cell_width + cell_width/2., 0, 
                meshSize=lc/_localrefine, 
                tag=_offset+i+k+2))

            points.append(_addPoint(
                px[2], py[2]+k*cell_width + cell_width/2., 0, 
                meshSize=lc/_localrefine, 
                tag=_offset+i+k+3))

            points.append(_addPoint(
                px[3], py[3]+k*cell_width + cell_width/2., 0, 
                meshSize=lc/_localrefine, 
                tag=_offset+i+k+4))
            
            _arc_points = points[-5::]
            # print("_arc_points", _arc_points)

            circle_arcs.append(_addCircleArc(
                _arc_points[1], _arc_points[0], _arc_points[2], tag=4*k + 0 + 1))
            circle_arcs.append(_addCircleArc(
                _arc_points[2], _arc_points[0], _arc_points[3], tag=4*k + 1 + 1))
            circle_arcs.append(_addCircleArc(
                _arc_points[3], _arc_points[0], _arc_points[4], tag=4*k + 2 + 1))
            circle_arcs.append(_addCircleArc(
                _arc_points[4], _arc_points[0], _arc_points[1], tag=4*k + 3 + 1))

            # print('k', k, 'circles', circles)
            # print('k', k, 'circle_arcs', circle_arcs)
            circles.append(_addCurveLoop(circle_arcs, tag = k+1))
            # print_info(gmsh, model, cy, cell_width, cx, points, circles, px, py, ax, k, circle_arcs, _offset)
            plot_info(gmsh, model, cy, cell_width, cx, points,
                      circles, px, py, ax, k, circle_arcs, _offset)
        
        plt.xlim(0, cell_width)
        plt.savefig('mesh.pdf')

        # holes = _addCurveLoop(circles)
        _addPlaneSurface(circles, tag = 10)
        model.geo.synchronize()

        surface_entities = [model[1] for model in model.getEntities(tdim)]
        _addPhysicalGroup(tdim, surface_entities, tag=1)
        model.setPhysicalName(tdim, 10, "Nails")

        gmsh.model.mesh.setOrder(order)

        model.mesh.generate(tdim)

        facet_tag_names = {
            "nails": 1,
        }
        cell_tag_names = {
            "nails": 1,
        }

        tag_names = {"facets": facet_tag_names, "cells": cell_tag_names}

    return gmsh.model if comm.rank == 0 else None, tdim, tag_names

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

    parameters = {
        'H': 1.,
        'L': 1.,
        'n': 10,
        'cx': .1,
        'h': .3,
        'cy': 0.,
    }

    # mesh_ikea_nails('ikea_nails',
    #                geom_parameters=parameters,
    #                lc=.1,
    #                tdim=2,
    #                order=0,
    #                msh_file='ikea_nails.msh'
    #                )

    mesh_ikea_real('ikea_real',
                   geom_parameters=parameters,
                    lc=.1,
                    tdim=2,
                    order=0,
                    msh_file='ikea_real.msh'
                    )
    import sys
    sys.exit()
