def mesh_error_gmshapi(name,
                    Lx,
                    Ly,
                    L0, 
                    s,
                    lc,
                    tdim,
                    order=1,
                    msh_file=None,
                    sep=0.1):

    import gmsh

    # Initialise gmsh and set options
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    gmsh.option.setNumber("Mesh.Algorithm", 6)
    model = gmsh.model()
    model.add("Rectangle")
    model.setCurrent("Rectangle")
    p0 = model.geo.addPoint(0.0, 0.0, 0, lc, tag=0)
    p1 = model.geo.addPoint(Lx, 0.0, 0, lc, tag=1)
    p2 = model.geo.addPoint(Lx, Ly, 0.0, lc, tag=2)
    p3 = model.geo.addPoint(0, Ly, 0, lc, tag=3)
    pLa= model.geo.addPoint(0, Ly/2-s/2-sep, 0, lc, tag=8)
    pLb= model.geo.addPoint(0, Ly/2-s/2+sep, 0, lc, tag=5)
    plM= model.geo.addPoint(L0, Ly/2-s/2, 0, lc, tag=9)
    bottom = model.geo.addLine(p0, p1, tag=0)
    right = model.geo.addLine(p1, p2, tag=1)
    top = model.geo.addLine(p2, p3, tag=5)
    leftT = model.geo.addLine(p3, pLb, tag=6)
    crackTL = model.geo.addLine(pLb, plM, tag=7)
    crackBL = model.geo.addLine(plM, pLa, tag=8)
    leftB = model.geo.addLine(pLa, p0, tag=9)
    cloop1 = model.geo.addCurveLoop([bottom, right, top, leftT, crackTL, crackBL, leftB])
    model.geo.addPlaneSurface([cloop1])

    model.geo.synchronize()
    surface_entities = [model[1] for model in model.getEntities(tdim)]
    model.addPhysicalGroup(tdim, surface_entities, tag=5)
    model.setPhysicalName(tdim, 5, "Rectangle surface")
    gmsh.model.mesh.setOrder(order)
        
    gmsh.model.addPhysicalGroup(tdim - 1, [0], tag=10)
    gmsh.model.setPhysicalName(tdim - 1, 10, "bottom")
    gmsh.model.addPhysicalGroup(tdim - 1, [5], tag=11)
    gmsh.model.setPhysicalName(tdim - 1, 11, "top")
    gmsh.model.addPhysicalGroup(tdim - 1, [6, 7, 8, 9], tag=12)
    gmsh.model.setPhysicalName(tdim - 1, 12, "left")
    gmsh.model.addPhysicalGroup(tdim - 1, [1], tag=13)
    gmsh.model.setPhysicalName(tdim - 1, 13, "right")
    model.mesh.generate(tdim)
    return gmsh.model, tdim

def main():

    Lx = 5
    Ly = 10
    s=0.0
    L0=1
    seedDist=0.5

    geom_type = "bar"

    gmsh_model, tdim = mesh_error_gmshapi(geom_type,
                                        Lx, 
                                        Ly,
                                        L0, 
                                        s,   
                                        seedDist, 
                                        sep=0.0,
                                        tdim=2)

if __name__=="__main__":
    main()