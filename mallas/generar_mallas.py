"""
Generación de mallas con Gmsh (mallador externo; el solver FEM es propio).

Cada malla se guarda en formato MSH 2.2 ASCII con grupos físicos con nombre:
dimensión 2 = materiales, dimensión 1 = tramos de frontera. Esos nombres son
los que se usan en los archivos JSON de entrada.

Uso:  python mallas/generar_mallas.py
Requiere:  pip install gmsh
"""
from pathlib import Path

import gmsh

AQUI = Path(__file__).parent


def _curvas_en_caja(x0, y0, x1, y1, tol=1e-6):
    return [t for _, t in gmsh.model.getEntitiesInBoundingBox(x0 - tol, y0 - tol, -tol,
                                                                x1 + tol, y1 + tol, tol, dim=1)]


def _guardar(ruta, h):
    gmsh.option.setNumber("Mesh.MeshSizeMin", h)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.model.mesh.generate(2)
    gmsh.write(str(ruta))


def malla_verificacion(h, ruta):
    """Cuadrado [0,1]² con dos materiales (interfaz en x = 0.5) y un agujero circular
    de radio 0.15 centrado en (0.75, 0.5), dentro del material 2."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("verificacion")
    occ = gmsh.model.occ
    r1 = occ.addRectangle(0, 0, 0, 0.5, 1)
    r2 = occ.addRectangle(0.5, 0, 0, 0.5, 1)
    disco = occ.addDisk(0.75, 0.5, 0, 0.15, 0.15)
    occ.cut([(2, r2)], [(2, disco)])
    occ.fragment([(2, r1)], [(2, r2)])
    occ.synchronize()

    sup = {}
    for _, s in gmsh.model.getEntities(2):
        cx = gmsh.model.occ.getCenterOfMass(2, s)[0]
        sup.setdefault("mat1" if cx < 0.5 else "mat2", []).append(s)
    for nombre, tags in sup.items():
        gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, tags), nombre)

    fronteras = {
        "izquierda": _curvas_en_caja(0, 0, 0, 1),
        "derecha": _curvas_en_caja(1, 0, 1, 1),
        "abajo_1": _curvas_en_caja(0, 0, 0.5, 0),
        "abajo_2": _curvas_en_caja(0.5, 0, 1, 0),
        "arriba_1": _curvas_en_caja(0, 1, 0.5, 1),
        "arriba_2": _curvas_en_caja(0.5, 1, 1, 1),
        "agujero": _curvas_en_caja(0.6, 0.35, 0.9, 0.65),
    }
    for nombre, tags in fronteras.items():
        gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, tags), nombre)
    _guardar(ruta, h)
    gmsh.finalize()


def malla_soporte(h, ruta):
    """Soporte en L de aluminio (unidades en m) con un inserto de cobre y un
    canal de refrigeración circular. Ejemplo de generalidad: no es el chip."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("soporte")
    occ = gmsh.model.occ
    a = occ.addRectangle(0, 0, 0, 0.10, 0.04)          # brazo horizontal
    b = occ.addRectangle(0, 0.04, 0, 0.04, 0.06)       # brazo vertical
    L, _ = occ.fuse([(2, a)], [(2, b)])
    canal = occ.addDisk(0.075, 0.02, 0, 0.008, 0.008)
    L, _ = occ.cut(L, [(2, canal)])
    inserto = occ.addRectangle(0.01, 0.06, 0, 0.02, 0.03)
    occ.fragment(L, [(2, inserto)])
    occ.synchronize()

    al, cu = [], []
    for _, s in gmsh.model.getEntities(2):
        xc, yc, _ = occ.getCenterOfMass(2, s)
        (cu if (0.01 < xc < 0.03 and 0.06 < yc < 0.09) else al).append(s)
    gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, al), "aluminio")
    gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, cu), "cobre")

    fronteras = {
        "base": _curvas_en_caja(0, 0, 0.10, 0),             # apoyada sobre una pieza caliente
        "extremo": _curvas_en_caja(0.10, 0, 0.10, 0.04),
        "canal": _curvas_en_caja(0.066, 0.011, 0.084, 0.029),
        "tope": _curvas_en_caja(0, 0.10, 0.04, 0.10),
    }
    usadas = {t for v in fronteras.values() for t in v}
    externas = [t for _, t in gmsh.model.getBoundary(gmsh.model.getEntities(2), oriented=False)]
    fronteras["aire"] = [t for t in set(externas) if t not in usadas]
    for nombre, tags in fronteras.items():
        gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, tags), nombre)
    _guardar(ruta, h)
    gmsh.finalize()


if __name__ == "__main__":
    for i, h in enumerate([0.1, 0.05, 0.025, 0.0125, 0.00625]):
        malla_verificacion(h, AQUI / f"verificacion_{i}.msh")
    malla_soporte(0.002, AQUI.parent / "ejemplos" / "soporte.msh")
    print("mallas generadas")
