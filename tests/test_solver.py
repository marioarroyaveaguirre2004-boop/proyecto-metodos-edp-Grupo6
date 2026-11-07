"""Pruebas del solver.  Correr desde la raíz del repo:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from fem_solver import malla_rectangular, leer_gmsh, SolverFEM, cargar_problema  # noqa: E402

MALLA_GMSH = RAIZ / "mallas" / "verificacion_1.msh"
necesita_gmsh = pytest.mark.skipif(not MALLA_GMSH.exists(),
                                   reason="correr antes: python mallas/generar_mallas.py")


def test_lineal_exacta_dirichlet():
    """P1 reproduce exactamente una solución lineal (prueba de parche)."""
    m = malla_rectangular(2.0, 1.0, 7, 5)
    Tex = lambda x, y: 1 + 2 * x - 3 * y
    s = SolverFEM(m, k=3.0, dirichlet={g: Tex for g in (1, 2, 3, 4)})
    T = s.resolver(np.zeros(s.n))
    assert np.allclose(T, Tex(*m.nodos.T), atol=1e-12)


@necesita_gmsh
def test_lineal_exacta_mixta_gmsh():
    """Lineal exacta en malla no estructurada con Dirichlet + Neumann + Robin y dos materiales
    con la misma k (con k distinta una lineal no cumple la interfaz)."""
    m = leer_gmsh(MALLA_GMSH)
    t = m.nombres
    k, h = 2.0, 7.0
    Tex = lambda x, y: 1 + 2 * x - 3 * y          # grad T = (2, -3)
    dn = {"abajo_1": 3, "abajo_2": 3, "arriba_1": -3, "arriba_2": -3}   # dT/dn
    s = SolverFEM(m, k=k, h={t[g]: h for g in dn},
                  T_inf={t[g]: (lambda x, y, d=d: Tex(x, y) + k / h * d) for g, d in dn.items()},
                  flujo={t["derecha"]: k * 2,
                         t["agujero"]: lambda x, y: -k * (2 * (x - 0.75) - 3 * (y - 0.5)) / 0.15},
                  dirichlet={t["izquierda"]: Tex})
    T = s.resolver(np.zeros(s.n))
    # el agujero es un polígono, así que la normal usada no es exacta: tolerancia pequeña
    assert np.abs(T - Tex(*m.nodos.T)).max() < 2e-2


def test_superposicion():
    """resolver(F1 + F2) = resolver(F1) + resolver_lineal(F2): base del Monte Carlo."""
    m = malla_rectangular(1.0, 1.0, 10, 10)
    s = SolverFEM(m, k=1.5, h={1: 4.0, 3: 2.0}, T_inf=20.0, beta=0.7, T_ref=25.0,
                  dirichlet={4: 30.0})
    rng = np.random.default_rng(0)
    F1, F2 = rng.random(s.n), rng.random(s.n)
    assert np.allclose(s.resolver(F1 + F2), s.resolver(F1) + s.resolver_lineal(F2))
    F = rng.random((s.n, 3))
    assert np.allclose(s.resolver(F)[:, 1], s.resolver(F[:, 1]))


def test_convergencia_robin_rectangular():
    """Caso de verificación 1 del avance: orden 2 en L2."""
    import sympy as sp
    x, y = sp.symbols("x y")
    k, h = 2.0, 5.0
    T = 1 + x * y**2 + sp.sin(sp.pi * x) * sp.cos(sp.pi * y / 2)
    q = sp.lambdify((x, y), -k * (sp.diff(T, x, 2) + sp.diff(T, y, 2)))
    n = {1: (0, -1), 2: (1, 0), 3: (0, 1), 4: (-1, 0)}
    Tinf = {g: sp.lambdify((x, y), T + k / h * (a * sp.diff(T, x) + b * sp.diff(T, y)))
            for g, (a, b) in n.items()}
    fT = sp.lambdify((x, y), T)
    err = []
    for nn in (8, 16, 32):
        m = malla_rectangular(1, 1, nn, nn)
        s = SolverFEM(m, k, h=h, T_inf=Tinf)
        e = s.resolver(s.carga_funcion(q)) - fT(*m.nodos.T)
        err.append(s.norma_L2(e))
    ordenes = np.log2(np.array(err[:-1]) / np.array(err[1:]))
    assert (ordenes > 1.85).all(), ordenes


@necesita_gmsh
def test_json_ejemplo():
    ruta = RAIZ / "ejemplos" / "soporte.json"
    if not (ruta.parent / "soporte.msh").exists():
        pytest.skip("falta ejemplos/soporte.msh")
    malla, s, F = cargar_problema(ruta)
    T = s.resolver(F)
    # principio del máximo: con fuentes nulas y flujo entrante pequeño, T queda acotada
    assert 20 <= T.min() and T.max() <= 90 + 1e-9
