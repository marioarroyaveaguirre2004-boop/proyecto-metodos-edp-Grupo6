"""
Proyecto métodos numéricos para EDPs - SOLVER FEM
Nancy Burgos
Carlos Arroyave
David Restrepo

Solver de elementos finitos (triángulos lineales P1) para el problema general

    -div(k grad T) + beta (T - T_ref) = q                 en Ω = ∪ Ω_m
                                   T = T_D                en Γ_D   (Dirichlet)
                         k dT/dn     = g                  en Γ_N   (Neumann: g = calor que ENTRA)
                        -k dT/dn     = h (T - T_inf)      en Γ_R   (Robin)

k, beta, T_ref y q se dan por región (material); el tipo de frontera y sus datos
se dan por etiqueta de frontera. Para el chip, beta = h_sup / H es la pérdida de
calor por la cara superior (espesor H) y T_ref la temperatura del disipador.

La malla puede ser la rectangular interna o un archivo .msh de Gmsh (formato 2.2
ASCII) con grupos físicos; el problema completo se puede leer de un JSON
(ver cargar_problema). Incluye el modelo probabilístico de fuentes (cópula
gaussiana / NORTA, sec. 4.5) y el Monte Carlo por superposición.
"""
import json
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace

import numpy as np
import sympy as sp
import scipy.sparse as sps
import scipy.sparse.linalg as spla

# =============================================================================
# 1. Matrices elementales de las ecs (26)-(29)
# =============================================================================
# Discretización de Galerkin (24)
# Cada triángulo se mapea desde el de referencia {ξ, η ≥ 0, ξ + η ≤ 1}:
#   (x, y) = (x1, y1) + J (ξ, η),   det J = 2·area  (>0 si el triángulo es antihorario)
xi, eta, s, L = sp.symbols("xi eta s L")
X = x1, y1, x2, y2, x3, y3 = sp.symbols("x1 y1 x2 y2 x3 y3")
N = sp.Matrix([1 - xi - eta, xi, eta])                  # funciones de forma P1
J = sp.Matrix([[x2 - x1, x3 - x1], [y2 - y1, y3 - y1]])
dN = N.jacobian([xi, eta]) * J.inv()                    # grad phi_i = grad_ref N · J⁻¹


def _int_ref(e):
    """∫_e (·) dΩ = ∫∫_ref (·) det J dξ dη"""
    return sp.simplify(sp.integrate(e * J.det(), (eta, 0, 1 - xi), (xi, 0, 1)))


Ke_sym = (dN * dN.T).applyfunc(_int_ref)    # ec. (26)  rigidez
Me_sym = (N * N.T).applyfunc(_int_ref)      # masa: término de reacción beta y norma L2
fe_sym = N.applyfunc(_int_ref)              # ec. (28)
Nb = sp.Matrix([1 - s, s])                  # phi sobre una arista de longitud L
Re_sym = (Nb * Nb.T * L).integrate((s, 0, 1))   # ec. (27)


def _vectorizada(M, args):
    f = sp.lambdify(args, M, "numpy")
    return lambda *v: np.moveaxis(np.array(f(*v), dtype=float), -1, 0)


_Ke, _Me, _fe = (_vectorizada(M, X) for M in (Ke_sym, Me_sym, fe_sym))
_Re = _vectorizada(Re_sym, (L,))


# =============================================================================
# 2. Mallas: rectangular interna y lector de Gmsh
# =============================================================================
# Una malla es un SimpleNamespace con:
#   nodos   (Nn, 2)   coordenadas
#   elems   (Ne, 3)   triángulos, orientados antihorario
#   region  (Ne,)     etiqueta del material de cada triángulo
#   aristas (Na, 2)   aristas de frontera
#   etiq    (Na,)     etiqueta de frontera de cada arista
#   nombres dict      nombre del grupo físico -> etiqueta (solo mallas de Gmsh)

def malla_rectangular(Lx, Ly, nx, ny):
    """Rectángulo [0,Lx]x[0,Ly]; fronteras Γ1 abajo, Γ2 derecha, Γ3 arriba, Γ4 izquierda."""
    gx, gy = np.meshgrid(np.linspace(0, Lx, nx + 1), np.linspace(0, Ly, ny + 1))
    idx = lambda i, j: j * (nx + 1) + i
    I, Jj = (a.ravel() for a in np.meshgrid(np.arange(nx), np.arange(ny)))
    n0, n1, n2, n3 = idx(I, Jj), idx(I + 1, Jj), idx(I + 1, Jj + 1), idx(I, Jj + 1)
    i, j = np.arange(nx), np.arange(ny)
    aristas = np.vstack([np.c_[idx(i, 0), idx(i + 1, 0)], np.c_[idx(nx, j), idx(nx, j + 1)],
                         np.c_[idx(i + 1, ny), idx(i, ny)], np.c_[idx(0, j + 1), idx(0, j)]])
    elems = np.vstack([np.c_[n0, n1, n2], np.c_[n0, n2, n3]])   # antihorario
    return SimpleNamespace(
        nodos=np.c_[gx.ravel(), gy.ravel()],
        elems=elems,
        region=np.ones(len(elems), dtype=int),
        aristas=aristas,
        etiq=np.repeat([1, 2, 3, 4], [nx, ny, nx, ny]),
        nombres={"abajo": 1, "derecha": 2, "arriba": 3, "izquierda": 4},
    )


def leer_gmsh(ruta):
    """Lee un .msh de Gmsh en formato 2.2 ASCII (Mesh.MshFileVersion = 2.2).

    Usa los grupos físicos: los de dimensión 2 son regiones (materiales) y los de
    dimensión 1 son tramos de frontera. Solo se conservan elementos con grupo físico.
    """
    lineas = Path(ruta).read_text().split("\n")
    pos = {l.strip(): i for i, l in enumerate(lineas) if l.startswith("$")}
    if "$MeshFormat" not in pos or not lineas[pos["$MeshFormat"] + 1].startswith("2"):
        raise ValueError(f"{ruta}: se espera formato MSH 2.2 ASCII "
                         "(en Gmsh: -format msh22 o Mesh.MshFileVersion = 2.2)")

    nombres = {}
    if "$PhysicalNames" in pos:
        i0 = pos["$PhysicalNames"]
        for l in lineas[i0 + 2: i0 + 2 + int(lineas[i0 + 1])]:
            dim, tag, nombre = l.split(maxsplit=2)
            nombres[nombre.strip('"')] = int(tag)

    i0 = pos["$Nodes"]
    datos = np.array([l.split() for l in lineas[i0 + 2: i0 + 2 + int(lineas[i0 + 1])]], dtype=float)
    ids, coords = datos[:, 0].astype(int), datos[:, 1:3]

    i0 = pos["$Elements"]
    tris, reg, segs, etq = [], [], [], []
    for l in lineas[i0 + 2: i0 + 2 + int(lineas[i0 + 1])]:
        v = list(map(int, l.split()))
        tipo, ntags = v[1], v[2]
        fisico, nod = (v[3] if ntags > 0 else 0), v[3 + ntags:]
        if fisico == 0:
            continue
        if tipo == 2:
            tris.append(nod); reg.append(fisico)
        elif tipo == 1:
            segs.append(nod); etq.append(fisico)
    if not tris:
        raise ValueError(f"{ruta}: no hay triángulos con grupo físico")

    # renumerar 0..n-1 conservando solo nodos usados por triángulos
    usados = np.unique(np.array(tris))
    mapa = np.full(ids.max() + 1, -1)
    mapa[usados] = np.arange(len(usados))
    coords_por_id = np.zeros((ids.max() + 1, 2)); coords_por_id[ids] = coords
    nodos = coords_por_id[usados]
    elems = mapa[np.array(tris)]
    aristas = mapa[np.array(segs, dtype=int).reshape(-1, 2)]
    if (aristas < 0).any():
        raise ValueError(f"{ruta}: hay aristas de frontera con nodos fuera de la malla 2D")

    # orientar todos los triángulos antihorario (det J > 0)
    P = nodos[elems]
    det = ((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
           - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1]))
    elems[det < 0] = elems[det < 0][:, [0, 2, 1]]

    return SimpleNamespace(nodos=nodos, elems=elems, region=np.array(reg), aristas=aristas,
                           etiq=np.array(etq, dtype=int), nombres=nombres)


def indicadora(malla, rect):
    c = malla.nodos[malla.elems].mean(axis=1)
    x0, x1_, y0, y1_ = rect
    return ((c[:, 0] > x0) & (c[:, 0] < x1_) & (c[:, 1] > y0) & (c[:, 1] < y1_)).astype(float)


def por_regiones(malla, regiones, fondo):
    """k(x,y) = sum k_m χ_Ωm (ec. 7). regiones = [(x0, x1, y0, y1, k_m), ...]."""
    k = np.full(len(malla.elems), float(fondo))
    for *rect, km in regiones:
        k[indicadora(malla, rect) == 1] = km
    return k


# =============================================================================
# 3. Evaluación de datos: escalar, dict {etiqueta: valor}, o función
# =============================================================================
def _en_frontera(v, pts, etiq):
    """Dato en puntos de frontera: escalar, dict {gamma: valor/función(x,y)} o f(x, y, gamma)."""
    if v is None:
        return np.zeros(len(etiq))
    if callable(v):
        return np.asarray(v(pts[:, 0], pts[:, 1], etiq), dtype=float) * np.ones(len(etiq))
    if isinstance(v, dict):
        out = np.zeros(len(etiq))
        for g, val in v.items():
            sel = etiq == g
            out[sel] = val(pts[sel, 0], pts[sel, 1]) if callable(val) else float(val)
        return out
    return np.full(len(etiq), float(v))


def _por_elemento(v, malla):
    """Coeficiente constante por elemento: escalar, arreglo (Ne,) o dict {region: valor}."""
    ne = len(malla.elems)
    if isinstance(v, dict):
        out = np.zeros(ne)
        for r, val in v.items():
            out[malla.region == r] = float(val)
        return out
    return np.broadcast_to(np.asarray(v, dtype=float), (ne,)).copy()


def _ensamblar(con, loc, n):
    """Suma las matrices locales (Ne, m, m) en la global dispersa (n, n)"""
    m = con.shape[1]
    filas, cols = np.repeat(con, m, axis=1).ravel(), np.tile(con, m).ravel()
    return sps.coo_matrix((loc.ravel(), (filas, cols)), shape=(n, n)).tocsr()


# =============================================================================
# 4. Solver: sistema de Galerkin (25)  (K + R + M_beta) T = F + G
# =============================================================================
class SolverFEM:
    """Ensambla y factoriza una sola vez; luego resuelve para cualquier carga F.

    k, beta, T_ref : escalar, arreglo por elemento o dict {region: valor}.
    h, T_inf       : Robin. Escalar, dict {gamma: valor o f(x,y)} o f(x, y, gamma).
                     h = 0 en una frontera equivale a Neumann homogénea (aislada).
    flujo          : Neumann, calor que entra (W/m²). Mismo formato que h.
    dirichlet      : dict {gamma: valor o f(x,y)} con la temperatura impuesta.
    """

    def __init__(self, malla, k, h=0.0, T_inf=0.0, beta=0.0, T_ref=0.0, flujo=None, dirichlet=None):
        m, n = malla, len(malla.nodos)
        P = m.nodos[m.elems]
        c = (P[:, 0, 0], P[:, 0, 1], P[:, 1, 0], P[:, 1, 1], P[:, 2, 0], P[:, 2, 1])
        k, beta, T_ref = (_por_elemento(v, m) for v in (k, beta, T_ref))
        self.malla, self.n = m, n
        self.area = 0.5 * ((c[2] - c[0]) * (c[5] - c[1]) - (c[4] - c[0]) * (c[3] - c[1]))
        if (self.area <= 0).any():
            raise ValueError("hay triángulos con orientación horaria o área nula")

        Me = _Me(*c)
        # (26) rigidez y masa
        self.K = _ensamblar(m.elems, k[:, None, None] * _Ke(*c), n)
        self.M = _ensamblar(m.elems, Me, n)
        # (28) F_i = Σ_e q_e ∫_e φi : se guarda como matriz B (n × Ne) para que F = B q
        self.B = sps.csr_matrix((_fe(*c).ravel(), (m.elems.ravel(), np.repeat(np.arange(len(m.elems)), 3))),
                                shape=(n, len(m.elems)))
        # reacción: ∫ beta T φi  y  ∫ beta T_ref φi
        self.Mb = _ensamblar(m.elems, beta[:, None, None] * Me, n)
        G_sup = self.B @ (beta * T_ref)

        # (27) Robin, (29) su carga; Neumann con la misma matriz de arista con h = 1
        a, b = m.nodos[m.aristas[:, 0]], m.nodos[m.aristas[:, 1]]
        Re1 = _Re(np.linalg.norm(b - a, axis=1))
        Re = _en_frontera(h, (a + b) / 2, m.etiq)[:, None, None] * Re1
        self.R = _ensamblar(m.aristas, Re, n)
        Tn = np.c_[_en_frontera(T_inf, a, m.etiq), _en_frontera(T_inf, b, m.etiq)]
        gn = np.c_[_en_frontera(flujo, a, m.etiq), _en_frontera(flujo, b, m.etiq)]
        G_rob = np.bincount(m.aristas.ravel(), (Re @ Tn[:, :, None]).ravel(), n)
        G_neu = np.bincount(m.aristas.ravel(), (Re1 @ gn[:, :, None]).ravel(), n)
        self.G = G_rob + G_neu + G_sup

        # Dirichlet: nodos fijos con valor T_D; se elimina por levantamiento
        A = (self.K + self.R + self.Mb).tocsr()
        self.TD = np.zeros(n)
        fijo = np.zeros(n, dtype=bool)
        for g, val in (dirichlet or {}).items():
            sel = m.etiq == g
            if not sel.any():
                raise ValueError(f"frontera Dirichlet {g} no existe en la malla")
            nod = np.unique(m.aristas[sel])
            x_, y_ = m.nodos[nod].T
            self.TD[nod] = val(x_, y_) if callable(val) else float(val)
            fijo[nod] = True
        self.libres, self.fijos = np.flatnonzero(~fijo), np.flatnonzero(fijo)
        self.A_ld = A[self.libres][:, self.fijos]
        self.lu = spla.splu(A[self.libres][:, self.libres].tocsc())

    # --- cargas ---
    def carga(self, q_elem):
        """F para q constante por elemento (arreglo (Ne,) o (Ne, m))."""
        return self.B @ q_elem

    def carga_nodal(self, q_nodal):
        """F para q interpolada linealmente desde los nodos."""
        return self.M @ q_nodal

    def carga_funcion(self, f):
        """F para q = f(x, y) con cuadratura de puntos medios (exacta para grado 2)."""
        P = self.malla.nodos[self.malla.elems]
        mid = np.stack([(P[:, 1] + P[:, 2]) / 2, (P[:, 2] + P[:, 0]) / 2, (P[:, 0] + P[:, 1]) / 2], axis=1)
        fm = np.asarray(f(mid[..., 0], mid[..., 1]), dtype=float) * np.ones(mid.shape[:2])
        # φ_i vale 1/2 en los dos puntos medios de sus aristas y 0 en el opuesto
        loc = self.area[:, None] / 6 * (fm.sum(axis=1, keepdims=True) - fm)
        return np.bincount(self.malla.elems.ravel(), loc.ravel(), self.n)

    # --- solución ---
    def resolver_lineal(self, F):
        """Parte lineal de la solución: datos de frontera y T_ref nulos. Sirve para superposición."""
        F = np.asarray(F, dtype=float)
        T = np.zeros_like(F)
        T[self.libres] = self.lu.solve(F[self.libres])
        return T

    def resolver(self, F):
        """Solución completa con todos los datos de frontera. F: (n,) o (n, m)."""
        F = np.asarray(F, dtype=float)
        col = (lambda v: v) if F.ndim == 1 else (lambda v: v[:, None])
        rhs = (F + col(self.G))[self.libres] - col(self.A_ld @ self.TD[self.fijos])
        T = np.array(np.broadcast_to(col(self.TD), F.shape))
        T[self.libres] = self.lu.solve(rhs)
        return T

    def norma_L2(self, v):
        return np.sqrt(v @ (self.M @ v))


# =============================================================================
# 5. Problema desde archivo JSON
# =============================================================================
_x, _y = sp.symbols("x y")


def _dato(v):
    """Número -> float. Texto -> expresión en x, y (sintaxis de SymPy) -> función numpy."""
    if isinstance(v, str):
        f = sp.lambdify((_x, _y), sp.sympify(v), "numpy")
        return lambda x, y: np.asarray(f(x, y), dtype=float) * np.ones_like(x, dtype=float)
    return float(v)


def cargar_problema(ruta):
    """Lee un problema completo desde un JSON y devuelve (malla, solver, F).

    Formato:
    {
      "malla": "archivo.msh",                       (relativo al JSON)
      "materiales": {"<grupo físico 2D>": {"k": 150, "q": 0, "beta": 0, "T_ref": 25}},
      "fronteras":  {"<grupo físico 1D>": {"tipo": "dirichlet", "T": 80}
                                          {"tipo": "robin", "h": 10, "T_inf": 25}
                                          {"tipo": "neumann", "flujo": 0}}
    }
    "q" (por material), "T", "T_inf" y "flujo" pueden ser texto con una expresión en x, y.
    Las fronteras de la malla que no aparezcan en el JSON quedan aisladas.
    """
    ruta = Path(ruta)
    cfg = json.loads(ruta.read_text(encoding="utf-8"))
    malla = leer_gmsh(ruta.parent / cfg["malla"])
    tag = lambda nombre: malla.nombres[nombre] if nombre in malla.nombres else int(nombre)

    mats = {tag(nom): d for nom, d in cfg["materiales"].items()}
    faltan = set(np.unique(malla.region)) - set(mats)
    if faltan:
        raise ValueError(f"regiones sin material en el JSON: {sorted(faltan)}")
    coef = {c: {r: float(d.get(c, 0.0)) for r, d in mats.items()} for c in ("k", "beta", "T_ref")}

    h, T_inf, flujo, dirichlet = {}, {}, {}, {}
    for nom, d in cfg.get("fronteras", {}).items():
        g, tipo = tag(nom), d["tipo"].lower()
        if tipo == "dirichlet":
            dirichlet[g] = _dato(d["T"])
        elif tipo == "robin":
            h[g], T_inf[g] = float(d["h"]), _dato(d["T_inf"])
        elif tipo == "neumann":
            flujo[g] = _dato(d.get("flujo", 0.0))
        else:
            raise ValueError(f"tipo de frontera desconocido: {d['tipo']}")

    solver = SolverFEM(malla, coef["k"], h=h, T_inf=T_inf, beta=coef["beta"], T_ref=coef["T_ref"],
                       flujo=flujo, dirichlet=dirichlet)

    F = np.zeros(solver.n)
    for r, d in mats.items():
        q = _dato(d.get("q", 0.0))
        if callable(q):
            ind = malla.region == r
            F += solver.carga_funcion(lambda x, y, ind=ind: q(x, y) * ind[:, None])
        else:
            F += solver.carga(q * (malla.region == r))
    return malla, solver, F


# =============================================================================
# 6. Modelo probabilístico de fuentes (secs. 4.4-4.5)
# =============================================================================
class ModeloFuentes:
    """10,12,13"""

    def __init__(self, malla, bloques, P0, H, p, ell):
        b = np.asarray(bloques, dtype=float)
        nb = len(b)
        self.bloques = b
        self.chi = np.column_stack([indicadora(malla, r) for r in b])           # χ_Ωi (Ne, Nb)
        self.qi = np.broadcast_to(P0, (nb,)) / ((b[:, 1] - b[:, 0]) * (b[:, 3] - b[:, 2]) * H)
        self.centros = np.c_[b[:, :2].mean(axis=1), b[:, 2:].mean(axis=1)]
        d = np.linalg.norm(self.centros[:, None] - self.centros[None], axis=2)  # distancias d_ij
        Sigma = np.exp(-d / ell) if ell > 0 else np.eye(nb)
        self.Lc = np.linalg.cholesky(Sigma + 1e-12 * np.eye(nb))
        self.c = np.array([NormalDist().inv_cdf(1 - pi) for pi in np.broadcast_to(p, (nb,))])

    def muestrear(self, n, rng):
        """14"""
        Z = rng.standard_normal((n, len(self.c))) @ self.Lc.T   # Z = L ε ~ N(0, Σ)
        return (Z > self.c).astype(float)                       # ec. (13)

    def q(self, A):
        """10"""
        return self.chi @ (self.qi * A).T


def montecarlo(solver, fuentes, n_real, T_crit, rng):
    """MONTE CARLO con (30) y (31). T = T0 + Σ A_i U_i (superposición)."""
    T0 = solver.resolver(np.zeros(solver.n))                         # sin fuentes
    U = solver.resolver_lineal(solver.carga(fuentes.chi * fuentes.qi))   # (n, Nb)
    A = fuentes.muestrear(n_real, rng)
    T_max = (T0[:, None] + U @ A.T).max(axis=0)                      # ec. (30)
    return A, T_max, np.mean(T_max > T_crit)                         # ec. (31)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    # Ejemplo: chip de 10 mm de silicio con 3×3 bloques de 2 mm
    malla = malla_rectangular(10e-3, 10e-3, 50, 50)
    solver = SolverFEM(malla, k=148.0, h=1e4, T_inf=25.0)
    pos = np.array([1.0, 4.0, 7.0]) * 1e-3
    bloques = [(x0, x0 + 2e-3, y0, y0 + 2e-3) for y0 in pos for x0 in pos]
    fuentes = ModeloFuentes(malla, bloques, P0=1.0, H=0.5e-3, p=0.5, ell=3e-3)
    A, T_max, pf = montecarlo(solver, fuentes, 2000, T_crit=70.0, rng=np.random.default_rng(0))
    print(f"T_max media = {T_max.mean():.2f} °C,  P(T_max > 70 °C) = {pf:.3f}")
