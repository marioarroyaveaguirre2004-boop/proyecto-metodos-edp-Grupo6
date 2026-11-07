"""
Segundo caso de verificación: solución manufacturada sobre mallas NO estructuradas
de Gmsh, con dos materiales, un agujero, término de reacción y los tres tipos de
frontera (Dirichlet, Neumann y Robin).

Dominio: [0,1]² con interfaz en x = 0.5 y agujero de radio 0.15 en (0.75, 0.5).
    material 1 (x < 0.5): k1 = 1,  beta1 = 0
    material 2 (x > 0.5): k2 = 4,  beta2 = 3, T_ref = 1   (como la pérdida por la cara superior)
Fronteras:
    izquierda          Dirichlet   T = T_ex
    abajo, arriba      Robin       h = 5,  T_inf de la ec. (17) despejada
    derecha, agujero   Neumann     g = k dT_ex/dn

Solución exacta T_m(x, y) = u_m(x) v(y). Para que sea solución del problema con dos
materiales debe cumplir en la interfaz x = 0.5 continuidad de temperatura
(u1 = u2) y de flujo (k1 u1' = k2 u2'). Con u1 = 1 + x²:  u1(0.5) = 1.25, k1 u1'(0.5) = 1,
así que u2(0.5) = 1.25 y u2'(0.5) = 1/4.

Uso:  python verificacion_mixta.py    (antes: python mallas/generar_mallas.py)
"""
import sys
from pathlib import Path

import numpy as np
import sympy as sp
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from fem_solver import leer_gmsh, SolverFEM

sys.stdout.reconfigure(encoding="utf-8")
AQUI = Path(__file__).parent
plt.style.use(AQUI / "estilo.mplstyle")

# =============================================================================
# 1. Solución manufacturada (SymPy)
# =============================================================================
x, y = sp.symbols("x y")
k = {1: 1.0, 2: 4.0}
beta = {1: 0.0, 2: 3.0}
T_ref, h = 1.0, 5.0
xc, yc, rc = 0.75, 0.5, 0.15

v = 2 + sp.sin(sp.pi * y)
u = {1: 1 + x**2,
     2: sp.Rational(5, 4) + (x - sp.Rational(1, 2)) / 4 + (1 - sp.cos(2 * sp.pi * (x - sp.Rational(1, 2)))) / 2}
T = {m: u[m] * v for m in (1, 2)}

# chequeo de las condiciones de interfaz
salto_T = sp.simplify((T[1] - T[2]).subs(x, sp.Rational(1, 2)))
salto_flujo = sp.simplify((k[1] * sp.diff(T[1], x) - k[2] * sp.diff(T[2], x)).subs(x, sp.Rational(1, 2)))
assert salto_T == 0 and salto_flujo == 0, (salto_T, salto_flujo)

grad = {m: (sp.diff(T[m], x), sp.diff(T[m], y)) for m in (1, 2)}
q = {m: sp.simplify(-k[m] * (sp.diff(T[m], x, 2) + sp.diff(T[m], y, 2)) + beta[m] * (T[m] - T_ref))
     for m in (1, 2)}

normales = {"abajo_1": (1, (0, -1)), "abajo_2": (2, (0, -1)),
            "arriba_1": (1, (0, 1)), "arriba_2": (2, (0, 1)),
            "derecha": (2, (1, 0)),
            "agujero": (2, (-(x - xc) / rc, -(y - yc) / rc))}      # normal exterior al dominio
dTdn = {g: grad[m][0] * n[0] + grad[m][1] * n[1] for g, (m, n) in normales.items()}

f = lambda e: sp.lambdify((x, y), e, "numpy")
f_T = {m: f(T[m]) for m in (1, 2)}
f_q = {m: f(q[m]) for m in (1, 2)}
T_inf = {g: f(T[normales[g][0]] + k[normales[g][0]] / h * dTdn[g]) for g in ("abajo_1", "abajo_2", "arriba_1", "arriba_2")}
flujo = {g: f(k[normales[g][0]] * dTdn[g]) for g in ("derecha", "agujero")}

print("Solución manufacturada")
for m in (1, 2):
    print(f"  T{m}(x,y) = {T[m]}")
print(f"  saltos en la interfaz: [T] = {salto_T}, [k dT/dx] = {salto_flujo}")


# =============================================================================
# 2. Resolver en una malla
# =============================================================================
def resolver(ruta):
    m = leer_gmsh(ruta)
    t = m.nombres
    reg = lambda d: {t[f"mat{i}"]: val for i, val in d.items()}
    s = SolverFEM(m, k=reg(k), beta=reg(beta), T_ref=T_ref,
                  h={t[g]: h for g in T_inf}, T_inf={t[g]: fn for g, fn in T_inf.items()},
                  flujo={t[g]: fn for g, fn in flujo.items()},
                  dirichlet={t["izquierda"]: f_T[1]})
    F = np.zeros(s.n)
    for i in (1, 2):
        dentro = (m.region == t[f"mat{i}"])[:, None]
        F += s.carga_funcion(lambda X, Y, i=i, dentro=dentro: f_q[i](X, Y) * dentro)
    Th = s.resolver(F)
    # solución exacta en los nodos (los nodos de la interfaz valen igual en ambos lados)
    X, Y = m.nodos.T
    Tex = np.where(X <= 0.5, f_T[1](X, Y), f_T[2](X, Y))
    P = m.nodos[m.elems]
    hmax = np.linalg.norm(P - np.roll(P, 1, axis=1), axis=2).max()   # arista más larga
    return m, s, Th, Tex, hmax


# =============================================================================
# 3. Convergencia
# =============================================================================
rutas = sorted((AQUI / "mallas").glob("verificacion_*.msh"))
if not rutas:
    sys.exit("No hay mallas: correr primero  python mallas/generar_mallas.py")

print("\nConvergencia en mallas no estructuradas (orden esperado en L2: 2)")
print(f"  {'nodos':>7} {'h':>8} {'error L2':>11} {'orden':>6} {'error max':>11} {'orden':>6}")
res = []
for r in rutas:
    m, s, Th, Tex, hm = resolver(r)
    e = Th - Tex
    res.append((hm, s.norma_L2(e), np.abs(e).max(), len(m.nodos)))
    if len(res) == 1:
        o = ("-", "-")
    else:
        dh = np.log(res[-2][0] / res[-1][0])
        o = tuple(f"{np.log(res[-2][j] / res[-1][j]) / dh:.2f}" for j in (1, 2))
    print(f"  {res[-1][3]:>7} {hm:8.4f} {res[-1][1]:11.3e} {o[0]:>6} {res[-1][2]:11.3e} {o[1]:>6}")
r_arr = np.array(res)

salida = AQUI / "resultados"
salida.mkdir(exist_ok=True)
np.savetxt(salida / "verificacion_mixta.csv", r_arr, delimiter=",",
           header="h,error_L2,error_max,nodos", comments="")

# =============================================================================
# 4. Figuras (tamaño real de página: ancho de texto ≈ 6.3 in)
# =============================================================================
m, s, Th, Tex, _ = resolver(rutas[2])
tri = mtri.Triangulation(*m.nodos.T, m.elems)

m0 = leer_gmsh(rutas[0])                      # la malla más gruesa se ve mejor
tri0 = mtri.Triangulation(*m0.nodos.T, m0.elems)
fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.3), constrained_layout=True)
ax[0].tripcolor(tri0, facecolors=(m0.region == m0.nombres["mat2"]).astype(float), cmap="Blues",
                vmin=-0.5, vmax=2)
ax[0].triplot(tri0, lw=0.4, color="0.2")
ax[0].text(0.25, 1.04, "$k=1$", ha="center", va="bottom", fontsize=8)
ax[0].text(0.75, 1.04, r"$k=4,\ \beta=3$", ha="center", va="bottom", fontsize=8)
ax[0].set_title("Malla y materiales", pad=14)
for a, campo, tit in ((ax[1], Th, "$T_h$"), (ax[2], np.abs(Th - Tex), r"$|T_h - T_{ex}|$")):
    fig.colorbar(a.tripcolor(tri, campo, shading="gouraud"), ax=a, shrink=0.85)
    a.set_title(tit)
for a in ax:
    a.set(aspect="equal", xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
fig.savefig(salida / "verificacion_mixta_campos.pdf")

fig, ax = plt.subplots(figsize=(3.6, 2.8))
ax.loglog(r_arr[:, 0], r_arr[:, 1], "o-", label=r"$\|e\|_{L^2}$")
ax.loglog(r_arr[:, 0], r_arr[:, 2], "s-", label=r"$\|e\|_\infty$")
ax.loglog(r_arr[:, 0], r_arr[0, 1] * (r_arr[:, 0] / r_arr[0, 0]) ** 2, "k--", label="pendiente 2")
ax.set(xlabel="tamaño máximo de elemento $h$", ylabel="error")
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xticks([0.01, 0.02, 0.05, 0.1], ["0.01", "0.02", "0.05", "0.1"])
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.savefig(salida / "verificacion_mixta_convergencia.pdf")
fig.savefig(salida / "verificacion_mixta_convergencia.png", dpi=200)
print(f"\nFiguras y tabla en {salida}")
