"""
Verificación del solver con una solución manufacturada (caso 1: rectángulo, Robin) 

 SOLUCIÓN MANUFACTURADA: se elige primero T_ex y se calcula con SymPy qué fuente q
y qué temperatura ambiente T_inf hacen que T_ex sea la solución exacta:
    q     = -k ΔT_ex                          (de la ec 5)
    T_inf = T_ex + (k/h) ∇T_ex · n             (despejando la ec 17 en cada gama)
El solver recibe solo k, h, q y T_inf (los mismos datos que en el chip) y debe
calcular T_ex por su cuenta.

Al final se verifica también el generador NORTA de fuentes (sec. 4.5).
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

from fem_solver import malla_rectangular, SolverFEM, ModeloFuentes

sys.stdout.reconfigure(encoding="utf-8")
plt.style.use(Path(__file__).parent / "estilo.mplstyle")

# =============================================================================
#1. Solucion analítica con sympy
# =============================================================================
# Problema: Ω = [0,1]², k = 2, h = 5 en las cuatro fronteras Γ1..Γ4.
x, y = sp.symbols("x y")
k, h = 2.0, 5.0
T_ex = 1 + x * y**2 + sp.sin(sp.pi * x) * sp.cos(sp.pi * y / 2)

#datos
q_ex = sp.simplify(-k * (sp.diff(T_ex, x, 2) + sp.diff(T_ex, y, 2)))          # ec5
normales = {1: (0, -1), 2: (1, 0), 3: (0, 1), 4: (-1, 0)}                      # n exterior en Γ1..Γ4
dTdn = {g: nx * sp.diff(T_ex, x) + ny * sp.diff(T_ex, y) for g, (nx, ny) in normales.items()}
Tinf_ex = {g: T_ex + k / h * dTdn[g] for g in normales}                        # ec17

res_edp = sp.simplify(-k * (sp.diff(T_ex, x, 2) + sp.diff(T_ex, y, 2)) - q_ex)
res_robin = [sp.simplify(-k * dTdn[g] - h * (T_ex - Tinf_ex[g])) for g in normales]
print("PASO 1. Solucion analítica")
print("  T_exacta(x,y) =", T_ex)
print("  q(x,y)        =", q_ex)
for g in normales:
    print(f"  T_inf en Γ{g}   =", sp.simplify(Tinf_ex[g]))
print(f"  residuo de la EDP = {res_edp},  residuo de Robin en Γ1..Γ4 = {res_robin}")

#numpy para evaluar en los nodos de la malla
f_T, f_q = (sp.lambdify((x, y), e, "numpy") for e in (T_ex, q_ex))
f_Tinf = {g: sp.lambdify((x, y), e, "numpy") for g, e in Tinf_ex.items()}


def T_inf(xv, yv, etiq):
    out = np.empty(len(xv))
    for g, f in f_Tinf.items():
        out[etiq == g] = f(xv[etiq == g], yv[etiq == g])
    return out


# =============================================================================
#solver fem y comparacion con la sln analitica
# =============================================================================
n = 32
malla = malla_rectangular(1.0, 1.0, n, n)
X, Y = malla.nodos.T

solver = SolverFEM(malla, k, h, T_inf)                       
T_fem = solver.resolver(solver.carga_nodal(f_q(X, Y)))  #(K + R) T = F + G
T_ana = f_T(X, Y)                                            
error = T_fem - T_ana

print(f"\nFEM vs analítica en ciertos puntos (malla {n}x{n})")
print(f"  {'(x, y)':>14} {'T analítica':>12} {'T FEM':>12} {'error':>10}")
for px, py in [(0.5, 0.5), (0.25, 0.75), (0.75, 0.25), (0.0, 0.0), (1.0, 1.0)]:
    i = np.argmin((X - px) ** 2 + (Y - py) ** 2)
    print(f"  ({px:4.2f}, {py:4.2f})   {T_ana[i]:12.6f} {T_fem[i]:12.6f} {error[i]:10.2e}")
print(f"  error máximo en los nodos = {np.abs(error).max():.2e}")
print(f"  error en norma L2         = {solver.norma_L2(error):.2e}")

#==========================================================================
#convergencia al refinar la malla
# =============================================================================
print("\nPASO 4. Convergencia al refinar la malla (orden esperado en L2: 2)")
print(f"  {'malla':>7} {'h':>8} {'error L2':>11} {'orden':>6} {'error max':>11} {'orden':>6}")
res = []
for nn in (4, 8, 16, 32, 64):
    m = malla_rectangular(1.0, 1.0, nn, nn)
    s = SolverFEM(m, k, h, T_inf)
    Xm, Ym = m.nodos.T
    e = s.resolver(s.carga_nodal(f_q(Xm, Ym))) - f_T(Xm, Ym)
    res.append((1 / nn, s.norma_L2(e), np.abs(e).max()))
    orden = ("-", "-") if len(res) == 1 else (f"{np.log2(res[-2][1] / res[-1][1]):.2f}",
                                              f"{np.log2(res[-2][2] / res[-1][2]):.2f}")
    print(f"  {nn:>3}x{nn:<3} {1 / nn:8.4f} {res[-1][1]:11.3e} {orden[0]:>6} {res[-1][2]:11.3e} {orden[1]:>6}")

#plot: dos figuras a tamaño real de página (ancho de texto ≈ 6.3 in) para que se lean
r = np.array(res)
tri = mtri.Triangulation(X, Y, malla.elems)
salida = Path(__file__).parent / "resultados"
salida.mkdir(exist_ok=True)

fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.2), constrained_layout=True)
for a, campo, titulo in ((ax[0], T_ana, "$T$ exacta"), (ax[1], T_fem, f"$T_h$ ({n}×{n})"),
                         (ax[2], np.abs(error), r"$|T_h - T|$")):
    fig.colorbar(a.tripcolor(tri, campo, shading="gouraud"), ax=a, shrink=0.85)
    a.set(aspect="equal", title=titulo, xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
fig.savefig(salida / "verificacion_campos.pdf")

fig, ax = plt.subplots(figsize=(3.6, 2.8))
ax.loglog(r[:, 0], r[:, 1], "o-", label=r"$\|e\|_{L^2}$")
ax.loglog(r[:, 0], r[:, 2], "s-", label=r"$\|e\|_\infty$")
ax.loglog(r[:, 0], r[0, 1] * (r[:, 0] / r[0, 0]) ** 2, "k--", label="pendiente 2")
ax.set(xlabel="tamaño de elemento $h$", ylabel="error")
ax.set_xticks(r[:, 0], [f"1/{round(1 / v)}" for v in r[:, 0]])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.savefig(salida / "verificacion_convergencia.pdf")

# =============================================================================
#NORTA (sec. 4.5)
# =============================================================================
m = malla_rectangular(10e-3, 10e-3, 50, 50)
pos = np.array([1.0, 4.0, 7.0]) * 1e-3
bloques = [(x0, x0 + 2e-3, y0, y0 + 2e-3) for y0 in pos for x0 in pos]
fuentes = ModeloFuentes(m, bloques, P0=1.0, H=0.5e-3, p=0.3, ell=3e-3)
A = fuentes.muestrear(200_000, np.random.default_rng(1))
C = np.corrcoef(A.T)
print("\nPASO 5. NORTA, p = 0.3, ℓ = 3 mm, 200000 muestras") #lambda si sale en print
print(f"  P(A_i = 1) estimada: {np.round(A.mean(axis=0), 3)}")
print(f"  corr(A_1, A_2) vecinos a 3 mm    = {C[0, 1]:.3f}")
print(f"  corr(A_1, A_9) esquinas opuestas = {C[0, 8]:.3f}")
print(f"\nFiguras guardadas en {salida}")
