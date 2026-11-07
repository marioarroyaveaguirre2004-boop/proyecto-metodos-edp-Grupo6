"""
Resuelve cualquier problema descrito en un archivo JSON (ver fem_solver.cargar_problema)
y guarda un mapa de temperatura.

Uso:  python resolver.py ejemplos/soporte.json
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from fem_solver import cargar_problema

sys.stdout.reconfigure(encoding="utf-8")
plt.style.use(Path(__file__).parent / "estilo.mplstyle")

ruta = Path(sys.argv[1])
malla, solver, F = cargar_problema(ruta)
T = solver.resolver(F)

print(f"{ruta.name}: {len(malla.nodos)} nodos, {len(malla.elems)} triángulos")
print(f"  T mín = {T.min():.2f}   T máx = {T.max():.2f}   en {malla.nodos[T.argmax()]}")
for nombre, tag in malla.nombres.items():
    if (malla.region == tag).any():
        nod = np.unique(malla.elems[malla.region == tag])
        print(f"  {nombre:>10}: T media nodal = {T[nod].mean():.2f}")

tri = mtri.Triangulation(*malla.nodos.T, malla.elems)
fig, ax = plt.subplots(figsize=(4.0, 3.4))
fig.colorbar(ax.tripcolor(tri, T, shading="gouraud"), ax=ax, label="T")
ax.tricontour(tri, T, levels=12, colors="w", linewidths=0.4)
# interfaces entre materiales: aristas compartidas por triángulos de regiones distintas
ar = np.sort(malla.elems[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
reg = np.repeat(malla.region, 3)
_, inv = np.unique(ar, axis=0, return_inverse=True)
inv = inv.ravel()
for i in np.flatnonzero(np.bincount(inv) == 2):
    r = reg[inv == i]
    if r[0] != r[1]:
        ax.plot(*malla.nodos[ar[inv == i][0]].T, color="c", lw=1)
ax.set(aspect="equal", xlabel="x", ylabel="y")
salida = ruta.with_suffix(".pdf")
fig.savefig(salida)
fig.savefig(ruta.with_suffix(".png"), dpi=200)
print(f"  figura: {salida}")
