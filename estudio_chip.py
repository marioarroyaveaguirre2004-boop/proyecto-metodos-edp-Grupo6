"""
Estudio principal: probabilidad de falla térmica de un chip para tres estrategias de
disipación, con fuentes de calor aleatorias correlacionadas (NORTA).

Modelo (2D, promediado en el espesor H del chip):
    -div(k ∇T) + (h_sup(x,y)/H) (T - T_amb) = q(x,y; A)     en el chip
    -k ∂T/∂n = h_borde (T - T_amb)                           en los bordes
h_sup(x,y) representa todo el camino térmico desde la cara superior hasta el aire
(pasta térmica + disipador). La ESTRATEGIA es el mapa h_sup(x,y); las tres tienen la
misma capacidad total ∫ h_sup dA, así que la comparación es justa.

Cálculo:
  * Superposición: T(A) = T0 + Σ A_i U_i   (N_b + 1 = 10 soluciones por diseño).
  * Con Nb bloques binarios solo hay 2^Nb configuraciones (256 o 512), así que T_max se calcula
    EXACTAMENTE para cada una. La única incertidumbre es la probabilidad de cada
    configuración:
      - "exacto": probabilidad de cada configuración = probabilidad de un ortante de la
        normal latente (algoritmo de Genz, scipy).
      - Monte Carlo: muestreo NORTA; se usa como estimador independiente y para
        mostrar la convergencia O(1/√N) con intervalos de Wilson.
  * Números aleatorios comunes: las mismas muestras se usan para los tres diseños y
    para todos los valores de ℓ.

Uso:  python estudio_chip.py              plano realista (núcleos, caché, GPU, ...)
      python estudio_chip.py rejilla      caso idealizado: 9 bloques iguales en rejilla
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm, multivariate_normal
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.ticker

from fem_solver import malla_rectangular, indicadora, SolverFEM, ModeloFuentes

sys.stdout.reconfigure(encoding="utf-8")
AQUI = Path(__file__).parent
plt.style.use(AQUI / "estilo.mplstyle")

# =============================================================================
# Parámetros (todos en SI). Cambiar aquí y volver a correr.
# =============================================================================
PAR = dict(
    L=10e-3,          # lado del chip [m]
    H=0.5e-3,         # espesor del chip [m]
    k=148.0,          # conductividad del silicio a 300 K [W/m K]
    ell=3e-3,         # longitud de correlación de referencia [m]
    h_sup=2.0e4,      # coeficiente promedio por la cara superior [W/m² K] (R ≈ 0.5 K/W)
    h_borde=10.0,     # convección en los bordes laterales [W/m² K]
    T_amb=40.0,       # temperatura del aire dentro del equipo [°C]
    T_crit=100.0,     # temperatura máxima de unión [°C]
    f_bajo=0.5,       # fracción mínima de h_sup promedio en los diseños no uniformes
    n_malla=100,      # elementos por lado (malla de referencia; ver convergencia en malla)
    N_mc=200_000,     # realizaciones de Monte Carlo
    semilla=2026,
)
ELLS = np.array([0.0, 0.5, 1, 2, 3, 5, 8, 12, 20, 50]) * 1e-3
FACTORES = np.linspace(0.2, 1.8, 17)   # factor de carga: p_i -> min(factor · p_i, 0.99)
MALLAS = [25, 50, 100, 200]
DISENOS = {"uniforme": "Uniforme", "centro": "Centrado",
           "media": "Según potencia media", "pico": "Según potencia pico"}
COLORES = {"uniforme": "#4C72B0", "centro": "#55A868", "media": "#8172B3", "pico": "#DD8452"}

# -----------------------------------------------------------------------------
# Plano de bloques: (nombre, x0, x1, y0, y1 [mm], potencia activa [W], p de activación)
# Coordenadas múltiplos de 0.4 mm para que los bordes coincidan con todas las mallas.
# -----------------------------------------------------------------------------
PLANOS = {
    "realista": [
        ("Núcleo 1", 0.4, 2.4, 7.2, 9.2, 12.0, 0.5),
        ("Núcleo 2", 2.8, 4.8, 7.2, 9.2, 12.0, 0.5),
        ("Núcleo 3", 5.2, 7.2, 7.2, 9.2, 12.0, 0.5),
        ("Núcleo 4", 7.6, 9.6, 7.2, 9.2, 12.0, 0.5),
        ("Caché L3", 0.4, 9.6, 5.2, 6.8, 8.0, 0.7),
        ("GPU", 0.4, 6.0, 0.4, 4.8, 40.0, 0.3),
        ("Memoria", 6.4, 9.6, 2.8, 4.8, 6.0, 0.6),
        ("E/S", 6.4, 9.6, 0.4, 2.4, 4.0, 0.4),
    ],
    "rejilla": [(f"B{3 * j + i + 1}", x0, x0 + 2, y0, y0 + 2, 13.0, 0.5)
                for j, y0 in enumerate((1.0, 4.0, 7.0)) for i, x0 in enumerate((1.0, 4.0, 7.0))],
}
PLANO = sys.argv[1] if len(sys.argv) > 1 else "realista"
SALIDA = AQUI / "resultados" / f"chip_{PLANO}"
SALIDA.mkdir(parents=True, exist_ok=True)
NOMBRES = [b[0] for b in PLANOS[PLANO]]
BLOQUES = [tuple(np.array(b[1:5]) * 1e-3) for b in PLANOS[PLANO]]
POT = np.array([b[5] for b in PLANOS[PLANO]])
PBASE = np.array([b[6] for b in PLANOS[PLANO]])
NB = len(BLOQUES)
CONFIGS = np.array(list(itertools.product([0, 1], repeat=NB)), dtype=float)   # (2^Nb, Nb)
PESOS_BITS = 2 ** np.arange(NB)[::-1]          # para pasar una fila A a su índice de configuración


# =============================================================================
# Diseños de disipación: mapas h_sup(x,y) con la misma integral
# =============================================================================
def mapa_hsup(malla, diseno):
    hm, fb = PAR["h_sup"], PAR["f_bajo"]
    if diseno == "uniforme":
        return np.full(len(malla.elems), hm)
    P = malla.nodos[malla.elems]
    a = 0.5 * np.abs((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
                     - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1]))
    if diseno in ("media", "pico"):
        # mínimo f_bajo·h + parte proporcional a la densidad de potencia de cada bloque:
        # "media" usa la potencia esperada P_i p_i, "pico" la potencia con el bloque activo P_i
        peso = POT * PBASE if diseno == "media" else POT
        qd = sum(indicadora(malla, r) * w_ / ((r[1] - r[0]) * (r[3] - r[2])) for r, w_ in zip(BLOQUES, peso))
        c = (1 - fb) * hm * a.sum() / (a @ qd)
        return fb * hm + c * qd
    if diseno == "centro":       # placa de 6x6 mm centrada (p. ej. base de cobre pequeña)
        c = malla.nodos[malla.elems].mean(axis=1)
        zona = (np.abs(c - PAR["L"] / 2) < 3e-3).all(axis=1)
    f = a[zona].sum() / a.sum()
    alto = (hm - (1 - f) * fb * hm) / f          # ∫ h_sup dA = h_sup_prom · área
    return np.where(zona, alto, fb * hm)


def construir(n_malla, diseno):
    """Devuelve malla, T0 (n,), U (n, Nb) y las temperaturas máximas de las 2^Nb configuraciones."""
    m = malla_rectangular(PAR["L"], PAR["L"], n_malla, n_malla)
    hs = mapa_hsup(m, diseno)
    s = SolverFEM(m, PAR["k"], h=PAR["h_borde"], T_inf=PAR["T_amb"],
                  beta=hs / PAR["H"], T_ref=PAR["T_amb"])
    fu = ModeloFuentes(m, BLOQUES, P0=POT, H=PAR["H"], p=PBASE, ell=PAR["ell"])
    T0 = s.resolver(np.zeros(s.n))
    U = s.resolver_lineal(s.carga(fu.chi * fu.qi))
    T_cfg = T0[:, None] + U @ CONFIGS.T                  # (n, 2^Nb) todas las configuraciones
    return SimpleNS(malla=m, hs=hs, solver=s, fuentes=fu, T0=T0, U=U, T_cfg=T_cfg,
                    Tmax=T_cfg.max(axis=0))


class SimpleNS(dict):
    __getattr__ = dict.__getitem__


# =============================================================================
# Probabilidades de las configuraciones
# =============================================================================
def sigma(ell, centros):
    d = np.linalg.norm(centros[:, None] - centros[None], axis=2)
    return np.exp(-d / ell) if ell > 0 else np.eye(len(centros))


def prob_configs(p, ell, centros):
    """P(A = a) para las 2^Nb configuraciones: ortantes de la normal latente (Genz)."""
    c = norm.ppf(1 - p)
    S = sigma(ell, centros)
    if ell == 0:   # independientes: producto exacto
        return np.prod(np.where(CONFIGS == 1, p, 1 - p), axis=1)
    pr = np.empty(len(CONFIGS))
    for j, a in enumerate(CONFIGS):
        s = np.where(a == 1, -1.0, 1.0)          # A_i=1 ⇔ Z_i > c ⇔ -Z_i < -c
        pr[j] = multivariate_normal(mean=np.zeros(NB), cov=S * np.outer(s, s), allow_singular=True,
                                    maxpts=50_000 * NB, abseps=1e-6, releps=1e-5, seed=0
                                    ).cdf(s * c)
    return pr / pr.sum()


def wilson(k, n, z=1.96):
    ph = k / n
    den = 1 + z**2 / n
    c = (ph + z**2 / (2 * n)) / den
    w = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / den
    return c - w, c + w


def muestrear_A(eps, p, ell, centros):
    """NORTA con ruido base eps (N, Nb) fijo: números aleatorios comunes."""
    Lc = np.linalg.cholesky(sigma(ell, centros) + 1e-12 * np.eye(NB))
    return (eps @ Lc.T > norm.ppf(1 - p)).astype(float)


# =============================================================================
# Cálculos
# =============================================================================
t0 = time.time()
print("Parámetros:", json.dumps({k: v for k, v in PAR.items()}, default=float))
print(f"Plano '{PLANO}': " + "; ".join(f"{n} {P_:g} W, p={p_:g}" for n, P_, p_ in zip(NOMBRES, POT, PBASE)))
mod = {d: construir(PAR["n_malla"], d) for d in DISENOS}
centros = mod["uniforme"].fuentes.centros
falla = {d: mod[d].Tmax > PAR["T_crit"] for d in DISENOS}
print(f"\nConfiguraciones que fallan (de {len(CONFIGS)}): " + ", ".join(f"{d} {falla[d].sum()}" for d in DISENOS))
print("T_max con todos los bloques encendidos: "
      + ", ".join(f"{d} {mod[d].Tmax[-1]:.1f} °C" for d in DISENOS))

# --- referencia exacta en (p, ℓ) base ---
w_ref = prob_configs(PBASE, PAR["ell"], centros)
Pex = {d: w_ref @ falla[d] for d in DISENOS}

# --- Monte Carlo con números aleatorios comunes ---
rng = np.random.default_rng(PAR["semilla"])
eps = rng.standard_normal((PAR["N_mc"], NB))
A = muestrear_A(eps, PBASE, PAR["ell"], centros)
idx = (A @ PESOS_BITS).astype(int)
F_mc = {d: falla[d][idx] for d in DISENOS}

# chequeo de NORTA frente al modelo exacto
C = np.corrcoef(A.T)
d01 = np.linalg.norm(centros[0] - centros[1])
rho_lat = np.exp(-d01 / PAR["ell"])
c_ = norm.ppf(1 - PBASE[0])
p11 = multivariate_normal(cov=[[1, rho_lat], [rho_lat, 1]]).cdf([-c_, -c_])   # P(Z1>c, Z2>c)
rho_bin = (p11 - PBASE[0] ** 2) / (PBASE[0] * (1 - PBASE[0]))

resumen = {"parametros": PAR, "plano": PLANOS[PLANO], "disenos": {}}
print(f"\nReferencia (p base, ℓ = {PAR['ell']*1e3:.0f} mm), N = {PAR['N_mc']}:")
print(f"  {'diseño':>10} {'P exacta':>9} {'P Monte Carlo':>14} {'IC 95% (Wilson)':>20} {'E[T_max]':>9}")
for d in DISENOS:
    kf = F_mc[d].sum()
    lo, hi = wilson(kf, PAR["N_mc"])
    ET = w_ref @ mod[d].Tmax
    print(f"  {d:>10} {Pex[d]:9.4f} {kf / PAR['N_mc']:14.4f}   [{lo:.4f}, {hi:.4f}] {ET:8.1f}")
    resumen["disenos"][d] = dict(P_exacta=float(Pex[d]), P_mc=float(kf / PAR["N_mc"]),
                                 IC95=[float(lo), float(hi)], E_Tmax=float(ET),
                                 Tmax_todos=float(mod[d].Tmax[-1]),
                                 configs_falla=int(falla[d].sum()))

print("\nDiferencias pareadas (números aleatorios comunes) vs independientes:")
for a_, b_ in (("uniforme", "pico"), ("media", "pico")):
    dlt = F_mc[a_].astype(float) - F_mc[b_]
    se_par = dlt.std(ddof=1) / np.sqrt(len(dlt))
    pa, pb = F_mc[a_].mean(), F_mc[b_].mean()
    se_ind = np.sqrt(pa * (1 - pa) / len(dlt) + pb * (1 - pb) / len(dlt))
    print(f"  P({a_}) - P({b_}) = {dlt.mean():+.4f} ± {1.96 * se_par:.4f} "
          f"(con muestras independientes sería ± {1.96 * se_ind:.4f}; "
          f"reducción de varianza ×{(se_ind / se_par) ** 2 if se_par > 0 else float('nan'):.1f})")
    resumen[f"dif_{a_}_{b_}"] = dict(media=float(dlt.mean()), IC95=float(1.96 * se_par),
                                     IC95_indep=float(1.96 * se_ind))

print(f"\nNORTA: P(A_i=1) = {np.round(A.mean(axis=0), 3)} (objetivo {PBASE});  corr(A1,A2) Monte Carlo = {C[0, 1]:.3f}, "
      f"teórica = {rho_bin:.3f}  (latente ρ = {rho_lat:.3f})")
resumen["norta"] = dict(corr_mc=float(C[0, 1]), corr_bin_teorica=float(rho_bin), rho_latente=float(rho_lat))

# --- barrido en ℓ (exacto + MC con las mismas eps) ---
print("\nBarrido en ℓ (p base):")
barr_ell = {d: [] for d in DISENOS}
for ell in ELLS:
    w = prob_configs(PBASE, ell, centros)
    Ae = muestrear_A(eps, PBASE, ell, centros)
    ie = (Ae @ PESOS_BITS).astype(int)
    fila = []
    for d in DISENOS:
        kf = falla[d][ie].sum()
        lo, hi = wilson(kf, len(ie))
        barr_ell[d].append((ell, w @ falla[d], kf / len(ie), lo, hi))
        fila.append(f"{d} {w @ falla[d]:.4f}")
    print(f"  ℓ = {ell*1e3:5.1f} mm: " + ", ".join(fila))
barr_ell = {d: np.array(v) for d, v in barr_ell.items()}

# --- barrido en el factor de carga (exacto): p_i -> min(f · p_i, 0.99) ---
print("\nBarrido en el factor de carga f (ℓ de referencia), exacto:")
barr_p = {d: [] for d in DISENOS}
for fc in FACTORES:
    w = prob_configs(np.minimum(fc * PBASE, 0.99), PAR["ell"], centros)
    for d in DISENOS:
        barr_p[d].append((fc, w @ falla[d]))
barr_p = {d: np.array(v) for d, v in barr_p.items()}
for d in DISENOS:
    print(f"  {d:>10}: " + " ".join(f"{v:.3f}" for v in barr_p[d][::4, 1]) + f"   (f = {FACTORES[::4]})")

# --- convergencia en malla ---
print("\nConvergencia en malla (p, ℓ de referencia):")
print(f"  {'malla':>8} " + " ".join(f"{d + ' Tmax*':>16} {'P':>7}" for d in DISENOS))
conv = {d: [] for d in DISENOS}
for nm in MALLAS:
    fila = []
    for d in DISENOS:
        md = construir(nm, d)
        P = w_ref @ (md.Tmax > PAR["T_crit"])
        conv[d].append((PAR["L"] / nm, md.Tmax[-1], P, w_ref @ md.Tmax))
        fila.append(f"{md.Tmax[-1]:16.3f} {P:7.4f}")
    print(f"  {nm:>4}x{nm:<3} " + " ".join(fila))
conv = {d: np.array(v) for d, v in conv.items()}
resumen["convergencia_malla"] = {d: conv[d].tolist() for d in DISENOS}
resumen["barrido_ell"] = {d: barr_ell[d].tolist() for d in DISENOS}
resumen["barrido_p"] = {d: barr_p[d].tolist() for d in DISENOS}
(SALIDA / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False, default=float))


# =============================================================================
# Figuras
# =============================================================================
m = mod["uniforme"].malla
tri = mtri.Triangulation(*m.nodos.T, m.elems)
mm = lambda a: (a.set(aspect="equal", xticks=[0, 5, 10], yticks=[0, 5, 10]), None)
tri_mm = mtri.Triangulation(*(m.nodos.T * 1e3), m.elems)


def contornos_bloques(a, color="w"):
    for x0, x1, y0, y1 in BLOQUES:
        a.add_patch(plt.Rectangle((x0 * 1e3, y0 * 1e3), (x1 - x0) * 1e3, (y1 - y0) * 1e3,
                                  fill=False, ec=color, lw=0.6, ls="--"))


# Fig 1: los tres diseños (h_sup) y T con todos los bloques encendidos
ND = len(DISENOS)
fig, ax = plt.subplots(2, ND, figsize=(6.5, 3.6), constrained_layout=True)
vmax_h = max(mod[d].hs.max() for d in DISENOS) / 1e3
Tall = [mod[d].T_cfg[:, -1] for d in DISENOS]
vT = (min(t.min() for t in Tall), max(t.max() for t in Tall))
for j, d in enumerate(DISENOS):
    im0 = ax[0, j].tripcolor(tri_mm, facecolors=mod[d].hs / 1e3, cmap="viridis", vmin=0, vmax=vmax_h,
                               edgecolors="none", antialiased=False, rasterized=True)
    im1 = ax[1, j].tripcolor(tri_mm, mod[d].T_cfg[:, -1], shading="gouraud", vmin=vT[0], vmax=vT[1],
                                 rasterized=True)
    ax[1, j].tricontour(tri_mm, mod[d].T_cfg[:, -1], levels=[PAR["T_crit"]], colors="c", linewidths=1)
    ax[0, j].set_title(DISENOS[d], fontsize=8)
    for a in ax[:, j]:
        mm(a); contornos_bloques(a)
for nom, (x0, x1, y0, y1) in zip(NOMBRES, BLOQUES):
    corto = {"Caché L3": "L3", "Memoria": "Mem"}.get(nom, nom.replace("Núcleo ", "N"))
    ax[0, 0].text((x0 + x1) / 2e-3, (y0 + y1) / 2e-3, corto, ha="center", va="center", fontsize=6, color="w")
fig.colorbar(im0, ax=ax[0, :], label=r"$h_{sup}$ [kW/m²K]", shrink=0.9)
fig.colorbar(im1, ax=ax[1, :], label="T [°C], todos encendidos", shrink=0.9)
ax[1, 0].set(xlabel="x [mm]", ylabel="y [mm]")
fig.savefig(SALIDA / "fig_disenos.pdf")

# Fig 2: P_fail vs ℓ (figura central)
fig, ax = plt.subplots(figsize=(4.2, 3.0))
for d in DISENOS:
    b = barr_ell[d]
    x = np.where(b[:, 0] == 0, 0.25e-3, b[:, 0]) * 1e3
    ax.plot(x, b[:, 1], "-", color=COLORES[d], label=DISENOS[d])
    ax.errorbar(x, b[:, 2], yerr=[b[:, 2] - b[:, 3], b[:, 4] - b[:, 2]], fmt="o", ms=3,
                color=COLORES[d], capsize=2, lw=0.8)
ax.set_xscale("log")
ax.set_xticks([0.25, 1, 3, 10, 50], ["indep.", "1", "3", "10", "50"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set(xlabel=r"longitud de correlación $\ell$ [mm]", ylabel=r"$P_{falla}$")
ax.grid(True, which="major", alpha=0.3)
ax.set_ylim(0, 1)
ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.3),
          title_fontsize=8, frameon=False)
fig.savefig(SALIDA / "fig_pfalla_ell.pdf"); fig.savefig(SALIDA / "fig_pfalla_ell.png", dpi=200)

# Fig 3: P_fail vs p
fig, ax = plt.subplots(figsize=(4.2, 3.0))
for d in DISENOS:
    ax.plot(barr_p[d][:, 0], barr_p[d][:, 1], "o-", ms=3, color=COLORES[d], label=DISENOS[d])
ax.set(xlabel=r"factor de carga $f$  ($p_i \to \min(f\,p_i,\ 0.99)$)", ylabel=r"$P_{falla}$")
f_sat = (0.99 / PBASE).min()                      # primer bloque que se satura
if f_sat < FACTORES.max():
    ax.axvspan(f_sat, FACTORES.max() + 0.05, color="0.85", alpha=0.5, lw=0)
    ax.text(f_sat + 0.02, 0.02, "con\ntruncamiento", fontsize=7, color="0.35", va="bottom")
ax.grid(True, alpha=0.3); ax.legend(title=rf"$\ell$ = {PAR['ell']*1e3:.0f} mm", fontsize=8)
fig.savefig(SALIDA / "fig_pfalla_p.pdf")

# Fig 4: convergencia de Monte Carlo
fig, ax = plt.subplots(figsize=(4.2, 3.0))
n = np.unique(np.logspace(2, np.log10(PAR["N_mc"]), 60).astype(int))
for d in DISENOS:
    cs = np.cumsum(F_mc[d])[n - 1]
    lo, hi = wilson(cs, n)
    ax.fill_between(n, lo, hi, color=COLORES[d], alpha=0.25, lw=0)
    ax.plot(n, cs / n, color=COLORES[d], lw=1, label=DISENOS[d])
    ax.axhline(Pex[d], color=COLORES[d], ls="--", lw=0.8)
ax.set_xscale("log")
ax.set(xlabel="número de realizaciones $N$", ylabel=r"$P_{falla}$ estimada")
ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
fig.savefig(SALIDA / "fig_convergencia_mc.pdf")

# Fig 5: mapas por punto: P(T(x) > T_crit), media y desviación estándar (exactas)
EA = w_ref @ CONFIGS
CovA = (CONFIGS * w_ref[:, None]).T @ CONFIGS - np.outer(EA, EA)
campos = {}
for d in DISENOS:
    md = mod[d]
    campos[d] = (md.T0 + md.U @ EA,
                 np.sqrt(np.einsum("ij,jk,ik->i", md.U, CovA, md.U)),
                 (md.T_cfg > PAR["T_crit"]) @ w_ref)
lim = [(min(c[i].min() for c in campos.values()), max(c[i].max() for c in campos.values())) for i in range(3)]
fig, ax = plt.subplots(3, ND, figsize=(6.5, 5.0), constrained_layout=True)
cmaps = ("inferno", "magma", "Reds")
etiquetas = ("E[T] [°C]", "desv. est. de T [°C]", r"$P(T(\mathbf{x}) > T_{crit})$")
for j, d in enumerate(DISENOS):
    for i in range(3):
        im = ax[i, j].tripcolor(tri_mm, campos[d][i], shading="gouraud", cmap=cmaps[i],
                                vmin=0 if i == 2 else lim[i][0], vmax=lim[i][1], rasterized=True)
        mm(ax[i, j]); contornos_bloques(ax[i, j], "0.5" if i == 2 else "w")
        if j == ND - 1:
            fig.colorbar(im, ax=ax[i, :], label=etiquetas[i], shrink=0.9)
    ax[0, j].set_title(DISENOS[d], fontsize=8)
ax[2, 0].set(xlabel="x [mm]", ylabel="y [mm]")
fig.savefig(SALIDA / "fig_mapas.pdf"); fig.savefig(SALIDA / "fig_mapas.png", dpi=200)

# Fig 6: distribución de T_max
fig, ax = plt.subplots(figsize=(4.2, 3.0))
for d in DISENOS:
    o = np.argsort(mod[d].Tmax)
    ax.step(mod[d].Tmax[o], np.cumsum(w_ref[o]), where="post", color=COLORES[d], label=DISENOS[d])
ax.axvline(PAR["T_crit"], color="k", ls="--", lw=0.8)
ax.text(PAR["T_crit"], 0.03, r" $T_{crit}$", fontsize=8)
ax.set(xlabel=r"$T_{máx}$ [°C]", ylabel=r"$P(T_{máx} \leq t)$")
ax.set_xlim(left=PAR["T_amb"] + 30)
ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
fig.savefig(SALIDA / "fig_cdf_tmax.pdf")

# Fig 7: convergencia en malla
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.6), constrained_layout=True)
for d in DISENOS:
    c = conv[d]
    ax[0].semilogx(c[:, 0] * 1e3, c[:, 1], "o-", color=COLORES[d], label=DISENOS[d])
    ax[1].semilogx(c[:, 0] * 1e3, c[:, 2], "o-", color=COLORES[d])
ax[0].set(xlabel="tamaño de elemento [mm]", ylabel=r"$T_{máx}$, todos encendidos [°C]")
ax[1].set(xlabel="tamaño de elemento [mm]", ylabel=r"$P_{falla}$")
for a in ax:
    a.set_xticks([0.4, 0.2, 0.1, 0.05], ["0.4", "0.2", "0.1", "0.05"])
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.invert_xaxis(); a.grid(True, alpha=0.3)
ax[0].legend(fontsize=8)
fig.savefig(SALIDA / "fig_convergencia_malla.pdf")

print(f"\nListo en {time.time() - t0:.0f} s. Figuras y resumen.json en {SALIDA}")
