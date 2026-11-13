# Riesgo de sobrecalentamiento de un chip con carga incierta

Proyecto final de **Métodos numéricos para EDP** (Universidad EAFIT, 2026-2).
Autores: Nancy Burgos, Carlos Arroyave, David Restrepo.

¿Cuál es la probabilidad de que un chip supere su temperatura máxima cuando no se sabe qué
bloques (núcleos, caché, GPU…) van a estar encendidos? Se compara cómo distribuir el
disipador con la misma capacidad total de enfriamiento.

- **Informe:** [`informe/informe.pdf`](informe/informe.pdf) (fuente LaTeX en `informe/`)
- **Resultados e interpretación:** [`RESULTADOS.md`](RESULTADOS.md)
- **Uso de inteligencia artificial:** [`USO_DE_IA.md`](USO_DE_IA.md)

Solver de elementos finitos P1, escrito desde cero, para

```
-div(k ∇T) + β (T - T_ref) = q        en Ω = ∪ Ω_m   (k, β, T_ref, q por material)
                         T = T_D      en Γ_D         Dirichlet
                  k ∂T/∂n  = g        en Γ_N         Neumann (g = calor que entra)
                 -k ∂T/∂n  = h(T-T∞)  en Γ_R         Robin
```

En el chip, `β = h_sup / H` representa el calor que sale por la cara superior hacia el
disipador (espesor H) y `T_ref` es la temperatura del disipador.

Gmsh solo se usa para generar las mallas (mallador externo). Lectura de la malla,
ensamblaje, condiciones de frontera, solución y Monte Carlo son código propio.

## Instalación

```
pip install -r requirements.txt
```

Las mallas ya vienen generadas en `mallas/` y `ejemplos/`; Gmsh solo hace falta para
regenerarlas.

## Uso

| Comando | Qué hace |
|---|---|
| `python mallas/generar_mallas.py` | Genera las mallas `.msh` (verificación y ejemplo) |
| `python resolver.py ejemplos/soporte.json` | Resuelve cualquier problema descrito en un JSON |
| `python estudio_chip.py` | Estudio principal: plano realista (núcleos, caché, GPU), 4 diseños de disipador, P_falla exacta y Monte Carlo, barridos en ℓ, carga y malla (ver RESULTADOS.md) |
| `python estudio_chip.py rejilla` | Mismo estudio con el caso idealizado de 9 bloques iguales |
| `python verificacion.py` | Caso 1: rectángulo, Robin, malla estructurada |
| `python verificacion_mixta.py` | Caso 2: malla no estructurada, 2 materiales, agujero, Dirichlet + Neumann + Robin + reacción |
| `python -m pytest -q` | Pruebas (solución lineal exacta, superposición, convergencia, JSON) |

## Archivo de entrada

```json
{
  "malla": "soporte.msh",
  "materiales": {"aluminio": {"k": 205}, "cobre": {"k": 400, "q": 0, "beta": 0, "T_ref": 25}},
  "fronteras": {
    "base":  {"tipo": "dirichlet", "T": 90},
    "canal": {"tipo": "robin", "h": 2000, "T_inf": 20},
    "tope":  {"tipo": "neumann", "flujo": 5000}
  }
}
```

- Los nombres son los grupos físicos de Gmsh: dimensión 2 para materiales y dimensión 1
  para fronteras. La malla se guarda en formato MSH 2.2 ASCII.
- `q`, `T`, `T_inf` y `flujo` pueden ser texto con una expresión en `x, y`, por ejemplo `"25 + 3*x"`.
- Las fronteras que no aparecen en el JSON quedan aisladas.

## Resultados de verificación

Caso 2 (mallas no estructuradas, `resultados/verificacion_mixta.txt`):

| nodos | h | ‖e‖_L2 | orden | ‖e‖_∞ | orden |
|---|---|---|---|---|---|
| 150 | 0.1200 | 2.77e-02 | – | 9.90e-02 | – |
| 511 | 0.0634 | 8.30e-03 | 1.89 | 3.63e-02 | 1.57 |
| 1870 | 0.0344 | 2.11e-03 | 2.25 | 9.20e-03 | 2.25 |
| 7149 | 0.0167 | 5.24e-04 | 1.91 | 2.13e-03 | 2.02 |
| 28019 | 0.0085 | 1.33e-04 | 2.04 | 5.25e-04 | 2.09 |

## Estructura

```
fem_solver.py            solver, lector de Gmsh, cargar_problema (JSON), NORTA, Monte Carlo
resolver.py              resolver un JSON y graficar
verificacion.py          caso de verificación 1
verificacion_mixta.py    caso de verificación 2
estudio_chip.py          estudio principal del chip
RESULTADOS.md            resultados, parámetros e interpretación
estilo.mplstyle          estilo común de figuras (tamaño real de página)
mallas/                  generador de mallas de Gmsh
ejemplos/                problemas de ejemplo (JSON + malla)
tests/                   pruebas con pytest
resultados/              figuras y tablas generadas
informe/                 artículo en LaTeX (informe.tex, referencias.bib, figuras/) y PDF
```

Para compilar el informe: `cd informe && latexmk -pdf informe.tex`, o subir la carpeta a
Overleaf.
