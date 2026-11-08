# Resultados del estudio del chip

```
python estudio_chip.py            # plano realista (principal), ≈ 1 min  -> resultados/chip_realista/
python estudio_chip.py rejilla    # caso idealizado de 9 bloques iguales -> resultados/chip_rejilla/
```

Todos los números salen de `salida.txt` y `resumen.json` de cada carpeta. Para cambiar
parámetros, editen `PAR` o `PLANOS` en `estudio_chip.py` y vuelvan a correr el script.

## Motivación (responde la pregunta del profesor: "¿problema real o capricho matemático?")

El disipador se diseña **antes** de saber qué carga va a correr el chip. Qué bloques están
activos (núcleos, GPU, caché...) depende del programa y del usuario, así que el mapa de calor
solo se conoce de forma estadística. Las dos alternativas deterministas fallan:

- **Diseñar para el peor caso (todo encendido)**: el mejor diseño ("según potencia pico") llega
  a 100.5 °C con todo encendido, así que un análisis determinista diría que falla. En realidad
  falla el **4%** del tiempo, frente a un 45% del diseño uniforme. El análisis determinista no
  distingue entre ambos; el probabilístico sí.
- **Diseñar para el promedio** ("según potencia media"): parece razonable, pero falla **cada vez
  que se enciende la GPU** (127 de las 128 configuraciones con GPU activa fallan). Como la GPU se
  usa poco (p = 0.3), el promedio la subestima.

## Plano del chip (10 × 10 mm, 106 W con todo encendido)

| Bloque | Tamaño [mm] | Potencia activa [W] | p de activación |
|---|---|---|---|
| Núcleos 1–4 | 2 × 2 c/u | 12 c/u | 0.5 |
| Caché L3 | 9.2 × 1.6 | 8 | 0.7 |
| GPU | 5.6 × 4.4 | 40 | 0.3 |
| Controlador de memoria | 3.2 × 2.0 | 6 | 0.6 |
| E/S | 3.2 × 2.0 | 4 | 0.4 |

Las potencias y probabilidades son **supuestos representativos**: los núcleos tienen alta
densidad de potencia, la GPU es grande y potente pero se usa poco y la caché casi siempre está
activa. Deben presentarse como tales en el documento.

## Parámetros físicos

| Parámetro | Valor | Justificación / fuente |
|---|---|---|
| Espesor del chip | 0.5 mm | supuesto, valor típico |
| k del silicio | 148 W/m·K | Glassbrenner y Slack (1964), Phys. Rev. 134, A1058 (verificar la cita) |
| h_sup promedio | 2·10⁴ W/m²·K | R ≈ 1/(h_sup·A) = 0.5 K/W de la unión al aire (supuesto) |
| h_borde | 10 W/m²·K | convección natural; los bordes casi no disipan |
| T_amb | 40 °C | aire dentro del equipo (supuesto) |
| T_crit | 100 °C | Intel: el límite de unión suele estar entre 100 y 110 °C |
| ℓ de referencia | 3 mm | igual que en el avance |

## Los cuatro diseños (misma capacidad total ∫ h_sup dA)

| Diseño | Idea de ingeniería |
|---|---|
| Uniforme | disipador estándar |
| Centrado | placa pequeña en el centro (diseño ingenuo) |
| Según potencia media | más enfriamiento donde la potencia esperada P_i·p_i es alta |
| Según potencia pico | más enfriamiento donde la potencia con el bloque activo P_i es alta |

En los dos últimos, h_sup = 0.5·h̄ + c·(densidad de potencia), con c elegido para conservar el total.

## Resultados clave (ℓ = 3 mm)

| Diseño | P_falla exacta | Monte Carlo (N = 2·10⁵) e IC 95% | E[T_máx] | T_máx con todo encendido |
|---|---|---|---|---|
| Uniforme | 0.451 | 0.450 [0.448, 0.453] | 93.0 °C | 111.9 °C |
| Centrado | 0.755 | 0.755 [0.753, 0.756] | 107.8 °C | 131.6 °C |
| Según potencia media | 0.295 | 0.296 [0.294, 0.298] | 86.9 °C | 106.1 °C |
| **Según potencia pico** | **0.042** | 0.042 [0.041, 0.043] | 86.4 °C | 100.5 °C |

## Conclusiones para la discusión

1. **Lo que importa es dónde se pone el disipador, no cuánto disipador hay.** Con la misma
   capacidad total, P_falla va de 0.04 a 0.75 según cómo se reparta.

2. **Diseñar para el promedio es una trampa.** El diseño "según potencia media" tiene casi la
   misma E[T_máx] que el de potencia pico (86.9 frente a 86.4 °C), pero una P_falla 7 veces mayor.
   Mirar solo la temperatura esperada esconde el riesgo. Además, su P_falla es casi
   exactamente p_GPU = 0.3 para todo ℓ, porque falla siempre que la GPU se enciende.

3. **El ranking depende de la carga** (`fig_pfalla_p`). Con carga baja (factor f < 0.6), el diseño
   "según potencia media" es **peor que el uniforme**; con carga alta es mejor. El mejor diseño
   depende del uso esperado del chip.
   Nota: en el barrido las probabilidades se truncan como p_i' = min(f·p_i, 0.99); la caché
   se satura desde f ≈ 1.41 y la memoria desde f = 1.65 (zona sombreada de la figura).

4. **La correlación afecta de forma distinta a cada diseño** (`fig_pfalla_ell`, la figura central):
   - Con el diseño pico, la falla es rara y requiere muchos bloques encendidos a la vez. La
     correlación hace más probable ese caso: P_falla pasa de 0.008 (bloques independientes) a
     0.27 (ℓ = 50 mm), es decir, se multiplica por 30.
   - Con el diseño centrado, la falla es común. La correlación hace más probable que todo esté
     apagado, así que P_falla baja.
   - Con el diseño media no cambia, porque solo depende de la GPU.
   - Consecuencia: **un análisis con bloques independientes subestima mucho el riesgo del mejor
     diseño** si la carga real es correlacionada (por ejemplo, programas que usan todos los
     núcleos a la vez).

5. **El caso idealizado (rejilla de 9 bloques iguales) no distingue entre diseñar para la media
   y para el pico**: dan exactamente lo mismo porque todos los bloques son iguales. Esto
   justifica usar un plano realista. Los resultados del caso rejilla están en
   `resultados/chip_rejilla/` por si quieren mostrarlos como caso base.

## Aspectos numéricos

- **Cálculo exacto:** con 8 bloques binarios hay 2⁸ = 256 configuraciones. Por superposición,
  T_máx de todas se calcula de forma exacta con 9 soluciones FEM. La probabilidad de cada
  configuración es un ortante de la normal multivariada (algoritmo de Genz). Monte Carlo coincide
  con el valor exacto dentro del IC, lo que verifica el muestreador NORTA.
  - Pregunta probable en la sustentación: "¿para qué Monte Carlo si son 256 casos?". Respuesta:
    con un plano más fino (por ejemplo, 64 unidades) habría 2⁶⁴ configuraciones; la enumeración
    es imposible y Monte Carlo cuesta lo mismo.
- **NORTA:** las frecuencias de activación coinciden con las p objetivo. Entre los núcleos 1 y 2,
  la correlación de las Bernoulli es 0.295 (teórica 0.297), menor que la latente (0.449).
- **Números aleatorios comunes:** reducen la varianza de la diferencia entre diseños
  (×1.2–1.3 aquí; ×1.9 en el caso rejilla).
- **Malla:** T_máx cambia menos de 0.02 °C entre 100×100 y 200×200, y P_falla ya no cambia
  desde 50×50. Se usa 100×100.

## Limitaciones

- El modelo es 2D, promediado en el espesor, y h_sup resume en un solo coeficiente todo el
  camino hacia el aire.
- El análisis es estacionario: supone que cada configuración dura más que la constante de tiempo
  térmica del chip.
- Los bloques solo están encendidos o apagados, y las potencias y probabilidades son supuestas.
- La correlación es espacial (exp(−d/ℓ)). En la realidad depende también de la función de cada
  bloque, por ejemplo núcleos que trabajan juntos.
- La cópula gaussiana no tiene dependencia en las colas.

## Figuras (`resultados/chip_realista/`)

| Archivo | Contenido |
|---|---|
| `fig_disenos.pdf` | Plano, mapas de h_sup y temperatura con todo encendido |
| `fig_pfalla_ell.pdf` | **Figura central**: P_falla frente a ℓ (línea exacta, puntos de Monte Carlo con IC) |
| `fig_pfalla_p.pdf` | P_falla frente al factor de carga (cruce entre los diseños media y uniforme) |
| `fig_mapas.pdf` | E[T], desviación estándar y P(T(x) > T_crit) por punto |
| `fig_cdf_tmax.pdf` | Distribución de T_máx |
| `fig_convergencia_mc.pdf` | Monte Carlo frente a N, con bandas de Wilson |
| `fig_convergencia_malla.pdf` | T_máx y P_falla frente al tamaño de elemento |
