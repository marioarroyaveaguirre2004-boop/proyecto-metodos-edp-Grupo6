# Declaración sobre el uso de inteligencia artificial

Durante este trabajo usamos un asistente de inteligencia artificial generativa (Claude, de
Anthropic) como apoyo en el desarrollo y la depuración de partes del código, y en la
revisión de la redacción del texto. Revisamos, ejecutamos y verificamos todo el código y
los resultados, y editamos el contenido. Asumimos total responsabilidad por el contenido
del proyecto.

Es la misma declaración que aparece al final del informe (`informe/informe.pdf`).

## Qué hicimos nosotros

- Definimos el problema, el modelo físico y matemático, los casos de estudio y los
  criterios para comparar los diseños de disipador.
- Verificamos el solver con soluciones manufacturadas (`verificacion.py`,
  `verificacion_mixta.py`) y con las pruebas automáticas de `tests/`.
- Revisamos y editamos el informe final.

## Cómo se ve en el repositorio

Los commits hechos con ayuda del asistente incluyen la línea `Co-Authored-By: Claude`, que
GitHub muestra como coautoría del commit.
