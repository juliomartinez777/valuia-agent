# Plan de pruebas de VALUIA-AGENT

Este plan se fija ANTES de ejecutar las pruebas. Los umbrales no se modifican después de ver los resultados.

## Medidas de desempeño

| ID | Medida | Fórmula | Umbral |
|----|--------|---------|--------|
| M1 | Error porcentual absoluto medio (MAPE) | promedio de |precio_final - precio_referencia| / precio_final x 100 | <= 10 % en escenarios normales (E1-E4); <= 15 % en escenarios límite (E5-E8) |
| M2 | Cobertura del rango publicado | ventas cuyo precio final cae dentro de [rango_inferior, rango_superior] / ventas valoradas x 100 | >= 75 % en escenarios normales (E1-E4) |
| M3 | Manejo de entradas inválidas | entradas inválidas que generan alerta de revisión humana (sin publicar precio) / entradas inválidas inyectadas x 100 | 100 % (E9-E11). En E12 (barrio desconocido) se espera valoración con confianza baja: 100 % |
| M4 | Adaptación a cambios del mercado | MAPE sin aprendizaje - MAPE con aprendizaje | >= 1,0 punto porcentual en E6 |

## Escenarios (cada uno con 200 propiedades del conjunto de prueba, 5 repeticiones con semillas 1 a 5)

| ID | Tipo | Ruido de negociación | Deriva mensual del mercado | Falla inyectada |
|----|------|----------------------|----------------------------|-----------------|
| E1 | Normal | 0 % | 0 % | ninguna |
| E2 | Normal | 3 % | 0 % | ninguna |
| E3 | Normal | 3 % | 1 % | ninguna |
| E4 | Normal | 5 % | 0 % | ninguna |
| E5 | Límite | 8 % | 0 % | ninguna |
| E6 | Límite | 3 % | 3 % | ninguna |
| E7 | Límite | 8 % | 3 % | ninguna |
| E8 | Límite | 10 % | 1 % | ninguna |
| E9 | Falla esperada | 3 % | 0 % | dato faltante (sin_dato) |
| E10 | Falla esperada | 3 % | 0 % | valor imposible (invalido) |
| E11 | Falla esperada | 3 % | 0 % | tipo no numérico |
| E12 | Falla esperada | 3 % | 0 % | barrio desconocido |

## Reglas de reporte

- Para M1, M2 y M4 se reporta la media y la desviación estándar de las 5 repeticiones.
- Una medida se declara cumplida solo si TODOS los escenarios que le corresponden pasan su umbral.
- Si una medida no se cumple, se reporta tal cual, con su análisis de causa. No se cambia el umbral.
- Mejoras al agente después de ver resultados se documentan como cambios, y las pruebas se repiten completas.