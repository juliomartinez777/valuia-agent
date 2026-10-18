# VALUIA-AGENT

Agente inteligente que estima el precio de venta de una propiedad residencial a partir de sus características, entrega un precio de referencia con un rango razonable y aprende del precio final de cada venta.

Proyecto del curso **Inteligencia Artificial 2026-II**, Universidad Tecnológica de Bolívar.
Autores: Julio Martinez y Jairo Mendoza.

## Qué hace el agente

- **Tipo de agente:** basado en modelo, con elementos de agente de aprendizaje (Russell y Norvig).
- **Entorno:** simulador de mercado inmobiliario construido con el dataset público Ames Housing. Es parcialmente observable, estocástico (ruido de negociación), dinámico (deriva de precios por mes) y de agente único.
- **Ciclo del agente** (`src/agente.py`):
  1. `perceive()`: recibe una propiedad y valida cada dato (faltante, no numérico, fuera de rango).
  2. `decide()`: estima el precio con un modelo Gradient Boosting, lo ajusta con las ventas recientes de la misma zona y calcula un rango razonable.
  3. `act()`: publica el precio y el rango en un sistema de listados simulado, o una alerta de revisión humana si no puede valorar.
  4. `learn()`: recibe el precio final de venta, actualiza su memoria y cada 50 ventas reentrena el modelo.
- **Registro:** cada ejecución guarda percepciones, decisiones y acciones con marca de tiempo en `logs/` (archivos `.jsonl`).

## Estructura del repositorio

| Carpeta o archivo | Contenido |
|---|---|
| `data/AmesHousing.csv` | Dataset Ames Housing (De Cock), incluido para que el proyecto sea reproducible |
| `notebooks/01_modelo.ipynb` | Construcción y evaluación del modelo de regresión |
| `models/` | Modelo entrenado (`valuia.joblib`) y lista de variables (`columnas.json`) |
| `src/datos.py` | Carga de datos y separación 80/20 (`random_state=42`) |
| `src/simulador.py` | Simulador del entorno y generador de fallas |
| `src/agente.py` | Clase `ValuiaAgent` |
| `tests/plan_de_pruebas.md` | Medidas, umbrales y escenarios, fijados antes de ejecutar las pruebas |
| `tests/ejecutar_pruebas.py` | Ejecuta los 12 escenarios y calcula las medidas |
| `tests/diagnostico_m2.py` | Diagnóstico de la medida M2 a partir de los logs |
| `tests/resultados/` | Tablas de resultados (CSV) |
| `logs/pruebas/` | Logs completos de las 65 ejecuciones de prueba |

## Cómo ejecutarlo

Probado con Python 3.14.8 en Windows. Los comandos son para el Símbolo del sistema o PowerShell; en Mac o Linux usa `python3` y `source .venv/bin/activate`.

```
git clone https://github.com/juliomartinez777/valuia-agent.git
cd valuia-agent
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Demostración del agente** (60 propiedades y 3 fallas esperadas, menos de un minuto):

```
py -m src.agente
```

**Pruebas completas** (65 ejecuciones, entre 7 y 20 minutos según el equipo):

```
py -m tests.ejecutar_pruebas
```

**Diagnóstico de la medida M2** (lee los logs ya generados):

```
py -m tests.diagnostico_m2
```

Para reconstruir el modelo desde cero, abre `notebooks/01_modelo.ipynb` y ejecuta las celdas en orden.

## Resultados del modelo

Evaluado una sola vez sobre 586 casas de prueba que el modelo no vio al entrenar: **MAPE 8,20 %** y **R² 0,919**. Las variables se eligieron con validación cruzada solo sobre las casas de entrenamiento.

## Resultados de las pruebas contra las medidas de desempeño

12 escenarios (normales, límite y de falla esperada), 200 propiedades por escenario y 5 repeticiones con semillas 1 a 5. Los umbrales se fijaron en `tests/plan_de_pruebas.md` y se subieron al repositorio antes de ejecutar las pruebas.

| Medida | Umbral | Resultado | Cumple |
|---|---|---|---|
| M1 MAPE | ≤ 10 % (E1-E4) y ≤ 15 % (E5-E8) | E1-E4 entre 8,29 % y 9,33 %; E5-E8 entre 10,78 % y 14,41 % | Sí |
| M2 Cobertura del rango | ≥ 75 % (E1-E4) | E3: 74,20 %; los demás entre 78,7 % y 78,9 % | **No** |
| M3 Entradas inválidas | 100 % (E9-E12) | 100 % en los cuatro escenarios | Sí |
| M4 Adaptación al mercado | ≥ 1,0 punto (E6) | Mejora de 1,50 puntos (14,92 % sin aprendizaje frente a 13,42 % con aprendizaje) | Sí |

**M2 no se cumplió.** El diagnóstico de los logs muestra que, cuando el mercado sube, el agente subestima de forma creciente y las ventas que quedan fuera del rango caen sobre todo por encima (83 % en E3). No se cambiaron umbrales ni agente después de ver el resultado.

## Diferencias entre el diseño y la implementación

- El modelo usa 4 variables adicionales a las del diseño inicial: `OverallCond`, `TotalBsmtSF`, `LotArea` y `Fireplaces`.
- Ames Housing usa pies cuadrados y dólares; el diseño hablaba de metros cuadrados.
- Se agregaron las medidas M2, M3 y M4 a la medida original (M1, MAPE menor al 10 %).
- Los valores de MAPE y R² del diseño inicial eran proyecciones; los valores reales son los de este repositorio.

## Limitaciones

- El entorno es simulado: los precios finales salen del propio dataset, con ruido y deriva artificiales.
- El dataset corresponde a una ciudad de Estados Unidos, no a un mercado local.
- La adaptación del agente a mercados con deriva fuerte es lenta (ver resultados de M2 y de E6 y E7).

## Evidencia de funcionamiento

Enlace al video o a la plataforma desplegada: *(pendiente)*