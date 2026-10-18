"""Ejecuta el plan de pruebas (tests/plan_de_pruebas.md) y genera la evidencia.

Salidas:
  logs/pruebas/<escenario>_s<semilla>.jsonl   log completo del agente en cada ejecución
  tests/resultados/resumen_por_ejecucion.csv  una fila por ejecución
  tests/resultados/resumen_por_escenario.csv  media y desviación estándar por escenario
  tests/resultados/tabla_resultados.csv       tabla final por medida de desempeño
"""
import time

import numpy as np
import pandas as pd

from src.agente import ValuiaAgent
from src.datos import RAIZ, cargar_datos, separar
from src.simulador import SimuladorMercado, inyectar_falla

N_PROPIEDADES = 200
SEMILLAS = [1, 2, 3, 4, 5]
FRACCION_FALLAS = 0.25      # en E9-E12, una de cada 4 propiedades llega con la falla

# (id, tipo, ruido de negociación, deriva mensual, falla inyectada)
ESCENARIOS = [
    ("E1", "normal", 0.00, 0.00, None),
    ("E2", "normal", 0.03, 0.00, None),
    ("E3", "normal", 0.03, 0.01, None),
    ("E4", "normal", 0.05, 0.00, None),
    ("E5", "limite", 0.08, 0.00, None),
    ("E6", "limite", 0.03, 0.03, None),
    ("E7", "limite", 0.08, 0.03, None),
    ("E8", "limite", 0.10, 0.01, None),
    ("E9", "falla", 0.03, 0.00, "sin_dato"),
    ("E10", "falla", 0.03, 0.00, "invalido"),
    ("E11", "falla", 0.03, 0.00, "no_numerico"),
    ("E12", "falla", 0.03, 0.00, "barrio_desconocido"),
]
UMBRAL_M1 = {"normal": 10.0, "limite": 15.0}
UMBRAL_M2 = 75.0
UMBRAL_M4 = 1.0

DIR_LOGS = RAIZ / "logs" / "pruebas"
DIR_RES = RAIZ / "tests" / "resultados"


def inyectar(propiedad, tipo):
    """Fallas del simulador, más el caso 'no_numerico' propio de las pruebas."""
    if tipo == "no_numerico":
        p = dict(propiedad)
        p["GrLivArea"] = "abc"
        return p
    return inyectar_falla(propiedad, tipo)


def ejecutar(X_train, y_train, X_test, y_test, id_esc, tipo, ruido, deriva, falla,
             semilla, aprende=True):
    nombre = f"{id_esc}{'' if aprende else '_sin_aprendizaje'}_s{semilla}"
    ruta_log = DIR_LOGS / f"{nombre}.jsonl"
    agente = ValuiaAgent(X_train, y_train, ruta_log=ruta_log)
    sim = SimuladorMercado(X_test, np.exp(y_test), semilla=semilla,
                           ruido_negociacion=ruido, deriva_mensual=deriva)
    rng = np.random.default_rng(1000 + semilla)
    con_falla = set(rng.choice(N_PROPIEDADES, int(N_PROPIEDADES * FRACCION_FALLAS),
                               replace=False)) if falla else set()

    errores, dentro_rango = [], []
    inyectadas = alertas = baja_confianza = 0
    for i in range(N_PROPIEDADES):
        id_, prop = sim.nueva_propiedad()
        if i in con_falla:
            inyectadas += 1
            pub = agente.valorar(id_, inyectar(prop, falla))
            sim.precio_final_venta(id_)           # el agente no aprende de estos casos
            if pub["tipo"] == "alerta_revision_humana":
                alertas += 1
            elif pub["confianza"].startswith("baja"):
                baja_confianza += 1
            continue
        pub = agente.valorar(id_, prop)
        final = sim.precio_final_venta(id_)
        errores.append(abs(final - pub["precio_referencia"]) / final)
        dentro_rango.append(pub["rango_inferior"] <= final <= pub["rango_superior"])
        if aprende:
            agente.learn(id_, final)

    if falla in ("sin_dato", "invalido", "no_numerico"):
        correctas = 100.0 * alertas / inyectadas
    elif falla == "barrio_desconocido":
        correctas = 100.0 * baja_confianza / inyectadas
    else:
        correctas = np.nan
    return {
        "escenario": id_esc, "tipo": tipo, "semilla": semilla, "aprende": aprende,
        "n_propiedades": N_PROPIEDADES, "n_valoradas": len(errores),
        "n_fallas_inyectadas": inyectadas,
        "mape": 100 * np.mean(errores), "cobertura_rango": 100 * np.mean(dentro_rango),
        "fallas_bien_manejadas_pct": correctas,
        "recalibraciones": agente.n_recalibraciones,
        "log": str(ruta_log.relative_to(RAIZ)).replace("\\", "/"),
    }


def ms(serie):
    return f"{serie.mean():.2f} ± {serie.std(ddof=1):.2f}"


def main():
    DIR_LOGS.mkdir(parents=True, exist_ok=True)
    DIR_RES.mkdir(parents=True, exist_ok=True)
    df = cargar_datos()
    X_train, X_test, y_train, y_test = separar(df)

    filas, t0 = [], time.time()
    total = len(ESCENARIOS) * len(SEMILLAS) + len(SEMILLAS)
    hecho = 0
    for (id_esc, tipo, ruido, deriva, falla) in ESCENARIOS:
        for s in SEMILLAS:
            filas.append(ejecutar(X_train, y_train, X_test, y_test,
                                  id_esc, tipo, ruido, deriva, falla, s))
            hecho += 1
            print(f"[{hecho}/{total}] {id_esc} semilla {s} listo "
                  f"({time.time() - t0:.0f} s)", flush=True)
    # M4: el escenario E6 repetido con el agente SIN aprendizaje
    _, tipo, ruido, deriva, falla = next(e for e in ESCENARIOS if e[0] == "E6")
    for s in SEMILLAS:
        filas.append(ejecutar(X_train, y_train, X_test, y_test,
                              "E6", tipo, ruido, deriva, falla, s, aprende=False))
        hecho += 1
        print(f"[{hecho}/{total}] E6 sin aprendizaje semilla {s} listo "
              f"({time.time() - t0:.0f} s)", flush=True)

    res = pd.DataFrame(filas)
    res.to_csv(DIR_RES / "resumen_por_ejecucion.csv", index=False)

    # ---- resumen por escenario (media y desviación estándar de las repeticiones)
    por_esc = (res.groupby(["escenario", "tipo", "aprende"], sort=False)
               .agg(mape_media=("mape", "mean"), mape_std=("mape", lambda x: x.std(ddof=1)),
                    cobertura_media=("cobertura_rango", "mean"),
                    cobertura_std=("cobertura_rango", lambda x: x.std(ddof=1)),
                    fallas_ok_media=("fallas_bien_manejadas_pct", "mean"),
                    fallas_ok_min=("fallas_bien_manejadas_pct", "min"),
                    recalibraciones_media=("recalibraciones", "mean"))
               .reset_index())
    por_esc.to_csv(DIR_RES / "resumen_por_escenario.csv", index=False)

    # ---- evaluación de las medidas contra los umbrales fijados en el plan
    ap = res[res["aprende"]]
    # M1
    m1_ok, m1_det = True, []
    for esc in [e[0] for e in ESCENARIOS if e[1] in ("normal", "limite")]:
        sub = ap[ap["escenario"] == esc]
        umbral = UMBRAL_M1[sub["tipo"].iloc[0]]
        ok = sub["mape"].mean() <= umbral
        m1_ok &= ok
        m1_det.append(f"{esc}: {ms(sub['mape'])} % (umbral {umbral:g})" + ("" if ok else " NO CUMPLE"))
    # M2
    m2_ok, m2_det = True, []
    for esc in ["E1", "E2", "E3", "E4"]:
        sub = ap[ap["escenario"] == esc]
        ok = sub["cobertura_rango"].mean() >= UMBRAL_M2
        m2_ok &= ok
        m2_det.append(f"{esc}: {ms(sub['cobertura_rango'])} %" + ("" if ok else " NO CUMPLE"))
    # M3
    sub3 = ap[ap["escenario"].isin(["E9", "E10", "E11", "E12"])]
    m3_val = sub3["fallas_bien_manejadas_pct"].min()
    m3_ok = bool(m3_val >= 100.0)
    m3_det = [f"{e}: {ap[ap['escenario'] == e]['fallas_bien_manejadas_pct'].mean():.1f} %"
              for e in ["E9", "E10", "E11", "E12"]]
    # M4
    con = res[(res["escenario"] == "E6") & res["aprende"]]["mape"].mean()
    sin = res[(res["escenario"] == "E6") & ~res["aprende"]]["mape"].mean()
    mejora = sin - con
    m4_ok = bool(mejora >= UMBRAL_M4)

    tabla = pd.DataFrame([
        {"Medida": "M1 MAPE", "Formula": "prom(|final - ref|/final) x 100",
         "Umbral": "<=10 % (E1-E4); <=15 % (E5-E8)",
         "Casos": f"8 escenarios x {len(SEMILLAS)} repeticiones x {N_PROPIEDADES} propiedades",
         "Resultado": " | ".join(m1_det), "Cumple": "SI" if m1_ok else "NO",
         "Evidencia": "tests/resultados/resumen_por_ejecucion.csv; logs/pruebas/E1..E8_s*.jsonl"},
        {"Medida": "M2 Cobertura del rango", "Formula": "ventas dentro del rango / valoradas x 100",
         "Umbral": ">=75 % (E1-E4)",
         "Casos": f"4 escenarios x {len(SEMILLAS)} repeticiones x {N_PROPIEDADES} propiedades",
         "Resultado": " | ".join(m2_det), "Cumple": "SI" if m2_ok else "NO",
         "Evidencia": "tests/resultados/resumen_por_ejecucion.csv; logs/pruebas/E1..E4_s*.jsonl"},
        {"Medida": "M3 Entradas invalidas", "Formula": "fallas bien manejadas / inyectadas x 100",
         "Umbral": "100 % (E9-E12)",
         "Casos": f"4 escenarios x {len(SEMILLAS)} repeticiones x {int(N_PROPIEDADES * FRACCION_FALLAS)} fallas",
         "Resultado": " | ".join(m3_det), "Cumple": "SI" if m3_ok else "NO",
         "Evidencia": "tests/resultados/resumen_por_ejecucion.csv; logs/pruebas/E9..E12_s*.jsonl"},
        {"Medida": "M4 Adaptacion al mercado", "Formula": "MAPE sin aprendizaje - MAPE con aprendizaje",
         "Umbral": ">=1,0 punto porcentual (E6)",
         "Casos": f"E6 x {len(SEMILLAS)} repeticiones x 2 variantes x {N_PROPIEDADES} propiedades",
         "Resultado": f"sin aprendizaje {sin:.2f} % | con aprendizaje {con:.2f} % | mejora {mejora:.2f} pp",
         "Cumple": "SI" if m4_ok else "NO",
         "Evidencia": "logs/pruebas/E6_s*.jsonl y E6_sin_aprendizaje_s*.jsonl"},
    ])
    tabla.to_csv(DIR_RES / "tabla_resultados.csv", index=False)

    print("\n=== RESULTADOS POR ESCENARIO (media ± desviación estándar) ===")
    for _, r in por_esc.iterrows():
        extra = "" if r["aprende"] else "  [sin aprendizaje]"
        print(f"{r['escenario']:>4}{extra}: MAPE {r['mape_media']:.2f} ± {r['mape_std']:.2f} % | "
              f"cobertura {r['cobertura_media']:.1f} % | fallas bien manejadas "
              f"{r['fallas_ok_media']:.1f} %")
    print("\n=== MEDIDAS DE DESEMPEÑO ===")
    for _, r in tabla.iterrows():
        print(f"{r['Medida']}: {r['Cumple']}  ->  {r['Resultado']}")
    print(f"\nTiempo total: {(time.time() - t0) / 60:.1f} minutos")
    print("Archivos guardados en tests/resultados y logs/pruebas")


if __name__ == "__main__":
    main()