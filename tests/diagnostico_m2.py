"""Diagnóstico de M2 (cobertura del rango): ¿cuándo y por qué falla?

Lee los logs de las pruebas ya ejecutadas (logs/pruebas) y calcula, por mes simulado:
  - cobertura: % de ventas cuyo precio final cayó dentro del rango publicado
  - sesgo: error medio con signo (precio final - precio de referencia) / precio final
           (positivo = el agente subestimó, el mercado estaba más caro que su estimación)
No ejecuta el agente: solo analiza la evidencia ya guardada.
"""
import json

import numpy as np
import pandas as pd

from src.datos import RAIZ

DIR_LOGS = RAIZ / "logs" / "pruebas"
DIR_RES = RAIZ / "tests" / "resultados"
ESCENARIOS = ["E1", "E3", "E6"]     # sin deriva, deriva 1 %/mes, deriva 3 %/mes
PROPIEDADES_POR_MES = 20


def leer_log(ruta):
    """Devuelve una fila por propiedad valorada: rango publicado y precio final."""
    publicaciones, finales = {}, {}
    orden = []
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            r = json.loads(linea)
            if r["etapa"] == "accion" and r.get("tipo") == "valoracion":
                publicaciones[r["id_propiedad"]] = r
                orden.append(r["id_propiedad"])
            elif r["etapa"] == "aprendizaje":
                finales[r["id_propiedad"]] = r["precio_final"]
    filas = []
    for i, id_ in enumerate(orden):
        if id_ not in finales:
            continue
        p, final = publicaciones[id_], finales[id_]
        filas.append({
            "mes": i // PROPIEDADES_POR_MES,
            "dentro": p["rango_inferior"] <= final <= p["rango_superior"],
            "arriba": final > p["rango_superior"],
            "abajo": final < p["rango_inferior"],
            "sesgo": (final - p["precio_referencia"]) / final,
        })
    return pd.DataFrame(filas)


def main():
    partes = []
    for esc in ESCENARIOS:
        for ruta in sorted(DIR_LOGS.glob(f"{esc}_s[0-9]*.jsonl")):
            df = leer_log(ruta)
            df["escenario"] = esc
            partes.append(df)
    datos = pd.concat(partes, ignore_index=True)

    cobertura = datos.pivot_table(index="mes", columns="escenario", values="dentro",
                                  aggfunc="mean") * 100
    sesgo = datos.pivot_table(index="mes", columns="escenario", values="sesgo",
                              aggfunc="mean") * 100
    fallos = datos[~datos["dentro"]].groupby("escenario").agg(
        total=("dentro", "size"), arriba=("arriba", "sum"), abajo=("abajo", "sum"))
    todos = datos.groupby("escenario").size()

    print("=== COBERTURA DEL RANGO POR MES (%) ===")
    print(cobertura.round(1).to_string())
    print("\n=== SESGO MEDIO POR MES (%): positivo = el agente subestima ===")
    print(sesgo.round(1).to_string())
    print("\n=== DÓNDE CAEN LAS VENTAS QUE QUEDAN FUERA DEL RANGO ===")
    for esc in ESCENARIOS:
        t = fallos.loc[esc]
        print(f"{esc}: {int(t['total'])} de {todos[esc]} ventas fuera del rango | "
              f"por encima: {100 * t['arriba'] / t['total']:.0f} % | "
              f"por debajo: {100 * t['abajo'] / t['total']:.0f} %")
    mitad = datos[datos["escenario"] == "E3"].groupby(datos["mes"] >= 5)["dentro"].mean() * 100
    print(f"\nE3: cobertura meses 0-4 = {mitad[False]:.1f} % | meses 5-9 = {mitad[True]:.1f} %")

    DIR_RES.mkdir(parents=True, exist_ok=True)
    cobertura.round(2).to_csv(DIR_RES / "diagnostico_m2_cobertura_por_mes.csv")
    sesgo.round(2).to_csv(DIR_RES / "diagnostico_m2_sesgo_por_mes.csv")
    print("\nArchivos guardados en tests/resultados")


if __name__ == "__main__":
    main()