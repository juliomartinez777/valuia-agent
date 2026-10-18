"""VALUIA-AGENT: agente basado en modelo, con elementos de agente de aprendizaje.

Ciclo del agente (por cada propiedad):
    1. PERCEPCIÓN  -> perceive():  recibe y valida las características de la propiedad
    2. DECISIÓN    -> decide():    estima el precio con su modelo interno y lo ajusta
                                   con las ventas recientes de la misma zona
    3. ACCIÓN      -> act():       publica precio de referencia y rango en el sistema
                                   de listados (o una alerta si no puede valorar)
Aprendizaje:
    learn(): cuando llega el precio final de venta, el agente actualiza su memoria
             y, cada cierto número de ventas, recalibra (reentrena) el modelo.
"""
import json
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from src.datos import RAIZ, RUTA_MODELO, cargar_columnas, cargar_datos, separar

# Rangos físicamente razonables para validar las percepciones (sensor de entrada)
RANGOS_VALIDOS = {
    "GrLivArea": (300, 6000),      # pies cuadrados construidos
    "BedroomAbvGr": (0, 10),
    "FullBath": (0, 6),
    "HalfBath": (0, 4),
    "YearBuilt": (1800, 2026),
    "YearRemodAdd": (1800, 2026),
    "OverallQual": (1, 10),
    "GarageCars": (0, 6),
    "OverallCond": (1, 10),
    "TotalBsmtSF": (0, 6500),
    "LotArea": (1000, 250000),
    "Fireplaces": (0, 6),
}


class ValuiaAgent:
    RECALIBRAR_CADA = 50      # ventas nuevas necesarias para reentrenar
    MIN_VENTAS_ZONA = 3       # ventas recientes mínimas para ajustar una zona
    VENTANA_ZONA = 10         # cuántas ventas recientes de la zona recuerda
    MARGEN_INICIAL = 0.15     # rango +-15 % mientras no tenga historial propio
    PESO_AJUSTE_ZONA = 0.7    # qué tanto confía en el ajuste por ventas recientes

    def __init__(self, X_hist, y_hist_log, ruta_log=None, ruta_modelo=RUTA_MODELO):
        # --- Estado interno (agente basado en modelo) ---
        self.modelo = joblib.load(ruta_modelo)
        self.cols_num, self.cols_cat = cargar_columnas()
        self.columnas = self.cols_num + self.cols_cat
        self.barrios_conocidos = set(
            self.modelo.named_steps["prep"].named_transformers_["cat"].categories_[0])

        # --- Memoria de largo plazo ---
        self.X_hist = X_hist.copy().reset_index(drop=True)      # historial de transacciones
        self.y_hist = pd.Series(np.asarray(y_hist_log)).reset_index(drop=True)
        self.residuos_zona = defaultdict(lambda: deque(maxlen=self.VENTANA_ZONA))
        self.errores_recientes = deque(maxlen=50)               # error relativo absoluto
        self.ventas_nuevas = []                                  # (fila, log precio real)
        self.n_recalibraciones = 0

        # --- Memoria de corto plazo: propiedades valoradas sin precio final todavía ---
        self._pendientes = {}

        # --- Actuador hacia el sistema de listados (simulado) ---
        self.listados = []     # lo publicado: precio, rango, alertas

        # --- Log de ejecución ---
        marca = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ruta_log = Path(ruta_log) if ruta_log else RAIZ / "logs" / f"ejecucion_{marca}.jsonl"
        self.ruta_log.parent.mkdir(parents=True, exist_ok=True)
        self.ruta_log.write_text("", encoding="utf-8")

    # ------------------------------------------------------------------ log
    def _log(self, etapa, id_propiedad, **datos):
        registro = {"timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    "etapa": etapa, "id_propiedad": id_propiedad, **datos}
        with open(self.ruta_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, default=str, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------ percepción
    def perceive(self, id_propiedad, propiedad):
        """Valida la percepción. Devuelve (percepto, estado, problemas)."""
        problemas = []
        percepto = {}
        for col in self.cols_num:
            valor = propiedad.get(col)
            if valor is None or (isinstance(valor, float) and np.isnan(valor)):
                problemas.append(f"{col}: sin dato")
                continue
            try:
                valor = float(valor)
            except (TypeError, ValueError):
                problemas.append(f"{col}: no numérico ({valor!r})")
                continue
            minimo, maximo = RANGOS_VALIDOS[col]
            if not (minimo <= valor <= maximo):
                problemas.append(f"{col}: fuera de rango ({valor})")
                continue
            percepto[col] = valor
        barrio = propiedad.get("Neighborhood")
        if not isinstance(barrio, str) or not barrio.strip():
            problemas.append("Neighborhood: sin dato")
        else:
            percepto["Neighborhood"] = barrio

        # Un dato faltante o inválido en las variables clave impide valorar.
        estado = "ok" if not problemas else (
            "sin_dato" if any("sin dato" in p for p in problemas) else "invalido")
        self._log("percepcion", id_propiedad, estado=estado,
                  problemas=problemas, entrada=propiedad)
        return percepto, estado, problemas

    # -------------------------------------------------------------- decisión
    def decide(self, id_propiedad, percepto, estado, problemas):
        if estado != "ok":
            decision = {"valorada": False, "motivo": "; ".join(problemas)}
            self._log("decision", id_propiedad, **decision)
            return decision

        fila = pd.DataFrame([percepto])[self.columnas]
        precio_modelo = float(np.exp(self.modelo.predict(fila)[0]))

        # Ajuste con ventas recientes de propiedades similares (misma zona)
        barrio = percepto["Neighborhood"]
        residuos = self.residuos_zona[barrio]
        ajuste = 1.0
        if len(residuos) >= self.MIN_VENTAS_ZONA:
            ajuste = float(np.exp(self.PESO_AJUSTE_ZONA * np.mean(residuos)))
        precio_ref = precio_modelo * ajuste

        # Rango razonable: aprende de sus propios errores recientes
        if len(self.errores_recientes) >= 10:
            margen = max(0.05, float(np.percentile(self.errores_recientes, 80)))
        else:
            margen = self.MARGEN_INICIAL
        confianza = "normal"
        if barrio not in self.barrios_conocidos:
            margen *= 1.5
            confianza = "baja (zona sin historial)"

        decision = {
            "valorada": True,
            "precio_modelo": round(precio_modelo, 2),
            "factor_ajuste_zona": round(ajuste, 4),
            "precio_referencia": round(precio_ref, 2),
            "rango_inferior": round(precio_ref * (1 - margen), 2),
            "rango_superior": round(precio_ref * (1 + margen), 2),
            "confianza": confianza,
            "ventas_zona_usadas": len(residuos),
        }
        self._pendientes[id_propiedad] = percepto
        self._log("decision", id_propiedad, **decision)
        return decision

    # ---------------------------------------------------------------- acción
    def act(self, id_propiedad, decision):
        """Publica en el sistema de listados (simulado) y devuelve lo publicado."""
        if decision["valorada"]:
            publicacion = {
                "id_propiedad": id_propiedad, "tipo": "valoracion",
                "precio_referencia": decision["precio_referencia"],
                "rango_inferior": decision["rango_inferior"],
                "rango_superior": decision["rango_superior"],
                "confianza": decision["confianza"],
            }
        else:
            publicacion = {"id_propiedad": id_propiedad, "tipo": "alerta_revision_humana",
                           "motivo": decision["motivo"]}
        self.listados.append(publicacion)
        self._log("accion", id_propiedad,
                  **{k: v for k, v in publicacion.items() if k != "id_propiedad"})
        return publicacion

    def valorar(self, id_propiedad, propiedad):
        """Ciclo completo percepción -> decisión -> acción."""
        percepto, estado, problemas = self.perceive(id_propiedad, propiedad)
        decision = self.decide(id_propiedad, percepto, estado, problemas)
        return self.act(id_propiedad, decision)

    # ------------------------------------------------------------ aprendizaje
    def learn(self, id_propiedad, precio_final):
        """Recibe el precio final de venta, actualiza memoria y recalibra si toca."""
        if id_propiedad not in self._pendientes:
            return None
        percepto = self._pendientes.pop(id_propiedad)
        publicada = next(p for p in reversed(self.listados)
                         if p["id_propiedad"] == id_propiedad and p["tipo"] == "valoracion")
        precio_ref = publicada["precio_referencia"]
        error_rel = abs(precio_final - precio_ref) / precio_final

        # Residuo del MODELO (sin ajuste): lo que la zona necesita corregir
        fila = pd.DataFrame([percepto])[self.columnas]
        precio_modelo = float(np.exp(self.modelo.predict(fila)[0]))
        residuo = float(np.log(precio_final / precio_modelo))
        self.residuos_zona[percepto["Neighborhood"]].append(residuo)
        self.errores_recientes.append(error_rel)
        self.ventas_nuevas.append((percepto, float(np.log(precio_final))))

        self._log("aprendizaje", id_propiedad, precio_final=round(precio_final, 2),
                  precio_referencia=precio_ref, error_relativo=round(error_rel, 4),
                  ventas_acumuladas=len(self.ventas_nuevas))
        if len(self.ventas_nuevas) >= self.RECALIBRAR_CADA:
            self.recalibrar(id_propiedad)
        return error_rel

    def recalibrar(self, id_propiedad=None):
        """Señal de recalibración: reentrena con historial + ventas nuevas."""
        nuevas = pd.DataFrame([p for p, _ in self.ventas_nuevas])[self.columnas]
        y_nuevas = pd.Series([y for _, y in self.ventas_nuevas])
        self.X_hist = pd.concat([self.X_hist, nuevas], ignore_index=True)
        self.y_hist = pd.concat([self.y_hist, y_nuevas], ignore_index=True)
        nuevo = clone(self.modelo)
        nuevo.fit(self.X_hist, self.y_hist)
        self.modelo = nuevo
        n = len(self.ventas_nuevas)
        self.ventas_nuevas = []
        self.residuos_zona.clear()      # el modelo nuevo ya absorbió esos residuos
        self.n_recalibraciones += 1
        self._log("recalibracion", id_propiedad, ventas_incorporadas=n,
                  tamano_historial=len(self.X_hist), n_recalibraciones=self.n_recalibraciones)


if __name__ == "__main__":
    # Demostración rápida: 60 propiedades + 3 fallas esperadas
    from src.simulador import SimuladorMercado, inyectar_falla

    df = cargar_datos()
    X_train, X_test, y_train, y_test = separar(df)
    agente = ValuiaAgent(X_train, y_train)
    sim = SimuladorMercado(X_test, np.exp(y_test), semilla=1,
                           ruido_negociacion=0.03, deriva_mensual=0.01)

    errores = []
    for _ in range(60):
        id_, prop = sim.nueva_propiedad()
        pub = agente.valorar(id_, prop)
        err = agente.learn(id_, sim.precio_final_venta(id_))
        errores.append(err)
    print(f"Valoraciones: {len(errores)} | MAPE: {100 * np.mean(errores):.2f} % "
          f"| recalibraciones: {agente.n_recalibraciones}")

    for tipo in ["sin_dato", "invalido", "barrio_desconocido"]:
        id_, prop = sim.nueva_propiedad()
        pub = agente.valorar(id_, inyectar_falla(prop, tipo))
        print(f"Falla '{tipo}': {pub['tipo']}", pub.get("motivo", pub.get("confianza", "")))
    print("Log guardado en:", agente.ruta_log)