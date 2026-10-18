"""Simulador del entorno inmobiliario donde opera VALUIA-AGENT."""
import numpy as np

from src.datos import cargar_datos, separar


class SimuladorMercado:
    PROPIEDADES_POR_MES = 20   # cada 20 propiedades "pasa un mes"

    def __init__(self, X_prueba, precios_reales, semilla=0,
                 ruido_negociacion=0.0, deriva_mensual=0.0):
        self.rng = np.random.default_rng(semilla)
        self.X = X_prueba.reset_index(drop=True)
        self.precios = np.asarray(precios_reales, dtype=float)
        self.ruido = ruido_negociacion      # desviación del ruido (0.03 = 3 %)
        self.deriva = deriva_mensual        # cambio del mercado por mes (0.01 = +1 %)
        self.orden = self.rng.permutation(len(self.X))   # orden aleatorio
        self.pos = 0
        self._precio_pendiente = {}

    def quedan_propiedades(self):
        return self.pos < len(self.orden)

    def mes_actual(self):
        return self.pos // self.PROPIEDADES_POR_MES

    def nueva_propiedad(self):
        """Entrega una propiedad (percepción). El precio final queda oculto."""
        idx = int(self.orden[self.pos])
        factor_mercado = (1 + self.deriva) ** self.mes_actual()
        ruido = np.exp(self.rng.normal(0, self.ruido)) if self.ruido > 0 else 1.0
        self._precio_pendiente[idx] = self.precios[idx] * factor_mercado * ruido
        self.pos += 1
        return idx, self.X.iloc[idx].to_dict()

    def precio_final_venta(self, id_propiedad):
        """Se llama DESPUÉS de que el agente da su estimación."""
        return self._precio_pendiente.pop(id_propiedad)


def inyectar_falla(propiedad, tipo):
    """Crea casos de falla esperada para las pruebas de la Fase 3."""
    p = dict(propiedad)
    if tipo == "sin_dato":
        p["GrLivArea"] = None                  # el sensor no entrega el área
    elif tipo == "invalido":
        p["GrLivArea"] = -50                   # área imposible
    elif tipo == "barrio_desconocido":
        p["Neighborhood"] = "BarrioInexistente"  # zona sin historial
    else:
        raise ValueError(f"Tipo de falla desconocido: {tipo}")
    return p


if __name__ == "__main__":
    df = cargar_datos()
    X_train, X_test, y_train, y_test = separar(df)
    sim = SimuladorMercado(X_test, np.exp(y_test), semilla=1,
                           ruido_negociacion=0.03, deriva_mensual=0.01)
    for _ in range(3):
        id_, prop = sim.nueva_propiedad()
        print(f"Propiedad {id_}: barrio={prop['Neighborhood']}, "
              f"área={prop['GrLivArea']} pies2, calidad={prop['OverallQual']}")
        print(f"   precio final (oculto para el agente): {sim.precio_final_venta(id_):,.0f}")
    print("Con falla sin_dato, el área queda en:", inyectar_falla(prop, "sin_dato")["GrLivArea"])