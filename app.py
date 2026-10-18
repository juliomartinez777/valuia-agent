"""VALUIA-AGENT: aplicación web (Streamlit) para demostrar el agente funcionando.

Pestañas:
  1. Valorar una propiedad : ciclo percepción -> decisión -> acción -> aprendizaje en vivo
  2. Simular el mercado    : un escenario del plan de pruebas, con el agente corriendo desde cero
  3. Pruebas               : resultados oficiales y ejecución del plan de pruebas
"""
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import tests.ejecutar_pruebas as ep
from src.agente import ValuiaAgent
from src.datos import RAIZ, RUTA_MODELO, cargar_columnas, cargar_datos, separar
from tests.diagnostico_m2 import leer_log

REPO = "https://github.com/juliomartinez777/valuia-agent"
DIR_APP = RAIZ / "logs" / "app"
DIR_APP.mkdir(parents=True, exist_ok=True)
ep.DIR_LOGS = DIR_APP / "pruebas"          # la app nunca sobrescribe los logs oficiales

st.set_page_config(page_title="VALUIA-AGENT", page_icon="🏠", layout="wide")


# --------------------------------------------------------------------------- entorno
def asegurar_modelo(X_train, y_train):
    """Usa models/valuia.joblib. Si no carga en este servidor, lo reentrena al iniciar."""
    try:
        modelo = joblib.load(RUTA_MODELO)
        modelo.predict(X_train.head(1))
        return None
    except Exception as e:
        num, cat = cargar_columnas()
        modelo = Pipeline([
            ("prep", ColumnTransformer([
                ("num", SimpleImputer(strategy="median"), num),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])),
            ("gb", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                             max_depth=3, random_state=42)),
        ])
        modelo.fit(X_train, y_train)
        joblib.dump(modelo, RUTA_MODELO)
        return f"El modelo guardado no cargó en este servidor ({type(e).__name__}); se reentrenó al iniciar."


@st.cache_resource(show_spinner="Cargando datos y modelo...")
def cargar_entorno():
    df = cargar_datos()
    X_train, X_test, y_train, y_test = separar(df)
    aviso = asegurar_modelo(X_train, y_train)
    return X_train, X_test, y_train, y_test, aviso


X_train, X_test, y_train, y_test, AVISO_MODELO = cargar_entorno()
COLS_NUM, COLS_CAT = cargar_columnas()


def agente_sesion():
    """Un agente por sesión del navegador: conserva su memoria mientras la usas."""
    if "agente" not in st.session_state:
        marca = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        st.session_state.ruta_log = DIR_APP / f"sesion_{marca}.jsonl"
        st.session_state.agente = ValuiaAgent(X_train, y_train, ruta_log=st.session_state.ruta_log)
        st.session_state.n_prop = 0
        st.session_state.ultima = None
        st.session_state.aprendidos = set()
    return st.session_state.agente


def eventos_de(id_propiedad):
    with open(st.session_state.ruta_log, encoding="utf-8") as f:
        eventos = [json.loads(linea) for linea in f]
    return [e for e in eventos if e["id_propiedad"] == id_propiedad]


# ------------------------------------------------------------- evaluación de medidas
def evaluar_medidas(res):
    """Aplica a un conjunto de ejecuciones los mismos umbrales fijados en el plan de pruebas."""
    ap = res[res["aprende"]]
    presentes = set(ap["escenario"])
    tipo_de = {e[0]: e[1] for e in ep.ESCENARIOS}

    def ms(serie):
        return f"{serie.mean():.2f}" + (f" ± {serie.std(ddof=1):.2f}" if len(serie) > 1 else "")

    filas = []
    # M1
    detalle, ok, n = [], True, 0
    for esc in [e[0] for e in ep.ESCENARIOS if e[1] in ("normal", "limite")]:
        if esc in presentes:
            n += 1
            umbral = ep.UMBRAL_M1[tipo_de[esc]]
            sub = ap[ap["escenario"] == esc]["mape"]
            ok &= sub.mean() <= umbral
            detalle.append(f"{esc}: {ms(sub)} % (umbral {umbral:g})")
    filas.append(("M1 MAPE", "≤10 % (E1-E4); ≤15 % (E5-E8)", " | ".join(detalle),
                  ("Sí" if ok else "No") if n else "Sin datos"))
    # M2
    detalle, ok, n = [], True, 0
    for esc in ["E1", "E2", "E3", "E4"]:
        if esc in presentes:
            n += 1
            sub = ap[ap["escenario"] == esc]["cobertura_rango"]
            ok &= sub.mean() >= ep.UMBRAL_M2
            detalle.append(f"{esc}: {ms(sub)} %")
    filas.append(("M2 Cobertura del rango", "≥75 % (E1-E4)", " | ".join(detalle),
                  ("Sí" if ok else "No") if n else "Sin datos"))
    # M3
    sub3 = ap[ap["escenario"].isin(["E9", "E10", "E11", "E12"])]
    if len(sub3):
        det = [f"{e}: {ap[ap['escenario'] == e]['fallas_bien_manejadas_pct'].mean():.1f} %"
               for e in ["E9", "E10", "E11", "E12"] if e in presentes]
        filas.append(("M3 Entradas inválidas", "100 % (E9-E12)", " | ".join(det),
                      "Sí" if sub3["fallas_bien_manejadas_pct"].min() >= 100.0 else "No"))
    else:
        filas.append(("M3 Entradas inválidas", "100 % (E9-E12)", "", "Sin datos"))
    # M4
    con = res[(res["escenario"] == "E6") & res["aprende"]]["mape"]
    sin = res[(res["escenario"] == "E6") & ~res["aprende"]]["mape"]
    if len(con) and len(sin):
        mejora = sin.mean() - con.mean()
        filas.append(("M4 Adaptación al mercado", "≥1,0 punto (E6)",
                      f"sin aprendizaje {sin.mean():.2f} % | con aprendizaje {con.mean():.2f} % | "
                      f"mejora {mejora:.2f} pp", "Sí" if mejora >= ep.UMBRAL_M4 else "No"))
    else:
        filas.append(("M4 Adaptación al mercado", "≥1,0 punto (E6)", "", "Sin datos"))
    return pd.DataFrame(filas, columns=["Medida", "Umbral", "Resultado", "¿Cumple?"])


# ------------------------------------------------------------------------- encabezado
st.title("🏠 VALUIA-AGENT")
st.caption("Agente inteligente que estima el precio de venta de una propiedad residencial, "
           "publica un rango razonable y aprende del precio final de cada venta. "
           "Tópicos Especiales 2026-II · Julio Martinez y Jairo Mendoza")
if AVISO_MODELO:
    st.warning(AVISO_MODELO)

with st.expander("Instrucciones de uso y casos de prueba sugeridos", expanded=False):
    st.markdown(f"""
**Datos y unidades.** El agente trabaja con el dataset Ames Housing: áreas en **pies cuadrados** y precios en **dólares**.
Código y documentación: {REPO}

**Casos de prueba sugeridos**
1. *Escenario típico:* pestaña **Valorar una propiedad**, deja los datos por defecto y pulsa **Valorar propiedad**.
   Luego enséñale el precio final de venta y observa cómo cambia su memoria.
2. *Falla esperada:* en la misma pestaña, elige en el desplegable una falla (dato faltante, área imposible,
   dato no numérico o zona sin historial) y valora de nuevo. El agente debe publicar una alerta de revisión humana,
   o una valoración con confianza baja si la zona es nueva.
3. *Entorno simulado:* pestaña **Simular el mercado**, elige un escenario (por ejemplo E3, mercado que sube 1 % al mes)
   y pulsa **Ejecutar escenario**. El agente corre desde cero sobre 200 propiedades.
4. *Pruebas:* pestaña **Pruebas**, ejecuta el plan con 1 repetición para ver cómo se calculan las medidas M1 a M4.
   Los resultados oficiales (65 ejecuciones) aparecen en esa misma pestaña.
""")

tab1, tab2, tab3 = st.tabs(["Valorar una propiedad", "Simular el mercado", "Pruebas"])

# ------------------------------------------------------------------- pestaña 1
with tab1:
    ag = agente_sesion()
    barrios = sorted(ag.barrios_conocidos)
    st.subheader("1. Percepción: datos de la propiedad")
    c1, c2, c3 = st.columns(3)
    with c1:
        area = st.number_input("Área construida (pies²)", 0, 20000, 1500, key="area")
        hab = st.number_input("Habitaciones", 0, 20, 3, key="hab")
        banos = st.number_input("Baños completos", 0, 10, 2, key="banos")
        medios = st.number_input("Medios baños", 0, 10, 0, key="medios")
    with c2:
        anio = st.number_input("Año de construcción", 1700, 2100, 1995, key="anio")
        remo = st.number_input("Año de remodelación", 1700, 2100, 2005, key="remo")
        calidad = st.number_input("Calidad general (1-10)", 0, 20, 6, key="calidad")
        condicion = st.number_input("Condición general (1-10)", 0, 20, 5, key="condicion")
    with c3:
        garaje = st.number_input("Capacidad de garaje (autos)", 0, 10, 2, key="garaje")
        sotano = st.number_input("Área del sótano (pies²)", 0, 10000, 1000, key="sotano")
        lote = st.number_input("Área del terreno (pies²)", 0, 400000, 9500, key="lote")
        chimeneas = st.number_input("Chimeneas", 0, 10, 1, key="chimeneas")
    idx = barrios.index("NAmes") if "NAmes" in barrios else 0
    barrio = st.selectbox("Barrio", barrios, index=idx, key="barrio")
    caso = st.selectbox("Caso de prueba", [
        "Propiedad normal (datos del formulario)",
        "Falla: dato de área faltante",
        "Falla: área imposible (-50 pies²)",
        "Falla: área no numérica ('abc')",
        "Falla: zona sin historial",
    ], key="caso")

    if st.button("Valorar propiedad", type="primary", key="btn_valorar"):
        prop = {"GrLivArea": area, "BedroomAbvGr": hab, "FullBath": banos, "HalfBath": medios,
                "YearBuilt": anio, "YearRemodAdd": remo, "OverallQual": calidad,
                "GarageCars": garaje, "OverallCond": condicion, "TotalBsmtSF": sotano,
                "LotArea": lote, "Fireplaces": chimeneas, "Neighborhood": barrio}
        if caso.startswith("Falla: dato de área"):
            prop["GrLivArea"] = None
        elif caso.startswith("Falla: área imposible"):
            prop["GrLivArea"] = -50
        elif caso.startswith("Falla: área no numérica"):
            prop["GrLivArea"] = "abc"
        elif caso.startswith("Falla: zona"):
            prop["Neighborhood"] = "BarrioInexistente"
        st.session_state.n_prop += 1
        id_ = st.session_state.n_prop
        st.session_state.ultima = {"id": id_, "pub": ag.valorar(id_, prop)}
        st.session_state.msg_aprendizaje = None

    ultima = st.session_state.ultima
    if ultima:
        pub, id_ = ultima["pub"], ultima["id"]
        st.subheader("2. Decisión y acción: lo que publica el agente")
        if pub["tipo"] == "valoracion":
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Precio de referencia", f"${pub['precio_referencia']:,.0f}")
            m2.metric("Rango inferior", f"${pub['rango_inferior']:,.0f}")
            m3.metric("Rango superior", f"${pub['rango_superior']:,.0f}")
            m4.metric("Confianza", pub["confianza"].split(" ")[0])
            if pub["confianza"] != "normal":
                st.info(f"Confianza {pub['confianza']}: el agente ensanchó el rango.")
        else:
            st.error("Alerta de revisión humana: el agente no publicó un precio.")
            st.write("Motivo:", pub["motivo"])
        with st.expander("Ver el ciclo completo en el log (percepción → decisión → acción)"):
            st.json(eventos_de(id_))

        if pub["tipo"] == "valoracion" and id_ not in st.session_state.aprendidos:
            st.subheader("3. Aprendizaje: enseñarle el precio final de venta")
            precio = st.number_input("Precio final de venta (USD)", min_value=1000.0,
                                     value=float(round(pub["precio_referencia"])),
                                     step=1000.0, key=f"precio_{id_}")
            if st.button("Enseñar al agente este precio", key="btn_ensenar"):
                error = ag.learn(id_, float(precio))
                st.session_state.aprendidos.add(id_)
                st.session_state.msg_aprendizaje = f"Error relativo de esta valoración: {100 * error:.1f} %"
                st.rerun()
        if st.session_state.get("msg_aprendizaje"):
            st.success(st.session_state.msg_aprendizaje)
        st.caption(f"Memoria del agente: {len(ag.ventas_nuevas)} ventas nuevas acumuladas "
                   f"(recalibra cada {ag.RECALIBRAR_CADA}) · recalibraciones: {ag.n_recalibraciones}")

# ------------------------------------------------------------------- pestaña 2
with tab2:
    st.subheader("Simular el mercado con el agente corriendo desde cero")
    st.write("Cada ejecución crea un agente nuevo, genera 200 propiedades del conjunto de prueba "
             "(que el modelo no vio al entrenar) y registra todo en un log.")
    etiquetas = {}
    for (id_e, tipo, ruido, deriva, falla) in ep.ESCENARIOS:
        txt = f"{id_e} · {tipo} · ruido {ruido:.0%} · deriva {deriva:.0%}/mes"
        etiquetas[txt + (f" · falla: {falla}" if falla else "")] = (id_e, tipo, ruido, deriva, falla)
    sel = st.selectbox("Escenario", list(etiquetas), index=2, key="escenario")
    semilla = st.number_input("Semilla aleatoria", 1, 1000, 1, key="semilla")
    if st.button("Ejecutar escenario", type="primary", key="btn_escenario"):
        id_e, tipo, ruido, deriva, falla = etiquetas[sel]
        with st.spinner("Ejecutando el agente sobre 200 propiedades..."):
            r = ep.ejecutar(X_train, y_train, X_test, y_test, id_e, tipo, ruido, deriva,
                            falla, int(semilla))
        st.session_state.resultado_escenario = (id_e, tipo, r)
    if st.session_state.get("resultado_escenario"):
        id_e, tipo, r = st.session_state.resultado_escenario
        a, b, c, d = st.columns(4)
        a.metric("MAPE", f"{r['mape']:.2f} %")
        b.metric("Cobertura del rango", f"{r['cobertura_rango']:.1f} %")
        c.metric("Fallas bien manejadas",
                 "—" if np.isnan(r["fallas_bien_manejadas_pct"]) else f"{r['fallas_bien_manejadas_pct']:.0f} %")
        d.metric("Recalibraciones", r["recalibraciones"])
        if tipo in ("normal", "limite"):
            umbral = ep.UMBRAL_M1[tipo]
            (st.success if r["mape"] <= umbral else st.error)(
                f"M1: MAPE {r['mape']:.2f} % frente al umbral de {umbral:g} % para escenarios "
                f"{'normales' if tipo == 'normal' else 'límite'}.")
        ruta = RAIZ / r["log"]
        datos = leer_log(ruta)
        if len(datos):
            por_mes = datos.groupby("mes").agg(cobertura=("dentro", "mean"), sesgo=("sesgo", "mean"))
            por_mes = por_mes * 100
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**Cobertura del rango por mes (%)**")
                st.line_chart(por_mes[["cobertura"]])
            with g2:
                st.markdown("**Sesgo medio por mes (%)** (positivo = el agente subestima)")
                st.line_chart(por_mes[["sesgo"]])
        with st.expander("Primeros eventos del log"):
            with open(ruta, encoding="utf-8") as f:
                st.code("".join(f.readlines()[:8]), language="json")
        st.download_button("Descargar el log completo (.jsonl)", data=ruta.read_bytes(),
                           file_name=ruta.name, key="descarga_log")

# ------------------------------------------------------------------- pestaña 3
with tab3:
    st.subheader("Resultados oficiales del plan de pruebas")
    st.write("Umbrales fijados en `tests/plan_de_pruebas.md` antes de ejecutar. "
             "12 escenarios, 200 propiedades cada uno, 5 repeticiones (65 ejecuciones).")
    ruta_tabla = RAIZ / "tests" / "resultados" / "tabla_resultados.csv"
    ruta_esc = RAIZ / "tests" / "resultados" / "resumen_por_escenario.csv"
    if ruta_tabla.exists():
        st.dataframe(pd.read_csv(ruta_tabla)[["Medida", "Umbral", "Resultado", "Cumple"]],
                     hide_index=True, width="stretch")
        st.caption("M2 no se cumplió: en E3 la cobertura fue 74,2 % frente al umbral de 75 %. "
                   "El análisis de causa está en el README del repositorio.")
    if ruta_esc.exists():
        with st.expander("Resumen por escenario (media y desviación estándar)"):
            st.dataframe(pd.read_csv(ruta_esc).round(2), hide_index=True, width="stretch")

    st.divider()
    st.subheader("Ejecutar el plan de pruebas aquí")
    st.write("Con 1 repetición y todos los escenarios son 13 ejecuciones (unos minutos). "
             "Con 5 repeticiones es la prueba completa (puede tardar 15 minutos o más en este servidor).")
    ids = [e[0] for e in ep.ESCENARIOS]
    escogidos = st.multiselect("Escenarios", ids, default=ids, key="escogidos")
    reps = st.slider("Repeticiones por escenario", 1, 5, 1, key="reps")
    con_m4 = st.checkbox("Incluir E6 sin aprendizaje (necesario para M4)", value=True, key="con_m4")
    if st.button("Ejecutar pruebas", type="primary", key="btn_pruebas"):
        plan = [(e, s, True) for e in escogidos for s in range(1, reps + 1)]
        if con_m4 and "E6" in escogidos:
            plan += [("E6", s, False) for s in range(1, reps + 1)]
        por_id = {e[0]: e for e in ep.ESCENARIOS}
        barra = st.progress(0.0, text="Preparando...")
        filas = []
        for i, (e, s, aprende) in enumerate(plan, start=1):
            _, tipo, ruido, deriva, falla = por_id[e]
            filas.append(ep.ejecutar(X_train, y_train, X_test, y_test, e, tipo, ruido,
                                     deriva, falla, s, aprende=aprende))
            barra.progress(i / len(plan), text=f"[{i}/{len(plan)}] {e} semilla {s}"
                           + ("" if aprende else " (sin aprendizaje)"))
        barra.empty()
        st.session_state.res_pruebas = pd.DataFrame(filas)
    if st.session_state.get("res_pruebas") is not None:
        res = st.session_state.res_pruebas
        st.markdown("**Medidas de desempeño con esta ejecución**")
        st.dataframe(evaluar_medidas(res), hide_index=True, width="stretch")
        st.caption("Si no ejecutaste todos los escenarios o las 5 repeticiones, el resultado es parcial "
                   "y no reemplaza al oficial.")
        st.dataframe(res.drop(columns=["log"]).round(2), hide_index=True, width="stretch")