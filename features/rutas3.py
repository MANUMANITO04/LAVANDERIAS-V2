import streamlit as st
import pandas as pd
from datetime import datetime
import time as tiempo

import firebase_admin
from firebase_admin import credentials, firestore

import googlemaps
from googlemaps.convert import decode_polyline

import folium
from streamlit_folium import st_folium

from core.firebase import db
from core.constants import GOOGLE_MAPS_API_KEY

from algorithms.algoritmo1 import optimizar_ruta_algoritmo22, cargar_pedidos, _crear_data_model, agrupar_puntos_aglomerativo, MARGEN
from algorithms.algoritmo2 import optimizar_ruta_cw_tabu
from algorithms.algoritmo3 import optimizar_ruta_cp_sat
from algorithms.algoritmo4 import optimizar_ruta_algoritmo4

COCHERA = {
    "lat": -16.4141434959913,
    "lon": -71.51839574233342,
    "direccion": "Cochera",
    "hora": "08:00",
}

gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

ALG_MAP = {
    "Algoritmo 1 - PCA - GLS": optimizar_ruta_algoritmo22,
    "Algoritmo 2 - Clarke Wrigth + Tabu Search": optimizar_ruta_cw_tabu,
    "Algoritmo 3 - CP - SAT": optimizar_ruta_cp_sat,
    "Algoritmo 4 - PAC + LNS": optimizar_ruta_algoritmo4,
}

def _hora_a_segundos(hhmm: str) -> int | None:
    if not isinstance(hhmm, str):
        return None
    parts = hhmm.split(":")
    if len(parts) < 2:
        return None
    try:
        h = int(parts[0])
        m = int(parts[1])
        return h * 3600 + m * 60
    except:
        return None

def _segundos_a_hora(segs: int) -> str:
    h = segs // 3600
    m = (segs % 3600) // 60
    return f"{h:02}:{m:02}"

def _ventana_extendida(row: pd.Series) -> str:
    ini = _hora_a_segundos(row["time_start"])
    fin = _hora_a_segundos(row["time_end"])
    if ini is None or fin is None:
        return "No especificado"
    ini_m = max(0, ini - MARGEN)
    fin_m = min(24 * 3600, fin + MARGEN)
    return f"{_segundos_a_hora(ini_m)} - {_segundos_a_hora(fin_m)}"

# ---- FUNCION PRINCIPAL ----
def ver_ruta_optimizada():
    st.title("🚚 Ver Ruta Optimizada")
    c1, c2 = st.columns(2)
    with c1:
        fecha = st.date_input("Fecha", value=datetime.now().date())
    with c2:
        algoritmo = st.selectbox("Algoritmo", list(ALG_MAP.keys()))

    if (st.session_state.get("fecha_actual") != fecha or
        st.session_state.get("algoritmo_actual") != algoritmo):
        for k in ["res","df_clusters","df_etiquetado","df_final","df_ruta","solve_t"]:
            st.session_state[k] = None
        st.session_state["leg_0"] = 0
        st.session_state["fecha_actual"] = fecha
        st.session_state["algoritmo_actual"] = algoritmo

    if st.session_state["res"] is None:
        pedidos = cargar_pedidos(fecha, "Todos")
        if not pedidos:
            st.info("No hay pedidos para esa fecha.")
            return

        df_original = pd.DataFrame(pedidos)
        df_clusters, df_et = agrupar_puntos_aglomerativo(df_original, eps_metros=5)
        st.session_state["df_clusters"] = df_clusters.copy()
        st.session_state["df_etiquetado"] = df_et.copy()

        df_final = df_clusters.copy()
        st.session_state["df_final"] = df_final.copy()

        data = _crear_data_model(df_final, vehiculos=1)

        alg_fn = ALG_MAP[algoritmo]
        t0 = tiempo.time()
        res = alg_fn(data, tiempo_max_seg=120)
        st.session_state["solve_t"] = tiempo.time() - t0

        if not res:
            st.error("😕 Sin solución factible.")
            return

        st.session_state["res"] = res

        ruta = res["routes"][0]["route"]
        arr  = res["routes"][0]["arrival_sec"]
        df_r = df_final.loc[ruta, ["nombre_cliente","direccion","time_start","time_end"]].copy()
        df_r["ventana_con_margen"] = df_r.apply(_ventana_extendida, axis=1)
        df_r["ETA"]   = [ _segundos_a_hora(t) for t in arr ]
        df_r["orden"] = range(len(ruta))
        st.session_state["df_ruta"] = df_r.copy()

    df_r = st.session_state["df_ruta"]
    filas = []
    vent_coch = _ventana_extendida(pd.Series({
        "time_start": COCHERA["hora"],
        "time_end":   COCHERA["hora"]
    }))
    filas.append({
        "orden": 0,
        "nombre_cliente": COCHERA["direccion"],
        "direccion": COCHERA["direccion"],
        "ventana_con_margen": vent_coch,
        "ETA": COCHERA["hora"]
    })
    for _, row in df_r.sort_values("orden").iterrows():
        filas.append({
            "orden": int(row["orden"]) + 1,
            "nombre_cliente": row["nombre_cliente"],
            "direccion": row["direccion"],
            "ventana_con_margen": row["ventana_con_margen"],
            "ETA": row["ETA"]
        })
    df_display = pd.DataFrame(filas).sort_values("orden").reset_index(drop=True)
    st.subheader("📋 Orden de visita optimizada")
    st.dataframe(df_display, use_container_width=True)
