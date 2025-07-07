# features/rutas3.py

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

from algorithms.algoritmo1 import optimizar_ruta_algoritmo22, cargar_pedidos, _crear_data_model, agrupar_puntos_aglomerativo, MARGEN #PCA-GLS
from algorithms.algoritmo2 import optimizar_ruta_cw_tabu
from algorithms.algoritmo3 import optimizar_ruta_cp_sat
from algorithms.algoritmo4 import optimizar_ruta_algoritmo4

# Coordenadas fijas de la cochera
COCHERA = {
    "lat": -16.4141434959913,
    "lon": -71.51839574233342,
    "direccion": "Cochera",
    "hora": "08:00",
}

# Cliente de Google Maps
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)


def optimizar_ruta_placeholder(data, tiempo_max_seg=60):
    """Placeholder para algoritmos no implementados aún."""
    return None


ALG_MAP = {
    "Algoritmo 1 - PCA - GLS": optimizar_ruta_algoritmo22,
    "Algoritmo 2 - Clarke Wrigth + Tabu Search": optimizar_ruta_cw_tabu,
    "Algoritmo 3 - CP - SAT": optimizar_ruta_cp_sat,
    "Algoritmo 4 - PAC + LNS": optimizar_ruta_algoritmo4,
}


def _hora_a_segundos(hhmm: str) -> int | None:
    """Convierte 'HH:MM' o 'HH:MM:SS' a segundos desde medianoche."""
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
    """Convierte segundos desde medianoche a 'HH:MM'."""
    h = segs // 3600
    m = (segs % 3600) // 60
    return f"{h:02}:{m:02}"


def _ventana_extendida(row: pd.Series) -> str:
    """Calcula la ventana time_start–time_end extendida ±MARGEN."""
    ini = _hora_a_segundos(row["time_start"])
    fin = _hora_a_segundos(row["time_end"])
    if ini is None or fin is None:
        return "No especificado"
    ini_m = max(0, ini - MARGEN)
    fin_m = min(24 * 3600, fin + MARGEN)
    return f"{_segundos_a_hora(ini_m)} - {_segundos_a_hora(fin_m)}"


def ver_ruta_optimizada():
    st.title("🚚 Ver Ruta Optimizada")
    c1, c2 = st.columns(2)
    with c1:
        fecha = st.date_input("Fecha", value=datetime.now().date())
    with c2:
        algoritmo = "Algoritmo 1 - PCA - GLS"

    if (st.session_state.get("fecha_actual") != fecha or
        st.session_state.get("algoritmo_actual") != algoritmo):
        for k in ["res", "df_clusters", "df_etiquetado", "df_final", "df_ruta", "solve_t"]:
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

        # Nodos especiales
        COCH_INI = {
            "id": "COCH_INI",
            "operacion": "Cochera",
            "nombre_cliente": "Cochera (Inicio)",
            "direccion": "Cochera",
            "lat": COCHERA["lat"],
            "lon": COCHERA["lon"],
            "time_start": COCHERA["hora"],
            "time_end": COCHERA["hora"],
            "demand": 0
        }

        PLANTA_INI = {
            "id": "PLANTA_INI",
            "operacion": "Planta",
            "nombre_cliente": "Planta (Recojo)",
            "direccion": "Planta Lavandería",
            "lat": df_clusters.iloc[0]["lat"],
            "lon": df_clusters.iloc[0]["lon"],
            "time_start": "08:00",
            "time_end": "09:00",
            "demand": 0
        }

        PLANTA_FIN = {
            "id": "PLANTA_FIN",
            "operacion": "Planta",
            "nombre_cliente": "Planta (Descarga)",
            "direccion": "Planta Lavandería",
            "lat": PLANTA_INI["lat"],
            "lon": PLANTA_INI["lon"],
            "time_start": "13:00",
            "time_end": "18:00",
            "demand": 0
        }

        COCH_FIN = {
            "id": "COCH_FIN",
            "operacion": "Cochera",
            "nombre_cliente": "Cochera (Fin)",
            "direccion": "Cochera",
            "lat": COCHERA["lat"],
            "lon": COCHERA["lon"],
            "time_start": "18:15",
            "time_end": "20:00",
            "demand": 0
        }

        # Concatenar ordenadamente
        df_final = pd.concat([
            pd.DataFrame([COCH_INI]),
            pd.DataFrame([PLANTA_INI]),
            df_clusters,
            pd.DataFrame([PLANTA_FIN]),
            pd.DataFrame([COCH_FIN])
        ], ignore_index=True)
        st.session_state["df_final"] = df_final.copy()

        # Crear modelo de datos
        data = _crear_data_model(df_final, vehiculos=1)
        st.write("🔍 Ventanas (segundos) =", data["time_windows"])

        # Ejecutar algoritmo
        t0 = tiempo.time()
        res = optimizar_ruta_algoritmo22(data, tiempo_max_seg=60)
        st.session_state["solve_t"] = tiempo.time() - t0

        if not res:
            st.error("❌ No se encontró solución con OR-Tools.")
            st.write("🔍 Ventanas de tiempo por nodo:\n")
            for i, (ini, fin) in enumerate(data["time_windows"]):
                st.write(f"Nodo {i:<2} → {_segundos_a_hora(ini)} - {_segundos_a_hora(fin)}")
            st.write("📦 Demandas por nodo:\n")
            for i, d in enumerate(data["demands"]):
                st.write(f"Nodo {i:<2}: demanda = {d}")
            return

        st.session_state["res"] = res

        def _seg_a_hhmm(segs: int) -> str:
            h = segs // 3600
            m = (segs % 3600) // 60
            return f"{h:02}:{m:02}"

        ruta = res["routes"][0]["route"]
        arr = res["routes"][0]["arrival_sec"]
        df_r = df_final.loc[ruta, ["nombre_cliente", "direccion", "time_start", "time_end"]].copy()
        df_r["ventana_con_margen"] = df_r.apply(_ventana_extendida, axis=1)
        df_r["ETA"] = [_seg_a_hhmm(t) for t in arr]
        df_r["orden"] = range(len(ruta))
        st.session_state["df_ruta"] = df_r.copy()

    # Mostrar tabla final
    df_r = st.session_state["df_ruta"]
    st.subheader("📋 Orden de visita optimizada")
    st.dataframe(df_r[["orden", "nombre_cliente", "direccion", "ventana_con_margen", "ETA"]], use_container_width=True)

    tiempo_total_min = (max(st.session_state["res"]["routes"][0]["arrival_sec"]) - 8 * 3600) / 60
    st.markdown(f"🕒 Tiempo estimado total: **{tiempo_total_min:.2f} minutos**")
    st.markdown(f"🧭 Puntos visitados: **{len(st.session_state['res']['routes'][0]['route'])}**")
    st.markdown(f"⚙️ Tiempo de cómputo: **{st.session_state['solve_t']:.2f} s**")

