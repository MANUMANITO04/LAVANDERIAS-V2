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
        algoritmo = st.selectbox("Algoritmo", list(ALG_MAP.keys()))

    # Reset state (Si la fecha es cambiada o el algoritmo diferente)
    if (st.session_state.get("fecha_actual") != fecha or
        st.session_state.get("algoritmo_actual") != algoritmo):
        for k in ["res","df_clusters","df_etiquetado","df_final","df_ruta","solve_t"]:
            st.session_state[k] = None
        st.session_state["leg_0"] = 0
        st.session_state["fecha_actual"] = fecha
        st.session_state["algoritmo_actual"] = algoritmo

    # Compute route if not yet done
    if st.session_state["res"] is None:
        pedidos = cargar_pedidos(fecha, "Todos")
        if not pedidos:
            st.info("No hay pedidos para esa fecha.")
            return

        df_original = pd.DataFrame(pedidos)
        df_clusters, df_et = agrupar_puntos_aglomerativo(df_original, eps_metros=5)
        st.session_state["df_clusters"] = df_clusters.copy()
        st.session_state["df_etiquetado"] = df_et.copy()

        # Depósito (Planta)
        DEP = {
            "id": "DEP",
            "operacion": "Depósito",
            "nombre_cliente": "Depósito",
            "direccion": "Planta Lavandería",
            "lat": -16.40904,
            "lon": -71.53745,
            "time_start": "08:00",
            "time_end": "18:00",
            "demand": 0
        }
        df_final = pd.concat([pd.DataFrame([DEP]), df_clusters], ignore_index=True)
        st.session_state["df_final"] = df_final.copy()

        # — DEBUG: imprime las ventanas de tiempo —
        data = _crear_data_model(df_final, vehiculos=1)
        st.write("🔍 Ventanas (segundos) =", data["time_windows"])

        # Resolver VRP
        alg_fn = ALG_MAP[algoritmo]
        t0 = tiempo.time()
        res = alg_fn(data, tiempo_max_seg=45)
        st.session_state["solve_t"] = tiempo.time() - t0

        if not res:
            st.error("😕 Sin solución factible.")
            return

        st.session_state["res"] = res

        # — DEBUG: imprime las ETAs en segundos —
        st.write("🔍 ETA de llegada (segundos) =", res["routes"][0]["arrival_sec"])

        # — DEBUG: tabla de verificación —
        def _seg_a_hhmm(segs: int) -> str:
            h = segs // 3600
            m = (segs % 3600) // 60
            return f"{h:02}:{m:02}"
        df_check = pd.DataFrame({
            "nodo": list(range(len(data["time_windows"]))),
            "ventana_inicio": [ _seg_a_hhmm(w[0]) for w in data["time_windows"] ],
            "ventana_fin":    [ _seg_a_hhmm(w[1]) for w in data["time_windows"] ],
            "arrival":        [ _seg_a_hhmm(t)    for t in res["routes"][0]["arrival_sec"] ],
        })
        st.table(df_check)


        
        # Construir df_r con ventana y ETA
        ruta = res["routes"][0]["route"]
        arr  = res["routes"][0]["arrival_sec"]
        df_r = df_final.loc[ruta, ["nombre_cliente","direccion","time_start","time_end"]].copy()
        df_r["ventana_con_margen"] = df_r.apply(_ventana_extendida, axis=1)
        df_r["ETA"]   = [ _seg_a_hhmm(t) for t in arr ]
        df_r["orden"] = range(len(ruta))
        st.session_state["df_ruta"] = df_r.copy()

    # Construir tabla de orden de visita (Cochera antes del Depósito)
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
    dep = df_r[df_r["orden"] == 0].iloc[0]
    filas.append({
        "orden": 1,
        "nombre_cliente": dep["nombre_cliente"],
        "direccion": dep["direccion"],
        "ventana_con_margen": dep["ventana_con_margen"],
        "ETA": dep["ETA"]
    })
    for _, row in df_r[df_r["orden"] >= 1].sort_values("orden").iterrows():
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

    # — Pestañas —
    tab1, tab2 = st.tabs(["🚀 Tramo actual","ℹ️ Info general"])
    df_f    = st.session_state["df_final"]
    df_et   = st.session_state["df_etiquetado"]
    res     = st.session_state["res"]
    ruta    = res["routes"][0]["route"]
    leg     = st.session_state["leg_0"]
    L       = len(ruta)

    # Tramo actual
    with tab1:
        total_legs = L + 1
        if leg > total_legs:
            st.success("✅ Ruta completada")
            return

        if leg == 0:
            orig = (COCHERA["lat"], COCHERA["lon"])
            dest_idx = ruta[0]
            dest = (df_f.loc[dest_idx,"lat"], df_f.loc[dest_idx,"lon"])
            nombre_dest = df_f.loc[dest_idx,"nombre_cliente"]
            ETA_dest = df_display.loc[df_display["orden"] == 1, "ETA"].iloc[0]
        elif 1 <= leg < L:
            idx_o = ruta[leg - 1]
            idx_d = ruta[leg]
            orig = (df_f.loc[idx_o,"lat"], df_f.loc[idx_o,"lon"])
            dest = (df_f.loc[idx_d,"lat"], df_f.loc[idx_d,"lon"])
            nombre_dest = df_f.loc[idx_d,"nombre_cliente"]
            ETA_dest = df_display.loc[df_display["orden"] == leg + 1, "ETA"].iloc[0]
        """else:
            idx_o = ruta[L - 1]
            orig = (df_f.loc[idx_o,"lat"], df_f.loc[idx_o,"lon"])
            dest = (COCHERA["lat"], COCHERA["lon"])
            nombre_dest = COCHERA["direccion"]
            ETA_dest = "—""""
        else:
            # Buscar índices del nodo "Depósito" (puede aparecer 2 veces: recojo y descarga)
            planta_idxs = [i for i in reversed(ruta) if df_f.loc[i, "nombre_cliente"] == "Depósito"]
            planta_descarga_idx = planta_idxs[1] if len(planta_idxs) >= 2 else planta_idxs[0] if planta_idxs else None
    
            if leg == L:
                # Último cliente → Planta (descarga)
                idx_o = ruta[L - 1]
                idx_d = planta_descarga_idx
                orig = (df_f.loc[idx_o, "lat"], df_f.loc[idx_o, "lon"])
                dest = (df_f.loc[idx_d, "lat"], df_f.loc[idx_d, "lon"])
                nombre_dest = "Depósito (Descarga)"
                ETA_dest = df_display[df_display["nombre_cliente"] == "Depósito"].iloc[-1]["ETA"]
            else:
                # Planta (descarga) → Cochera
                idx_o = planta_descarga_idx
                orig = (df_f.loc[idx_o, "lat"], df_f.loc[idx_o, "lon"])
                dest = (COCHERA["lat"], COCHERA["lon"])
                nombre_dest = COCHERA["direccion"]
                ETA_dest = "—"


        st.markdown(
            f"### Próximo → **{nombre_dest}**  \n"
            f"📍 {dest[0]:.6f},{dest[1]:.6f} (ETA {ETA_dest})",
            unsafe_allow_html=True
        )
        if st.button(f"✅ Llegué a {nombre_dest}"):
            st.session_state["leg_0"] += 1
            st.rerun()

        try:
            directions = gmaps.directions(
                f"{orig[0]},{orig[1]}",
                f"{dest[0]},{dest[1]}",
                mode="driving",
                departure_time=datetime.now(),
                traffic_model="best_guess"
            )
            leg0 = directions[0]["legs"][0]
            tiempo_traffic = leg0.get("duration_in_traffic", leg0["duration"])["text"]
            overview = directions[0]["overview_polyline"]["points"]
            segmento = [(p["lat"], p["lng"]) for p in decode_polyline(overview)]
        except:
            tiempo_traffic = None
            segmento = [orig, dest]

        m = folium.Map(location=segmento[0], zoom_start=14)
        folium.PolyLine(
            segmento,
            weight=5, opacity=0.8,
            tooltip=f"⏱ {tiempo_traffic}" if tiempo_traffic else None
        ).add_to(m)
        folium.Marker(segmento[0], icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(m)
        folium.Marker(segmento[-1], icon=folium.Icon(color="blue", icon="flag", prefix="fa")).add_to(m)
        st_folium(m, width=700, height=400)

    # Info general con API y métricas
    with tab2:
        st.subheader("🗺️ Mapa de toda la ruta (via API)")
        origin = f"{COCHERA['lat']},{COCHERA['lon']}"
        depot_idx = ruta[0]
        depot = f"{df_f.loc[depot_idx,'lat']},{df_f.loc[depot_idx,'lon']}"
        waypoints = [depot] + [f"{df_f.loc[i,'lat']},{df_f.loc[i,'lon']}" for i in ruta[1:]] + [depot]
        destination = origin

        directions = gmaps.directions(
            origin,
            destination,
            mode="driving",
            departure_time=datetime.now(),
            optimize_waypoints=False,
            waypoints=waypoints
        )

        overview = directions[0]["overview_polyline"]["points"]
        path = [(p["lat"], p["lng"]) for p in decode_polyline(overview)]

        total_m = sum(leg["distance"]["value"] for leg in directions[0]["legs"])
        total_s = sum(leg["duration"]["value"] for leg in directions[0]["legs"])

        m = folium.Map(location=path[0], zoom_start=13)
        folium.PolyLine(path, weight=4, opacity=0.7).add_to(m)

        # Marcadores con tooltip
        folium.Marker(
            (COCHERA["lat"],COCHERA["lon"]),
            popup="Cochera", tooltip="Cochera",
            icon=folium.Icon(color="purple",icon="building",prefix="fa")
        ).add_to(m)
        folium.Marker(
            (df_f.loc[ruta[0],"lat"],df_f.loc[ruta[0],"lon"]),
            popup="Planta Lavandería", tooltip="Depósito",
            icon=folium.Icon(color="green",icon="home",prefix="fa")
        ).add_to(m)
        for idx in ruta[1:]:
            lat, lon = df_f.loc[idx,["lat","lon"]]
            nombre = df_f.loc[idx,"nombre_cliente"]
            direccion = df_f.loc[idx,"direccion"]
            folium.Marker(
                (lat,lon),
                popup=f"{nombre}<br>{direccion}",
                tooltip=nombre,
                icon=folium.Icon(color="orange",icon="flag",prefix="fa")
            ).add_to(m)
        folium.Marker(
            (COCHERA["lat"],COCHERA["lon"]),
            popup="Cochera", tooltip="Cochera",
            icon=folium.Icon(color="purple",icon="building",prefix="fa")
        ).add_to(m)

        for _, row in df_et.iterrows():
            folium.CircleMarker(
                (row["lat"],row["lon"]),
                radius=4, color="red", fill=True, fill_opacity=0.7
            ).add_to(m)

        st_folium(m, width=700, height=500)

        # Métricas de la ruta real
        st.markdown("## 🔍 Métricas de la ruta real")
        st.markdown(f"- Distancia total (Driving): **{total_m/1000:.2f} km**")
        st.markdown(f"- Duración estimada (Driving): **{total_s//60:.0f} min**")

        # Métricas finales del VRP
        st.markdown("## 🔍 Métricas Finales")
        st.markdown(f"- Kilometraje total: **{res['distance_total_m']/1000:.2f} km**")
        st.markdown(f"- Tiempo de cómputo: **{st.session_state['solve_t']:.2f} s**")
        tiempo_total_min = (max(res["routes"][0]["arrival_sec"]) - 9*3600) / 60
        st.markdown(f"- Tiempo estimado total: **{tiempo_total_min:.2f} min**")
        st.markdown(f"- Puntos visitados: **{len(ruta)}**")
