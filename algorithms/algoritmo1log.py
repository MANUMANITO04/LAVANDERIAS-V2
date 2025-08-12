#################################################################################################################
# :) –  Streamlit App Integrado:
# → GLS + PCI + OR-Tools
# → Firebase Firestore (usando service account JSON)
# → Google Maps Distance Matrix & Directions
# → OR-Tools VRP-TW con servicio, ventanas, tráfico real
# → Se empleó el algoritmo de agrupación: Agglomerative Clustering para agrupar pedidos cercanos.
# → Página única: Ver Ruta Optimizada
# → En caso el algoritmo no dé respuesta, usa distancias euclidianas
##################################################################################################################

import os
import math
import time as tiempo
from datetime import datetime
import logging

import streamlit as st
import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
import googlemaps
from googlemaps.convert import decode_polyline
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from sklearn.cluster import AgglomerativeClustering
import folium
from streamlit_folium import st_folium

# -------------------- INICIALIZAR FIREBASE --------------------
# Usa el JSON de servicio: 'lavanderia_key.json'
# if not firebase_admin._apps:
#     cred = credentials.Certificate("lavanderia_key.json")
#     firebase_admin.initialize_app(cred)
db = firestore.client()

# -------------------- CONFIG GOOGLE MAPS --------------------
GOOGLE_MAPS_API_KEY = st.secrets.get("google_maps", {}).get("api_key") or os.getenv("GOOGLE_MAPS_API_KEY")
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

# -------------------- CONSTANTES VRP --------------------
SERVICE_TIME    = 8 * 60         # 8 minutos de servicio en cada parada (excepto depósito)
MAX_ELEMENTS    = 100            # límite de celdas por petición Distance Matrix API
SHIFT_START_SEC = 8 * 3600 + 30 * 60   # 08:30 en segundos
SHIFT_END_SEC   = 17 * 3600           # 17:00 en segundos
MARGEN          = 15 * 60        # 15 minutos en segundos

# ===================== FUNCIONES AUXILIARES =====================
def _hora_a_segundos(hhmm):
    """Convierte 'HH:MM' o 'HH:MM:SS' a segundos desde medianoche."""
    if hhmm is None or pd.isna(hhmm) or hhmm == "":
        return None
    try:
        parts = str(hhmm).split(":")
        h = int(parts[0])
        m = int(parts[1])
        return h * 3600 + m * 60
    except Exception:
        return None

def _haversine_dist_dur(coords, vel_kmh=40.0):
    """
    Calcula matrices de distancias (en metros) y duraciones (en segundos)
    basadas en fórmula de Haversine asumiendo velocidad vel_kmh para la duración.
    coords = [(lat1, lon1), (lat2, lon2), ...]
    """
    R = 6371e3  # radio terrestre en metros
    n = len(coords)
    dist = [[0] * n for _ in range(n)]
    dur = [[0] * n for _ in range(n)]
    v_ms = vel_kmh * 1000 / 3600  # convertir km/h a m/s
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            lat1, lon1 = map(math.radians, coords[i])
            lat2, lon2 = map(math.radians, coords[j])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            d = 2 * R * math.asin(math.sqrt(a))
            dist[i][j] = int(d)
            dur[i][j] = int(d / v_ms)
    return dist, dur

@st.cache_data(ttl=3600, show_spinner=False)
def _distancia_duracion_matrix(coords):
    """
    Llama a la Distance Matrix API de Google Maps para obtener distancias (m) y duraciones (s)
    entre cada par de coords = [(lat, lon), ...].
    Si falta clave de la API del archivo st.secrets, usa la aproximación Haversine.
    """
    if not GOOGLE_MAPS_API_KEY:
        return _haversine_dist_dur(coords)
    n = len(coords)
    dist = [[0] * n for _ in range(n)]
    dur = [[0] * n for _ in range(n)]
    # Dividimos en lotes para no exceder MAX_ELEMENTS celdas
    batch = max(1, min(n, MAX_ELEMENTS // n))
    for i0 in range(0, n, batch):
        resp = gmaps.distance_matrix(
            origins=coords[i0:i0 + batch],
            destinations=coords,
            mode="driving",
            units="metric",
            departure_time=datetime.now(),
            traffic_model="best_guess"
        )
        for i, row in enumerate(resp["rows"]):
            for j, el in enumerate(row["elements"]):
                dist[i0 + i][j] = el.get("distance", {}).get("value", 1)
                dur[i0 + i][j] = el.get("duration_in_traffic", {}).get(
                    "value",
                    el.get("duration", {}).get("value", 1)
                )
    return dist, dur

def _crear_data_model(df, vehiculos=1, capacidad_veh=None):
    coords = list(zip(df["lat"], df["lon"]))
    dist_m, dur_s = _distancia_duracion_matrix(coords)

    time_windows = []
    demandas = []
    service_times = []

    for _, row in df.iterrows():
        ini = _hora_a_segundos(row.get("time_start"))
        fin = _hora_a_segundos(row.get("time_end"))
        if ini is None or fin is None:
            ini, fin = SHIFT_START_SEC, SHIFT_END_SEC
        else:
            ini = max(0, ini - MARGEN)
            fin = min(24 * 3600, fin + MARGEN)
        time_windows.append((ini, fin))
        demandas.append(row.get("demand", 1))

        tipo = row.get("tipo", "").strip()
        if tipo == "Sucursal":
            service_times.append(10 * 60)  # 10 minutos
        elif tipo == "Planta":
            service_times.append(10 * 60)  # 10 minutos
        else:
            service_times.append(10 * 60)  # Cliente Delivery o indefinido

    return {
        "distance_matrix": dist_m,
        "duration_matrix": dur_s,
        "time_windows": time_windows,
        "demands": demandas,
        "num_vehicles": vehiculos,
        "vehicle_capacities": [capacidad_veh or 10**9] * vehiculos,
        "depot": 0,
        "service_times": service_times
    }

def optimizar_ruta_algoritmo22(data, tiempo_max_seg=60, reintento=False):
    """
    VRPTW con OR-Tools (Routing Solver):
    - El tiempo de servicio se suma en el callback (en el nodo origen).
    - La salida del depósito se fija a SHIFT_START_SEC (no se usa ventana del depósito en el bucle).
    - Si falla, reintenta ampliando ventanas cortas.
    """
    # --- Manager & Model
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]),
        data["num_vehicles"],
        data["depot"]
    )
    routing = pywrapcp.RoutingModel(manager)

    depot_node   = data["depot"]
    durations    = data["duration_matrix"]
    service_times = data.get("service_times", [0] * len(data["time_windows"]))

    # --- Callback: viaje + servicio del nodo 'from' (excepto depósito)
    def time_cb(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        travel  = int(durations[i][j])
        service = 0 if i == depot_node else int(service_times[i])
        return travel + service

    transit_cb_idx = routing.RegisterTransitCallback(time_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

    # --- Dimensión de tiempo
    routing.AddDimension(
        transit_cb_idx,
        24 * 3600,    # waiting permitido
        24 * 3600,    # máximo por vehículo
        False,
        "Time"
    )
    time_dim = routing.GetDimensionOrDie("Time")
    time_dim.SetGlobalSpanCostCoefficient(1000)

    # --- Ventanas de tiempo: todos EXCEPTO el depósito
    for node, (ini, fin) in enumerate(data["time_windows"]):
        if node == depot_node:
            continue
        idx = manager.NodeToIndex(node)
        time_dim.CumulVar(idx).SetRange(int(ini), int(fin))

    # --- Depósito: salida fija a SHIFT_START_SEC
    depot_idx = manager.NodeToIndex(depot_node)
    time_dim.CumulVar(depot_idx).SetRange(int(SHIFT_START_SEC), int(SHIFT_START_SEC))

    # --- Capacidad (si aplica)
    if any(data["demands"]):
        def demand_cb(from_index):
            return int(data["demands"][manager.IndexToNode(from_index)])
        demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_cb)
        routing.AddDimensionWithVehicleCapacity(
            demand_cb_idx, 0, data["vehicle_capacities"], True, "Capacity"
        )

    # --- Búsqueda
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.time_limit.FromSeconds(tiempo_max_seg)
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH

    sol = routing.SolveWithParameters(params)

    # --- Sin solución: diagnóstico + reintento
    if not sol:
        st.warning("❌ No se encontró solución con OR-Tools.")
        st.info("🔍 Ventanas de tiempo por nodo:")
        ventanas_cortas = []
        for node, (ini, fin) in enumerate(data["time_windows"]):
            dur   = fin - ini
            h_ini = f"{ini // 3600:02}:{(ini % 3600) // 60:02}"
            h_fin = f"{fin // 3600:02}:{(fin % 3600) // 60:02}"
            label = "[DEPÓSITO]" if node == depot_node else f"Nodo {node}"
            if node != depot_node and dur < 45 * 60:
                st.error(f"⚠️ {label:12} → {h_ini} - {h_fin}  (solo {dur // 60} min)")
                ventanas_cortas.append(node)
            else:
                st.text(f"{label:12} → {h_ini} - {h_fin}")

        st.info("📦 Demandas por nodo:")
        for i, d in enumerate(data["demands"]):
            st.text(f"Nodo {i}: demanda = {d}")

        if not reintento and ventanas_cortas:
            st.warning("🔄 Reintentando con ventanas ampliadas...")
            nueva_data = data.copy()
            nuevas_ventanas = []
            for i, (ini, fin) in enumerate(data["time_windows"]):
                if i in ventanas_cortas:
                    c = (ini + fin) // 2
                    nuevas_ventanas.append((max(0, c - 3600), min(86400, c + 3600)))
                else:
                    nuevas_ventanas.append((ini, fin))
            nueva_data["time_windows"] = nuevas_ventanas
            return optimizar_ruta_algoritmo22(nueva_data, tiempo_max_seg, reintento=True)

        st.error("😕 Sin solución factible incluso tras reintentar.")
        return None

    # --- Extraer solución (ETA = inicio servicio). Omitimos depósito en la ruta
    rutas = []
    dist_total_m = 0
    for v in range(data["num_vehicles"]):
        idx = routing.Start(v)
        route, arrival_sec, departure_sec = [], [], []
        while not routing.IsEnd(idx):
            n   = manager.IndexToNode(idx)
            nxt = sol.Value(routing.NextVar(idx))
            dest = manager.IndexToNode(nxt)

            dist_total_m += data["distance_matrix"][n][dest]

            eta = int(sol.Min(time_dim.CumulVar(idx)))
            srv = 0 if n == depot_node else int(service_times[n])

            if n != depot_node:
                route.append(n)
                arrival_sec.append(eta)
                departure_sec.append(eta + srv)

            idx = nxt

        rutas.append({
            "vehicle": v,
            "route": route,
            "arrival_sec": arrival_sec,
            "departure_sec": departure_sec,
        })

    st.success("✅ Ruta encontrada con éxito.")
    return {"routes": rutas, "distance_total_m": dist_total_m}


def agregar_ventana_margen(df, margen_segundos=15 * 60):
    def expandir_fila(row):
        ini = _hora_a_segundos(row["time_start"])
        fin = _hora_a_segundos(row["time_end"])
        if ini is None or fin is None:
            return "No especificado"
        ini = max(0, ini - margen_segundos)
        fin = min(24 * 3600, fin + margen_segundos)
        h_ini = f"{ini // 3600:02}:{(ini % 3600) // 60:02}"
        h_fin = f"{fin // 3600:02}:{(fin % 3600) // 60:02}"
        return f"{h_ini} - {h_fin} h"

    df["ventana_con_margen"] = df.apply(expandir_fila, axis=1)
    return df

# ===================== FUNCIONES PARA CLUSTERING =====================
def _haversine_meters(lat1, lon1, lat2, lon2):
    """Retorna distancia en metros entre dos puntos (lat, lon) usando Haversine."""
    R = 6371e3  # metros
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def agrupar_puntos_aglomerativo(df, eps_metros=5):
    """
    Agrupa pedidos cercanos mediante AgglomerativeClustering. 
    eps_metros: umbral de distancia máxima en metros para que queden en el mismo cluster.
    Retorna (df_clusters, df_etiquetado), donde:
      - df_etiquetado = df original con columna 'cluster' indicando etiqueta de cluster.
      - df_clusters  = DataFrame de centroides con columnas ['id','operacion','nombre_cliente',
                        'dirección','lat','lon','time_start','time_end','demand'].
    """
    # Si no hay pedidos, retorno vacíos
    if df.empty:
        return pd.DataFrame(), df.copy()

    coords = df[["lat", "lon"]].to_numpy()
    n = len(coords)
    # 1) Construir matriz de distancias en metros
    dist_m = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                dist_m[i, j] = 0.0
            else:
                dist_m[i, j] = _haversine_meters(
                    coords[i, 0], coords[i, 1],
                    coords[j, 0], coords[j, 1]
                )

    # 2) Aplicar AgglomerativeClustering con distancia precomputada
    clustering = AgglomerativeClustering(
        n_clusters=None,  # 1
        metric="precomputed",
        linkage="average",
        distance_threshold=eps_metros
    )
    labels = clustering.fit_predict(dist_m)
    df_labeled = df.copy()
    df_labeled["cluster"] = labels

    # 3) Construir DataFrame de centroides
    agrupados = []
    for clus in sorted(np.unique(labels)):
        members = df_labeled[df_labeled["cluster"] == clus]
        centro_lat = members["lat"].mean()
        centro_lon = members["lon"].mean()
        # Nombre descriptivo: primeros dos clientes del cluster
        nombres = list(members["nombre_cliente"].unique())
        nombre_desc = ", ".join(nombres[:2]) + ("..." if len(nombres) > 2 else "")
        # Concatenar direcciones de los pedidos (hasta 2)
        direcciones = list(members["direccion"].unique())
        direccion_desc = ", ".join(direcciones[:2]) + ("..." if len(direcciones) > 2 else "")
        # Ventana: tomo min y max de time_start/time_end
        ts_vals = members["time_start"].tolist()
        te_vals = members["time_end"].tolist()
        ts_vals = [t for t in ts_vals if t]
        te_vals = [t for t in te_vals if t]
        inicio_cluster = min(ts_vals) if ts_vals else ""
        fin_cluster = max(te_vals) if te_vals else ""
        demanda_total = int(members["demand"].sum())
        agrupados.append({
            "id":             f"cluster_{clus}",
            "operacion":      "Agrupado",
            # "nombre_cliente": f"Grupo {clus}: {nombre_desc}",
            "nombre_cliente": nombre_desc,
            "direccion":      direccion_desc,
            "lat":            centro_lat,
            "lon":            centro_lon,
            "time_start":     inicio_cluster,
            "time_end":       fin_cluster,
            "demand":         demanda_total
        })

    df_clusters = pd.DataFrame(agrupados)
    return df_clusters, df_labeled

# ===================== CARGAR PEDIDOS DESDE FIRESTORE =====================
@st.cache_data(ttl=300)
def cargar_pedidos(fecha, tipo):
    """
    Lee de Firestore la colección 'recogidas' filtrando por fecha (recojo/entrega)
    y tipo de servicio. Soporta dos esquemas de datos:

      Esquema unificado:
        - direccion
        - coordenadas: {lat, lon}
        - fecha
        - hora   (opcional)

      Esquema histórico (doble):
        - direccion_recojo / direccion_entrega
        - coordenadas_recojo / coordenadas_entrega: {lat, lon}
        - fecha_recojo / fecha_entrega
        - hora_recojo / hora_entrega  (opcional)

    Retorna una lista de dicts con:
      - id, operacion, nombre_cliente, direccion, lat, lon, time_start, time_end, demand, tipo
    """
    def _to_float(x):
        try:
            return float(x) if x is not None else None
        except Exception:
            return None

    def _pick_dir_coords(data, is_recojo):
        """
        Selecciona dirección y coordenadas:
        1) Prioriza esquema unificado (direccion, coordenadas{lat,lon})
        2) Si falta algo, usa lado correspondiente (recojo/entrega)
        3) Si aún falta, intenta nuevamente por compatibilidad)
        """
        # 1) Unificado
        direccion = data.get("direccion")
        coords_uni = data.get("coordenadas") or {}
        lat = _to_float(coords_uni.get("lat"))
        lon = _to_float(coords_uni.get("lon"))

        # 2) Lado principal (recojo/entrega)
        lado = "recojo" if is_recojo else "entrega"
        if not direccion:
            direccion = data.get(f"direccion_{lado}")
        if lat is None or lon is None:
            coords_lado = data.get(f"coordenadas_{lado}") or {}
            lat = _to_float(lat if lat is not None else coords_lado.get("lat"))
            lon = _to_float(lon if lon is not None else coords_lado.get("lon"))

        # 3) Lado opuesto como último recurso
        if (lat is None or lon is None) or not direccion:
            opuesto = "entrega" if is_recojo else "recojo"
            direccion = direccion or data.get(f"direccion_{opuesto}")
            coords_op = data.get(f"coordenadas_{opuesto}") or {}
            lat = _to_float(lat if lat is not None else coords_op.get("lat"))
            lon = _to_float(lon if lon is not None else coords_op.get("lon"))

        return direccion or "", lat, lon

    def _pick_hora(data, is_recojo):
        """
        Elige hora de servicio:
        1) hora_recojo / hora_entrega si existen
        2) 'hora' unificada si existe
        3) default 09:00–16:00
        """
        lado = "recojo" if is_recojo else "entrega"
        h = data.get(f"hora_{lado}") or data.get("hora")
        if h:
            return h, h
        return "08:30", "17:00"

    col = db.collection('recogidas')
    docs = []
    # Coincidencias por fecha en ambos campos del esquema histórico
    hoy_str = fecha.strftime("%Y-%m-%d")
    docs += col.where("fecha_recojo", "==", hoy_str).stream()
    docs += col.where("fecha_entrega", "==", hoy_str).stream()
    # También considerar el esquema unificado (si hubiera 'fecha')
    try:
        docs += col.where("fecha", "==", hoy_str).stream()
    except Exception:
        # Si el índice/propiedad no existe, seguimos sin romper
        pass

    # Filtrar por tipo si corresponde
    if tipo != "Todos":
        tf = "Sucursal" if tipo == "Sucursal" else "Cliente Delivery"
        docs = [d for d in docs if (d.to_dict().get("tipo_solicitud") or "").strip() == tf]

    out = []
    seen_ids = set()  # evitar duplicados si un doc matchea varias consultas
    for d in docs:
        if d.id in seen_ids:
            continue
        data = d.to_dict()
        seen_ids.add(d.id)

        # Determinar si el registro aplica como Recojo o Entrega hoy.
        # Regla:
        #   - Si coincide fecha_recojo -> Recojo
        #   - elif coincide fecha_entrega -> Entrega
        #   - elif coincide 'fecha' unificada -> Recojo (por defecto)
        is_recojo = False
        if data.get("fecha_recojo") == hoy_str:
            is_recojo = True
        elif data.get("fecha_entrega") == hoy_str:
            is_recojo = False
        elif data.get("fecha") == hoy_str:
            is_recojo = True  # por defecto tratamos como "Recojo"

        op = "Recojo" if is_recojo else "Entrega"

        # Dirección y coordenadas 
        direccion, lat, lon = _pick_dir_coords(data, is_recojo)

        # Nombre (cliente o sucursal)
        nombre = data.get("nombre_cliente") or data.get("sucursal") or "Sin nombre"

        # Hora de servicio (start/end)
        ts, te = _pick_hora(data, is_recojo)

        # Si no hay coords válidas
        if lat is None or lon is None:
            # st.warning(f"[{d.id}] sin coordenadas válidas. Se omite.")
            continue

        out.append({
            "id":             d.id,
            "operacion":      op,
            "nombre_cliente": nombre,
            "direccion":      direccion or "",
            "lat":            lat,
            "lon":            lon,
            "time_start":     ts,
            "time_end":       te,
            "demand":         1,
            "tipo":           (data.get("tipo_solicitud") or "").strip()
        })

    return out
