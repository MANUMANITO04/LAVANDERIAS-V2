# Lógica para el Algoritmo 3:
import streamlit as st
from core.constants import GOOGLE_MAPS_API_KEY
from googlemaps.convert import decode_polyline
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import firebase_admin
import os
from firebase_admin import credentials, firestore
import math
import pandas as pd
import datetime
from datetime import timedelta
import time
import googlemaps
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import folium
from streamlit_folium import st_folium

db = firestore.client()

# -------------------- CONFIG GOOGLE MAPS --------------------
GOOGLE_MAPS_API_KEY = st.secrets.get("google_maps", {}).get("api_key") or os.getenv("GOOGLE_MAPS_API_KEY")
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

# -------------------- INICIALIZAR FIREBASE --------------------
#if not firebase_admin._apps:
 #   cred = credentials.Certificate("lavanderia_key.json")
 #   firebase_admin.initialize_app(cred)
#db = firestore.client()

# -------------------- CONSTANTES VRP --------------------
SERVICE_TIME    = 10 * 60        # 10 minutos de servicio en cada parada (excepto depósito)
MAX_ELEMENTS    = 100            # límite de celdas por petición Distance Matrix API
SHIFT_START_SEC =  9 * 3600      # 09:00 en segundos
SHIFT_END_SEC   = 16*3600 +30*60 # 16:30 en segundos
MARGEN = 15 * 60  # 15 minutos en segundos
# 100 kg <------------------------------------------------ #Preguntar
# ===================== FUNCIONES AUXILIARES =====================

def _hora_a_segundos(hhmm):
    """Convierte 'HH:MM' o 'HH:MM:SS' a segundos desde medianoche."""
    if hhmm is None or pd.isna(hhmm) or hhmm == "":
        return None
    try:
        parts = str(hhmm).split(":")
        h = int(parts[0])
        m = int(parts[1])
        return h*3600 + m*60
    except:
        return None


def _haversine_dist_dur(coords, vel_kmh=40.0):
    """
    Calcula matrices de distancias (en metros) y duraciones (en segundos)
    basadas en fórmula de Haversine asumiendo velocidad vel_kmh para la duración.
    coords = [(lat1, lon1), (lat2, lon2), ...]
    """
    R = 6371e3  # radio terrestre en metros
    n = len(coords)
    dist = [[0]*n for _ in range(n)]
    dur  = [[0]*n for _ in range(n)]
    v_ms = vel_kmh * 1000 / 3600  # convertir km/h a m/s
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            lat1, lon1 = map(math.radians, coords[i])
            lat2, lon2 = map(math.radians, coords[j])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
            d = 2 * R * math.asin(math.sqrt(a))
            dist[i][j] = int(d)
            dur[i][j]  = int(d / v_ms)
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
    dist = [[0]*n for _ in range(n)]
    dur  = [[0]*n for _ in range(n)]
    # Dividimos en lotes para no exceder MAX_ELEMENTS celdas
    batch = max(1, min(n, MAX_ELEMENTS // n))
    for i0 in range(0, n, batch):
        resp = gmaps.distance_matrix(
            origins=coords[i0:i0+batch],
            destinations=coords,
            mode="driving",
            units="metric",
            departure_time=datetime.now(),
            traffic_model="best_guess"
        )
        for i, row in enumerate(resp["rows"]):
            for j, el in enumerate(row["elements"]):
                dist[i0 + i][j] = el.get("distance", {}).get("value", 1)
                dur[i0 + i][j]  = el.get("duration_in_traffic", {}).get(
                    "value",
                    el.get("duration", {}).get("value", 1)
                )
    return dist, dur

def _crear_data_model(df, vehiculos=1, capacidad_veh=None):
    coords = list(zip(df["lat"], df["lon"]))
    dist_m, dur_s = _distancia_duracion_matrix(coords)
    
    MARGEN = 15 * 60  # 15 minutos en segundos
    
    time_windows = []
    demandas = []
    for _, row in df.iterrows():
        ini = _hora_a_segundos(row.get("time_start"))
        fin = _hora_a_segundos(row.get("time_end"))
        if ini is None or fin is None:
            ini, fin = SHIFT_START_SEC, SHIFT_END_SEC
        else:
            ini = max(0, ini - MARGEN)
            fin = min(24*3600, fin + MARGEN)
        time_windows.append((ini, fin))
        demandas.append(row.get("demand", 1))
    
    return {
        "distance_matrix": dist_m,
        "duration_matrix": dur_s,
        "time_windows": time_windows,
        "demands": demandas,
        "num_vehicles": vehiculos,
        "vehicle_capacities": [capacidad_veh or 10**9] * vehiculos,
        "depot": 0,
    }

# Función para obtener geometría de ruta con Directions API
@st.cache_data(ttl=3600)
# Algoritmo 3: CP-SAT
def optimizar_ruta_cp_sat(data, tiempo_max_seg=60):
    if "service_times" not in data:
        data["service_times"] = [13 * 60] * len(data["duration_matrix"])

    # Si no hay duración, calcular a partir de distancia
    if all(all(t == 0 for t in fila) for fila in data["duration_matrix"]):
        st.write("⚠️ Matriz de tiempos vacía o con ceros. Generando nueva matriz basada en distancia...")
        velocidad_mps = 20 * 1000 / 3600
        data["duration_matrix"] = [
            [int(dist / velocidad_mps) for dist in fila]
            for fila in data["distance_matrix"]
        ]

    # Configuración del modelo
    manager = pywrapcp.RoutingIndexManager(len(data["duration_matrix"]), data["num_vehicles"], data["depot"])
    routing = pywrapcp.RoutingModel(manager)

    # Callback: distancia
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Callback: tiempo de viaje + servicio
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        travel_time = data["duration_matrix"][from_node][to_node]
        service_time = data["service_times"][from_node]
        return travel_time + service_time

    time_callback_index = routing.RegisterTransitCallback(time_callback)

    # Dimensión de tiempo (respetará tiempo de viaje + servicio)
    routing.AddDimension(
        time_callback_index,
        3 * 60 * 60,   # holgura (buffer entre visitas)
        24 * 3600,     # tiempo total máximo por ruta
        False,
        "Time"
    )
    time_dimension = routing.GetDimensionOrDie("Time")

    # Penalizar slack (tiempo de espera entre visitas)
    for node_idx in range(len(data["duration_matrix"])):
        index = manager.NodeToIndex(node_idx)
        slack_var = time_dimension.SlackVar(index)
        routing.AddVariableMinimizedByFinalizer(slack_var)

    # Fijar hora de inicio en el depósito (ej. 9:00 am)
    index_depot = manager.NodeToIndex(data["depot"])
    time_dimension.CumulVar(index_depot).SetRange(9 * 3600, 9 * 3600 + 900)

     # Aplicar ventanas estrictas
    for i, (start, end) in enumerate(data["time_windows"]):
        index = manager.NodeToIndex(i)
        time_dimension.CumulVar(index).SetRange(start, end)
        # ⬇️ Tiempo obligatorio de espera/servicio en ese nodo
        # time_dimension.SlackVar(index).SetValue(data["service_times"][i])
        # Penalizar llegar antes del inicio (tiempo muerto)
        time_dimension.SetCumulVarSoftUpperBound(index, start, 5000)
        # Minimizar slack por nodo
        routing.AddVariableMinimizedByFinalizer(time_dimension.SlackVar(index))

    # Minimizar tiempo total por ruta (fin del recorrido)
    for v in range(data["num_vehicles"]):
        end_index = routing.End(v)
        routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(end_index))

    # Verificar si alguna ventana es muy ajustada
    for i, (start, end) in enumerate(data["time_windows"]):
        service = data["service_times"][i]
        travel = data["duration_matrix"][data["depot"]][i]
        if end - start < service + travel:
            st.warning(f"⚠️ Nodo {i}: ventana puede ser muy ajustada para viajar y atender")

    # Configurar parámetros del solver
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.time_limit.seconds = tiempo_max_seg
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.log_search = True

    # Resolver
    sol = routing.SolveWithParameters(search_parameters)

    if not sol:
        st.error("🚫 No se encontró una solución factible.")
        return None

    # Procesar solución
    arrival_sec_all_nodes = [None] * len(data["duration_matrix"])
    dist_total_m = 0
    rutas = []

    for v in range(data["num_vehicles"]):
        idx = routing.Start(v)
        route, llegada = [], []
        while not routing.IsEnd(idx):
            n = manager.IndexToNode(idx)
            nxt = sol.Value(routing.NextVar(idx))
            dest = manager.IndexToNode(nxt)

            # Distancia total
            dist_total_m += data["distance_matrix"][n][dest]

            # Hora de llegada
            arrival_time = sol.Value(time_dimension.CumulVar(idx))
            arrival_sec_all_nodes[n] = arrival_time
            route.append(n)
            llegada.append(arrival_time)

            idx = nxt

        rutas.append({
            "vehicle": v,
            "route": route,
            "arrival_sec": llegada
        })

    return {
        "routes": rutas,
        "distance_total_m": dist_total_m,
        "arrival_sec_all_nodes": arrival_sec_all_nodes
    }
    
def agregar_ventana_margen(df, margen_segundos=15*60):
    def expandir_fila(row):
        ini = _hora_a_segundos(row["time_start"])
        fin = _hora_a_segundos(row["time_end"])
        if ini is None or fin is None:
            return "No especificado"
        ini = max(0, ini - margen_segundos)
        fin = min(24*3600, fin + margen_segundos)
        h_ini = f"{ini//3600:02}:{(ini%3600)//60:02}"
        h_fin = f"{fin//3600:02}:{(fin%3600)//60:02}"
        return f"{h_ini} - {h_fin} h"
    
    df["ventana_con_margen"] = df.apply(expandir_fila, axis=1)
    return df
