#Algoritmo 4: OR + PCA + LNS
import os
import math
import time as tiempo
from datetime import datetime
from io import BytesIO

import streamlit as st
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
import googlemaps
from googlemaps.convert import decode_polyline
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import folium
from streamlit_folium import st_folium
from core.constants import GOOGLE_MAPS_API_KEY

gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)


# -------------------- CONSTANTES VRP --------------------
SERVICE_TIME    = 600       # 10 minutos de servicio
MAX_ELEMENTS    = 100            # límite de celdas por petición DM API
SHIFT_START_SEC =  9 * 3600      # 09:00
SHIFT_END_SEC   = 16*3600 +30*60 # 16:30

# ===================== AUXILIARES VRP =====================
db = firestore.client()

def _hora_a_segundos(hhmm):
    if hhmm is None or pd.isna(hhmm):
        return None
    h, m = map(int, str(hhmm).split(":"))
    return h*3600 + m*60
#Distancia euclidiana - Drones - Emergencia - Botar sí o sí una tabla ordenada considerando distancias, 
# no tiempo real, sino estimación
def _haversine_dist_dur(coords, vel_kmh=30.0):
    R = 6371e3
    n = len(coords)
    dist = [[0]*n for _ in range(n)]
    dur  = [[0]*n for _ in range(n)]
    v_ms = vel_kmh * 1000 / 3600
    for i in range(n):
        for j in range(n):
            if i==j: continue
            la1,lo1 = map(math.radians, coords[i])
            la2,lo2 = map(math.radians, coords[j])
            dlat = la2-la1; dlon=lo2-lo1
            a = math.sin(dlat/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dlon/2)**2
            d = 2*R*math.asin(math.sqrt(a))
            dist[i][j] = int(d)
            dur [i][j] = int(d/v_ms)
    return dist, dur

#Tomar los puntos de la API - Matriz para el algoritmo las reciba 
@st.cache_data(ttl="1h", show_spinner=False)
def _distancia_duracion_matrix(coords):
    if not GOOGLE_MAPS_API_KEY:
        return _haversine_dist_dur(coords)
    n = len(coords)
    dist = [[0]*n for _ in range(n)]
    dur  = [[0]*n for _ in range(n)]
    batch = max(1, min(n, MAX_ELEMENTS//n))
    for i0 in range(0, n, batch):
        resp = gmaps.distance_matrix(
            origins=coords[i0:i0+batch],
            destinations=coords,
            mode="driving",
            units="metric",
            departure_time=datetime.now(),
            traffic_model="best_guess" #pessimistic / optimistic
        )
        for i,row in enumerate(resp["rows"]):
            for j,el in enumerate(row["elements"]):
                dist[i0+i][j] = el.get("distance",{}).get("value",1)
                dur [i0+i][j] = el.get("duration_in_traffic",{}).get("value",
                                      el.get("duration",{}).get("value",1))
    return dist, dur

#Datos para la visualización del usuario + Consideraciones del algoritmo
def _crear_data_model(df, vehiculos=1, capacidad_veh=None):
    coords = list(zip(df["lat"], df["lon"]))
    dist_m, dur_s = _distancia_duracion_matrix(coords)
    time_windows, demandas = [], []
    for _, row in df.iterrows():
        ini = _hora_a_segundos(row.get("time_start"))
        fin = _hora_a_segundos(row.get("time_end"))
        if ini is None or fin is None:
            ini, fin = SHIFT_START_SEC, SHIFT_END_SEC
        time_windows.append((ini, fin))
        demandas.append(row.get("demand", 1))
    return {
        "distance_matrix":    dist_m,
        "duration_matrix":    dur_s,
        "time_windows":       time_windows,
        "demands":            demandas,
        "num_vehicles":       vehiculos,
        "vehicle_capacities": [capacidad_veh or 10**9] * vehiculos,
        "depot":              0,
    }


#Algoritmos diversos
#OR-Tool + LNS + PCA
def optimizar_ruta_algoritmo4(data, tiempo_max_seg=120):
    manager = pywrapcp.RoutingIndexManager(
        len(data["duration_matrix"]),
        data["num_vehicles"],
        data["depot"]
    )
    routing = pywrapcp.RoutingModel(manager)

    # Callback de tiempo (duración + tiempo de servicio)
    def time_cb(from_idx, to_idx):
        i = manager.IndexToNode(from_idx)
        j = manager.IndexToNode(to_idx)
        travel = data["duration_matrix"][i][j]
        service = SERVICE_TIME if i != data["depot"] else 0
        return travel + service

    transit_cb_index = routing.RegisterTransitCallback(time_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_index)

    # Dimensión de tiempo con inicio fijado a las 08:00
    routing.AddDimension(
        transit_cb_index,
        1800,                # tiempo de espera permitido (slack)
        24 * 3600,           # límite total de ruta
        False,                # <- fijar el tiempo inicial a 0 (necesario para controlarlo)
        "Time"
    )
    time_dimension = routing.GetDimensionOrDie("Time")

    # Fijar salida del depósito a las 08:00
    for vehicle_id in range(data["num_vehicles"]):
        start_index = routing.Start(vehicle_id)
        time_dimension.CumulVar(start_index).SetRange(SHIFT_START_SEC, SHIFT_START_SEC)

    # Aplicar ventanas de tiempo a cada nodo
    for node_index, (ini, fin) in enumerate(data["time_windows"]):
        index = manager.NodeToIndex(node_index)
        time_dimension.SetCumulVarSoftLowerBound(index, 600, 1000)
        time_dimension.CumulVar(index).SetRange(ini, fin)

    # Capacidad (si hay demandas)
    if any(data["demands"]):
        def demand_cb(index):
            return data["demands"][manager.IndexToNode(index)]

        demand_cb_index = routing.RegisterUnaryTransitCallback(demand_cb)
        routing.AddDimensionWithVehicleCapacity(
            demand_cb_index,
            0,  # sin capacidad extra
            data["vehicle_capacities"],
            True,
            "Capacity"
        )

    # Parámetros de búsqueda con LNS
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    # Metaheurística para explorar soluciones vecinas
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = tiempo_max_seg

    solution = routing.SolveWithParameters(search_parameters)

    if not solution:
        return None

    rutas, dist_total = [], 0
    for v in range(data["num_vehicles"]):
        idx = routing.Start(v)
        route, llegada = [], []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            route.append(node)
            llegada.append(solution.Min(time_dimension.CumulVar(idx)))
            next_idx = solution.Value(routing.NextVar(idx))
            dist_total += data["distance_matrix"][node][manager.IndexToNode(next_idx)]
            idx = next_idx

        rutas.append({
            "vehicle": v,
            "route": route,
            "arrival_sec": llegada
        })

    return {
        "routes": rutas,
        "distance_total_m": dist_total
    }
# ============= CARGAR PEDIDOS DESDE FIRESTORE =============

@st.cache_data(ttl=300)
def cargar_pedidos(fecha, tipo):
    col = db.collection('recogidas')
    docs = []
    docs += col.where("fecha_recojo", "==", fecha.strftime("%Y-%m-%d")).stream()
    docs += col.where("fecha_entrega", "==", fecha.strftime("%Y-%m-%d")).stream()
    if tipo != "Todos":
        tf = "Sucursal" if tipo == "Sucursal" else "Cliente Delivery"
        docs = [d for d in docs if d.to_dict().get("tipo_solicitud")==tf]
    out = []
    for d in docs:
        data = d.to_dict()
        is_recojo = data.get("fecha_recojo")==fecha.strftime("%Y-%m-%d")
        op = "Recojo" if is_recojo else "Entrega"
        coords = data.get(f"coordenadas_{'recojo' if is_recojo else 'entrega'}",{})
        lat, lon = coords.get("lat"), coords.get("lon")
        hs = data.get(f"hora_{'recojo' if is_recojo else 'entrega'}","")
        ts, te = (hs,hs) if hs else ("08:00","18:00")
        out.append({
            "id":d.id,
            "operacion":op,
            "nombre_cliente":data.get("nombre_cliente",""),
            "lat":lat, "lon":lon,
            "time_start":ts, "time_end":te,
            "demand":1
        })
    return out



