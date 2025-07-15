# Algoritmo 4: OR + PCA + LNS 
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
SERVICE_TIME = 10 * 60        # 10 minutos de servicio 
MAX_ELEMENTS = 100            # límite de celdas por petición DM API
SHIFT_START_SEC = 9 * 3600    # 09:00 
SHIFT_END_SEC = 16*3600 +30*60 # 16:30 
MARGEN = 15 * 60              # 15 minutos de margen 

# ===================== AUXILIARES VRP =====================
db = firestore.client()

def _hora_a_segundos(hhmm):
    """Versión idéntica a la de algoritmo1 que funciona correctamente"""
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
    """Versión idéntica a la de algoritmo1"""
    R = 6371e3
    n = len(coords)
    dist = [[0]*n for _ in range(n)]
    dur = [[0]*n for _ in range(n)]
    v_ms = vel_kmh * 1000 / 3600
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
            dur[i][j] = int(d / v_ms)
    return dist, dur

@st.cache_data(ttl="1h", show_spinner=False)
def _distancia_duracion_matrix(coords):
    """Versión idéntica a la de algoritmo1"""
    if not GOOGLE_MAPS_API_KEY:
        return _haversine_dist_dur(coords)
    n = len(coords)
    dist = [[0]*n for _ in range(n)]
    dur = [[0]*n for _ in range(n)]
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
                dist[i0+i][j] = el.get("distance",{}).get("value",1)
                dur[i0+i][j] = el.get("duration_in_traffic",{}).get(
                    "value",
                    el.get("duration",{}).get("value",1)
                )
    return dist, dur

def _crear_data_model(df, vehiculos=1, capacidad_veh=None):
    """Versión mejorada basada en algoritmo1 con márgenes"""
    coords = list(zip(df["lat"], df["lon"]))
    dist_m, dur_s = _distancia_duracion_matrix(coords)
    
    time_windows = []
    demandas = []
    for _, row in df.iterrows():
        ini = _hora_a_segundos(row.get("time_start"))
        fin = _hora_a_segundos(row.get("time_end"))
        if ini is None or fin is None:
            ini, fin = SHIFT_START_SEC, SHIFT_END_SEC
        else:
            # Aplicar márgenes como en algoritmo1
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

def optimizar_ruta_algoritmo4(data, tiempo_max_seg=120, reintento=False):
    """
    Versión completamente revisada basada en algoritmo1 pero manteniendo LNS
    """
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]),
        data["num_vehicles"],
        data["depot"]
    )
    routing = pywrapcp.RoutingModel(manager)

    # Callback de tiempo idéntico al de algoritmo1
    def time_cb(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        travel = data["duration_matrix"][i][j]
        service = SERVICE_TIME if i != data["depot"] else 0
        return travel + service

    transit_cb_idx = routing.RegisterTransitCallback(time_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

    routing.AddDimension(
        transit_cb_idx,
        1800,           # slack máximo
        24 * 3600,           # límite total de ruta
        False,               # no fijar tiempo inicial
        "Time"
    )
    time_dim = routing.GetDimensionOrDie("Time")
    time_dim.SetGlobalSpanCostCoefficient(1000)  # Importante para optimización de tiempo

    # Aplicar ventanas de tiempo como en algoritmo1
    for node, (ini, fin) in enumerate(data["time_windows"]):
        idx = manager.NodeToIndex(node)
        time_dim.CumulVar(idx).SetRange(ini, fin)

    # Fijar tiempo de salida del depósito
    depot_idx = manager.NodeToIndex(data["depot"])
    time_dim.CumulVar(depot_idx).SetRange(SHIFT_START_SEC, SHIFT_START_SEC)

    # Configuración de capacidad
    if any(data["demands"]):
        def demand_cb(from_index):
            return data["demands"][manager.IndexToNode(from_index)]
        demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_cb)
        routing.AddDimensionWithVehicleCapacity(
            demand_cb_idx, 0, data["vehicle_capacities"], True, "Capacity"
        )

    # Parámetros de búsqueda
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = tiempo_max_seg

    solution = routing.SolveWithParameters(search_parameters)

    if not solution:
        if not reintento:
            st.warning("❌ No se encontró solución inicial. Analizando posibles problemas...")
            
            ventanas_cortas = []
            for node, (ini, fin) in enumerate(data["time_windows"]):
                dur = fin - ini
                if dur < 45 * 60 and node != data["depot"]:
                    ventanas_cortas.append(node)
            
            if ventanas_cortas:
                st.warning("🔄 Intentando nuevamente con márgenes ampliados...")
                nueva_data = data.copy()
                nuevas_ventanas = []
                for i, (ini, fin) in enumerate(data["time_windows"]):
                    if i in ventanas_cortas:
                        centro = (ini + fin) // 2
                        nuevo_ini = max(0, centro - 3600)
                        nuevo_fin = min(86400, centro + 3600)
                        nuevas_ventanas.append((nuevo_ini, nuevo_fin))
                    else:
                        nuevas_ventanas.append((ini, fin))
                
                nueva_data["time_windows"] = nuevas_ventanas
                return optimizar_ruta_algoritmo4(nueva_data, tiempo_max_seg, reintento=True)
        
        st.error("😕 No se encontró solución factible.")
        return None

    # Procesamiento de solución
    rutas, dist_total = [], 0
    for v in range(data["num_vehicles"]):
        idx = routing.Start(v)
        route, llegada = [], []
        route_distance = 0
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            route.append(node)
            llegada.append(solution.Min(time_dim.CumulVar(idx)))
            next_idx = solution.Value(routing.NextVar(idx))
            
            # Cálculo de distancia usando la matriz real
            route_distance += data["distance_matrix"][node][manager.IndexToNode(next_idx)]
            
            idx = next_idx

        dist_total += route_distance
        rutas.append({
            "vehicle": v,
            "route": route,
            "arrival_sec": llegada
        })

    return {
        "routes": rutas,
        "distance_total_m": dist_total  # Distancia calculada correctamente
    }

@st.cache_data(ttl=300)
def cargar_pedidos(fecha, tipo):
    """Función idéntica a la de algoritmo1"""
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
