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
SERVICE_TIME    = 60        # 10 minutos de servicio
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
def _haversine_dist_dur(coords, vel_kmh=40.0):
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

# ============ ALGORITMO DE SOLUCION LNS ===================

def optimizar_ruta_algoritmo4(data, tiempo_max_seg=120):
    """
    Versión optimizada con:
    - Large Neighborhood Search (LNS)
    - Múltiples intentos internos
    - Ajuste automático de restricciones
    - Retorno EXACTO como versión original (solo rutas y distancia)
    - Llamada única desde externo
    """
    # 1. Configuración inicial común
    def setup_routing(data, attempt):
        manager = pywrapcp.RoutingIndexManager(
            len(data["duration_matrix"]),
            data["num_vehicles"],
            data["depot"]
        )
        routing = pywrapcp.RoutingModel(manager)

        # Callback de tiempo
        def time_cb(from_idx, to_idx):
            from_node = manager.IndexToNode(from_idx)
            to_node = manager.IndexToNode(to_idx)
            return data["duration_matrix"][from_node][to_node] + (SERVICE_TIME if from_node != data["depot"] else 0)

        transit_idx = routing.RegisterTransitCallback(time_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

        # Dimensión de tiempo
        time_dim = routing.AddDimension(
            transit_idx,
            10800 if attempt > 0 else 7200,  # Slack más flexible en reintentos
            24*3600,
            False,
            "Time"
        )
        time_dim = routing.GetDimensionOrDie("Time")
        time_dim.SetSpanCostCoefficientForAllVehicles(200 if attempt > 0 else 100)

        # Ventanas de tiempo
        for node, (start, end) in enumerate(data["time_windows"]):
            idx = manager.NodeToIndex(node)
            time_dim.CumulVar(idx).SetRange(start, end)

        # Capacidad (si aplica)
        if any(d > 0 for d in data["demands"]):
            def demand_cb(from_idx):
                return data["demands"][manager.IndexToNode(from_idx)]
            
            demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
            routing.AddDimensionWithVehicleCapacity(
                demand_idx,
                0,
                [int(c * (1.2 if attempt > 0 else 1.0)) for c in data["vehicle_capacities"]],
                True,
                "Capacity"
            )

        return routing, manager

    # 2. Estrategias de resolución progresivas
    def solve_attempt(data, attempt, time_left):
        routing, manager = setup_routing(data, attempt)
        
        params = pywrapcp.DefaultRoutingSearchParameters()
        
        # Configuración progresiva por intento
        strategy = [
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
            routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
        ][min(attempt, 2)]
        
        params.first_solution_strategy = strategy
        params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        
        # Configuración específica LNS
        params.local_search_operators.use_path_lns = True
        params.local_search_operators.use_inactive_lns = True
        
        params.time_limit.seconds = max(20, time_left // (3 - attempt))  # Asegurar mínimo 20s
        
        return routing.SolveWithParameters(params), routing, manager

    # 3. Procesar solución
    def extract_solution(solution, routing, manager):
        if not solution:
            return None
            
        routes = []
        total_dist = 0
        time_dim = routing.GetDimensionOrDie("Time")
        
        for veh in range(data["num_vehicles"]):
            idx = routing.Start(veh)
            route, arrivals = [], []
            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                route.append(node)
                arrivals.append(solution.Min(time_dim.CumulVar(idx)))
                next_idx = solution.Value(routing.NextVar(idx))
                total_dist += routing.GetArcCostForVehicle(idx, next_idx, veh)
                idx = next_idx
                
            if route:
                routes.append({
                    "vehicle": veh,
                    "route": route,
                    "arrival_sec": arrivals
                })

        return {
            "routes": routes,
            "distance_total_m": total_dist
        }

    # 4. Sistema de intentos internos
    start_time = tiempo.time()
    
    for attempt in range(3):  # Máximo 3 intentos internos
        elapsed = tiempo.time() - start_time
        time_left = max(1, tiempo_max_seg - elapsed)
        
        # Ajustar datos en reintentos
        current_data = data if attempt == 0 else ajustar_restricciones(data, attempt)
        
        solution, routing, manager = solve_attempt(current_data, attempt, time_left)
        
        if solution:
            result = extract_solution(solution, routing, manager)
            if result:
                return result  # Retorno EXACTO como versión original
        
        if tiempo.time() - start_time >= tiempo_max_seg:
            break

    return None  # Cumple con la interfaz original

def ajustar_restricciones(data, attempt):
    """Función interna para ajustar restricciones en reintentos"""
    new_data = data.copy()
    
    # Ampliar ventanas problemáticas
    if attempt > 0:
        new_windows = []
        for i, (start, end) in enumerate(data["time_windows"]):
            if i != data["depot"] and (end - start) < 3600:  # Ventanas <1h
                center = (start + end) // 2
                new_start = max(SHIFT_START_SEC, center - 90*60)  # ±1.5h
                new_end = min(SHIFT_END_SEC, center + 90*60)
                new_windows.append((new_start, new_end))
            else:
                new_windows.append((start, end))
        new_data["time_windows"] = new_windows
    
    return new_data

    
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

