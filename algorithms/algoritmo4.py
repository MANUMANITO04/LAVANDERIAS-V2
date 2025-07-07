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

def optimizar_ruta_algoritmo4(data, tiempo_max_seg=120, reintento=False):
    """
    Versión mejorada con:
    - LNS (Large Neighborhood Search) optimizado
    - Reintentos inteligentes con ajuste de ventanas
    - Márgenes flexibles en restricciones
    - Diagnóstico de problemas
    - Mismo formato de entrada/salida original
    """
    # 1. Validación inicial de datos
    if not data or len(data["duration_matrix"]) == 0:
        return None

    # 2. Configuración inicial del solver
    manager = pywrapcp.RoutingIndexManager(
        len(data["duration_matrix"]),
        data["num_vehicles"],
        data["depot"]
    )
    routing = pywrapcp.RoutingModel(manager)

    # 3. Callback de tiempo con servicio
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["duration_matrix"][from_node][to_node] + (SERVICE_TIME if from_node != data["depot"] else 0)

    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 4. Dimensión de tiempo con márgenes
    routing.AddDimension(
        transit_callback_index,
        7200,  # Slack máximo (2 horas)
        24 * 3600,  # Ventana máxima de tiempo
        False,  # No fijar inicio en cero
        "Time"
    )
    time_dimension = routing.GetDimensionOrDie("Time")
    time_dimension.SetSpanCostCoefficientForAllVehicles(100)  # Penalización moderada

    # 5. Aplicar ventanas de tiempo
    for node_index, (window_start, window_end) in enumerate(data["time_windows"]):
        index = manager.NodeToIndex(node_index)
        time_dimension.CumulVar(index).SetRange(window_start, window_end)

    # 6. Configuración de capacidad (si hay demandas)
    if any(d > 0 for d in data["demands"]):
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return data["demands"][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # slack
            [int(c * 1.1) for c in data["vehicle_capacities"]],  # 10% más capacidad
            True,  # start cumul to zero
            "Capacity"
        )

    # 7. Configurar parámetros de búsqueda para LNS
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    
    # Estrategias combinadas
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    
    # Configuración específica para LNS
    search_parameters.local_search_operators.use_path_lns = True
    search_parameters.local_search_operators.use_inactive_lns = True
    search_parameters.local_search_operators.use_tsp_opt = True
    
    search_parameters.time_limit.seconds = tiempo_max_seg
    search_parameters.log_search = False

    # 8. Resolver el problema
    solution = routing.SolveWithParameters(search_parameters)

    # 9. Sistema de reintentos inteligente
    if not solution and not reintento:
        # Identificar nodos problemáticos
        problematic_nodes = [
            i for i, (start, end) in enumerate(data["time_windows"])
            if (end - start) < 3600 and i != data["depot"]  # Ventanas < 1 hora
        ]
        
        if problematic_nodes:
            # Crear nueva data con ventanas ampliadas
            new_data = data.copy()
            new_windows = []
            for i, (start, end) in enumerate(data["time_windows"]):
                if i in problematic_nodes:
                    center = (start + end) // 2
                    new_windows.append((
                        max(SHIFT_START_SEC, center - 3600),
                        min(SHIFT_END_SEC, center + 3600)
                    ))
                else:
                    new_windows.append((start, end))
            
            new_data["time_windows"] = new_windows
            
            # Reintentar con parámetros más relajados
            return optimizar_ruta_algoritmo4(new_data, tiempo_max_seg, reintento=True)

    # 10. Procesar la solución si existe
    if not solution:
        # Diagnóstico detallado del fallo
        total_demand = sum(data["demands"])
        total_capacity = sum(data["vehicle_capacities"])
        narrow_windows = sum(1 for start, end in data["time_windows"] if (end - start) < 3600)
        
        print(f"⚠️ No se encontró solución. Diagnóstico:")
        print(f"- Demanda total: {total_demand} vs Capacidad: {total_capacity}")
        print(f"- Nodos con ventanas <1h: {narrow_windows}/{len(data['time_windows'])}")
        print(f"- Tiempo máximo agotado: {tiempo_max_seg} segundos")
        
        return None

    # 11. Extraer y formatear la solución
    routes = []
    total_distance = 0
    time_dimension = routing.GetDimensionOrDie("Time")
    
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        route_nodes = []
        arrival_times = []
        
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route_nodes.append(node)
            arrival_times.append(solution.Min(time_dimension.CumulVar(index)))
            next_index = solution.Value(routing.NextVar(index))
            total_distance += routing.GetArcCostForVehicle(index, next_index, vehicle_id)
            index = next_index

        if route_nodes:  # Solo agregar rutas no vacías
            routes.append({
                "vehicle": vehicle_id,
                "route": route_nodes,
                "arrival_sec": arrival_times
            })

    return {
        "routes": routes,
        "distance_total_m": total_distance,
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

