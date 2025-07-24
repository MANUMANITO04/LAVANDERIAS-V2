# algorithms/algoritmo5.py
import random
import copy
import math
from datetime import datetime, timedelta
import pandas as pd

# Configuración de la ruta
SERVICE_TIME = 600  # 10 minutos en segundos
SHIFT_START_SEC = 9 * 3600  # 9:00 AM
SHIFT_END_SEC = 16.5 * 3600  # 4:30 PM

def optimizar_ruta_42(data, tiempo_max_seg=120):
    # 1. Preparar los datos en formato para LNS
    pedidos = _preparar_datos_para_lns(data)
    
    # 2. Crear y ejecutar el optimizador LNS
    optimizador = LNSOptimizer(
        pedidos=pedidos,
        vehiculos=data.get("num_vehicles", 1),
        tiempo_max=tiempo_max_seg,
        dist_matrix=data["distance_matrix"],
        dur_matrix=data["duration_matrix"]
    )
    
    return optimizador.optimizar()

def _preparar_datos_para_lns(data):
    pedidos = []
    n = len(data["duration_matrix"])
    
    # Asumimos que el depósito es el índice 0
    pedidos.append({
        "lat": 0,
        "lon": 0,
        "time_start_sec": SHIFT_START_SEC,
        "time_end_sec": SHIFT_END_SEC,
        "is_depot": True
    })
    
    # Agregar los demás puntos (asumiendo estructura similar a tu aplicación)
    for i in range(1, n):
        pedidos.append({
            "lat": i,  # Esto es simbólico, se usará la matriz de distancias
            "lon": i,
            "time_start_sec": data["time_windows"][i][0],
            "time_end_sec": data["time_windows"][i][1],
            "is_depot": False
        })
    
    return pedidos

class LNSOptimizer:
    def __init__(self, pedidos, vehiculos=1, tiempo_max=120, dist_matrix=None, dur_matrix=None):
        self.pedidos = pedidos
        self.vehiculos = vehiculos
        self.tiempo_max = tiempo_max
        self.dist_matrix = dist_matrix
        self.dur_matrix = dur_matrix
        self.mejor_solucion = None
        self.mejor_costo = float('inf')
        
        # Configuración LNS
        self.iteraciones = 1000
        self.porcentaje_destruccion = 0.3
        self.tiempo_servicio = SERVICE_TIME
        
        # Horario de trabajo
        self.hora_inicio = SHIFT_START_SEC
        self.hora_fin = SHIFT_END_SEC

    def distancia_entre_puntos(self, idx1, idx2):
        """Usa la matriz de distancias precalculada"""
        return self.dist_matrix[idx1][idx2]

    def duracion_entre_puntos(self, idx1, idx2):
        """Usa la matriz de duraciones precalculada"""
        return self.dur_matrix[idx1][idx2]

    def calcular_costo_ruta(self, ruta):
        """Calcula el costo total de una ruta"""
        if not ruta:
            return float('inf')
            
        costo = 0
        tiempo_actual = self.hora_inicio
        
        for i in range(len(ruta)-1):
            idx_actual = ruta[i]
            idx_siguiente = ruta[i+1]
            
            # Obtener ventana de tiempo del nodo destino
            tw_start = self.pedidos[idx_siguiente]["time_start_sec"]
            tw_end = self.pedidos[idx_siguiente]["time_end_sec"]
            
            # Calcular tiempo de viaje
            tiempo_viaje = self.duracion_entre_puntos(idx_actual, idx_siguiente)
            
            # Verificar ventana de tiempo
            if tiempo_actual < tw_start:
                tiempo_actual = tw_start  # Esperar hasta que se abra la ventana
            elif tiempo_actual > tw_end:
                return float('inf')  # Penalización por violar ventana
            
            costo += tiempo_viaje
            tiempo_actual += tiempo_viaje + self.tiempo_servicio
            
        return costo

    def construir_solucion_inicial(self):

        depot_idx = 0
        pedidos_idx = [i for i in range(len(self.pedidos)) if not self.pedidos[i]["is_depot"]]

        random.shuffle(pedidos_idx)
        rutas = []
        pedidos_por_ruta = len(pedidos_idx) // self.vehiculos
        
        for i in range(self.vehiculos):
            inicio = i * pedidos_por_ruta
            fin = (i+1) * pedidos_por_ruta if i != self.vehiculos-1 else len(pedidos_idx)
            ruta_vehiculo = [depot_idx] + pedidos_idx[inicio:fin] + [depot_idx]
            rutas.append(ruta_vehiculo)
        
        return rutas

    def destruir_solucion(self, solucion):
        solucion_destruida = copy.deepcopy(solucion)
        pedidos_removidos = []
        
        for ruta in solucion_destruida:
            if len(ruta) > 2:  # No destruir rutas vacías
                num_remover = max(1, int(len(ruta) * self.porcentaje_destruccion))
                # No remover el depósito (primer y último elemento)
                indices_posibles = list(range(1, len(ruta)-1))
                if not indices_posibles:
                    continue
                    
                indices_remover = random.sample(indices_posibles, min(num_remover, len(indices_posibles)))
                
                for idx in sorted(indices_remover, reverse=True):
                    pedidos_removidos.append(ruta.pop(idx))
        
        return solucion_destruida, pedidos_removidos

    def reparar_solucion(self, solucion_destruida, pedidos_removidos):
        for pedido_idx in pedidos_removidos:
            mejor_costo = float('inf')
            mejor_posicion = (0, 1)  # (índice ruta, posición en ruta)
            
            # Evaluar todas las posibles inserciones
            for i_ruta, ruta in enumerate(solucion_destruida):
                for j in range(1, len(ruta)):
                    # Probar inserción en esta posición
                    ruta_temporal = ruta[:j] + [pedido_idx] + ruta[j:]
                    costo = self.calcular_costo_ruta(ruta_temporal)
                    
                    if costo < mejor_costo:
                        mejor_costo = costo
                        mejor_posicion = (i_ruta, j)
            
            # Insertar en la mejor posición encontrada
            solucion_destruida[mejor_posicion[0]].insert(mejor_posicion[1], pedido_idx)
        
        return solucion_destruida

    def optimizar(self):
        solucion_actual = self.construir_solucion_inicial()
        costo_actual = sum(self.calcular_costo_ruta(r) for r in solucion_actual)
        
        self.mejor_solucion = copy.deepcopy(solucion_actual)
        self.mejor_costo = costo_actual
        
        inicio = datetime.now()
        iteracion = 0
        
        while (datetime.now() - inicio).seconds < self.tiempo_max and iteracion < self.iteraciones:
            # Paso de destrucción
            solucion_destruida, pedidos_removidos = self.destruir_solucion(solucion_actual)
            
            # Paso de reparación
            nueva_solucion = self.reparar_solucion(solucion_destruida, pedidos_removidos)
            nuevo_costo = sum(self.calcular_costo_ruta(r) for r in nueva_solucion)
            
            # Criterio de aceptación (con probabilidad de aceptar peores soluciones)
            if nuevo_costo < costo_actual or random.random() < 0.1:
                solucion_actual = nueva_solucion
                costo_actual = nuevo_costo
                
                if nuevo_costo < self.mejor_costo:
                    self.mejor_solucion = copy.deepcopy(nueva_solucion)
                    self.mejor_costo = nuevo_costo
            
            iteracion += 1
        
        return self._formatear_solucion(self.mejor_solucion)

def _formatear_solucion(self, solucion):
    """Convierte la solución al formato compatible con OR-Tools"""
    rutas_formateadas = []
    distancia_total = 0
    tiempo_total = 0
    
    for i, ruta in enumerate(solucion):
        tiempos_llegada = []
        tiempo_actual = self.hora_inicio
        
        # Asegurarnos de incluir el tiempo de llegada al depósito inicial
        tiempos_llegada.append(tiempo_actual)
        
        for j in range(1, len(ruta)):
            idx_actual = ruta[j-1]
            idx_siguiente = ruta[j]
            
            distancia = self.distancia_entre_puntos(idx_actual, idx_siguiente)
            tiempo_viaje = self.duracion_entre_puntos(idx_actual, idx_siguiente)
            
            # Ajustar por ventanas de tiempo
            tw_start = self.pedidos[idx_siguiente]["time_start_sec"]
            if tiempo_actual < tw_start:
                tiempo_actual = tw_start
            
            tiempos_llegada.append(tiempo_actual)
            tiempo_actual += tiempo_viaje + self.tiempo_servicio
            distancia_total += distancia
        
        tiempo_total += tiempo_actual - self.hora_inicio
        
        # Asegurar que tenemos un tiempo por cada nodo en la ruta
        assert len(tiempos_llegada) == len(ruta), "Los tiempos de llegada deben coincidir con los nodos"
        
        rutas_formateadas.append({
            'vehicle': i,
            'route': ruta,
            'arrival_sec': tiempos_llegada
        })
    
    return {
        'routes': rutas_formateadas,
        'total_distance': distancia_total,
        'total_time': tiempo_total,
        'distance_total_m': distancia_total
    }
