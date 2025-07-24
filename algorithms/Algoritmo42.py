import random
import copy
import math
from datetime import datetime, timedelta
import pandas as pd

# Configuración de la ruta
SERVICE_TIME = 900  # 15 minutos en segundos
SHIFT_START_SEC = 9 * 3600  # 9:00 AM
SHIFT_END_SEC = 16.5 * 3600  # 4:30 PM

def optimizar_ruta_42(data, tiempo_max_seg=120):
    """
    Implementación pura de Large Neighborhood Search para el problema de rutas
    
    Args:
        data: Diccionario con los datos del problema (debe contener matrices de distancia y duración)
        tiempo_max_seg: Tiempo máximo de ejecución en segundos
    
    Returns:
        Diccionario con la solución en formato compatible
    """
    # Validar datos de entrada
    if not all(key in data for key in ['distance_matrix', 'duration_matrix', 'time_windows']):
        raise ValueError("Datos de entrada incompletos para el algoritmo")
    
    # Crear y ejecutar el optimizador LNS
    optimizador = LNSOptimizer(
        dist_matrix=data["distance_matrix"],
        dur_matrix=data["duration_matrix"],
        time_windows=data["time_windows"],
        vehiculos=data.get("num_vehicles", 1),
        tiempo_max=tiempo_max_seg
    )
    
    return optimizador.optimizar()

class LNSOptimizer:
    def __init__(self, dist_matrix, dur_matrix, time_windows, vehiculos=1, tiempo_max=120):
        self.dist_matrix = dist_matrix
        self.dur_matrix = dur_matrix
        self.time_windows = time_windows
        self.vehiculos = vehiculos
        self.tiempo_max = tiempo_max
        self.mejor_solucion = None
        self.mejor_costo = float('inf')
        
        # Configuración LNS
        self.iteraciones = 2000
        self.porcentaje_destruccion = 0.4
        self.tiempo_servicio = SERVICE_TIME
        self.hora_inicio = SHIFT_START_SEC
        self.hora_fin = SHIFT_END_SEC

    def calcular_costo_ruta(self, ruta):
        """Calcula el costo total de una ruta (incluyendo ventanas de tiempo)"""
        if not ruta or len(ruta) < 2:  # Ruta debe incluir al menos depósito de salida y llegada
            return float('inf')
            
        costo = 0
        tiempo_actual = self.hora_inicio
        
        for i in range(len(ruta)-1):
            idx_actual = ruta[i]
            idx_siguiente = ruta[i+1]
            
            # Obtener ventana de tiempo del nodo destino
            tw_start, tw_end = self.time_windows[idx_siguiente]
            
            # Calcular tiempo de viaje
            tiempo_viaje = self.dur_matrix[idx_actual][idx_siguiente]
            
            # Verificar ventana de tiempo
            if tiempo_actual < tw_start:
                tiempo_actual = tw_start  # Esperar hasta que se abra la ventana
            elif tiempo_actual > tw_end:
                return float('inf')  # Penalización por violar ventana
            
            costo += tiempo_viaje
            tiempo_actual += tiempo_viaje + self.tiempo_servicio
            
        return costo

    def construir_solucion_inicial(self):
        """Construye una solución inicial simple asignando pedidos a vehículos"""
        n = len(self.dist_matrix)
        depot_idx = 0  # Asumimos que el depósito es el índice 0
        pedidos_idx = [i for i in range(n) if i != depot_idx]
        
        # Distribuir pedidos entre vehículos
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
        """Fase de destrucción: elimina aleatoriamente parte de la solución"""
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
        """Fase de reparación: reinserta pedidos usando criterio greedy"""
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
        """Ejecuta el algoritmo LNS"""
        # 1. Construir solución inicial
        solucion_actual = self.construir_solucion_inicial()
        costo_actual = sum(self.calcular_costo_ruta(r) for r in solucion_actual)
        
        self.mejor_solucion = copy.deepcopy(solucion_actual)
        self.mejor_costo = costo_actual
        
        # 2. Bucle principal de LNS
        inicio = datetime.now()
        iteracion = 0
        
        while (datetime.now() - inicio).seconds < self.tiempo_max and iteracion < self.iteraciones:
            # Paso de destrucción
            solucion_destruida, pedidos_removidos = self.destruir_solucion(solucion_actual)
            
            # Paso de reparación
            nueva_solucion = self.reparar_solucion(solucion_destruida, pedidos_removidos)
            nuevo_costo = sum(self.calcular_costo_ruta(r) for r in nueva_solucion)
            
            # Criterio de aceptación
            if nuevo_costo < costo_actual or random.random() < 0.1:  # 10% de probabilidad de aceptar peores soluciones
                solucion_actual = nueva_solucion
                costo_actual = nuevo_costo
                
                if nuevo_costo < self.mejor_costo:
                    self.mejor_solucion = copy.deepcopy(nueva_solucion)
                    self.mejor_costo = nuevo_costo
            
            iteracion += 1
        
        return self._formatear_solucion()

    def _formatear_solucion(self):
        """Convierte la solución al formato esperado por la aplicación"""
        if not self.mejor_solucion:
            return None
            
        rutas_formateadas = []
        distancia_total = 0
        tiempo_total = 0
        
        for i, ruta in enumerate(self.mejor_solucion):
            tiempos_llegada = []
            tiempo_actual = self.hora_inicio
            
            # Tiempo de salida del depósito
            tiempos_llegada.append(tiempo_actual)
            
            for j in range(1, len(ruta)):
                idx_actual = ruta[j-1]
                idx_siguiente = ruta[j]
                
                distancia = self.dist_matrix[idx_actual][idx_siguiente]
                tiempo_viaje = self.dur_matrix[idx_actual][idx_siguiente]
                
                # Ajustar por ventanas de tiempo
                tw_start, tw_end = self.time_windows[idx_siguiente]
                if tiempo_actual < tw_start:
                    tiempo_actual = tw_start
                
                tiempos_llegada.append(tiempo_actual)
                tiempo_actual += tiempo_viaje + self.tiempo_servicio
                distancia_total += distancia
            
            # Verificar que tenemos un tiempo por cada nodo en la ruta
            if len(tiempos_llegada) != len(ruta):
                raise ValueError("Número de tiempos de llegada no coincide con la ruta")
            
            rutas_formateadas.append({
                'vehicle': i,
                'route': ruta,
                'arrival_sec': tiempos_llegada
            })
        
        return {
            'routes': rutas_formateadas,
            'total_distance': distancia_total,
            'total_time': tiempo_total,
            'distance_total_m': distancia_total  # Para compatibilidad
        }
