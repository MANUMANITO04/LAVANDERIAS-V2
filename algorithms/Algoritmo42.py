import random
import copy
import math
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Tuple, Dict

# Configuración avanzada
SERVICE_TIME = 600  # 10 minutos en segundos
SHIFT_START_SEC = 9 * 3600  # 9:00 AM
SHIFT_END_SEC = 16.5 * 3600  # 4:30 PM
PENALIZACION_VIOLACION = 3600  # 1 hora de penalización por violar ventana

class LNSOptimizer:
    def __init__(self, dist_matrix: List[List[int]], dur_matrix: List[List[int]], 
                 time_windows: List[Tuple[int, int]], vehiculos: int = 1, 
                 tiempo_max: int = 120):
        self.dist_matrix = dist_matrix
        self.dur_matrix = dur_matrix
        self.time_windows = time_windows
        self.vehiculos = vehiculos
        self.tiempo_max = tiempo_max
        self.n = len(dist_matrix)
        
        # Mejor solución encontrada
        self.mejor_solucion = None
        self.mejor_costo = float('inf')
        self.mejor_factibilidad = float('inf')
        
        # Configuración avanzada LNS
        self.iteraciones = 0
        self.max_iteraciones = 5000
        self.porcentaje_destruccion = 0.4
        self.tiempo_servicio = SERVICE_TIME
        self.hora_inicio = SHIFT_START_SEC
        self.hora_fin = SHIFT_END_SEC
        
        # Estadísticas
        self.iteraciones_mejora = 0
        self.llamadas_costo = 0

    def calcular_costo(self, ruta: List[int]) -> Tuple[float, float]:
        """
        Calcula costo total y grado de violación de restricciones
        Devuelve: (costo_total, violacion_total)
        """
        self.llamadas_costo += 1
        if len(ruta) < 2:
            return (float('inf'), float('inf'))
            
        costo = 0
        violacion = 0
        tiempo_actual = self.hora_inicio
        
        for i in range(len(ruta)-1):
            actual, siguiente = ruta[i], ruta[i+1]
            tw_inicio, tw_fin = self.time_windows[siguiente]
            
            # Tiempo de viaje
            tiempo_viaje = self.dur_matrix[actual][siguiente]
            
            # Llegada antes de la ventana -> esperar
            if tiempo_actual < tw_inicio:
                tiempo_actual = tw_inicio
            # Llegada después de la ventana -> penalizar
            elif tiempo_actual > tw_fin:
                violacion += (tiempo_actual - tw_fin)
            
            costo += tiempo_viaje
            tiempo_actual += tiempo_viaje + self.tiempo_servicio
        
        return (costo, violacion)

    def construir_solucion_inicial(self) -> List[List[int]]:
        """Construye solución inicial con enfoque en ventanas de tiempo"""
        # Ordenar pedidos por ventana de tiempo más temprana
        pedidos = sorted(
            [i for i in range(self.n) if i != 0],  # Excluir depósito (0)
            key=lambda x: self.time_windows[x][0]
        )
        
        # Asignar a vehículos considerando ventanas
        rutas = []
        pedidos_por_vehiculo = math.ceil(len(pedidos) / self.vehiculos)
        
        for i in range(self.vehiculos):
            inicio = i * pedidos_por_vehiculo
            fin = min((i+1) * pedidos_por_vehiculo, len(pedidos))
            ruta = [0] + pedidos[inicio:fin] + [0]  # Depósito al inicio y final
            rutas.append(ruta)
        
        return rutas

    def destruir_solucion(self, solucion: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
        """Destrucción más inteligente enfocada en puntos problemáticos"""
        solucion_dest = copy.deepcopy(solucion)
        removidos = []
        
        # Identificar los puntos con mayores violaciones
        puntos_problematicos = []
        for ruta in solucion_dest:
            for i in range(1, len(ruta)-1):
                costo, violacion = self.calcular_costo([ruta[i-1], ruta[i], ruta[i+1]])
                if violacion > 0:
                    puntos_problematicos.append((violacion, ruta[i]))
        
        # Ordenar por mayor violación
        puntos_problematicos.sort(reverse=True, key=lambda x: x[0])
        
        # Destruir los más problemáticos primero
        num_destruir = max(1, int(self.n * self.porcentaje_destruccion))
        for _, punto in puntos_problematicos[:num_destruir]:
            for ruta in solucion_dest:
                if punto in ruta:
                    idx = ruta.index(punto)
                    if 0 < idx < len(ruta)-1:  # No remover depósitos
                        removidos.append(ruta.pop(idx))
                    break
        
        # Si no hay puntos problemáticos, destrucción aleatoria
        if not removidos:
            for ruta in solucion_dest:
                if len(ruta) > 2:
                    num_remover = max(1, int(len(ruta) * self.porcentaje_destruccion/2))
                    indices = random.sample(range(1, len(ruta)-1), min(num_remover, len(ruta)-2))
                    for idx in sorted(indices, reverse=True):
                        removidos.append(ruta.pop(idx))
        
        return solucion_dest, removidos

    def reparar_solucion(self, solucion: List[List[int]], removidos: List[int]]) -> List[List[int]]:
        """Reparación con búsqueda tabú simple"""
        tabu_list = []
        tabu_tenure = 5
        
        for pedido in removidos:
            mejor_costo = float('inf')
            mejor_violacion = float('inf')
            mejor_pos = (0, 1)  # (ruta_idx, pos_idx)
            
            for ruta_idx in range(len(solucion)):
                ruta = solucion[ruta_idx]
                
                # Evaluar todas las posibles inserciones
                for pos_idx in range(1, len(ruta)):
                    if (ruta_idx, pos_idx) in tabu_list:
                        continue
                        
                    ruta_temp = ruta[:pos_idx] + [pedido] + ruta[pos_idx:]
                    costo, violacion = self.calcular_costo(ruta_temp)
                    
                    # Criterio de aceptación (primero factibilidad, luego costo)
                    if (violacion < mejor_violacion or 
                        (violacion == mejor_violacion and costo < mejor_costo)):
                        mejor_costo = costo
                        mejor_violacion = violacion
                        mejor_pos = (ruta_idx, pos_idx)
            
            # Aplicar la mejor inserción
            solucion[mejor_pos[0]].insert(mejor_pos[1], pedido)
            tabu_list.append(mejor_pos)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)
        
        return solucion

    def optimizar(self) -> Dict:
        """Algoritmo LNS mejorado con manejo explícito de restricciones"""
        # 1. Solución inicial
        solucion_actual = self.construir_solucion_inicial()
        costo_actual, violacion_actual = self.evaluar_solucion(solucion_actual)
        
        self.mejor_solucion = copy.deepcopy(solucion_actual)
        self.mejor_costo, self.mejor_factibilidad = costo_actual, violacion_actual
        
        # 2. Búsqueda LNS
        inicio = datetime.now()
        sin_mejora = 0
        max_sin_mejora = 50
        
        while ((datetime.now() - inicio).seconds < self.tiempo_max and 
               self.iteraciones < self.max_iteraciones and
               sin_mejora < max_sin_mejora):
            
            # Paso de destrucción-reparación
            solucion_dest, removidos = self.destruir_solucion(solucion_actual)
            nueva_solucion = self.reparar_solucion(solucion_dest, removidos)
            nuevo_costo, nueva_violacion = self.evaluar_solucion(nueva_solucion)
            
            # Criterio de aceptación
            if self.aceptar_solucion(
                (nuevo_costo, nueva_violacion),
                (costo_actual, violacion_actual),
                self.iteraciones
            ):
                solucion_actual = nueva_solucion
                costo_actual, violacion_actual = nuevo_costo, nueva_violacion
                
                # Actualizar mejor solución global
                if (nueva_violacion < self.mejor_factibilidad or 
                    (nueva_violacion == self.mejor_factibilidad and 
                     nuevo_costo < self.mejor_costo)):
                    self.mejor_solucion = copy.deepcopy(nueva_solucion)
                    self.mejor_costo, self.mejor_factibilidad = nuevo_costo, nueva_violacion
                    sin_mejora = 0
                    self.iteraciones_mejora = self.iteraciones
                else:
                    sin_mejora += 1
            
            self.iteraciones += 1
        
        return self._formatear_solucion()

    def evaluar_solucion(self, solucion: List[List[int]]) -> Tuple[float, float]:
        """Evalúa solución completa"""
        costo_total = 0
        violacion_total = 0
        
        for ruta in solucion:
            costo, violacion = self.calcular_costo(ruta)
            costo_total += costo
            violacion_total += violacion
        
        return (costo_total, violacion_total)

    def aceptar_solucion(self, nueva: Tuple[float, float], actual: Tuple[float, float], 
                        iteracion: int) -> bool:
        """Criterio de aceptación con enfriamiento simulado"""
        # Siempre aceptar soluciones mejores
        if (nueva[1] < actual[1] or 
            (nueva[1] == actual[1] and nueva[0] < actual[0])):
            return True
        
        # Probabilidad de aceptar soluciones peores (disminuye con el tiempo)
        temperatura = max(0.1, 1 - (iteracion / self.max_iteraciones))
        probabilidad = math.exp(-(nueva[1] - actual[1]) / temperatura)
        
        return random.random() < probabilidad

    def _formatear_solucion(self) -> Dict:
        """Formatea la solución para la aplicación"""
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
                actual, siguiente = ruta[j-1], ruta[j]
                distancia = self.dist_matrix[actual][siguiente]
                tiempo_viaje = self.dur_matrix[actual][siguiente]
                
                # Ajustar por ventana de tiempo
                tw_inicio, _ = self.time_windows[siguiente]
                if tiempo_actual < tw_inicio:
                    tiempo_actual = tw_inicio
                
                tiempos_llegada.append(tiempo_actual)
                tiempo_actual += tiempo_viaje + self.tiempo_servicio
                distancia_total += distancia
            
            rutas_formateadas.append({
                'vehicle': i,
                'route': ruta,
                'arrival_sec': tiempos_llegada
            })
            tiempo_total = max(tiempo_total, tiempo_actual - self.hora_inicio)
        
        return {
            'routes': rutas_formateadas,
            'total_distance': distancia_total,
            'total_time': tiempo_total,
            'distance_total_m': distancia_total,
        }

def optimizar_ruta_42(data: Dict, tiempo_max_seg: int = 120) -> Dict:
    """Función principal para integración con la aplicación"""
    # Validación de datos de entrada
    required_keys = ['distance_matrix', 'duration_matrix', 'time_windows']
    if not all(k in data for k in required_keys):
        raise ValueError(f"Datos incompletos. Requeridos: {required_keys}")
    
    # Configurar optimizador
    optimizador = LNSOptimizer(
        dist_matrix=data['distance_matrix'],
        dur_matrix=data['duration_matrix'],
        time_windows=data['time_windows'],
        vehiculos=data.get('num_vehicles', 1),
        tiempo_max=tiempo_max_seg
    )
    
    # Ejecutar optimización
    return optimizador.optimizar()
