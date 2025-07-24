import random
import copy
import math
from datetime import datetime, timedelta
import pandas as pd

# Configuración de la ruta
SERVICE_TIME = 600  # 10 minutos en segundos
SHIFT_START_SEC = 9 * 3600  # 9:00 AM
SHIFT_END_SEC = 16.5 * 3600  # 4:30 PM
PENALIZACION_VIOLACION = 300  # Penalización alta por violar ventana
MAX_TIEMPO_ENTRE_PUNTOS = 1800  # 30 minutos en segundos (ajustable)

def optimizar_ruta_42(data, tiempo_max_seg=120):

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
        
        # Nuevos parámetros para manejo de ventanas
        self.penalizacion_violacion = PENALIZACION_VIOLACION
        self.mejor_violacion = float('inf')  # Trackear violación de restricciones
        
        # Ajustar parámetros de búsqueda
        self.iteraciones = 2000  # Más iteraciones para mejor exploración
        self.porcentaje_destruccion = 0.4  # Mayor destrucción para diversificación
        self.tiempo_servicio = SERVICE_TIME
        self.hora_inicio = SHIFT_START_SEC
        self.hora_fin = SHIFT_END_SEC

    def calcular_costo_ruta(self, ruta):
        """Versión modificada para penalizar saltos temporales largos"""
        if not ruta or len(ruta) < 2:
            return float('inf')
        
        costo = 0
        tiempo_actual = self.hora_inicio
        penalizacion_saltos = 0
    
        for i in range(len(ruta)-1):
            idx_actual = ruta[i]
            idx_siguiente = ruta[i+1]
            tw_start, tw_end = self.time_windows[idx_siguiente]
        
            tiempo_viaje = self.dur_matrix[idx_actual][idx_siguiente]
        
            # Penalización por saltos largos
            if tiempo_viaje > MAX_TIEMPO_ENTRE_PUNTOS:
                penalizacion_saltos += (tiempo_viaje - MAX_TIEMPO_ENTRE_PUNTOS) * 10
            
            # Manejo de ventanas de tiempo
            if tiempo_actual < tw_start:
                tiempo_actual = tw_start
            elif tiempo_actual > tw_end:
                return float('inf')  # Solución inválida
            
            costo += tiempo_viaje + penalizacion_saltos
            tiempo_actual += tiempo_viaje + self.tiempo_servicio
        
        return costo

    def construir_solucion_inicial(self):
        """Construcción inicial que agrupa puntos cercanos temporalmente"""
        # Ordenar pedidos por ventana de tiempo más temprana
        pedidos = sorted(
            [i for i in range(1, self.n)],  # Excluir depósito (0)
            key=lambda x: self.time_windows[x][0]
        )
    
        # Agrupar pedidos con ventanas similares
        grupos = []
        grupo_actual = []
        ultimo_fin = 0
    
        for pedido in pedidos:
            inicio, fin = self.time_windows[pedido]
            if not grupo_actual or inicio <= ultimo_fin + MAX_TIEMPO_ENTRE_PUNTOS:
                grupo_actual.append(pedido)
                ultimo_fin = max(ultimo_fin, fin)
            else:
                grupos.append(grupo_actual)
                grupo_actual = [pedido]
                ultimo_fin = fin
    
        if grupo_actual:
            grupos.append(grupo_actual)
    
        # Asignar grupos a vehículos
        rutas = []
        for i in range(self.vehiculos):
            if i < len(grupos):
                rutas.append([0] + grupos[i] + [0])
            else:
                rutas.append([0, 0])  # Vehículo sin asignación
    
        return rutas

    def destruir_solucion(self, solucion):
        """Versión que prioriza destruir nodos con violaciones"""
        solucion_destruida = copy.deepcopy(solucion)
        pedidos_removidos = []
        
        # Identificar nodos problemáticos primero
        nodos_problematicos = []
        for ruta in solucion:
            for i in range(1, len(ruta)-1):
                tw_start, tw_end = self.time_windows[ruta[i]]
                tiempo_llegada = self.estimar_tiempo_llegada(ruta, i)
                if tiempo_llegada > tw_end:
                    nodos_problematicos.append(ruta[i])
        
        # Destruir nodos problemáticos primero
        if nodos_problematicos:
            num_destruir = min(len(nodos_problematicos), int(len(nodos_problematicos) * 0.7))
            nodos_a_remover = random.sample(nodos_problematicos, num_destruir)
            
            for nodo in nodos_a_remover:
                for ruta in solucion_destruida:
                    if nodo in ruta:
                        idx = ruta.index(nodo)
                        if 0 < idx < len(ruta)-1:
                            pedidos_removidos.append(ruta.pop(idx))
                            break
        
        # Destrucción aleatoria para el resto
        for ruta in solucion_destruida:
            if len(ruta) > 2:
                num_remover = max(1, int(len(ruta) * self.porcentaje_destruccion/2))
                indices = random.sample(range(1, len(ruta)-1), min(num_remover, len(ruta)-2))
                for idx in sorted(indices, reverse=True):
                    pedidos_removidos.append(ruta.pop(idx))
        
        return solucion_destruida, pedidos_removidos

    def estimar_tiempo_llegada(self, ruta, pos):
        """Estimar tiempo de llegada a un nodo en la ruta"""
        tiempo = self.hora_inicio
        for i in range(1, pos+1):
            tiempo += self.dur_matrix[ruta[i-1]][ruta[i]] + self.tiempo_servicio
        return tiempo

    def reparar_solucion(self, solucion_destruida, pedidos_removidos):
        """Reparación que evita insertar puntos que creen saltos largos"""
        for pedido_idx in pedidos_removidos:
            mejor_posicion = None
            mejor_costo = float('inf')
            
            for i_ruta, ruta in enumerate(solucion_destruida):
                for j in range(1, len(ruta)):
                    # Calcular tiempo de viaje antes y después de la inserción
                    tiempo_antes = self.dur_matrix[ruta[j-1]][pedido_idx]
                    tiempo_despues = self.dur_matrix[pedido_idx][ruta[j]]
                
                    # Rechazar inserciones que creen saltos largos
                    if tiempo_antes > MAX_TIEMPO_ENTRE_PUNTOS or tiempo_despues > MAX_TIEMPO_ENTRE_PUNTOS:
                        continue
                    
                    # Calcular costo de esta inserción
                    ruta_temp = ruta[:j] + [pedido_idx] + ruta[j:]
                    costo = self.calcular_costo_ruta(ruta_temp)
                
                    if costo < mejor_costo:
                        mejor_costo = costo
                        mejor_posicion = (i_ruta, j)
            
            if mejor_posicion:
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
        sin_mejora = 0
        
        while (datetime.now() - inicio).seconds < self.tiempo_max and iteracion < self.iteraciones:
            # Paso de destrucción
            solucion_destruida, pedidos_removidos = self.destruir_solucion(solucion_actual)
            
            # Paso de reparación
            nueva_solucion = self.reparar_solucion(solucion_destruida, pedidos_removidos)
            nuevo_costo = sum(self.calcular_costo_ruta(r) for r in nueva_solucion)
            
            
            # Criterio de aceptación mejorado
            violacion_actual = self.calcular_violacion(solucion_actual)
            violacion_nueva = self.calcular_violacion(nueva_solucion)
            
            if (violacion_nueva < violacion_actual or 
                (violacion_nueva == violacion_actual and nuevo_costo < costo_actual) or
                random.random() < 0.1):
                
                solucion_actual = nueva_solucion
                costo_actual = nuevo_costo
                
                if (violacion_nueva < self.mejor_violacion or
                    (violacion_nueva == self.mejor_violacion and nuevo_costo < self.mejor_costo)):
                    
                    self.mejor_solucion = copy.deepcopy(nueva_solucion)
                    self.mejor_costo = nuevo_costo
                    self.mejor_violacion = violacion_nueva
                    sin_mejora = 0
                else:
                    sin_mejora += 1
            
            iteracion += 1
            
            # Terminar antes si no hay mejora en 100 iteraciones
            if sin_mejora > 100:
                break
        
        return self._formatear_solucion()

    def calcular_violacion(self, solucion):
        """Calcula la violación total de ventanas"""
        violacion_total = 0
        for ruta in solucion:
            tiempo_actual = self.hora_inicio
            for i in range(1, len(ruta)):
                tw_start, tw_end = self.time_windows[ruta[i]]
                tiempo_actual += self.dur_matrix[ruta[i-1]][ruta[i]] + self.tiempo_servicio
                if tiempo_actual > tw_end:
                    violacion_total += (tiempo_actual - tw_end)
        return violacion_total


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
