import random
import copy
import math
from datetime import datetime

# Configuración de la ruta
SERVICE_TIME = 10 * 60  # 10 minutos en segundos
SHIFT_START_SEC = 9 * 3600  # 9:00 AM
SHIFT_END_SEC = 16.5 * 3600  # 4:30 PM
MAX_TIEMPO_ENTRE_PUNTOS = 25 * 60
PENALIZACION_SALTOS_LARGOS = 50

class LNSOptimizer:
    def __init__(self, dist_matrix, dur_matrix, time_windows, vehiculos=1, tiempo_max=120):
        # Validar matrices de entrada
        if len(dist_matrix) != len(dur_matrix) or len(dist_matrix) != len(time_windows):
            raise ValueError("Las matrices y ventanas de tiempo deben tener el mismo tamaño")
        
        self.dist_matrix = dist_matrix
        self.dur_matrix = dur_matrix
        self.time_windows = time_windows
        self.n = len(dist_matrix)  # Número total de nodos (incluyendo depósito)
        self.vehiculos = vehiculos
        self.tiempo_max = tiempo_max
        
        # Solución
        self.mejor_solucion = None
        self.mejor_costo = float('inf')
        
        # Configuración LNS
        self.iteraciones = 1000
        self.porcentaje_destruccion = 0.3
        self.tiempo_servicio = SERVICE_TIME
        self.hora_inicio = SHIFT_START_SEC
        self.hora_fin = SHIFT_END_SEC

    def calcular_costo_ruta(self, ruta):
        """Calcula costo con penalizaciones mejoradas"""
        if len(ruta) < 2:
            return float('inf')
    
        costo = 0
        tiempo_actual = self.hora_inicio
        penalizacion = 0
        
        for i in range(len(ruta)-1):
            actual, siguiente = ruta[i], ruta[i+1]
            tw_start, tw_end = self.time_windows[siguiente]
            tiempo_viaje = self.dur_matrix[actual][siguiente]
            
            # Penalización exponencial por saltos largos
            if tiempo_viaje > MAX_TIEMPO_ENTRE_PUNTOS:
                exceso = tiempo_viaje - MAX_TIEMPO_ENTRE_PUNTOS
                penalizacion += (exceso ** 2) * 10  # Penalización cuadrática
            
            # Manejo estricto de ventanas
            if tiempo_actual < tw_start:
                tiempo_actual = tw_start
            elif tiempo_actual > tw_end:
                return float('inf')
            
            costo += tiempo_viaje + penalizacion
            tiempo_actual += tiempo_viaje + self.tiempo_servicio
        
        # Penalización adicional por dispersión temporal
        tiempo_total = tiempo_actual - self.hora_inicio
        costo += tiempo_total * 0.1  # Favorecer rutas más compactas
        
        return costo
    def construir_solucion_inicial(self):
        """Construye solución inicial agrupando puntos geográfica y temporalmente cercanos"""
        pedidos_idx = [i for i in range(1, self.n)]
    
        # Ordenar por proximidad geográfica y temporal
        pedidos_idx.sort(key=lambda x: (
            self.time_windows[x][0],  # Hora inicio ventana
            self.dist_matrix[0][x]    # Distancia al depósito
        ))
    
        # Agrupar considerando distancia y tiempo
        grupos = [[] for _ in range(self.vehiculos)]
        distancias = [0] * self.vehiculos
        
        for pedido in pedidos_idx:
            # Encontrar vehículo más cercano con tiempo compatible
            mejor_vehiculo = None
            mejor_tiempo_extra = float('inf')
            
            for i in range(self.vehiculos):
                if not grupos[i]:
                    tiempo_extra = self.dur_matrix[0][pedido]
                else:
                    ultimo_punto = grupos[i][-1]
                    tiempo_extra = self.dur_matrix[ultimo_punto][pedido]
                
                # Verificar compatibilidad temporal
                tiempo_llegada = distancias[i] + tiempo_extra
                tw_start, tw_end = self.time_windows[pedido]
                if tiempo_llegada <= tw_end + MAX_TIEMPO_ENTRE_PUNTOS:
                    if tiempo_extra < mejor_tiempo_extra:
                        mejor_tiempo_extra = tiempo_extra
                        mejor_vehiculo = i
        
            if mejor_vehiculo is not None:
                grupos[mejor_vehiculo].append(pedido)
                distancias[mejor_vehiculo] += mejor_tiempo_extra + self.tiempo_servicio
        
        # Construir rutas completas
        return [[0] + grupo + [0] for grupo in grupos]

    def destruir_solucion(self, solucion):
        """Destrucción que prioriza puntos con saltos largos"""
        solucion_dest = copy.deepcopy(solucion)
        removidos = []
        
        # Identificar puntos problemáticos
        problematicos = []
        for ruta in solucion:
            for i in range(1, len(ruta)-1):
                tiempo_viaje = self.dur_matrix[ruta[i-1]][ruta[i]]
                if tiempo_viaje > MAX_TIEMPO_ENTRE_PUNTOS:
                    problematicos.append(ruta[i])
        
        # Destruir primero los problemáticos
        if problematicos:
            num_remover = min(len(problematicos), int(len(problematicos) * 0.7))
            for punto in random.sample(problematicos, num_remover):
                for ruta in solucion_dest:
                    if punto in ruta:
                        idx = ruta.index(punto)
                        if 0 < idx < len(ruta)-1:
                            removidos.append(ruta.pop(idx))
                        break
        
        # Destrucción aleatoria complementaria
        for ruta in solucion_dest:
            if len(ruta) > 2:
                num_remover = max(1, int(len(ruta) * self.porcentaje_destruccion/2))
                indices = random.sample(range(1, len(ruta)-1), min(num_remover, len(ruta)-2))
                for idx in sorted(indices, reverse=True):
                    removidos.append(ruta.pop(idx))
        
        return solucion_dest, removidos

    def reparar_solucion(self, solucion, removidos):
        """Reparación con búsqueda local mejorada"""
        for punto in removidos:
            tw_punto = self.time_windows[punto]
            
            # Generar todas las posibles inserciones factibles
            candidatos = []
            for i_ruta, ruta in enumerate(solucion):
                for j in range(1, len(ruta)):
                    # Calcular tiempo de llegada estimado
                    tiempo_llegada = self.hora_inicio
                    for k in range(1, j):
                        tiempo_llegada += self.dur_matrix[ruta[k-1]][ruta[k]] + self.tiempo_servicio
                    
                    tiempo_llegada += self.dur_matrix[ruta[j-1]][punto]
                    
                    # Verificar factibilidad
                    if (tiempo_llegada <= tw_punto[1] and 
                        self.dur_matrix[punto][ruta[j]] <= MAX_TIEMPO_ENTRE_PUNTOS):
                        
                        ruta_temp = ruta[:j] + [punto] + ruta[j:]
                        costo = self.calcular_costo_ruta(ruta_temp)
                        candidatos.append((costo, i_ruta, j))
            
            # Elegir la mejor inserción factible
            if candidatos:
                candidatos.sort()
                mejor_costo, i_ruta, j = candidatos[0]
                solucion[i_ruta].insert(j, punto)
        
        return solucion

    def optimizar(self):
        """Algoritmo LNS mejorado"""
        # Construir solución inicial
        solucion_actual = self.construir_solucion_inicial()
        costo_actual = sum(self.calcular_costo_ruta(r) for r in solucion_actual)
        
        self.mejor_solucion = copy.deepcopy(solucion_actual)
        self.mejor_costo = costo_actual
        
        # Búsqueda LNS
        inicio = datetime.now()
        iteracion = 0
        temperatura = 1.0
        enfriamiento = 0.995
        while (datetime.now() - inicio).seconds < self.tiempo_max:
            # Destruir y reparar
            solucion_dest, removidos = self.destruir_solucion(solucion_actual)
            nueva_solucion = self.reparar_solucion(solucion_dest, removidos)
            nuevo_costo = sum(self.calcular_costo_ruta(r) for r in nueva_solucion)
            
            # Criterio de aceptación con enfriamiento
            delta = nuevo_costo - costo_actual
            if delta < 0 or random.random() < math.exp(-delta/temperatura):
                solucion_actual = nueva_solucion
                costo_actual = nuevo_costo
            
                if nuevo_costo < self.mejor_costo:
                    self.mejor_solucion = copy.deepcopy(nueva_solucion)
                    self.mejor_costo = nuevo_costo
        
            temperatura *= enfriamiento
        
        return self._formatear_solucion()

    def _formatear_solucion(self):
        """Formatea la solución para la aplicación"""
        if not self.mejor_solucion:
            return None
            
        rutas_formateadas = []
        distancia_total = 0
        
        for i, ruta in enumerate(self.mejor_solucion):
            tiempos = []
            tiempo_actual = self.hora_inicio
            tiempos.append(tiempo_actual)
            
            for j in range(1, len(ruta)):
                actual, siguiente = ruta[j-1], ruta[j]
                distancia = self.dist_matrix[actual][siguiente]
                tiempo_viaje = self.dur_matrix[actual][siguiente]
                
                # Ajustar por ventana
                tw_start, _ = self.time_windows[siguiente]
                if tiempo_actual < tw_start:
                    tiempo_actual = tw_start
                
                tiempos.append(tiempo_actual)
                tiempo_actual += tiempo_viaje + self.tiempo_servicio
                distancia_total += distancia
            
            rutas_formateadas.append({
                'vehicle': i,
                'route': ruta,
                'arrival_sec': tiempos
            })
        
        return {
            'routes': rutas_formateadas,
            'total_distance': distancia_total,
            'distance_total_m': distancia_total
        }

def optimizar_ruta_42(data, tiempo_max_seg=120):
    """Función principal para integración"""
    # Validación de datos
    required = ['distance_matrix', 'duration_matrix', 'time_windows']
    if not all(k in data for k in required):
        raise ValueError(f"Faltan datos requeridos: {required}")
    
    # Crear optimizador
    optimizador = LNSOptimizer(
        dist_matrix=data['distance_matrix'],
        dur_matrix=data['duration_matrix'],
        time_windows=data['time_windows'],
        vehiculos=data.get('num_vehicles', 1),
        tiempo_max=tiempo_max_seg
    )
    
    return optimizador.optimizar()
