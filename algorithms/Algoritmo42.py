import random
import copy
import math
from datetime import datetime

# Configuración de la ruta
SERVICE_TIME = 10 * 60  # 10 minutos en segundos
SHIFT_START_SEC = 9 * 3600  # 9:00 AM
SHIFT_END_SEC = 16.5 * 3600  # 4:30 PM
MAX_TIEMPO_ENTRE_PUNTOS = 30 * 60
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
        if len(ruta) < 1:
            return float('inf')
        
        costo = 0
        tiempo_actual = self.hora_inicio
        
        for i in range(len(ruta)):
            # Primer punto de la ruta
            if i == 0:
                tw_start, tw_end = self.time_windows[ruta[i]]
                tiempo_actual = max(tiempo_actual, tw_start)  # Respetar ventana
            # Puntos subsiguientes
            else:
                tiempo_viaje = self.dur_matrix[ruta[i-1]][ruta[i]]
                tw_start, tw_end = self.time_windows[ruta[i]]
                tiempo_actual += tiempo_viaje
                tiempo_actual = max(tiempo_actual, tw_start)
                
                if tiempo_actual > tw_end:
                    return float('inf')
                
                costo += tiempo_viaje
            
            tiempo_actual += self.tiempo_servicio
        
        return costo
    
    def construir_solucion_inicial(self):
        """Asigna puntos aleatoriamente a vehículos, todos tratados igual"""
        puntos = list(range(self.n))  # Todos los puntos son normales
        random.shuffle(puntos)
        
        # Dividir equitativamente entre vehículos
        rutas = []
        puntos_por_vehiculo = math.ceil(len(puntos) / self.vehiculos)
        
        for i in range(self.vehiculos):
            inicio = i * puntos_por_vehiculo
            fin = min((i + 1) * puntos_por_vehiculo, len(puntos))
            ruta = puntos[inicio:fin]  # No hay depósito inicial/final
            rutas.append(ruta)
        
        return rutas
    

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
        for punto in removidos:
            mejor_costo = float('inf')
            mejor_posicion = None
            
            for i_ruta, ruta in enumerate(solucion):
                for j in range(len(ruta) + 1):  # Permite insertar al inicio/final
                    ruta_temp = ruta[:j] + [punto] + ruta[j:]
                    costo = self.calcular_costo_ruta(ruta_temp)
                    
                    if costo < mejor_costo:
                        mejor_costo = costo
                        mejor_posicion = (i_ruta, j)
            
            if mejor_posicion:
                solucion[mejor_posicion[0]].insert(mejor_posicion[1], punto)
        
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
        
        while (datetime.now() - inicio).seconds < self.tiempo_max and iteracion < self.iteraciones:
            # Destruir y reparar
            solucion_dest, removidos = self.destruir_solucion(solucion_actual)
            nueva_solucion = self.reparar_solucion(solucion_dest, removidos)
            nuevo_costo = sum(self.calcular_costo_ruta(r) for r in nueva_solucion)
            
            # Criterio de aceptación
            if nuevo_costo < costo_actual or random.random() < 0.1:
                solucion_actual = nueva_solucion
                costo_actual = nuevo_costo
                
                if nuevo_costo < self.mejor_costo:
                    self.mejor_solucion = copy.deepcopy(nueva_solucion)
                    self.mejor_costo = nuevo_costo
            
            iteracion += 1
        
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
            
            for j in range(len(ruta)):
                # Todos los puntos (incluido el primero) calculan su tiempo
                if j > 0:
                    distancia = self.dist_matrix[ruta[j-1]][ruta[j]]
                    tiempo_viaje = self.dur_matrix[ruta[j-1]][ruta[j]]
                    tiempo_actual += tiempo_viaje
                    distancia_total += distancia
                
                tw_start, _ = self.time_windows[ruta[j]]
                tiempo_actual = max(tiempo_actual, tw_start)
                tiempos.append(tiempo_actual)
                tiempo_actual += self.tiempo_servicio
            
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
