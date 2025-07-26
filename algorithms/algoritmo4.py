import random
import copy
import math
from datetime import datetime

# Configuración de la ruta
SERVICE_TIME = 13 * 60  # 10 minutos en segundos
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

    def calcular_costo_ruta(self, ruta, strict=False):
        if len(ruta) < 1:
            return float('inf')
    
        costo = 0
        penalizacion = 0
        tiempo_actual = self.hora_inicio
        
        for i in range(len(ruta)):
            punto = ruta[i]
            tw_start, tw_end = self.time_windows[punto]
            
            # Tiempo de llegada al punto
            if i > 0:
                tiempo_viaje = self.dur_matrix[ruta[i-1]][punto]
                tiempo_actual += tiempo_viaje
                costo += tiempo_viaje
                
                # Penalizar saltos largos
                if tiempo_viaje > MAX_TIEMPO_ENTRE_PUNTOS:
                    penalizacion += PENALIZACION_SALTOS_LARGOS
            
            # Verificar ventana de tiempo
            if tiempo_actual < tw_start:
                tiempo_actual = tw_start
            elif tiempo_actual > tw_end:
                if strict:
                    return float('inf')
                penalizacion += (tiempo_actual - tw_end) * 10
            
            # Tiempo de servicio
            tiempo_actual += self.tiempo_servicio
            
            # Verificar fin de jornada
            if tiempo_actual > self.hora_fin:
                if strict:
                    return float('inf')
                penalizacion += (tiempo_actual - self.hora_fin) * 5
        
        return costo + penalizacion
    
    def construir_solucion_inicial(self):
        puntos = list(range(self.n))
        random.shuffle(puntos)
        
        rutas = []
        puntos_por_vehiculo = math.ceil(len(puntos) / self.vehiculos)
        
        for i in range(self.vehiculos):
            inicio = i * puntos_por_vehiculo
            fin = min((i + 1) * puntos_por_vehiculo, len(puntos))
            ruta = puntos[inicio:fin]
            rutas.append(ruta)
        
        return rutas

    def destruir_solucion(self, solucion):
        """Versión robusta de destrucción que evita errores de índice"""
        solucion_dest = copy.deepcopy(solucion)
        removidos = []
        
        # 1. Identificar puntos problemáticos de manera segura
        problematicos = []
        for i_ruta, ruta in enumerate(solucion):
            if len(ruta) < 2:  # Saltar rutas demasiado cortas
                continue
            for i in range(1, len(ruta)):
                try:
                    if self.dur_matrix[ruta[i-1]][ruta[i]] > MAX_TIEMPO_ENTRE_PUNTOS:
                        problematicos.append((i_ruta, i))
                except IndexError:
                    continue  # Si hay error en la matriz, saltar este punto
    
        # 2. Remover puntos problemáticos de manera controlada
        if problematicos:
            num_remover = min(len(problematicos), int(len(problematicos) * 0.7))
            for i_ruta, idx in random.sample(problematicos, num_remover):
                try:
                    if 0 <= idx < len(solucion_dest[i_ruta]):
                        removidos.append(solucion_dest[i_ruta].pop(idx))
                except (IndexError, TypeError):
                    continue  # Si falla, continuar con el siguiente
    
        # 3. Destrucción aleatoria complementaria con verificación
        puntos_disponibles = []
        for i_ruta, ruta in enumerate(solucion_dest):
            if len(ruta) > 1:  # Solo rutas con múltiples puntos
                puntos_disponibles.extend((i_ruta, i) for i in range(len(ruta)))
        
        if puntos_disponibles:
            num_aleatorio = max(1, int(self.n * self.porcentaje_destruccion/3))
            num_aleatorio = min(num_aleatorio, len(puntos_disponibles))
            
            for i_ruta, idx in random.sample(puntos_disponibles, num_aleatorio):
                try:
                    if 0 <= idx < len(solucion_dest[i_ruta]):
                        removidos.append(solucion_dest[i_ruta].pop(idx))
                except (IndexError, TypeError):
                    continue
    
        return solucion_dest, removidos

    def reparar_solucion(self, solucion, removidos):
        for punto in removidos:
            mejor_costo = float('inf')
            mejor_posicion = (0, 0)  # Posición por defecto
            
            for i_ruta, ruta in enumerate(solucion):
                for j in range(len(ruta) + 1):
                    ruta_temp = ruta[:j] + [punto] + ruta[j:]
                    costo = self.calcular_costo_ruta(ruta_temp, strict=False)
                    
                    if costo < mejor_costo:
                        mejor_costo = costo
                        mejor_posicion = (i_ruta, j)
            
            # Insertar en la mejor posición encontrada
            solucion[mejor_posicion[0]].insert(mejor_posicion[1], punto)
        
        return solucion

    def optimizar(self):
        """Algoritmo LNS con garantía de cobertura completa"""
        solucion_actual = self.construir_solucion_inicial()
        costo_actual = sum(self.calcular_costo_ruta(r, strict=False) for r in solucion_actual)
        
        self.mejor_solucion = copy.deepcopy(solucion_actual)
        self.mejor_costo = costo_actual
        
        inicio = datetime.now()
        iteracion = 0
        
        while (datetime.now() - inicio).seconds < self.tiempo_max and iteracion < self.iteraciones:
            solucion_dest, removidos = self.destruir_solucion(solucion_actual)
            nueva_solucion = self.reparar_solucion(solucion_dest, removidos)
            nuevo_costo = sum(self.calcular_costo_ruta(r, strict=False) for r in nueva_solucion)
            
            if nuevo_costo < costo_actual or random.random() < 0.1:
                solucion_actual = nueva_solucion
                costo_actual = nuevo_costo
                
                if nuevo_costo < self.mejor_costo:
                    self.mejor_solucion = copy.deepcopy(nueva_solucion)
                    self.mejor_costo = nuevo_costo
            
            iteracion += 1
        
        # Verificación final de cobertura
        puntos_cubiertos = {p for ruta in self.mejor_solucion for p in ruta}
        if len(puntos_cubiertos) != self.n:
            puntos_faltantes = set(range(self.n)) - puntos_cubiertos
            for punto in puntos_faltantes:
                self._insertar_punto_forzado(punto)
        
        return self._formatear_solucion()


    def _insertar_punto_forzado(self, punto):
        """Inserta un punto en la posición menos mala (último recurso)"""
        mejor_costo = float('inf')
        mejor_posicion = (0, 0)
        
        for i_ruta, ruta in enumerate(self.mejor_solucion):
            for j in range(len(ruta) + 1):
                ruta_temp = ruta[:j] + [punto] + ruta[j:]
                costo = self.calcular_costo_ruta(ruta_temp, strict=False)
                
                if costo < mejor_costo:
                    mejor_costo = costo
                    mejor_posicion = (i_ruta, j)
        
        self.mejor_solucion[mejor_posicion[0]].insert(mejor_posicion[1], punto)
        self.mejor_costo += mejor_costo  # Ajustar costo total


    def _formatear_solucion(self):
        """Formatea la solución garantizando estructura consistente"""
        if not self.mejor_solucion or all(len(r) == 0 for r in self.mejor_solucion):
            return {
                'routes': [],
                'total_distance': 0,
                'distance_total_m': 0,
                'error': 'No se pudo generar una solución válida'
            }
        
        rutas_formateadas = []
        distancia_total = 0
        
        for i, ruta in enumerate(self.mejor_solucion):
            if not ruta:  # Si la ruta está vacía
                continue
                
            tiempos = []
            tiempo_actual = self.hora_inicio
            
            for j in range(len(ruta)):
                # Calcular tiempo de llegada
                if j > 0:
                    distancia_total += self.dist_matrix[ruta[j-1]][ruta[j]]
                    tiempo_actual += self.dur_matrix[ruta[j-1]][ruta[j]]
                
                tw_start, _ = self.time_windows[ruta[j]]
                tiempo_actual = max(tiempo_actual, tw_start)
                tiempos.append(tiempo_actual)
                tiempo_actual += self.tiempo_servicio
            
            rutas_formateadas.append({
                'vehicle': i,
                'route': ruta,
                'arrival_sec': tiempos,
                'num_points': len(ruta)
            })
        
        return {
            'routes': rutas_formateadas,
            'total_distance': distancia_total,
            'distance_total_m': distancia_total,
        }

def optimizar_ruta_lns(data, tiempo_max_seg=120):
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
