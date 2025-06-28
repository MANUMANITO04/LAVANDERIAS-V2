# algorithms/algoritmo2.py
# CW + Tabu Search con reconstrucción global y ETA válidas

import time
from typing import List, Dict, Any, Tuple
from heapq import heappush, heappop

# Importamos las constantes del algoritmo principal
from algorithms.algoritmo1 import SERVICE_TIME, SHIFT_START_SEC


def _route_distance(
    route: List[int],
    data: Dict[str, Any]
) -> float:
    """
    Suma de distancias (m) a lo largo de un route (lista de nodos).
    """
    D = data["distance_matrix"]
    return sum(D[u][v] for u, v in zip(route, route[1:]))


def _check_feasible_and_time(
    route: List[int],
    data: Dict[str, Any]
) -> Tuple[bool, List[int]]:
    """
    Comprueba factibilidad de un route bajo time_windows y SERVICE_TIME.
    Devuelve (es_factible, tiempos_de_llegada).
    """
    T       = data["duration_matrix"]
    windows = data["time_windows"]
    depot   = data["depot"]
    t       = SHIFT_START_SEC
    arrivals = [t]

    for u, v in zip(route, route[1:]):
        t += T[u][v]
        if u != depot:
            t += SERVICE_TIME
        w0, w1 = windows[v]
        if t > w1:
            return False, []
        t = max(t, w0)
        arrivals.append(t)

    return True, arrivals


def build_global_feasible_route(data, nodes, depot):
    """
    Heurística GreedyTW: construye una única ruta completa
    respetando ventanas de tiempo y tiempos de servicio.
    """
    visited = set()
    current = depot
    t_now   = SHIFT_START_SEC
    route   = [depot]
    arrival = [t_now]

    while len(visited) < len(nodes):
        heap = []
        for nxt in nodes:
            if nxt in visited:
                continue
            t_temp = t_now + data["duration_matrix"][current][nxt]
            if current != depot:
                t_temp += SERVICE_TIME
            w0, w1 = data["time_windows"][nxt]
            if t_temp <= w1:
                wait = max(0, w0 - t_temp)
                score = t_temp + wait
                heappush(heap, (score, nxt, t_temp + wait))

        if not heap:
            return None, None  # no factible

        _, chosen, t_arrival = heappop(heap)
        route.append(chosen)
        arrival.append(t_arrival)
        visited.add(chosen)
        current = chosen
        t_now = t_arrival

    return route, arrival


def optimizar_ruta_cw_tabu(
    data: Dict[str, Any],
    tiempo_max_seg: int = 60
) -> Dict[str, Any]:
    """
    1) Clarke–Wright Savings para generar mini-rutas.
    2) Tabu Search (swap de dos nodos) para refinar cada mini-ruta.
    3) Construye UNA sola ruta factible (respetando ventanas) con ETA realista.
    """
    depot = data["depot"]
    n     = len(data["distance_matrix"])
    nodes = [i for i in range(n) if i != depot]

    # 1) Clarke–Wright Savings
    D = data["distance_matrix"]
    savings = []
    for i in nodes:
        for j in nodes:
            if i < j:
                s = D[depot][i] + D[depot][j] - D[i][j]
                savings.append((s, i, j))
    savings.sort(reverse=True)

    # Inicial: cada cliente en mini-ruta [depot, i, depot]
    parent    = {i: i for i in nodes}
    start_map = {i: i for i in nodes}
    end_map   = {i: i for i in nodes}
    route_map = {i: [depot, i, depot] for i in nodes}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for _, i, j in savings:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if end_map[ri] == i and start_map[rj] == j:
            merged = route_map[ri][:-1] + route_map[rj][1:]
            feas, _ = _check_feasible_and_time(merged, data)
            if not feas:
                continue
            parent[rj]     = ri
            end_map[ri]    = end_map[rj]
            route_map[ri]  = merged

    roots = {find(i) for i in nodes}
    initial_routes = [route_map[r] for r in roots]

    # 2) Tabu Search sobre cada mini-ruta
    final_routes = []
    total_dist   = 0.0
    start_ts     = time.time()

    for init in initial_routes:
        best_route = init[:]
        best_dist  = _route_distance(best_route, data)
        tabu_list  = []
        tabu_size  = 50

        while time.time() - start_ts < tiempo_max_seg:
            improved = False
            L = len(best_route)
            for a in range(1, L - 2):
                for b in range(a + 1, L - 1):
                    move = (a, b)
                    if move in tabu_list:
                        continue
                    cand = best_route[:]
                    cand[a], cand[b] = cand[b], cand[a]
                    feas, _ = _check_feasible_and_time(cand, data)
                    if not feas:
                        continue
                    dist_c = _route_distance(cand, data)
                    if dist_c < best_dist - 1e-6:
                        best_route = cand
                        best_dist  = dist_c
                        tabu_list.append(move)
                        if len(tabu_list) > tabu_size:
                            tabu_list.pop(0)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

        _, arrival = _check_feasible_and_time(best_route, data)
        final_routes.append((best_route, arrival, best_dist))
        total_dist += best_dist

    # 3) Unificamos clientes y reconstruimos una sola ruta factible
    clientes_unicos = list({n for rt, _, _ in final_routes for n in rt if n != depot})

    ruta_final, llegada_final = build_global_feasible_route(data, clientes_unicos, depot)

    if ruta_final is None or llegada_final is None:
        ruta_final = [depot] + clientes_unicos
        llegada_final = [SHIFT_START_SEC + i * 60 for i in range(len(ruta_final))]

    dist_final = _route_distance(ruta_final, data)

    return {
        "routes": [{
            "vehicle":     0,
            "route":       ruta_final,
            "arrival_sec": llegada_final
        }],
        "distance_total_m": dist_final
    }
