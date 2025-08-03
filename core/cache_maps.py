import os
import json
import hashlib
from datetime import datetime
import googlemaps

from core.constants import GOOGLE_MAPS_API_KEY

gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

CACHE_DIR = "cache_directions"

def _hash_directions(origin, destination, waypoints):
    base = {
        "origin": origin,
        "destination": destination,
        "waypoints": waypoints or []
    }
    raw = json.dumps(base, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()

def get_cached_directions(origin, destination, waypoints=None, force_refresh=False):
    """
    Obtiene una respuesta cacheada de gmaps.directions(...) si existe.
    Si no, la realiza y la guarda. Usa departure_time = now y tráfico.
    """
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    key = _hash_directions(origin, destination, waypoints)
    path = os.path.join(CACHE_DIR, f"{key}.json")

    if os.path.exists(path) and not force_refresh:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    directions = gmaps.directions(
        origin,
        destination,
        mode="driving",
        departure_time=datetime.now(),
        traffic_model="best_guess",
        optimize_waypoints=False,
        waypoints=waypoints or []
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(directions, f, ensure_ascii=False, indent=2)

    return directions
