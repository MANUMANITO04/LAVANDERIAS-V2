# Aquí pondrás variables globales, claves y PUNTOS_FIJOS_COMPLETOS

import os
import streamlit as st

GOOGLE_MAPS_API_KEY = st.secrets.get("google_maps", {}).get("api_key") or os.getenv("GOOGLE_MAPS_API_KEY")

PUNTOS_FIJOS_COMPLETOS = [
    {"lat": -16.4141434959913, "lon": -71.51839574233342, "direccion": "Cochera", "tipo": "fijo", "orden": 0, "hora": "08:00"},
    {"lat": -16.398605226701633, "lon": -71.4376266111019, "direccion": "Planta", "tipo": "fijo", "orden": 1, "hora": "08:30"},
    {"lat": -16.43564123078658, "lon": -71.52216190495753, "direccion": "Sucursal Av Dolores", "tipo": "fijo", "orden": 2, "hora": "09:00"},
    {"lat": -16.43564123078658, "lon": -71.52216190495753, "direccion": "Sucursal Av Dolores", "tipo": "fijo", "orden": -3, "hora": "16:00"},
    {"lat": -16.398605226701633, "lon": -71.4376266111019, "direccion": "Planta", "tipo": "fijo", "orden": -2, "hora": "16:40"},
    {"lat": -16.4141434959913, "lon": -71.51839574233342, "direccion": "Cochera", "tipo": "fijo", "orden": -1, "hora": "17:00"}
]
