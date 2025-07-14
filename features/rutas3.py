
import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, firestore
from core.firebase import db
from core.constants import GOOGLE_MAPS_API_KEY, PUNTOS_FIJOS_COMPLETOS
import requests
from googlemaps.convert import decode_polyline
from streamlit_folium import st_folium
import folium
import time
import googlemaps
from core.firebase import db, obtener_sucursales
from core.geo_utils import obtener_sugerencias_direccion, obtener_direccion_desde_coordenadas

gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

@st.cache_data(ttl=300)
def cargar_ruta(fecha):
    try:
        query = db.collection('recogidas')
        docs = list(query.where("fecha_recojo", "==", fecha.strftime("%Y-%m-%d")).stream()) +                list(query.where("fecha_entrega", "==", fecha.strftime("%Y-%m-%d")).stream())

        datos = []
        for doc in docs:
            data = doc.to_dict()
            doc_id = doc.id

            if data.get("fecha_recojo") == fecha.strftime("%Y-%m-%d"):
                datos.append({
                    "id": doc_id,
                    "operacion": "Recojo",
                    "nombre_cliente": data.get("nombre_cliente"),
                    "sucursal": data.get("sucursal"),
                    "direccion": data.get("direccion_recojo", "N/A"),
                    "telefono": data.get("telefono", "N/A"),
                    "hora": data.get("hora_recojo", ""),
                    "tipo_solicitud": data.get("tipo_solicitud"),
                    "coordenadas": data.get("coordenadas_recojo", {"lat": -16.409047, "lon": -71.537451}),
                    "fecha": data.get("fecha_recojo"),
                })

            if data.get("fecha_entrega") == fecha.strftime("%Y-%m-%d"):
                datos.append({
                    "id": doc_id,
                    "operacion": "Entrega",
                    "nombre_cliente": data.get("nombre_cliente"),
                    "sucursal": data.get("sucursal"),
                    "direccion": data.get("direccion_entrega", "N/A"),
                    "telefono": data.get("telefono", "N/A"),
                    "hora": data.get("hora_entrega", ""),
                    "tipo_solicitud": data.get("tipo_solicitud"),
                    "coordenadas": data.get("coordenadas_entrega", {"lat": -16.409047, "lon": -71.537451}),
                    "fecha": data.get("fecha_entrega"),
                })

        return datos
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return []

def datos_ruta():
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://github.com/Melisa2303/LAVANDERIAS-V2/raw/main/data/LOGO.PNG", width=100)
    with col2:
        st.markdown("<h1 style='text-align: left; color: black;'>Lavanderías Americanas</h1>", unsafe_allow_html=True)
    st.title("📋 Ruta del Día")

    fecha_seleccionada = st.date_input("Seleccionar Fecha", value=datetime.now().date())
    datos = cargar_ruta(fecha_seleccionada)

    if datos:
        tabla_data = []
        for item in datos:
            nombre_mostrar = item["nombre_cliente"] if item["tipo_solicitud"] == "Cliente Delivery" else item["sucursal"]
            tabla_data.append({
                "Operación": item["operacion"],
                "Cliente/Sucursal": nombre_mostrar if nombre_mostrar else "N/A",
                "Dirección": item["direccion"],
                "Teléfono": item["telefono"],
                "Hora": item["hora"] if item["hora"] else "Sin hora",
            })

        df_tabla = pd.DataFrame(tabla_data)
        st.dataframe(df_tabla, height=600, use_container_width=True, hide_index=True)

        deliveries = [item for item in datos if item["tipo_solicitud"] == "Cliente Delivery"]
        if deliveries:
            st.markdown("---")
            st.subheader("🔄 Gestión de Deliveries")

            opciones = {f"{item['operacion']} - {item['nombre_cliente']}": item for item in deliveries}
            selected = st.selectbox("Seleccionar operación:", options=opciones.keys())
            delivery_data = opciones[selected]

            st.markdown(f"### Hora de {delivery_data['operacion']}")
            hora_col1, hora_col2 = st.columns([4, 1])
            with hora_col1:
                horas_sugeridas = [f"{h:02d}:{m:02d}" for h in range(7, 19) for m in (0, 30)]
                hora_actual = delivery_data.get("hora")

                if hora_actual and hora_actual[:5] not in horas_sugeridas:
                    horas_sugeridas.append(hora_actual[:5])
                    horas_sugeridas.sort()

                opciones_hora = ["-- Sin asignar --"] + horas_sugeridas
                if hora_actual and hora_actual[:5] in horas_sugeridas:
                    index_hora = opciones_hora.index(hora_actual[:5])
                else:
                    index_hora = 0

                nueva_hora = st.selectbox(
                    "Seleccionar o escribir hora (HH:MM):",
                    options=opciones_hora,
                    index=index_hora,
                    key=f"hora_combobox_{delivery_data['id']}"
                )

            with hora_col2:
                st.write("")
                st.write("")
                if st.button("💾 Guardar", key=f"guardar_btn_{delivery_data['id']}"):
                    try:
                        campo_hora = "hora_recojo" if delivery_data["operacion"] == "Recojo" else "hora_entrega"
                        if nueva_hora == "-- Sin asignar --":
                            db.collection('recogidas').document(delivery_data["id"]).update({
                                campo_hora: None
                            })
                            st.success("Hora eliminada")
                        else:
                            if len(nueva_hora.split(":")) != 2:
                                raise ValueError
                            hora, minutos = map(int, nueva_hora.split(":"))
                            if not (0 <= hora < 24 and 0 <= minutos < 60):
                                raise ValueError

                            db.collection('recogidas').document(delivery_data["id"]).update({
                                campo_hora: f"{hora:02d}:{minutos:02d}:00"
                            })
                            st.success("Hora actualizada")

                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()

                    except ValueError:
                        st.error("Formato inválido. Use HH:MM")
                    except Exception as e:
                        st.error(f"Error: {e}")
