import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime, timedelta
import time
import firebase_admin
from firebase_admin import credentials, firestore
from core.firebase import db, obtener_sucursales
from core.constants import GOOGLE_MAPS_API_KEY, PUNTOS_FIJOS_COMPLETOS
import requests  # Importar requests A
from googlemaps.convert import decode_polyline
from streamlit_folium import st_folium
import folium
import googlemaps
from core.geo_utils import obtener_sugerencias_direccion, obtener_direccion_desde_coordenadas

gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

@st.cache_data(ttl=300)
def cargar_ruta(fecha):
    """
    Carga las rutas de recogida y entrega desde la base de datos para una fecha específica.
    Retorna una lista de dict con los campos necesarios.
    """
    try:
        query = db.collection('recogidas')
        docs = (
            list(query.where("fecha_recojo", "==", fecha.strftime("%Y-%m-%d")).stream()) +
            list(query.where("fecha_entrega", "==", fecha.strftime("%Y-%m-%d")).stream())
        )

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
        st.markdown(
            "<h1 style='text-align: left; color: black;'>Lavanderías Americanas</h1>",
            unsafe_allow_html=True
        )
    st.title("📋 Ruta del Día")

    # Filtro: solo fecha
    fecha_seleccionada = st.date_input("Seleccionar Fecha", value=datetime.now().date())

    # Obtener datos
    datos = cargar_ruta(fecha_seleccionada)

    if datos:
        # Mostrar Tabla
        tabla_data = []
        for item in datos:
            nombre_mostrar = (
                item["nombre_cliente"]
                if item["tipo_solicitud"] == "Cliente Delivery"
                else item["sucursal"]
            )
            tabla_data.append({
                "Operación": item["operacion"],
                "Cliente/Sucursal": nombre_mostrar or "N/A",
                "Dirección": item["direccion"],
                "Teléfono": item["telefono"],
                "Hora": item["hora"] or "Sin hora",
            })

        df_tabla = pd.DataFrame(tabla_data)
        st.dataframe(df_tabla, height=600, use_container_width=True, hide_index=True)

        # Gestión de Deliveries
        deliveries = [item for item in datos if item["tipo_solicitud"] == "Cliente Delivery"]
        if deliveries:
            st.markdown("---")
            st.subheader("🔄 Gestión de Deliveries")

            opciones = {f"{item['operacion']} - {item['nombre_cliente']}": item for item in deliveries}
            selected = st.selectbox("Seleccionar operación:", opciones.keys())
            delivery_data = opciones[selected]

            # --- Selector de Hora Unificado ---
            st.markdown(f"### Hora de {delivery_data['operacion']}")
            hora_col1, hora_col2 = st.columns([4, 1])
            with hora_col1:
                horas_sugeridas = [f"{h:02d}:{m:02d}" for h in range(7, 19) for m in (0, 30)]
                hora_actual = delivery_data.get("hora", "12:00:00")[:5]
                if hora_actual not in horas_sugeridas:
                    horas_sugeridas.append(hora_actual)
                    horas_sugeridas.sort()
                nueva_hora = st.selectbox(
                    "Seleccionar o escribir hora (HH:MM):",
                    options=horas_sugeridas,
                    index=horas_sugeridas.index(hora_actual),
                    key=f"hora_combobox_{delivery_data['id']}"
                )
            with hora_col2:
                st.write("")
                st.write("")
                if st.button("💾 Guardar", key=f"guardar_btn_{delivery_data['id']}"):
                    try:
                        hora, minutos = map(int, nueva_hora.split(":"))
                        campo_hora = (
                            "hora_recojo"
                            if delivery_data["operacion"] == "Recojo"
                            else "hora_entrega"
                        )
                        db.collection('recogidas').document(delivery_data["id"]).update({
                            campo_hora: f"{hora:02d}:{minutos:02d}:00"
                        })
                        st.success("Hora actualizada")
                        st.cache_data.clear()
                        time.sleep(3)
                        st.rerun()
                    except:
                        st.error("Formato inválido. Use HH:MM")

            # Reprogramación de fecha y ubicación
            st.markdown(f"### 📅 Reprogramación de {delivery_data['operacion']}")
            with st.expander("Cambiar fecha y ubicación", expanded=True):
                # Inicializar en session_state si hace falta
                key_pref = f"reprogramar_{delivery_data['id']}_"
                if f"{key_pref}lat" not in st.session_state:
                    st.session_state[f"{key_pref}lat"] = delivery_data["coordenadas"]["lat"]
                    st.session_state[f"{key_pref}lon"] = delivery_data["coordenadas"]["lon"]
                    st.session_state[f"{key_pref}direccion"] = delivery_data["direccion"]
                    st.session_state[f"{key_pref}mapa"] = folium.Map(
                        location=[
                            st.session_state[f"{key_pref}lat"],
                            st.session_state[f"{key_pref}lon"]
                        ],
                        zoom_start=15
                    )
                    folium.Marker(
                        [
                            st.session_state[f"{key_pref}lat"],
                            st.session_state[f"{key_pref}lon"]
                        ],
                        tooltip="Punto seleccionado"
                    ).add_to(st.session_state[f"{key_pref}mapa"])

                # Dirección y sugerencias
                direccion_input = st.text_input(
                    "Dirección",
                    value=st.session_state[f"{key_pref}direccion"],
                    key=f"{key_pref}direccion_input"
                )
                sugerencias = []
                if direccion_input != st.session_state[f"{key_pref}direccion"]:
                    sugerencias = obtener_sugerencias_direccion(direccion_input)
                direccion_sel = st.selectbox(
                    "Sugerencias de Direcciones:",
                    (["Seleccione una dirección"] +
                     [s["display_name"] for s in sugerencias]) if sugerencias else ["No hay sugerencias"],
                    key=f"{key_pref}sugerencias"
                )
                if direccion_sel != "Seleccione una dirección":
                    for s in sugerencias:
                        if s["display_name"] == direccion_sel:
                            st.session_state[f"{key_pref}lat"] = float(s["lat"])
                            st.session_state[f"{key_pref}lon"] = float(s["lon"])
                            st.session_state[f"{key_pref}direccion"] = direccion_sel
                            # actualizar mapa
                            st.session_state[f"{key_pref}mapa"] = folium.Map(
                                location=[
                                    st.session_state[f"{key_pref}lat"],
                                    st.session_state[f"{key_pref}lon"]
                                ],
                                zoom_start=15
                            )
                            folium.Marker(
                                [
                                    st.session_state[f"{key_pref}lat"],
                                    st.session_state[f"{key_pref}lon"]
                                ],
                                tooltip="Punto seleccionado"
                            ).add_to(st.session_state[f"{key_pref}mapa"])
                            break

                # Mostrar mapa
                mapa = st_folium(
                    st.session_state[f"{key_pref}mapa"],
                    width=700,
                    height=500,
                    key=f"{key_pref}mapa_folium"
                )
                if mapa.get("last_clicked"):
                    lat_c, lon_c = mapa["last_clicked"]["lat"], mapa["last_clicked"]["lng"]
                    st.session_state[f"{key_pref}lat"] = lat_c
                    st.session_state[f"{key_pref}lon"] = lon_c
                    st.session_state[f"{key_pref}direccion"] = obtener_direccion_desde_coordenadas(lat_c, lon_c)
                    st.session_state[f"{key_pref}mapa"] = folium.Map(
                        location=[lat_c, lon_c],
                        zoom_start=15
                    )
                    folium.Marker([lat_c, lon_c], tooltip="Punto seleccionado")\
                        .add_to(st.session_state[f"{key_pref}mapa"])
                    st.rerun()

                st.markdown(f"""
                    <div style='background-color: #f0f8ff; padding: 10px; border-radius: 5px; margin-top: 10px;'>
                        <h4 style='color: #333; margin: 0;'>Dirección Final:</h4>
                        <p style='color: #555; font-size: 16px;'>{st.session_state[f"{key_pref}direccion"]}</p>
                    </div>
                """, unsafe_allow_html=True)

                # Selector de nueva fecha
                min_date = datetime.now().date() if delivery_data["operacion"] == "Recojo" \
                    else datetime.strptime(delivery_data["fecha"], "%Y-%m-%d").date()
                nueva_fecha = st.date_input(
                    "Nueva fecha:",
                    value=min_date + timedelta(days=1),
                    min_value=min_date,
                    key=f"{key_pref}fecha_input"
                )

                if st.button(f"💾 Guardar Cambios de {delivery_data['operacion']}"):
                    try:
                        field_date = "fecha_recojo" if delivery_data["operacion"] == "Recojo" else "fecha_entrega"
                        field_dir = "direccion_recojo" if delivery_data["operacion"] == "Recojo" else "direccion_entrega"
                        field_coord = "coordenadas_recojo" if delivery_data["operacion"] == "Recojo" else "coordenadas_entrega"
                        db.collection('recogidas').document(delivery_data["id"]).update({
                            field_date: nueva_fecha.strftime("%Y-%m-%d"),
                            field_dir: st.session_state[f"{key_pref}direccion"],
                            field_coord: {
                                "lat": st.session_state[f"{key_pref}lat"],
                                "lon": st.session_state[f"{key_pref}lon"]
                            }
                        })
                        st.success("¡Reprogramación exitosa!")
                        st.cache_data.clear()
                        time.sleep(3)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error al guardar: {e}")

        # Botón de descarga
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df_tabla.to_excel(writer, index=False)
        st.download_button(
            label="Descargar Excel",
            data=excel_buffer.getvalue(),
            file_name=f"ruta_{fecha_seleccionada.strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("No hay datos para la fecha seleccionada con los filtros actuales.")
