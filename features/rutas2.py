import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from core.firebase import db
from core.constants import GOOGLE_MAPS_API_KEY, PUNTOS_FIJOS_COMPLETOS
import requests  # Importar requests A
from googlemaps.convert import decode_polyline
from streamlit_folium import st_folium
import folium
from datetime import datetime, timedelta
import time
import googlemaps
from core.firebase import db, obtener_sucursales
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
        docs = list(query.where("fecha_recojo", "==", fecha.strftime("%Y-%m-%d")).stream()) + \
               list(query.where("fecha_entrega", "==", fecha.strftime("%Y-%m-%d")).stream())

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

def datos_ruta()
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://github.com/Melisa2303/LAVANDERIAS-V2/raw/main/data/LOGO.PNG", width=100)
    with col2:
        st.markdown("<h1 style='text-align: left; color: black;'>Lavanderías Americanas</h1>", unsafe_allow_html=True)
    st.title("📋 Ruta del Día")

        st.markdown("### 📤 Subir CSV para importar Deliveries")

    archivo_csv = st.file_uploader("Selecciona el archivo CSV", type=["csv"])
    if archivo_csv is not None:
        try:
            df_csv = pd.read_csv(archivo_csv)

            st.success(f"Archivo cargado con {len(df_csv)} filas.")
            if st.checkbox("📦 Mostrar datos del archivo cargado"):
                st.dataframe(df_csv)

            if st.button("⬆️ Subir a Firebase"):
                contador = 0
                for _, fila in df_csv.iterrows():
                    try:
                        data = {
                            "tipo_solicitud": fila["tipo_solicitud"],
                            "telefono": str(fila["telefono"]),
                            "nombre_cliente": fila["nombre_cliente"],
                            "sucursal": fila.get("sucursal", ""),
                            "fecha_entrega": fila["fecha"],
                            "fecha_recojo": fila["fecha"],
                            "direccion_entrega": fila["direccion"],
                            "direccion_recojo": fila["direccion"],
                            "hora_recojo": "12:00:00",  # Puedes personalizar esto si el CSV lo trae
                            "coordenadas_entrega": {
                                "lat": float(fila["coordenadas.lat"]),
                                "lon": float(fila["coordenadas.lon"]),
                            },
                            "coordenadas_recojo": {
                                "lat": float(fila["coordenadas.lat"]),
                                "lon": float(fila["coordenadas.lon"]),
                            }
                        }
                        db.collection("recogidas").add(data)
                        contador += 1
                    except Exception as e:
                        st.error(f"Error en fila {_ + 2}: {e}")

                st.success(f"Se subieron {contador} registros correctamente.")
                st.cache_data.clear()
                st.rerun()

        except Exception as e:
            st.error(f"No se pudo procesar el archivo: {e}")


    # Filtro: solo fecha
    fecha_seleccionada = st.date_input("Seleccionar Fecha", value=datetime.now().date())

    # Obtener datos
    datos = cargar_ruta(fecha_seleccionada)
    # Mostrar Tabla
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

        # Gestión de Deliveries
        deliveries = [item for item in datos if item["tipo_solicitud"] == "Cliente Delivery"]
        
        if deliveries:
            st.markdown("---")
            st.subheader("🔄 Gestión de Deliveries")
            
            opciones = {f"{item['operacion']} - {item['nombre_cliente']}": item for item in deliveries}
            selected = st.selectbox("Seleccionar operación:", options=opciones.keys())
            delivery_data = opciones[selected]

            # --- Selector de Hora Unificado ---
            st.markdown(f"### Hora de {delivery_data['operacion']}")
            
            # Crear una fila con el combobox y botón
            hora_col1, hora_col2 = st.columns([4, 1])
            
            with hora_col1:
                # Generar opciones de hora (7:00 a 18:00 cada 30 min)
                horas_sugeridas = [f"{h:02d}:{m:02d}" for h in range(7, 19) for m in (0, 30)]
                hora_actual = delivery_data.get("hora", "12:00:00")[:5]  # Formato HH:MM
                
                # Si la hora actual no está en las sugeridas, la agregamos
                if hora_actual not in horas_sugeridas:
                    horas_sugeridas.append(hora_actual)
                    horas_sugeridas.sort()
                
                # Combobox unificado
                nueva_hora = st.selectbox(
                    "Seleccionar o escribir hora (HH:MM):",
                    options=horas_sugeridas,
                    index=horas_sugeridas.index(hora_actual) if hora_actual in horas_sugeridas else 0,
                    key=f"hora_combobox_{delivery_data['id']}"
                )
            
            with hora_col2:
                st.write("")  # Espaciado
                st.write("")  # Espaciado
                if st.button("💾 Guardar", key=f"guardar_btn_{delivery_data['id']}"):
                    try:
                        # Validar formato HH:MM
                        if len(nueva_hora.split(":")) != 2:
                            raise ValueError
                        hora, minutos = map(int, nueva_hora.split(":"))
                        if not (0 <= hora < 24 and 0 <= minutos < 60):
                            raise ValueError
                        
                        campo_hora = "hora_recojo" if delivery_data["operacion"] == "Recojo" else "hora_entrega"
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

            # Sección de Dirección y Mapa (Versión idéntica a Solicitar Recogida)
          
            st.markdown(f"### 📅 Reprogramación de {delivery_data['operacion']}")
            with st.expander("Cambiar fecha y ubicación", expanded=True):
                # Inicialización independiente (usando prefijo "reprogramar_" en lugar de "delivery_")
                if "reprogramar_lat" not in st.session_state:
                    st.session_state.reprogramar_lat = delivery_data["coordenadas"]["lat"]
                    st.session_state.reprogramar_lon = delivery_data["coordenadas"]["lon"]
                    st.session_state.reprogramar_direccion = delivery_data["direccion"]
                    st.session_state.reprogramar_mapa = folium.Map(
                        location=[st.session_state.reprogramar_lat, st.session_state.reprogramar_lon],
                        zoom_start=15
                    )
                    st.session_state.reprogramar_marker = folium.Marker(
                        [st.session_state.reprogramar_lat, st.session_state.reprogramar_lon],
                        tooltip="Punto seleccionado"        
                    ).add_to(st.session_state.reprogramar_mapa)

                # Campo de dirección
                direccion_input = st.text_input(
                    "Dirección",
                    value=st.session_state.reprogramar_direccion,
                    key=f"reprogramar_direccion_input_{delivery_data['id']}"
                )

                # Buscar sugerencias
                sugerencias = []
                if direccion_input and direccion_input != st.session_state.reprogramar_direccion:
                    sugerencias = obtener_sugerencias_direccion(direccion_input)
    
                direccion_seleccionada = st.selectbox(
                    "Sugerencias de Direcciones:",
                    ["Seleccione una dirección"] + [sug["display_name"] for sug in sugerencias] if sugerencias else ["No hay sugerencias"],
                    key=f"reprogramar_sugerencias_{delivery_data['id']}"
                )

                # Actualizar al seleccionar sugerencia
                if direccion_seleccionada and direccion_seleccionada != "Seleccione una dirección":
                    for sug in sugerencias:
                        if direccion_seleccionada == sug["display_name"]:
                            st.session_state.reprogramar_lat = float(sug["lat"])
                            st.session_state.reprogramar_lon = float(sug["lon"])
                            st.session_state.reprogramar_direccion = direccion_seleccionada
                
                            # Actualizar mapa y marcador
                            st.session_state.reprogramar_mapa = folium.Map(
                                location=[st.session_state.reprogramar_lat, st.session_state.reprogramar_lon],
                                zoom_start=15
                            )
                            st.session_state.reprogramar_marker = folium.Marker(
                                [st.session_state.reprogramar_lat, st.session_state.reprogramar_lon],
                                tooltip="Punto seleccionado"
                            ).add_to(st.session_state.reprogramar_mapa)
                            break

                # Renderizar mapa
                mapa = st_folium(
                    st.session_state.reprogramar_mapa,
                    width=700,
                    height=500,
                    key=f"reprogramar_mapa_{delivery_data['id']}"
                )

                # Actualizar al hacer clic
                if mapa.get("last_clicked"):
                    st.session_state.reprogramar_lat = mapa["last_clicked"]["lat"]
                    st.session_state.reprogramar_lon = mapa["last_clicked"]["lng"]
                    st.session_state.reprogramar_direccion = obtener_direccion_desde_coordenadas(
                        st.session_state.reprogramar_lat, st.session_state.reprogramar_lon
                    )
        
                    # Actualizar mapa y marcador
                    st.session_state.reprogramar_mapa = folium.Map(
                        location=[st.session_state.reprogramar_lat, st.session_state.reprogramar_lon],
                        zoom_start=15
                    )
                    st.session_state.reprogramar_marker = folium.Marker(
                        [st.session_state.reprogramar_lat, st.session_state.reprogramar_lon],
                        tooltip="Punto seleccionado"
                    ).add_to(st.session_state.reprogramar_mapa)
                    st.rerun()

                # Mostrar dirección final
                st.markdown(f"""
                    <div style='background-color: #f0f8ff; padding: 10px; border-radius: 5px; margin-top: 10px;'>
                        <h4 style='color: #333; margin: 0;'>Dirección Final:</h4>
                        <p style='color: #555; font-size: 16px;'>{st.session_state.reprogramar_direccion}</p>
                    </div>
                """, unsafe_allow_html=True)

                # Selector de fecha (manteniendo tu lógica original)
                min_date = datetime.now().date() if delivery_data["operacion"] == "Recojo" else datetime.strptime(delivery_data["fecha"], "%Y-%m-%d").date()
                nueva_fecha = st.date_input(
                    "Nueva fecha:",
                    value=min_date + timedelta(days=1),
                    min_value=min_date
                )

                # Botón para guardar cambios
                if st.button(f"💾 Guardar Cambios de {delivery_data['operacion']}"):
                    try:
                        updates = {
                            "fecha_recojo" if delivery_data["operacion"] == "Recojo" else "fecha_entrega": nueva_fecha.strftime("%Y-%m-%d"),
                            "direccion_recojo" if delivery_data["operacion"] == "Recojo" else "direccion_entrega": st.session_state.reprogramar_direccion,
                            "coordenadas_recojo" if delivery_data["operacion"] == "Recojo" else "coordenadas_entrega": {
                                "lat": st.session_state.reprogramar_lat,
                                "lon": st.session_state.reprogramar_lon
                            }
                        }
            
                        db.collection('recogidas').document(delivery_data["id"]).update(updates)
                        st.success("¡Reprogramación exitosa!")
                        st.cache_data.clear()
                        time.sleep(2)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error al guardar: {e}")

        # Botón de Descarga
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
