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
        fecha_str = fecha.strftime("%Y-%m-%d")
        query = db.collection('recogidas')
        docs = list(query.stream())  # Traemos todos para controlar duplicación

        datos = []
        for doc in docs:
            data = doc.to_dict()
            doc_id = doc.id

            if data.get("fecha_recojo") == fecha_str:
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

            # Solo agregar como entrega si no es la misma fecha que recojo
            if data.get("fecha_entrega") == fecha_str and data.get("fecha_entrega") != data.get("fecha_recojo"):
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

            # Dirección y mapa
            st.markdown(f"### 📅 Reprogramación de {delivery_data['operacion']}")
            with st.expander("Cambiar fecha y ubicación", expanded=True):
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

                direccion_input = st.text_input(
                    "Dirección",
                    value=st.session_state.reprogramar_direccion,
                    key=f"reprogramar_direccion_input_{delivery_data['id']}"
                )

                sugerencias = []
                if direccion_input and direccion_input != st.session_state.reprogramar_direccion:
                    sugerencias = obtener_sugerencias_direccion(direccion_input)

                direccion_seleccionada = st.selectbox(
                    "Sugerencias de Direcciones:",
                    ["Seleccione una dirección"] + [sug["display_name"] for sug in sugerencias] if sugerencias else ["No hay sugerencias"],
                    key=f"reprogramar_sugerencias_{delivery_data['id']}"
                )

                if direccion_seleccionada and direccion_seleccionada != "Seleccione una dirección":
                    for sug in sugerencias:
                        if direccion_seleccionada == sug["display_name"]:
                            st.session_state.reprogramar_lat = float(sug["lat"])
                            st.session_state.reprogramar_lon = float(sug["lon"])
                            st.session_state.reprogramar_direccion = direccion_seleccionada
                            st.session_state.reprogramar_mapa = folium.Map(
                                location=[st.session_state.reprogramar_lat, st.session_state.reprogramar_lon],
                                zoom_start=15
                            )
                            st.session_state.reprogramar_marker = folium.Marker(
                                [st.session_state.reprogramar_lat, st.session_state.reprogramar_lon],
                                tooltip="Punto seleccionado"
                            ).add_to(st.session_state.reprogramar_mapa)
                            break

                mapa = st_folium(
                    st.session_state.reprogramar_mapa,
                    width=700,
                    height=500,
                    key=f"reprogramar_mapa_{delivery_data['id']}"
                )

                if mapa.get("last_clicked"):
                    st.session_state.reprogramar_lat = mapa["last_clicked"]["lat"]
                    st.session_state.reprogramar_lon = mapa["last_clicked"]["lng"]
                    st.session_state.reprogramar_direccion = obtener_direccion_desde_coordenadas(
                        st.session_state.reprogramar_lat, st.session_state.reprogramar_lon
                    )
                    st.rerun()

                st.markdown(f"""
                    <div style='background-color: #f0f8ff; padding: 10px; border-radius: 5px; margin-top: 10px;'>
                        <h4 style='color: #333; margin: 0;'>Dirección Final:</h4>
                        <p style='color: #555; font-size: 16px;'>{st.session_state.reprogramar_direccion}</p>
                    </div>
                """, unsafe_allow_html=True)

                min_date = datetime.now().date() if delivery_data["operacion"] == "Recojo" else datetime.strptime(delivery_data["fecha"], "%Y-%m-%d").date()
                nueva_fecha = st.date_input("Nueva fecha:", value=min_date + timedelta(days=1), min_value=min_date)

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

    
    # -----------------------------------------------
    # 📤 CARGA DE CSV A FIRESTORE
    # -----------------------------------------------
    st.markdown("---")
    st.subheader("📤 Cargar datos desde archivo CSV")

    uploaded_file = st.file_uploader("Selecciona el archivo CSV", type=["csv"], key="cargar_csv")

    if uploaded_file:
        df_csv = pd.read_csv(uploaded_file, dtype={"telefono": str})
        st.dataframe(df_csv)

        if st.button("🚀 Subir a Firestore", key="boton_subir_csv"):
            total = len(df_csv)
            errores = 0

            for i, row in df_csv.iterrows():
                try:
                    tipo_solicitud = str(row["tipo_solicitud"]).strip()
                    nombre_cliente = row.get("nombre_cliente", "")
                    sucursal = row.get("sucursal", "")
                    telefono = str(row["telefono"]).strip()
                    fecha_str = str(row["fecha"]).strip()
                    direccion = str(row["direccion"]).strip()
                    lat = float(row["coordenadas.lat"])
                    lon = float(row["coordenadas.lon"])

                    fecha = datetime.strptime(fecha_str, "%Y-%m-%d").strftime("%Y-%m-%d")

                    doc_data = {
                        "tipo_solicitud": tipo_solicitud,
                        "telefono": telefono,
                        "nombre_cliente": nombre_cliente if tipo_solicitud == "Cliente Delivery" else None,
                        "sucursal": sucursal if tipo_solicitud == "Sucursal" else None,
                        "coordenadas_recojo": {"lat": lat, "lon": lon},
                        "coordenadas_entrega": {"lat": lat, "lon": lon},
                        "direccion_recojo": direccion,
                        "direccion_entrega": None,
                        "fecha_recojo": fecha,
                        "fecha_entrega": None,
                        "hora_recojo": None,
                        "hora_entrega": None
                    }

                    db.collection("recogidas").add(doc_data)

                except Exception as e:
                    errores += 1
                    st.warning(f"⚠️ Error en fila {i + 2}: {e}")

            st.success(f"✅ Se subieron {total - errores} registros correctamente. {errores} errores.")
            st.cache_data.clear()
            # -----------------------------------------------
    # ❌ ELIMINAR RUTAS DE LA FECHA SELECCIONADA
    # -----------------------------------------------
    st.markdown("---")
    st.subheader("❌ Eliminar rutas del día")

    with st.expander("⚠️ Esta acción eliminará todos los pedidos (recogidas y entregas) de la fecha seleccionada."):
        confirmar = st.checkbox("Sí, quiero eliminar todos los registros de esta fecha")

        if confirmar:
            if st.button("🗑️ Eliminar todas las rutas de esta fecha"):
                try:
                    fecha_str = fecha_seleccionada.strftime("%Y-%m-%d")
                    recojo_docs = list(db.collection("recogidas").where("fecha_recojo", "==", fecha_str).stream())
                    entrega_docs = list(db.collection("recogidas").where("fecha_entrega", "==", fecha_str).stream())

                    todos_ids = set([doc.id for doc in recojo_docs + entrega_docs])

                    for doc_id in todos_ids:
                        db.collection("recogidas").document(doc_id).delete()

                    st.success(f"✅ Se eliminaron {len(todos_ids)} documentos correspondientes a {fecha_str}.")
                    st.cache_data.clear()
                    time.sleep(2)
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error al eliminar rutas: {e}")
