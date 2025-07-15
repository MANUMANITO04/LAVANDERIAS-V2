import streamlit as st
import re
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
from core.firebase import db, obtener_sucursales
from core.geo_utils import obtener_sugerencias_direccion, obtener_direccion_desde_coordenadas

def solicitar_recogida():
    # Reinicio seguro de campos si se activó la bandera
    if st.session_state.get("reset_solicitud", False):
        for key in [
            "delivery_lat", "delivery_lon", "delivery_direccion", "delivery_data",
            "delivery_mapa", "delivery_marker", "nombre_cliente", "telefono"
        ]:
            st.session_state.pop(key, None)
        st.session_state["reset_solicitud"] = False
        st.rerun()

    # Inicialización segura
    st.session_state.setdefault("delivery_data", {"direccion": "", "lat": None, "lon": None})
    st.session_state.setdefault("delivery_lat", -16.409047)
    st.session_state.setdefault("delivery_lon", -71.537451)
    st.session_state.setdefault("delivery_direccion", "Arequipa, Perú")
    st.session_state.setdefault("delivery_mapa", folium.Map(
        location=[st.session_state["delivery_lat"], st.session_state["delivery_lon"]],
        zoom_start=15
    ))
    st.session_state.setdefault("nombre_cliente", "")
    st.session_state.setdefault("telefono", "")

    # Encabezado
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://github.com/Melisa2303/LAVANDERIAS-V2/raw/main/data/LOGO.PNG", width=100)
    with col2:
        st.markdown("<h1 style='text-align: left; color: black;'>Lavanderías Americanas</h1>", unsafe_allow_html=True)
    st.title("🛒 Solicitar Recogida")

    def calcular_fecha_entrega(fecha_recojo):
        dia_semana = fecha_recojo.weekday()
        if dia_semana == 5: return fecha_recojo + timedelta(days=0)
        return fecha_recojo + timedelta(days=0)

    tipo_solicitud = st.radio("Tipo de Solicitud", ["Sucursal", "Cliente Delivery"], horizontal=True)

    if tipo_solicitud == "Sucursal":
        sucursales = obtener_sucursales()
        nombres_sucursales = [s["nombre"] for s in sucursales]
        nombre_sucursal = st.selectbox("Seleccionar Sucursal", nombres_sucursales)
        sucursal_seleccionada = next((s for s in sucursales if s["nombre"] == nombre_sucursal), None)
        if not sucursal_seleccionada:
            st.error("Datos de sucursal incompletos.")
            return

        lat = sucursal_seleccionada["coordenadas"]["lat"]
        lon = sucursal_seleccionada["coordenadas"]["lon"]
        direccion = sucursal_seleccionada["direccion"]
        st.markdown(f"**Dirección:** {direccion}")
        fecha_recojo = st.date_input("Fecha de Recojo", min_value=datetime.now().date())

        if st.button("💾 Solicitar Recogida"):
            #fecha_entrega = calcular_fecha_entrega(fecha_recojo)
            fecha_entrega = calcular_fecha_entrega(fecha_recojo)
            solicitud = {
                "tipo_solicitud": tipo_solicitud,
                "sucursal": nombre_sucursal,
                "direccion_recojo": direccion,
                "coordenadas_recojo": {"lat": lat, "lon": lon},
                "direccion_entrega": direccion,
                "coordenadas_entrega": {"lat": lat, "lon": lon},
                "fecha_recojo": fecha_recojo.strftime("%Y-%m-%d"),
                "fecha_entrega": fecha_entrega.strftime("%Y-%m-%d"),
            }
            try:
                db.collection('recogidas').add(solicitud)
                st.success(f"Recogida agendada. Entrega el {fecha_entrega.strftime('%d/%m/%Y')}")
                st.session_state["reset_solicitud"] = True
                st.rerun()
            except Exception as e:
                st.error(f"Error al guardar: {e}")

    elif tipo_solicitud == "Cliente Delivery":
        col1, col2 = st.columns(2)
        with col1:
            nombre_cliente = st.text_input("Nombre del Cliente", value=st.session_state.get("nombre_cliente", ""), key="nombre_cliente")
        with col2:
            telefono = st.text_input("Teléfono", max_chars=9, value=st.session_state.get("telefono", ""), key="telefono")

        direccion_input = st.text_input(
            "Dirección",
            value=st.session_state.get("delivery_direccion", ""),
            key="delivery_direccion_input"
        )

        sugerencias = []
        if direccion_input and direccion_input != st.session_state["delivery_direccion"]:
            sugerencias = obtener_sugerencias_direccion(direccion_input)

        direccion_seleccionada = st.selectbox(
            "Sugerencias de Direcciones:",
            ["Seleccione una dirección"] + [sug["display_name"] for sug in sugerencias] if sugerencias else ["No hay sugerencias"],
            key="delivery_sugerencias"
        )

        if direccion_seleccionada and direccion_seleccionada != "Seleccione una dirección":
            for sug in sugerencias:
                if direccion_seleccionada == sug["display_name"]:
                    st.session_state["delivery_lat"] = float(sug["lat"])
                    st.session_state["delivery_lon"] = float(sug["lon"])
                    st.session_state["delivery_direccion"] = direccion_seleccionada
                    st.session_state["delivery_mapa"] = folium.Map(
                        location=[st.session_state["delivery_lat"], st.session_state["delivery_lon"]],
                        zoom_start=15
                    )
                    folium.Marker(
                        [st.session_state["delivery_lat"], st.session_state["delivery_lon"]],
                        tooltip="Punto seleccionado"
                    ).add_to(st.session_state["delivery_mapa"])
                    break

        mapa = st_folium(
            st.session_state["delivery_mapa"],
            width=700,
            height=500,
            key="delivery_mapa_folium"
        )

        if mapa.get("last_clicked"):
            st.session_state["delivery_lat"] = mapa["last_clicked"]["lat"]
            st.session_state["delivery_lon"] = mapa["last_clicked"]["lng"]
            st.session_state["delivery_direccion"] = obtener_direccion_desde_coordenadas(
                st.session_state["delivery_lat"], st.session_state["delivery_lon"]
            )
            st.session_state["delivery_mapa"] = folium.Map(
                location=[st.session_state["delivery_lat"], st.session_state["delivery_lon"]],
                zoom_start=15
            )
            folium.Marker(
                [st.session_state["delivery_lat"], st.session_state["delivery_lon"]],
                tooltip="Punto seleccionado"
            ).add_to(st.session_state["delivery_mapa"])
            st.rerun()

        st.markdown(f"""
            <div style='background-color: #f0f8ff; padding: 10px; border-radius: 5px; margin-top: 10px;'>
                <h4 style='color: #333; margin: 0;'>Dirección Final:</h4>
                <p style='color: #555; font-size: 16px;'>{st.session_state["delivery_direccion"]}</p>
            </div>
        """, unsafe_allow_html=True)

        fecha_recojo = st.date_input("Fecha de Recojo", min_value=datetime.now().date())

        if st.button("💾 Solicitar Recogida"):
            if not nombre_cliente:
                st.error("El nombre del cliente es obligatorio.")
                return
            if not re.match(r"^\d{9}$", telefono):
                st.error("El teléfono debe tener 9 dígitos.")
                return

            fecha_entrega = calcular_fecha_entrega(fecha_recojo)
            solicitud = {
                "tipo_solicitud": tipo_solicitud,
                "nombre_cliente": nombre_cliente,
                "telefono": telefono,
                "direccion_recojo": st.session_state["delivery_direccion"],
                "coordenadas_recojo": {
                    "lat": st.session_state["delivery_lat"],
                    "lon": st.session_state["delivery_lon"]
                },
                "direccion_entrega": st.session_state["delivery_direccion"],
                "coordenadas_entrega": {
                    "lat": st.session_state["delivery_lat"],
                    "lon": st.session_state["delivery_lon"]
                },
                "fecha_recojo": fecha_recojo.strftime("%Y-%m-%d"),
                "fecha_entrega": fecha_entrega.strftime("%Y-%m-%d")
            }

            try:
                db.collection('recogidas').add(solicitud)
                st.success(f"Recogida agendada. Entrega el {fecha_entrega.strftime('%d/%m/%Y')}")
                st.session_state["reset_solicitud"] = True
                st.rerun()
            except Exception as e:
                st.error(f"Error al guardar: {e}")
