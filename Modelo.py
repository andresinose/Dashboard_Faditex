import os
import time
import pandas as pd
import joblib
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from Adafruit_IO import Client

# 1. CONFIGURACIÓN E IMPORTACIÓN 
st.set_page_config(page_title="Monitor en Vivo - Faditex", page_icon="🔴", layout="wide")

def crear_gauge(valor, titulo, rango, unidad, color_barra):
    umbral_aviso = rango[0] + (rango[1]-rango[0])*0.6
    umbral_peligro = rango[0] + (rango[1]-rango[0])*0.85
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = valor,
        title = {'text': f"{titulo} ({unidad})", 'font': {'size': 14, 'color': '#aaaaaa'}},
        number = {'font': {'size': 20, 'color': '#ffffff'}},
        gauge = {
            'axis': {'range': [rango[0], rango[1]], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color_barra},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [rango[0], umbral_aviso], 'color': 'rgba(0, 230, 118, 0.1)'},
                {'range': [umbral_aviso, umbral_peligro], 'color': 'rgba(255, 170, 0, 0.1)'},
                {'range': [umbral_peligro, rango[1]], 'color': 'rgba(255, 75, 75, 0.1)'}
            ]
        }
    ))
    fig.update_layout(
        height=220, 
        margin=dict(l=15, r=15, t=40, b=15),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#aaaaaa')
    )
    return fig

# Credenciales SECURE de Adafruit IO (La clave actual se lee de Secrets)
ADAFRUIT_IO_USERNAME = st.secrets["ADAFRUIT_IO_USERNAME"]
ADAFRUIT_IO_KEY = st.secrets["ADAFRUIT_IO_KEY"]
aio = Client(ADAFRUIT_IO_USERNAME, ADAFRUIT_IO_KEY)

# 2. CARGAR EL CEREBRO DE LA IA 
@st.cache_resource
def cargar_ia():
    ruta_base = os.path.dirname(os.path.abspath(__file__))
    ruta_scaler = os.path.join(ruta_base, 'escalador_faditex.pkl')
    ruta_modelo = os.path.join(ruta_base, 'modelo_if_faditex.pkl')
    
    scaler = joblib.load(ruta_scaler)
    modelo = joblib.load(ruta_modelo)
    return scaler, modelo

try:
    scaler, modelo_if = cargar_ia()
    ia_lista = True
except FileNotFoundError:
    st.error("ERROR: No se encuentran los archivos .pkl del modelo entrenado en esta carpeta.")
    ia_lista = False

# 3. FUNCIÓN PARA LEER LA API
def obtener_datos_actuales():
    try:
        feeds = ['co2', 'humedad', 'sensor-de-sonido', 'temperatura', 'tvoc']
        valores = {}
        for feed in feeds:
            dato = aio.receive(feed)
            valores[feed] = float(dato.value)
            
        df_live = pd.DataFrame([{
            'co2': valores['co2'],
            'humedad': valores['humedad'],
            'ruido': valores['sensor-de-sonido'],
            'temperatura': valores['temperatura'],
            'tvoc': valores['tvoc']
        }])
        return df_live
    except Exception as e:
        return str(e)

#4. INTERFAZ GRÁFICA Y CALIBRACIÓN DE RUIDO
if ia_lista:

    st.markdown("""
        <style>
        [data-testid="stMetric"] {
            background-color: #1e1e2e;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.4);
            border: 1px solid #333;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.8rem;
            color: #00e676;
        }
        h1 {
            color: #ff4b4b;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Monitor Ambiental Avanzado")
    st.markdown("<p style='text-align: center; color: #aaaaaa; font-size: 1.2rem; margin-bottom: 30px;'>Sistema Inteligente IoT de Faditex enlazado con Adafruit IO</p>", unsafe_allow_html=True)
    
    # Menú de Navegación Lateral
    with st.sidebar:
        st.markdown("## 🧭 Menú de Navegación")
        menu_seleccionado = st.radio("Secciones", [
            "📍 Monitor Principal", 
            "📈 Gráficas Históricas", 
            "📑 Reportes", 
            "⚙️ Calibración"
        ])

    if "ventana_suavizado" not in st.session_state:
        st.session_state.ventana_suavizado = 3
    if "forzar_alarma" not in st.session_state:
        st.session_state.forzar_alarma = False

    if "monitoreando" not in st.session_state:
        st.session_state.monitoreando = False
    if "st.session_state.historial_lecturas" not in st.session_state:
        st.session_state.historial_lecturas = []
    if "st.session_state.historial_grafico" not in st.session_state:
        st.session_state.historial_grafico = pd.DataFrame()
    if "st.session_state.total_incidencias" not in st.session_state:
        st.session_state.total_incidencias = 0

    if menu_seleccionado == "⚙️ Calibración":
        st.markdown("### Configuración del Filtro Analítico")
        ventana_suavizado = st.slider(
            "Filtro Anti-Ruido (Lecturas a promediar)", 
            min_value=1, max_value=10, value=st.session_state.ventana_suavizado, 
            help="Absorbe picos eléctricos falsos promediando las últimas N lecturas antes de pasarlas a la IA.",
            key="slider_ventana"
        )
        st.session_state.ventana_suavizado = ventana_suavizado
        
        st.divider()
        st.markdown("### Área de Pruebas (Herramientas Tesis)")
        forzar_alarma = st.checkbox("🧪 Forzar Sistema a Incidencia Global (Simular Riesgo Alto para prueba de sonido)", value=st.session_state.forzar_alarma, key="check_alarma")
        st.session_state.forzar_alarma = forzar_alarma
    else:
        ventana_suavizado = st.session_state.ventana_suavizado
        forzar_alarma = st.session_state.forzar_alarma

    placeholder_principal = st.empty() if menu_seleccionado == "📍 Monitor Principal" else None
    placeholder_graficas = st.empty() if menu_seleccionado == "📈 Gráficas Históricas" else None
    placeholder_reportes = st.empty() if menu_seleccionado == "📑 Reportes" else None

    placeholder_boton = st.empty()
    if not st.session_state.monitoreando:
        if placeholder_boton.button("Iniciar Monitoreo en Vivo", type="primary", use_container_width=True):
            st.session_state.monitoreando = True
            st.session_state.historial_lecturas = []
            st.session_state.historial_grafico = pd.DataFrame()
            st.session_state.total_incidencias = 0
            placeholder_boton.empty()
            st.rerun()
            
    if st.session_state.monitoreando:
        placeholder_boton.empty() # Desvanece el botón para limpiar la interfaz
        while True:
            df_actual = obtener_datos_actuales()
            
            if isinstance(df_actual, pd.DataFrame):
                # 1. Ingresar lectura al buffer
                st.session_state.historial_lecturas.append(df_actual)
                
                # 2. Eliminar lecturas viejas si superamos el límite del slider
                if len(st.session_state.historial_lecturas) > ventana_suavizado:
                    st.session_state.historial_lecturas.pop(0)
                    
                # 3. Fase de Calibración
                if len(st.session_state.historial_lecturas) < ventana_suavizado:
                    if placeholder_principal is not None:
                        with placeholder_principal.container():
                            st.info(f"⏳ **Calibrando IA...** Por favor espera mientras el filtro anti-ruido absorbe y estabiliza el impacto eléctrico inicial del encendido ({len(st.session_state.historial_lecturas)} de {ventana_suavizado} lecturas requeridas).")
                            st.progress(len(st.session_state.historial_lecturas) / float(ventana_suavizado))
                    time.sleep(10)
                    continue
                
                # 4. Aplicar Filtro: Promediar el buffer
                df_calibrado = pd.concat(st.session_state.historial_lecturas).mean().to_frame().T
                
                # 5. Inferencia y Nivel de Severidad
                dato_normalizado = scaler.transform(df_calibrado)
                prediccion = modelo_if.predict(dato_normalizado)
                
                # Extraer el "Anomaly Score" bruto del modelo para el termómetro
                decision_score = modelo_if.decision_function(dato_normalizado)[0]
                # Convertir el score matemático a un porcentaje de Peligro (0 a 100%)
                riesgo_porcentaje = max(0, min(100, int(50 - (decision_score * 200))))
                
                # Inyección forzada desde interfaz de pruebas
                if forzar_alarma:
                    riesgo_porcentaje = 95
                    df_calibrado['co2'] = 2500
                    df_calibrado['temperatura'] = 60
                    df_calibrado['tvoc'] = 600
                
                # Enriquecer datos para el historial de 24h
                df_calibrado['Timestamp'] = pd.Timestamp.now()
                df_calibrado['Riesgo'] = riesgo_porcentaje
                df_calibrado['Estado'] = 'Incidencia' if riesgo_porcentaje >= 50 else 'Normal'
                
                # Guardar para la gráfica (24h = 8640 lecturas de 10s)
                st.session_state.historial_grafico = pd.concat([st.session_state.historial_grafico, df_calibrado], ignore_index=True).tail(8640)
                
                # --- 1. MONITOR PRINCIPAL ---
                if placeholder_principal is not None:
                    with placeholder_principal.container():
                        # PANEL DE ALERTAS Y TERMÓMETRO
                        if forzar_alarma:
                            st.markdown(f"### 🧪 MODO PRUEBA ACTIVO: Severidad Simulada al **{riesgo_porcentaje}%**")
                        else:
                            st.markdown(f"### Índice de Severidad: **{riesgo_porcentaje}%**")
                        st.progress(riesgo_porcentaje / 100.0)

                        # Mostrar resultados finales
                        if riesgo_porcentaje >= 50:
                            st.session_state.total_incidencias += 1
                            st.error("¡ALERTA ROJA! Anomalía ambiental detectada en la planta. Riesgo exponencial.")
                            # Alarma Sonora
                            audio_html = """
                                <audio autoplay>
                                    <source src="https://actions.google.com/sounds/v1/alarms/digital_watch_alarm_long.ogg" type="audio/ogg">
                                </audio>
                            """
                            components.html(audio_html, width=0, height=0)
                        else:
                            st.success("Estado del Sistema: Parámetros dentro del reglamento.")

                        st.markdown("### Tablero de Componentes Dinámicos")
                        g1, g2, g3, g4, g5 = st.columns(5)
                        # Generamos una llave única para esta iteración del loop
                        loop_k = int(time.time() * 1000)
                        with g1:
                            st.plotly_chart(crear_gauge(df_calibrado['co2'][0], "CO2", [400, 2000], "ppm", "#ff4b4b"), use_container_width=True, key=f"g_co2_{loop_k}")
                        with g2:
                            st.plotly_chart(crear_gauge(df_calibrado['humedad'][0], "Humedad", [0, 100], "%", "#00e676"), use_container_width=True, key=f"g_hum_{loop_k}")
                        with g3:
                            st.plotly_chart(crear_gauge(df_calibrado['ruido'][0], "Ruido", [40, 130], "dB", "#1e90ff"), use_container_width=True, key=f"g_rui_{loop_k}")
                        with g4:
                            st.plotly_chart(crear_gauge(df_calibrado['temperatura'][0], "Temp.", [0, 50], "°C", "#ffa500"), use_container_width=True, key=f"g_tem_{loop_k}")
                        with g5:
                            st.plotly_chart(crear_gauge(df_calibrado['tvoc'][0], "TVOC", [0, 500], "ppb", "#9400d3"), use_container_width=True, key=f"g_tvc_{loop_k}")

                        st.divider()

                        # PANEL DE INCIDENCIAS (24 HORAS)
                        st.markdown("### Flujo de Estado e Incidencias (Últimas 24 horas)")
                        col_inc1, col_inc2 = st.columns([1, 3])
                        with col_inc1:
                            st.metric(label="Número de Incidencias", value=st.session_state.total_incidencias)
                        with col_inc2:
                            fig = px.scatter(
                                st.session_state.historial_grafico, 
                                x='Timestamp', 
                                y='Riesgo', 
                                color='Estado',
                                color_discrete_map={'Normal': '#00e676', 'Incidencia': '#ff4b4b'},
                            )
                            fig.update_layout(
                                xaxis_title="Hora",
                                yaxis_title="Nivel de Riesgo (%)",
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#aaaaaa'),
                                margin=dict(l=0, r=0, t=0, b=0),
                                legend_title_text=''
                            )
                            st.plotly_chart(fig, use_container_width=True, key=f"24h_scatter_{loop_k}")
                
                # --- 2. GRÁFICAS INDIVIDUALES ---
                if placeholder_graficas is not None:
                    with placeholder_graficas.container():
                        # GRÁFICOS INDIVIDUALES POR SENSOR
                        st.markdown("### Análisis de Tendencia Individual (Histórico)")

                        # Fila 1 de gráficos
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.markdown("**CO2 (ppm)**")
                            st.line_chart(st.session_state.historial_grafico['co2'], height=180, color="#ff4b4b")
                        with c2:
                            st.markdown("**Humedad (%)**")
                            st.line_chart(st.session_state.historial_grafico['humedad'], height=180, color="#00e676")
                        with c3:
                            st.markdown("**Temperatura (°C)**")
                            st.line_chart(st.session_state.historial_grafico['temperatura'], height=180, color="#ffa500")

                        # Fila 2 de gráficos
                        c4, c5 = st.columns(2)
                        with c4:
                            st.markdown("**Ruido Ambiental (dB)**")
                            st.line_chart(st.session_state.historial_grafico['ruido'], height=180, color="#1e90ff")
                        with c5:
                            st.markdown("**Gases TVOC (ppb)**")
                            st.line_chart(st.session_state.historial_grafico['tvoc'], height=180, color="#9400d3")
                
                # --- 3. REPORTES ---
                if placeholder_reportes is not None:
                    with placeholder_reportes.container():
                        st.markdown("### Resumen Estadístico de la Sesión")
                        cols_sensor = ['co2', 'humedad', 'ruido', 'temperatura', 'tvoc']
                        est_df = st.session_state.historial_grafico[cols_sensor].describe().T[['min', 'mean', 'max']]
                        est_df.columns = ['Mínimo Histórico', 'Promedio', 'Máximo Alcanzado']
                        st.dataframe(est_df.style.format("{:.1f}"))

                        st.divider()

                        # TABLA DE DATOS Y EXPORTACIÓN
                        st.markdown("### Registro de Datos Histórico (Crudo)")
                        st.info("Descarga estos datos como CSV desplazando el cursor sobre la tabla inferior y dando click al botón de descarga flotante.")
                        st.dataframe(st.session_state.historial_grafico)
            else:
                if placeholder_principal is not None:
                    with placeholder_principal.container():
                        st.error("¡Fallo Crítico de Conexión en Adafruit IO!")
                        st.error(f"Motivo Real: {df_actual}")  
                        st.warning("Reintentando conexión automática a Adafruit en 10 segundos...")
                if placeholder_graficas is not None:
                    with placeholder_graficas.container():
                        st.warning("Detenido por error de sincronización de Adafruit...")
                if placeholder_reportes is not None:
                    with placeholder_reportes.container():
                        st.warning("Proceso estancado esperando a Adafruit...")
                    
            time.sleep(10)