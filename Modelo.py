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

# ══════════════════════════════════════════════════════════
# SISTEMA DE LOGIN — Faditex IoT
# ══════════════════════════════════════════════════════════
USERS = {
    'admin':   {'password': 'admin2026',   'role': 'Administrador',   'name': 'Administrador Sistema', 'initial': 'A'},
    'Faditex': {'password': 'Faditex2026', 'role': 'Usuario Faditex', 'name': 'Operario Faditex',      'initial': 'F'},
}

for _k, _v in [('logged_in', False), ('username', ''), ('role', ''), ('display_name', ''), ('initial', 'U')]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

if not st.session_state.logged_in:
    # Inyectar CSS de la pantalla de login
    components.html("""
    <script>
    (function(){
      var doc = window.parent.document;
      if (!doc.getElementById('faditex-login-css')) {
        var s = doc.createElement('style'); s.id='faditex-login-css';
        s.textContent = [
          "[data-testid='stApp']{background:linear-gradient(135deg,#0a0a1a 0%,#0d1428 60%,#0a1020 100%)!important}",
          "[data-testid='stMain']{background:transparent!important}",
          "[data-testid='stVerticalBlock']{align-items:center}",
          ".login-card{background:rgba(255,255,255,.04);border:1px solid rgba(0,229,255,.15);border-radius:24px;padding:40px 36px;backdrop-filter:blur(20px);box-shadow:0 24px 60px rgba(0,0,0,.6);max-width:420px;margin:0 auto}",
          ".login-logo{font-size:3rem;text-align:center;margin-bottom:8px}",
          ".login-brand{font-size:1.8rem;font-weight:800;text-align:center;color:#ffffff;letter-spacing:2px;font-family:Inter,sans-serif}",
          ".login-sub{font-size:.8rem;text-align:center;color:#8899bb;margin-top:6px;margin-bottom:28px;font-family:Inter,sans-serif;letter-spacing:.5px}",
          ".login-divider{border:none;border-top:1px solid rgba(255,255,255,.08);margin:20px 0}",
          ".login-hint{font-size:.72rem;color:#556688;text-align:center;margin-top:16px;font-family:Inter,sans-serif}",
          "[data-testid='stSidebar']{display:none!important}"
        ].join(' ');
        doc.head.appendChild(s);
      }
    })();
    </script>
    """, height=0)

    st.markdown("""
    <div style='height:60px'></div>
    <div class='login-card'>
        <div class='login-logo'>🏭</div>
        <div class='login-brand'>FADITEX</div>
        <div class='login-sub'>Sistema de Monitoreo Ambiental Industrial<br>Adafruit IO &middot; Isolation Forest ML</div>
        <hr class='login-divider'>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 1.4, 1])
    with col_c:
        with st.form('login_form', clear_on_submit=False):
            usuario    = st.text_input("👤  Usuario",    placeholder="Ingresa tu usuario")
            contrasena = st.text_input("🔑  Contraseña", placeholder="Ingresa tu contraseña", type="password")
            submitted  = st.form_submit_button("Iniciar Sesión", use_container_width=True, type="primary")

        if submitted:
            if usuario in USERS and USERS[usuario]['password'] == contrasena:
                st.session_state.logged_in    = True
                st.session_state.username     = usuario
                st.session_state.role         = USERS[usuario]['role']
                st.session_state.display_name = USERS[usuario]['name']
                st.session_state.initial      = USERS[usuario]['initial']
                st.rerun()
            else:
                st.error("❌ Usuario o contraseña incorrectos. Intenta de nuevo.")

        st.markdown("<div class='login-hint'>© 2026 Faditex S.A. &nbsp;·&nbsp; Tesis IoT</div>",
                    unsafe_allow_html=True)
    st.stop()


def crear_gauge(valor, titulo, rango, unidad, color_barra):
    umbral_aviso = rango[0] + (rango[1]-rango[0])*0.6
    umbral_peligro = rango[0] + (rango[1]-rango[0])*0.85
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = valor,
        domain = {'x': [0, 1], 'y': [0.15, 1]},   # ← sube el arco, deja espacio para el número
        title = {'text': f"{titulo} ({unidad})", 'font': {'size': 13, 'color': '#aaaaaa'}},
        number = {'font': {'size': 18, 'color': '#888888'}, 'suffix': f" {unidad}"},
        gauge = {
            'axis': {'range': [rango[0], rango[1]], 'tickwidth': 1, 'tickcolor': "#888888",
                     'tickfont': {'size': 10}},
            'bar': {'color': color_barra, 'thickness': 0.7},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [rango[0], umbral_aviso],    'color': 'rgba(0, 230, 118, 0.08)'},
                {'range': [umbral_aviso, umbral_peligro], 'color': 'rgba(255, 170, 0, 0.08)'},
                {'range': [umbral_peligro, rango[1]],  'color': 'rgba(255, 75, 75, 0.08)'}
            ]
        }
    ))
    fig.update_layout(
        height=230,
        margin=dict(l=20, r=20, t=36, b=55),   # ← b=55 evita el solapamiento
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#aaaaaa', family='Inter, sans-serif')
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

@st.cache_data(ttl=3600)
def cargar_datos_historicos_adafruit():
    feeds = ['co2', 'humedad', 'sensor-de-sonido', 'temperatura', 'tvoc']
    dataframes = []
    
    for feed in feeds:
        try:
            # Obtener TODOS los registros históricos almacenados en el feed de Adafruit IO
            registros = aio.data(feed, max_results=None)
            if not registros:
                continue
                
            df_f = pd.DataFrame([{
                'Timestamp': pd.to_datetime(r.created_at),
                feed: float(r.value)
            } for r in registros])
            
            df_f['Timestamp'] = df_f['Timestamp'].dt.tz_localize(None)
            df_f.set_index('Timestamp', inplace=True)
            dataframes.append(df_f)
            
            # Pausa para no saturar los límites de la API de Adafruit
            time.sleep(1.5)
        except Exception:
            continue
            
    if not dataframes:
        return pd.DataFrame()
        
    # Unir todos los dataframes por su marca de tiempo y sincronizarlos (Resample 10s)
    df_historico = dataframes[0]
    for i in range(1, len(dataframes)):
        df_historico = df_historico.join(dataframes[i], how='outer')
        
    df_historico = df_historico.resample('10s').mean().ffill().dropna()
    df_historico.reset_index(inplace=True)
    
    if 'sensor-de-sonido' in df_historico.columns:
        df_historico.rename(columns={'sensor-de-sonido': 'ruido'}, inplace=True)
        
    # Formatear el Timestamp a string para asegurar compatibilidad en el CSV
    df_historico['Timestamp'] = df_historico['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
    return df_historico

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

    components.html("""
    <script>
    (function(){
      var doc = window.parent.document;
      if (!doc.getElementById('faditex-font')) {
        var lk = doc.createElement('link'); lk.id='faditex-font'; lk.rel='stylesheet';
        lk.href='https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap';
        doc.head.appendChild(lk);
      }
      function isDark() {
        // Streamlit 1.x stores theme in multiple possible keys
        var keys = ['stActiveTheme', 'theme', 'streamlit:theme'];
        for (var i=0; i<keys.length; i++) {
          try {
            var raw = window.parent.localStorage.getItem(keys[i]);
            if (raw) {
              // Could be JSON {"name":"Dark"} or plain string "Dark"
              var val = raw.trim().replace(/^"|"$/g,'');
              try { val = JSON.parse(raw); if(val && val.name) val = val.name; } catch(e){}
              if (typeof val === 'string') {
                if (val.toLowerCase() === 'light') return false;
                if (val.toLowerCase() === 'dark')  return true;
              }
            }
          } catch(e){}
        }
        // Fallback: check if Streamlit's native header is light (white bg = light mode)
        var toolbar = doc.querySelector('[data-testid="stHeader"]') || doc.querySelector('header');
        if (toolbar) {
          var bg = window.parent.getComputedStyle(toolbar).backgroundColor;
          var m = bg.match(/\d+/g);
          if (m && parseInt(m[0])+parseInt(m[1])+parseInt(m[2]) > 600) return false; // light header
        }
        // Final fallback: check app background
        var app = doc.querySelector('[data-testid="stApp"]');
        if (app) {
          var m2 = window.parent.getComputedStyle(app).backgroundColor.match(/\d+/g);
          if (m2) return parseInt(m2[0])+parseInt(m2[1])+parseInt(m2[2]) < 400;
        }
        return true; // default dark
      }
      function darkCSS(){return [
        /* Font solo en contenido del app, NO en iconos de Streamlit */
        "[data-testid='stApp'],[data-testid='stSidebar'],[data-testid='stMain'],p,h1,h2,h3,button,input,select,textarea,.z-header,.z-card,.z-section-title,.kpi-label,.kpi-value,.kpi-unit,.sidebar-brand-name,.sidebar-brand-sub,.sidebar-section-label,.sidebar-footer{font-family:'Inter',sans-serif!important}",
        "[data-testid='stApp']{background:linear-gradient(135deg,#0d0d1a,#111128,#0a1628)!important}",
        "section[data-testid='stSidebar']{background:#0e0e20!important;border-right:1px solid rgba(255,255,255,.06)!important}",
        /* ── Estilos Modernos del Menú Lateral (Radio) ── */
        "section[data-testid='stSidebar'] [data-testid='stRadio'] div[role='radiogroup'] > label > div:first-child { display: none !important; }",
        "section[data-testid='stSidebar'] [data-testid='stRadio'] div[role='radiogroup'] > label > div:last-child { margin-left: 0 !important; }",
        "section[data-testid='stSidebar'] [data-testid='stRadio'] div[role='radiogroup'] > label { padding: 12px 18px!important; border-radius: 10px!important; transition: all .2s ease-in-out!important; margin-bottom: 2px!important; cursor: pointer!important; background: transparent!important; width: 100%!important; border: 1px solid transparent!important; }",
        "section[data-testid='stSidebar'] [data-testid='stRadio'] div[role='radiogroup'] > label:hover { background: rgba(255,255,255,0.05)!important; border: 1px solid rgba(255,255,255,0.08)!important; }",
        "section[data-testid='stSidebar'] [data-testid='stRadio'] div[role='radiogroup'] > label:has(input:checked) { background: rgba(255,255,255,0.12)!important; border: 1px solid rgba(255,255,255,0.15)!important; box-shadow: 0 4px 12px rgba(0,0,0,0.2)!important; }",
        "section[data-testid='stSidebar'] [data-testid='stRadio'] p { color: #888899 !important; font-size: .88rem !important; font-weight: 500 !important; margin: 0 !important; transition: color .2s!important; }",
        "section[data-testid='stSidebar'] [data-testid='stRadio'] label:hover p { color: #ffffff !important; }",
        "section[data-testid='stSidebar'] [data-testid='stRadio'] label:has(input:checked) p { color: #ffffff !important; font-weight: 700 !important; }",
        /* Botón de Cerrar Sesión (Estilo similar al menú) */
        "section[data-testid='stSidebar'] button[kind='secondary'] { background: transparent !important; border: 1px solid transparent !important; padding: 12px 18px !important; border-radius: 10px !important; justify-content: flex-start !important; height: auto !important; margin-bottom: 8px !important; transition: all .2s!important; }",
        "section[data-testid='stSidebar'] button[kind='secondary'] > div { color: #888899 !important; font-size: .88rem !important; font-weight: 500 !important; transition: color .2s!important; }",
        "section[data-testid='stSidebar'] button[kind='secondary']:hover { background: rgba(255,75,75,0.1) !important; border: 1px solid rgba(255,75,75,0.2) !important; color: #ff4b4b !important; }",
        "section[data-testid='stSidebar'] button[kind='secondary']:hover > div { color: #ff4b4b !important; }",
        /* ── Sidebar user card — DARK ── */
        ".sb-user-card{display:flex;align-items:center;gap:14px;padding:20px 18px 18px;border-bottom:1px solid rgba(255,255,255,.08);margin-bottom:6px}",
        ".sb-avatar{width:46px;height:46px;border-radius:50%;background:linear-gradient(135deg,#00b4d8,#0077b6);color:#fff;font-size:1.2rem;font-weight:800;display:flex;align-items:center;justify-content:center;flex-shrink:0;box-shadow:0 4px 14px rgba(0,229,255,.3)}",
        ".sb-user-role{font-size:.58rem;font-weight:700;color:#00e5ff;text-transform:uppercase;letter-spacing:1.5px}",
        ".sb-user-name{font-size:.88rem;font-weight:700;color:#ffffff;margin-top:2px}",
        ".sb-nav-label{font-size:.58rem;font-weight:700;color:#444466;text-transform:uppercase;letter-spacing:2px;padding:14px 18px 6px}",
        ".sb-logout{margin:14px 10px 0;border-top:1px solid rgba(255,255,255,.08);padding-top:14px}",
        ".sidebar-brand{padding:24px 20px 16px;border-bottom:1px solid rgba(255,255,255,.07);margin-bottom:4px}",
        ".sidebar-brand-name{font-size:1.1rem;font-weight:800;color:#fff!important}",
        ".sidebar-brand-sub{font-size:.6rem;color:#888899!important;text-transform:uppercase;letter-spacing:1.5px;margin-top:3px}",
        ".sidebar-section-label{font-size:.6rem;font-weight:700;color:#666688!important;text-transform:uppercase;letter-spacing:2px;padding:14px 20px 6px}",
        ".sidebar-footer{margin-top:28px;padding:12px 20px;border-top:1px solid rgba(255,255,255,.07);font-size:.62rem;color:#555577!important;text-align:center;line-height:1.7}",
        ".z-header{background:linear-gradient(90deg,rgba(0,229,255,.08),rgba(255,75,75,.06));border:1px solid rgba(0,229,255,.18);border-radius:18px;padding:22px 28px;margin-bottom:20px}",
        ".z-logo-title{font-size:clamp(1.1rem,2.5vw,1.75rem);font-weight:800;color:#fff!important;white-space:normal;overflow-wrap:break-word}",
        ".z-logo-sub{font-size:.82rem;color:#aac!important;margin-top:5px}",
        ".z-section-title{font-size:.72rem;font-weight:700;color:#00e5ff!important;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;border-left:3px solid #00e5ff;padding-left:10px}",
        ".z-card{background:rgba(30,30,50,.8)!important;border:1px solid rgba(255,255,255,.08)!important;border-radius:16px;padding:18px 22px;backdrop-filter:blur(12px);box-shadow:0 8px 32px rgba(0,0,0,.4);margin-bottom:8px}",
        ".kpi-label{font-size:.7rem;color:#8888aa!important;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}",
        ".kpi-unit{font-size:.72rem;color:#8888aa!important}",
        ".status-badge{display:inline-flex;align-items:center;gap:8px;padding:7px 16px;border-radius:30px;font-size:.85rem;font-weight:600}",
        ".badge-ok{color:#00e676;border:1.5px solid #00e676;background:rgba(0,230,118,.1)}",
        ".badge-alert{color:#ff4b4b;border:1.5px solid #ff4b4b;background:rgba(255,75,75,.1)}",
        ".pulse-dot{width:10px;height:10px;border-radius:50%;display:inline-block}",
        ".dot-ok{background:#00e676;animation:pok 1.4s infinite}",
        ".dot-alert{background:#ff4b4b;animation:pal 1s infinite}",
        "@keyframes pok{0%,100%{box-shadow:0 0 0 0 rgba(0,230,118,.6)}50%{box-shadow:0 0 0 7px rgba(0,230,118,0)}}",
        "@keyframes pal{0%,100%{box-shadow:0 0 0 0 rgba(255,75,75,.6)}50%{box-shadow:0 0 0 7px rgba(255,75,75,0)}}",
        "[data-testid='stMetric']{background:rgba(30,30,50,.8)!important;border:1px solid rgba(255,255,255,.08)!important;border-radius:14px!important}",
        "[data-testid='stMetricValue']{color:#00e676!important;font-size:1.8rem;font-weight:700}",
        "h1,h2,h3{color:#fff!important}",
        /* Forzar color de los números Plotly en modo oscuro */
        "[data-testid='stPlotlyChart'] text.number,[data-testid='stPlotlyChart'] .number text,[data-testid='stPlotlyChart'] g.number{fill:#e0e0e0!important}"
      ].join(" ");}
      function lightCSS(){return [
        /* Font */
        "[data-testid='stApp'],[data-testid='stSidebar'],[data-testid='stMain'],p,h1,h2,h3,button,input,select,textarea,.z-header,.z-card,.z-section-title,.kpi-label,.kpi-value,.kpi-unit,.sidebar-brand-name,.sidebar-brand-sub,.sidebar-section-label,.sidebar-footer{font-family:'Inter',sans-serif!important}",
        /* Fondo de app más saturado para que las tarjetas blancas resalten */
        "[data-testid='stApp']{background:linear-gradient(150deg,#c8d8f0 0%,#d2ddf0 40%,#ccd6ee 100%)!important}",
        /* Sidebar */
        "section[data-testid='stSidebar']{background:#dde6f7!important;border-right:2px solid rgba(0,80,160,.15)!important}",
        /* ── Estilos Modernos del Menú Lateral (Radio) ── */
        "section[data-testid='stSidebar'] [data-testid='stRadio'] div[role='radiogroup'] > label > div:first-child { display: none !important; }",
        "section[data-testid='stSidebar'] [data-testid='stRadio'] div[role='radiogroup'] > label > div:last-child { margin-left: 0 !important; }",
        "section[data-testid='stSidebar'] [data-testid='stRadio'] div[role='radiogroup'] > label { padding: 12px 18px!important; border-radius: 10px!important; transition: all .2s ease-in-out!important; margin-bottom: 2px!important; cursor: pointer!important; background: transparent!important; width: 100%!important; border: 1px solid transparent!important; }",
        "section[data-testid='stSidebar'] [data-testid='stRadio'] div[role='radiogroup'] > label:hover { background: rgba(0,80,160,0.06)!important; border: 1px solid rgba(0,80,160,0.1)!important; }",
        "section[data-testid='stSidebar'] [data-testid='stRadio'] div[role='radiogroup'] > label:has(input:checked) { background: rgba(0,80,160,0.12)!important; border: 1px solid rgba(0,80,160,0.2)!important; box-shadow: 0 4px 12px rgba(0,50,150,0.08)!important; }",
        "section[data-testid='stSidebar'] [data-testid='stRadio'] p { color: #556688 !important; font-size: .88rem !important; font-weight: 500 !important; margin: 0 !important; transition: color .2s!important; }",
        "section[data-testid='stSidebar'] [data-testid='stRadio'] label:hover p { color: #0a1a33 !important; }",
        "section[data-testid='stSidebar'] [data-testid='stRadio'] label:has(input:checked) p { color: #0a1a33 !important; font-weight: 700 !important; }",
        /* Botón de Cerrar Sesión (Estilo similar al menú) */
        "section[data-testid='stSidebar'] button[kind='secondary'] { background: transparent !important; border: 1px solid transparent !important; padding: 12px 18px !important; border-radius: 10px !important; justify-content: flex-start !important; height: auto !important; margin-bottom: 8px !important; transition: all .2s!important; }",
        "section[data-testid='stSidebar'] button[kind='secondary'] > div { color: #556688 !important; font-size: .88rem !important; font-weight: 500 !important; transition: color .2s!important; }",
        "section[data-testid='stSidebar'] button[kind='secondary']:hover { background: rgba(191,32,24,0.1) !important; border: 1px solid rgba(191,32,24,0.2) !important; color: #bf2018 !important; }",
        "section[data-testid='stSidebar'] button[kind='secondary']:hover > div { color: #bf2018 !important; }",
        ".sidebar-brand{padding:24px 20px 16px;border-bottom:1px solid rgba(0,60,140,.15);margin-bottom:4px}",
        ".sidebar-brand-name{font-size:1.1rem;font-weight:800;color:#0a1a33!important}",
        ".sidebar-brand-sub{font-size:.6rem;color:#3a5577!important;text-transform:uppercase;letter-spacing:1.5px;margin-top:3px}",
        ".sidebar-section-label{font-size:.6rem;font-weight:700;color:#3a5577!important;text-transform:uppercase;letter-spacing:2px;padding:14px 20px 6px}",
        ".sidebar-footer{margin-top:28px;padding:12px 20px;border-top:1px solid rgba(0,60,140,.15);font-size:.62rem;color:#556688!important;text-align:center;line-height:1.7}",
        /* Header — blanco sólido con sombra para destacar del fondo azul */
        ".z-header{background:#ffffff!important;border:1px solid rgba(0,100,200,.2)!important;border-radius:18px;padding:22px 28px;margin-bottom:20px;box-shadow:0 4px 24px rgba(0,50,150,.14)}",
        ".z-logo-title{font-size:clamp(1.1rem,2.5vw,1.75rem);font-weight:800;color:#0a1a33!important;white-space:normal;overflow-wrap:break-word}",
        ".z-logo-sub{font-size:.82rem;color:#334466!important;margin-top:5px}",
        /* Títulos de sección — azul oscuro fuerte */
        ".z-section-title{font-size:.72rem;font-weight:800;color:#004488!important;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;border-left:3px solid #0066cc;padding-left:10px}",
        /* Tarjetas — blancas sólidas con sombra pronunciada para contraste con el fondo */
        ".z-card{background:#ffffff!important;border:1px solid rgba(0,80,200,.15)!important;border-radius:16px;padding:18px 22px;backdrop-filter:none;box-shadow:0 4px 20px rgba(0,50,150,.14),0 1px 4px rgba(0,0,0,.08);margin-bottom:8px}",
        /* KPI labels y units más oscuros */
        ".kpi-label{font-size:.7rem;color:#334466!important;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;font-weight:600}",
        ".kpi-unit{font-size:.72rem;color:#445577!important;font-weight:500}",
        /* Badges */
        ".status-badge{display:inline-flex;align-items:center;gap:8px;padding:7px 16px;border-radius:30px;font-size:.85rem;font-weight:600}",
        ".badge-ok{color:#0d6e3a;border:2px solid #0d6e3a;background:rgba(13,110,58,.1)}",
        ".badge-alert{color:#bf2018;border:2px solid #bf2018;background:rgba(191,32,24,.1)}",
        ".pulse-dot{width:10px;height:10px;border-radius:50%;display:inline-block}",
        ".dot-ok{background:#0d6e3a;animation:pok 1.4s infinite}",
        ".dot-alert{background:#bf2018;animation:pal 1s infinite}",
        "@keyframes pok{0%,100%{box-shadow:0 0 0 0 rgba(13,110,58,.6)}50%{box-shadow:0 0 0 7px rgba(13,110,58,0)}}",
        "@keyframes pal{0%,100%{box-shadow:0 0 0 0 rgba(191,32,24,.6)}50%{box-shadow:0 0 0 7px rgba(191,32,24,0)}}",
        /* Metrics */
        "[data-testid='stMetric']{background:#ffffff!important;border:1px solid rgba(0,80,200,.15)!important;border-radius:14px!important;box-shadow:0 4px 16px rgba(0,50,150,.12)}",
        "[data-testid='stMetricValue']{color:#0d6e3a!important;font-size:1.8rem;font-weight:700}",
        "h1,h2,h3{color:#0a1a33!important}",
        ".z-card [style*='border-bottom']{border-bottom-color:rgba(0,60,140,.12)!important}",
        /* ── Sidebar user card — LIGHT ── */
        ".sb-user-card{display:flex;align-items:center;gap:14px;padding:20px 18px 18px;border-bottom:1px solid rgba(0,60,140,.15);margin-bottom:6px}",
        ".sb-avatar{width:46px;height:46px;border-radius:50%;background:linear-gradient(135deg,#0066cc,#0044aa);color:#fff;font-size:1.2rem;font-weight:800;display:flex;align-items:center;justify-content:center;flex-shrink:0;box-shadow:0 4px 12px rgba(0,80,200,.35)}",
        ".sb-user-role{font-size:.58rem;font-weight:700;color:#0055aa;text-transform:uppercase;letter-spacing:1.5px}",
        ".sb-user-name{font-size:.88rem;font-weight:700;color:#0a1a33;margin-top:2px}",
        ".sb-nav-label{font-size:.58rem;font-weight:700;color:#4466aa;text-transform:uppercase;letter-spacing:2px;padding:14px 18px 6px}",
        ".sb-logout{margin:14px 10px 0;border-top:1px solid rgba(0,60,140,.15);padding-top:14px}",
        /* Forzar color oscuro en los números Plotly en modo claro */
        "[data-testid='stPlotlyChart'] text.number,[data-testid='stPlotlyChart'] .number text,[data-testid='stPlotlyChart'] g.number{fill:#112244!important}"
      ].join(" ");}

      function apply(){
        var loginCSS=doc.getElementById('faditex-login-css');
        if(loginCSS){loginCSS.remove();}
        var dark=isDark();
        var el=doc.getElementById('faditex-css');
        if(!el){el=doc.createElement('style');el.id='faditex-css';doc.head.appendChild(el);}
        el.textContent=dark?darkCSS():lightCSS();
      }
      apply();
      window.parent.addEventListener('storage',apply);
      new MutationObserver(apply).observe(doc.body,{attributes:true,attributeFilter:['class','data-theme']});
      setInterval(apply, 1500);
    })();
    </script>
    """, height=0)

    st.markdown("""
    <div class='z-header'>
        <div class='z-logo-title'>🏭 FADITEX &nbsp;·&nbsp; Monitor Ambiental Industrial</div>
        <div class='z-logo-sub'>Sistema Inteligente IoT — Adafruit IO + Isolation Forest ML</div>
    </div>
    """, unsafe_allow_html=True)


    # ── Menú de Navegación Lateral — Rediseñado con tarjeta de usuario ──
    _ini  = st.session_state.get('initial', 'U')
    _rol  = st.session_state.get('role', 'Usuario')
    _name = st.session_state.get('display_name', 'Usuario')

    with st.sidebar:
        # Tarjeta de usuario al estilo de la imagen de referencia
        st.markdown(f"""
        <div class='sb-user-card'>
            <div class='sb-avatar'>{_ini}</div>
            <div>
                <div class='sb-user-role'>{_rol}</div>
                <div class='sb-user-name'>{_name}</div>
            </div>
        </div>
        <div class='sb-nav-label'>Principal</div>
        """, unsafe_allow_html=True)

        opciones_menu = [
            "📍 Monitor Principal",
            "📈 Gráficas Históricas",
            "📑 Reportes",
            "📥 Descargar Reporte Histórico"
        ]
        
        # Solo permitir acceso a Calibración si es el Administrador
        if st.session_state.get('username') == 'admin':
            opciones_menu.append("⚙️ Calibración")

        menu_seleccionado = st.radio("Sección", opciones_menu, label_visibility="collapsed")

        st.markdown("<div class='sb-nav-label'>Sistema</div>", unsafe_allow_html=True)

        # Botón de cerrar sesión
        st.markdown("<div class='sb-logout'></div>", unsafe_allow_html=True)
        if st.button("🚪  Cerrar Sesión", use_container_width=True, key="btn_logout"):
            for _k in ['logged_in', 'username', 'role', 'display_name', 'initial',
                       'monitoreando', 'historial_lecturas', 'historial_grafico', 'total_incidencias']:
                if _k in st.session_state:
                    del st.session_state[_k]
            st.rerun()

        st.markdown("""
        <div class='sidebar-footer'>
            Sistema de Monitoreo Ambiental<br>Faditex S.A. &copy; 2026
        </div>
        """, unsafe_allow_html=True)

    if "ventana_suavizado" not in st.session_state:
        st.session_state.ventana_suavizado = 3
    if "forzar_alarma" not in st.session_state:
        st.session_state.forzar_alarma = False

    if "monitoreando" not in st.session_state:
        st.session_state.monitoreando = False
    if "historial_lecturas" not in st.session_state:
        st.session_state.historial_lecturas = []
    if "historial_grafico" not in st.session_state:
        st.session_state.historial_grafico = pd.DataFrame()
    if "total_incidencias" not in st.session_state:
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

    if not st.session_state.monitoreando:
        if st.button("Iniciar Monitoreo en Vivo", type="primary", use_container_width=True):
            st.session_state.monitoreando = True
            st.session_state.historial_lecturas = []
            st.session_state.historial_grafico = pd.DataFrame()
            st.session_state.total_incidencias = 0
            st.rerun()
            
    if st.session_state.monitoreando:
        
        @st.fragment(run_every=10)
        def ciclo_monitoreo():
            df_actual = obtener_datos_actuales()
            
            if isinstance(df_actual, pd.DataFrame):
                # 1. Ingresar lectura al buffer
                st.session_state.historial_lecturas.append(df_actual)
                
                # 2. Eliminar lecturas viejas si superamos el límite del slider
                if len(st.session_state.historial_lecturas) > ventana_suavizado:
                    st.session_state.historial_lecturas.pop(0)
                    
                # 3. Fase de Calibración
                if len(st.session_state.historial_lecturas) < ventana_suavizado:
                    msg = f"⏳ **Calibrando IA...** Por favor espera mientras el filtro anti-ruido absorbe y estabiliza el impacto eléctrico inicial del encendido ({len(st.session_state.historial_lecturas)} de {ventana_suavizado} lecturas requeridas)."
                    prog = len(st.session_state.historial_lecturas) / float(ventana_suavizado)
                    st.info(msg)
                    st.progress(prog)
                    return
                
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
                
                # ════════════════════════════════════════════════════════
                # MÉTODO Z — MONITOR PRINCIPAL
                # ════════════════════════════════════════════════════════
                if menu_seleccionado == "📍 Monitor Principal":

                    # Variables de estado para el HTML
                    es_alerta = riesgo_porcentaje >= 50
                    badge_class = "badge-alert" if es_alerta else "badge-ok"
                    dot_class  = "dot-alert"  if es_alerta else "dot-ok"
                    estado_txt  = "⚠ ALERTA ROJA" if es_alerta else "✔ Sistema Normal"
                    modo_txt    = f"🧪 MODO PRUEBA · Severidad Simulada" if forzar_alarma else ""

                    # ── ZONA Z PUNTO 1 & 2: KPIs Superiores (Izq → Der) ──
                    st.markdown("<div class='z-section-title'>▲ Punto 1 · 2 — KPIs Críticos</div>", unsafe_allow_html=True)
                    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns([2, 1, 1, 1])

                    with kpi_col1:
                        color_sev = "#ff4b4b" if riesgo_porcentaje >= 50 else ("#ffa726" if riesgo_porcentaje >= 30 else "#00e676")
                        st.markdown(f"""
                        <div class='z-card' style='border-color:{color_sev}44;'>
                            <div class='kpi-label'>Índice de Severidad IA</div>
                            <div class='kpi-value' style='color:{color_sev}; font-size:2.6rem;'>{riesgo_porcentaje}<span class='kpi-unit'>%</span></div>
                            <div style='margin-top:10px; background:rgba(255,255,255,0.06); border-radius:6px; height:10px; overflow:hidden;'>
                                <div style='width:{riesgo_porcentaje}%; height:100%; background:{color_sev}; border-radius:6px; transition:width 0.6s ease;'></div>
                            </div>
                            {f"<div style='color:#ffa726; font-size:0.75rem; margin-top:6px;'>{modo_txt}</div>" if forzar_alarma else ""}
                        </div>
                        """, unsafe_allow_html=True)

                    with kpi_col2:
                        co2_val = df_calibrado['co2'][0]
                        co2_color = "#ff4b4b" if co2_val > 1200 else ("#ffa726" if co2_val > 800 else "#00e676")
                        st.markdown(f"""
                        <div class='z-card' style='border-color:{co2_color}44;'>
                            <div class='kpi-label'>💨 CO₂</div>
                            <div class='kpi-value' style='color:{co2_color};'>{co2_val:.0f}</div>
                            <div class='kpi-unit'>ppm</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with kpi_col3:
                        temp_val = df_calibrado['temperatura'][0]
                        temp_color = "#ff4b4b" if temp_val > 35 else ("#ffa726" if temp_val > 28 else "#00e676")
                        st.markdown(f"""
                        <div class='z-card' style='border-color:{temp_color}44;'>
                            <div class='kpi-label'>🌡️ Temperatura</div>
                            <div class='kpi-value' style='color:{temp_color};'>{temp_val:.1f}</div>
                            <div class='kpi-unit'>°C</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with kpi_col4:
                        st.markdown(f"""
                        <div class='z-card' style='display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:100px;'>
                            <div class='kpi-label'>Estado del Sistema</div>
                            <div class='status-badge {badge_class}' style='margin-top:10px;'>
                                <div class='pulse-dot {dot_class}'></div>
                                {estado_txt}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Alarma sonora si aplica
                    if es_alerta:
                        st.session_state.total_incidencias += 1
                        components.html("""<audio autoplay><source src="https://actions.google.com/sounds/v1/alarms/digital_watch_alarm_long.ogg" type="audio/ogg"></audio>""", width=0, height=0)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # ── ZONA Z PUNTO 3: Gauges (Diagonal central) ──
                    st.markdown("<div class='z-section-title'>◆ Punto 3 — Sensores en Tiempo Real</div>", unsafe_allow_html=True)
                    g1, g2, g3, g4, g5 = st.columns(5)
                    with g1:
                        st.plotly_chart(crear_gauge(df_calibrado['co2'][0], "CO2", [400, 2000], "ppm", "#ff4b4b"), use_container_width=True, key="g_co2")
                    with g2:
                        st.plotly_chart(crear_gauge(df_calibrado['humedad'][0], "Humedad", [0, 100], "%", "#00e676"), use_container_width=True, key="g_hum")
                    with g3:
                        st.plotly_chart(crear_gauge(df_calibrado['ruido'][0], "Ruido", [40, 130], "dB", "#1e90ff"), use_container_width=True, key="g_rui")
                    with g4:
                        st.plotly_chart(crear_gauge(df_calibrado['temperatura'][0], "Temp.", [0, 50], "°C", "#ffa500"), use_container_width=True, key="g_tem")
                    with g5:
                        st.plotly_chart(crear_gauge(df_calibrado['tvoc'][0], "TVOC", [0, 500], "ppb", "#9400d3"), use_container_width=True, key="g_tvc")

                    # ── ZONA Z PUNTO 4: Panel Inferior (Izq → Der) ──
                    st.markdown("<div class='z-section-title'>▼ Punto 4 — Tendencia 24h & Resumen de Sesión</div>", unsafe_allow_html=True)
                    col_graf, col_resumen = st.columns([3, 1])

                    with col_graf:
                        fig = px.scatter(
                            st.session_state.historial_grafico,
                            x='Timestamp', y='Riesgo', color='Estado',
                            color_discrete_map={'Normal': '#00e676', 'Incidencia': '#ff4b4b'},
                            title="Flujo de Riesgo — Últimas 24 horas"
                        )
                        fig.update_layout(
                            xaxis_title="Hora", yaxis_title="Nivel de Riesgo (%)",
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#aaaaaa', family='Inter'),
                            margin=dict(l=0, r=0, t=36, b=0),
                            legend_title_text='',
                            title_font=dict(size=13, color='#8888aa')
                        )
                        st.plotly_chart(fig, use_container_width=True, key="24h_scatter")

                    with col_resumen:
                        tvoc_val = df_calibrado['tvoc'][0]
                        hum_val  = df_calibrado['humedad'][0]
                        rui_val  = df_calibrado['ruido'][0]
                        st.markdown(f"""
                        <div class='z-card' style='height:100%;'>
                            <div class='kpi-label' style='margin-bottom:14px;'>📋 Resumen de Sesión</div>
                            <div style='display:flex; flex-direction:column; gap:10px;'>
                                <div style='display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:8px;'>
                                    <span style='color:#8888aa; font-size:0.78rem;'>🔴 Incidencias</span>
                                    <span style='color:#ff4b4b; font-weight:700; font-size:1rem;'>{st.session_state.total_incidencias}</span>
                                </div>
                                <div style='display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:8px;'>
                                    <span style='color:#8888aa; font-size:0.78rem;'>💨 TVOC</span>
                                    <span style='color:#9400d3; font-weight:600;'>{tvoc_val:.0f} ppb</span>
                                </div>
                                <div style='display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:8px;'>
                                    <span style='color:#8888aa; font-size:0.78rem;'>💧 Humedad</span>
                                    <span style='color:#00e676; font-weight:600;'>{hum_val:.1f} %</span>
                                </div>
                                <div style='display:flex; justify-content:space-between; padding-bottom:4px;'>
                                    <span style='color:#8888aa; font-size:0.78rem;'>🔊 Ruido</span>
                                    <span style='color:#1e90ff; font-weight:600;'>{rui_val:.1f} dB</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # --- 2. GRÁFICAS INDIVIDUALES ---
                elif menu_seleccionado == "📈 Gráficas Históricas":
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
                elif menu_seleccionado == "📑 Reportes":
                    st.markdown("### Resumen Estadístico de la Sesión")
                    cols_sensor = ['co2', 'humedad', 'ruido', 'temperatura', 'tvoc']
                    est_df = st.session_state.historial_grafico[cols_sensor].describe().T[['min', 'mean', 'max']]
                    est_df.columns = ['Mínimo Histórico', 'Promedio', 'Máximo Alcanzado']
                    st.dataframe(est_df.style.format("{:.1f}"))

                    st.divider()

                # --- 4. DESCARGA HISTÓRICA ---
                elif menu_seleccionado == "📥 Descargar Reporte Histórico":
                    st.markdown("### 📥 Base de Datos Histórica (Adafruit IO)")
                    st.info("A continuación se presenta el registro histórico completo del sistema. Haz clic en el botón inferior para exportarlo a Excel (CSV).")
                    
                    # Cargar el histórico consolidado desde la API
                    with st.spinner("Conectando con Adafruit IO para descargar todo el historial... (Esto puede tardar unos minutos si hay mucha data)"):
                        df_historico = cargar_datos_historicos_adafruit()
                    
                    # Combinarlo con los datos de la sesión actual si existen
                    if not st.session_state.historial_grafico.empty:
                        df_mostrar = pd.concat([df_historico, st.session_state.historial_grafico], ignore_index=True)
                    else:
                        df_mostrar = df_historico
                        
                    st.dataframe(df_mostrar, use_container_width=True)
                    
                    if not df_mostrar.empty:
                        csv = df_mostrar.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Descargar Data Histórica Completa (CSV)",
                            data=csv,
                            file_name=f"historico_adafruit_completo_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv",
                            type="primary",
                            use_container_width=True
                        )
            else:
                st.error("¡Fallo Crítico de Conexión en Adafruit IO!")
                st.error(f"Motivo Real: {df_actual}")  
                st.warning("Reintentando conexión automática a Adafruit en 10 segundos...")

        # Iniciar el fragmento
        ciclo_monitoreo()