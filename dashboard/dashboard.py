# ============================================================
# VERSIÓN COMPLETA · CONSERVA LA ESTRUCTURA ORIGINAL
# Correcciones: claves únicas, arrays Plotly seguros y fondo verde claro.
# ============================================================
# Requisitos:
#   pip install streamlit pandas numpy plotly reportlab kaleido
#
# Ejecución:
#   streamlit run dashboard_ejecutivo.py
#
# El dashboard conserva las tablas/columnas del DW utilizado
# por la versión anterior, pero cambia las gráficas a Plotly:
# - información al pasar el cursor
# - zoom y selección
# - títulos y subtítulos claros
# - periodos de tiempo visibles
# - diseño más limpio y espacioso
# ============================================================

import json
import sqlite3
import re
from io import BytesIO
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="SIG-LOG | Dashboard Ejecutivo",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data_warehouse" / "siglog_dw.db"

MODELOS_DIR = BASE_DIR / "modelos_entrenados"
KMEANS_DIR = BASE_DIR / "no_supervisado" / "resultados"
PCA_DIR = BASE_DIR / "pca" / "resultados"
MANT_DIR = BASE_DIR / "mantenimiento" / "resultados"
REPORTES_MODELOS_DIR = BASE_DIR / "reportes_modelos"

try:
    from modulos_operativos_siglog import render_modulo
except Exception:
    render_modulo = None


# ============================================================
# COLORES
# ============================================================

COLORS = {
    "azul": "#2563EB",
    "azul_oscuro": "#1E3A8A",
    "azul_claro": "#60A5FA",
    "verde": "#16A34A",
    "amarillo": "#F59E0B",
    "naranja": "#EA580C",
    "rojo": "#DC2626",
    "morado": "#7C3AED",
    "turquesa": "#0891B2",
    "gris": "#64748B",
    "gris_claro": "#E2E8F0",
    "fondo": "#F0F8F3",
    "blanco": "#FFFFFF",
    "negro": "#0F172A",
}

STATUS_COLORS = {
    "Entregada": COLORS["verde"],
    "Tardía": COLORS["rojo"],
    "Tardia": COLORS["rojo"],
    "Pendiente": COLORS["amarillo"],
    "Cancelada": COLORS["gris"],
}

MAINT_COLORS = {
    "Preventivo": COLORS["verde"],
    "Correctivo": COLORS["amarillo"],
    "Emergencia": COLORS["rojo"],
}

RISK_COLORS = {
    "BAJO": COLORS["verde"],
    "MEDIO": COLORS["amarillo"],
    "ALTO": COLORS["naranja"],
    "CRÍTICO": COLORS["rojo"],
    "CRITICO": COLORS["rojo"],
}

MONTHS_ES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
}

MONTH_SHORT = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Ago",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic",
}


# ============================================================
# ESTILO
# ============================================================

st.markdown(
    """
    <style>
    /* ============================================================
       SIG-LOG · UI EJECUTIVA VERDE
       ============================================================ */
    :root{
        --bg:#EAF7F0;
        --surface:#FFFFFF;
        --surface-soft:#F6FCF8;
        --green:#159447;
        --green-dark:#08743A;
        --green-soft:#E7F7EC;
        --teal:#0B8791;
        --ink:#10231A;
        --muted:#60756B;
        --border:#CFE5D8;
        --danger:#E5484D;
        --shadow:0 10px 28px rgba(21,72,48,.07);
    }

    html, body, [class*="css"]{
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
                     "Segoe UI", Arial, sans-serif;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    section.main,
    section.main > div{
        background:var(--bg)!important;
        color:var(--ink)!important;
    }

    [data-testid="stHeader"]{
        background:rgba(234,247,240,.96)!important;
        border-bottom:0!important;
        box-shadow:none!important;
        height:2.4rem!important;
    }

    [data-testid="stToolbar"]{
        background:transparent!important;
        border:0!important;
    }

    .block-container{
        max-width:1540px;
        padding-top:.55rem!important;
        padding-bottom:3.5rem;
    }

    /* ========================= TOPBAR ========================= */
    .topbar{
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:18px;
        min-height:48px;
        padding:4px 4px 10px;
        border-bottom:1px solid rgba(207,229,216,.65);
        margin-bottom:12px;
    }
    .topbar-brand{
        display:flex;
        align-items:center;
        gap:10px;
        min-width:0;
    }
    .topbar-logo{
        font-size:27px;
        line-height:1;
    }
    .topbar-name{
        color:var(--green-dark);
        font-size:20px;
        font-weight:900;
        letter-spacing:-.35px;
    }
    .topbar-desc{
        color:#536B60;
        font-size:12px;
        font-weight:600;
        white-space:nowrap;
    }
    .topbar-right{
        display:flex;
        align-items:center;
        gap:18px;
        color:#476257;
        font-size:12px;
        font-weight:700;
        white-space:nowrap;
    }
    .topbar-date{
        display:flex;
        align-items:center;
        gap:6px;
    }
    .topbar-exit{
        border:1px solid #9ACFB0;
        border-radius:9px;
        background:#FFFFFF;
        color:var(--green-dark);
        padding:8px 14px;
        font-weight:800;
    }

    /* ========================= SIDEBAR ========================= */
    [data-testid="stSidebar"]{
        background:#FFFFFF!important;
        border-right:1px solid #D8E9DF!important;
        box-shadow:4px 0 18px rgba(16,64,40,.035);
    }
    [data-testid="stSidebar"] > div:first-child{
        padding-top:.8rem!important;
    }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span{
        color:#29473A!important;
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3{
        color:#103A26!important;
    }

    /* Radio navigation: remove the black circles and create menu rows. */
    [data-testid="stSidebar"] [role="radiogroup"]{
        gap:2px!important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] > label{
        border-radius:9px!important;
        padding:7px 9px!important;
        margin:0!important;
        transition:all .15s ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] > label:hover{
        background:#EFFAF3!important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked){
        background:#E7F7EC!important;
        box-shadow:inset 3px 0 0 #16A34A;
    }
    [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p,
    [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) span{
        color:#08743A!important;
        font-weight:850!important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] input{
        accent-color:#16A34A!important;
    }
    [data-testid="stSidebar"] .stButton button{
        background:#FFFFFF!important;
        color:#08743A!important;
        border:1px solid #BBDCC8!important;
        border-radius:10px!important;
        font-weight:800!important;
    }

    /* ========================= HERO ========================= */
    .hero{
        position:relative;
        overflow:hidden;
        background:linear-gradient(120deg,#078B4B 0%,#149D59 48%,#0D8D97 100%);
        border-radius:22px;
        padding:25px 30px 24px;
        color:#FFFFFF;
        margin:0 0 18px 0;
        box-shadow:0 14px 34px rgba(10,115,62,.16);
    }
    .hero:after{
        content:"";
        position:absolute;
        right:-90px;
        top:-100px;
        width:310px;
        height:310px;
        border-radius:50%;
        background:rgba(255,255,255,.08);
    }
    .hero-row{
        position:relative;
        z-index:1;
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:25px;
    }
    .hero-brand{
        display:flex;
        align-items:center;
        gap:15px;
        min-width:0;
    }
    .hero-icon{font-size:47px;line-height:1}
    .hero h1{
        margin:0;
        color:#FFFFFF!important;
        font-size:42px;
        line-height:1;
        font-weight:900;
        letter-spacing:-1.3px;
    }
    .hero p{
        margin:9px 0 0;
        color:#E9FFF1!important;
        font-size:14px;
        font-weight:600;
    }
    .hero-meta{
        position:relative;
        z-index:1;
        display:flex;
        gap:30px;
        align-items:center;
        color:#FFFFFF;
    }
    .hero-meta-item{
        display:flex;
        gap:10px;
        align-items:center;
        min-width:140px;
    }
    .hero-meta-icon{font-size:24px}
    .hero-meta-label{
        color:#D8F8E4;
        font-size:10px;
        font-weight:700;
    }
    .hero-meta-value{
        margin-top:2px;
        font-size:13px;
        font-weight:850;
    }
    .period-badge{
        display:inline-block;
        margin-top:14px;
        padding:7px 11px;
        border-radius:999px;
        background:rgba(255,255,255,.14);
        border:1px solid rgba(255,255,255,.24);
        color:#FFFFFF;
        font-size:11px;
        font-weight:800;
    }

    /* ========================= CONTROL / FILTERS ========================= */
    .filter-title{
        display:flex;
        align-items:center;
        gap:12px;
        background:#FFFFFF;
        border:1px solid var(--border);
        border-radius:18px 18px 0 0;
        padding:15px 19px 8px;
        margin:0;
        box-shadow:var(--shadow);
    }
    .filter-icon{
        width:34px;height:34px;border-radius:10px;
        display:flex;align-items:center;justify-content:center;
        background:#E7F7EC;color:#0B8B49;font-size:18px;
    }
    .filter-kicker{
        color:#0B743A;
        font-size:11px;
        font-weight:900;
        letter-spacing:.9px;
    }
    .filter-main{
        color:#10231A;
        font-size:16px;
        font-weight:900;
        margin-top:1px;
    }
    .filter-help{
        color:#61776C;
        font-size:11.5px;
        margin-top:2px;
    }
    .filter-box{
        background:#FFFFFF;
        border:1px solid var(--border);
        border-top:0;
        border-radius:0 0 18px 18px;
        padding:7px 14px 14px;
        box-shadow:var(--shadow);
        margin-bottom:11px;
    }

    /* BaseWeb / Streamlit multiselects: light, tall and readable. */
    div[data-baseweb="select"]{
        width:100%!important;
    }
    div[data-baseweb="select"] > div{
        border-radius:10px!important;
        border:1px solid #BFD9CA!important;
        background:#FFFFFF!important;
        box-shadow:none!important;
        min-height:48px!important;
        height:auto!important;
        padding:3px 5px!important;
    }
    div[data-baseweb="select"] > div:hover{
        border-color:#73B58C!important;
    }
    div[data-baseweb="select"] input{
        color:#183B2B!important;
        background:#FFFFFF!important;
    }
    div[data-baseweb="select"] span{
        color:#183B2B!important;
    }
    div[data-baseweb="select"] [data-baseweb="tag"]{
        background:#E7F7EC!important;
        border:1px solid #C7E8D1!important;
        border-radius:7px!important;
        margin:2px!important;
        max-width:100%!important;
    }
    div[data-baseweb="select"] [data-baseweb="tag"] span{
        color:#08743A!important;
        font-weight:750!important;
    }
    div[data-baseweb="select"] svg{
        color:#4D6D5C!important;
    }
    .stMultiSelect label,
    .stSelectbox label{
        color:#315445!important;
        font-size:11px!important;
        font-weight:850!important;
        margin-bottom:3px!important;
    }
    [data-testid="stMultiSelect"]{
        margin-bottom:3px!important;
    }
    .filter-summary{
        display:flex;
        flex-wrap:wrap;
        gap:7px;
        align-items:center;
        background:#F4FBF7;
        border:1px solid #D1E9D9;
        border-radius:12px;
        padding:9px 13px;
        margin:0 0 18px;
        color:#60756A;
        font-size:11px;
    }
    .filter-summary strong{color:#173D2B}
    .filter-summary .dot{color:#A5C3B1}

    .stButton>button{
        border-radius:10px!important;
        min-height:48px!important;
        border:1px solid #B6D8C3!important;
        background:#0D964B!important;
        color:#FFFFFF!important;
        font-weight:850!important;
        box-shadow:0 5px 12px rgba(13,150,75,.13)!important;
    }
    .stButton>button:hover{
        background:#08743A!important;
        border-color:#08743A!important;
        color:#FFFFFF!important;
    }

    /* ========================= KPI ========================= */
    .kpi{
        background:#FFFFFF;
        border:1px solid #D7E9DE;
        border-radius:16px;
        padding:15px 16px 13px;
        min-height:108px;
        box-shadow:var(--shadow);
        position:relative;
    }
    .kpi .icon{font-size:20px}
    .kpi .label{
        color:#60756A!important;
        font-size:11px;
        font-weight:800;
        margin-top:5px;
    }
    .kpi .value{
        color:#08743A!important;
        font-size:23px;
        font-weight:900;
        margin-top:2px;
        white-space:nowrap;
    }
    .kpi .extra{
        color:#7A8E84!important;
        font-size:10.5px;
        margin-top:3px;
    }

    /* ========================= CHART CARDS ========================= */
    .chart-card{
        background:#FFFFFF;
        border:1px solid #D7E9DE;
        border-radius:17px;
        padding:12px 14px 10px;
        margin-bottom:18px;
        box-shadow:var(--shadow);
        overflow:hidden;
    }
    .chart-heading{
        font-size:17px;
        line-height:1.3;
        font-weight:900;
        color:#173C2B!important;
        margin:3px 4px 2px;
    }
    .chart-description{
        color:#667B70!important;
        font-size:12px;
        line-height:1.45;
        margin:0 4px 7px;
    }

    /* Download buttons visually match the reference. */
    .chart-card .stDownloadButton button{
        min-height:38px!important;
        border-radius:9px!important;
        background:#FFFFFF!important;
        color:#08743A!important;
        border:1px solid #91CBA5!important;
        font-size:11px!important;
        font-weight:850!important;
        box-shadow:none!important;
    }
    .chart-card .stDownloadButton button:hover{
        background:#EFFAF3!important;
        border-color:#159447!important;
    }

    /* ========================= INTERPRETACIÓN ========================= */
    .interpretacion{
        background:#F3FBF6;
        border:1px solid #D6EBDD;
        border-left:4px solid #159447;
        border-radius:11px;
        padding:12px 15px;
        margin:2px 0 16px;
    }
    .interpretacion-title{
        color:#08743A!important;
        font-size:13px;
        font-weight:900;
        margin-bottom:4px;
    }
    .interpretacion-text{
        color:#4D665A!important;
        font-size:12.5px;
        line-height:1.5;
    }
    .accion-card{
        background:#FFFFFF;
        border:1px solid #D7E9DE;
        border-radius:14px;
        padding:15px 16px;
        margin-bottom:10px;
        box-shadow:0 5px 15px rgba(16,64,40,.035);
    }
    .accion-title{
        color:#173C2B!important;
        font-weight:900;
        font-size:13px;
        margin-bottom:5px;
    }
    .accion-text{
        color:#60756A!important;
        font-size:12.5px;
        line-height:1.5;
    }

    /* ========================= SECCIONES / TABLAS ========================= */
    .section-title{
        font-size:25px;
        font-weight:900;
        color:#123A27!important;
        margin:12px 0 3px;
        letter-spacing:-.3px;
    }
    .section-subtitle{
        color:#60756A!important;
        font-size:13px;
        line-height:1.55;
        margin-bottom:16px;
    }
    .notice{
        border-radius:12px;
        padding:12px 15px;
        margin:8px 0 16px;
        font-size:13px;
        line-height:1.5;
    }
    .notice.info{background:#EFFAF3;color:#08743A!important;border-left:4px solid #159447}
    .notice.warning{background:#FFF8EA;color:#9A5B00!important;border-left:4px solid #E9A52B}
    .notice.danger{background:#FFF1F1;color:#A72B30!important;border-left:4px solid #E5484D}

    .question-card{
        background:#FFFFFF;
        border:1px solid #D7E9DE;
        border-radius:18px;
        padding:18px 18px 14px;
        margin:0 0 18px;
        box-shadow:var(--shadow);
    }
    .question-title{font-size:19px;font-weight:900;color:#173C2B;margin-bottom:5px}
    .question-sub{font-size:12.5px;line-height:1.5;color:#667B70;margin-bottom:12px}
    .business-filter{
        background:#F5FCF7;
        border:1px solid #D5EBDD;
        border-radius:14px;
        padding:12px 14px;
        margin:8px 0 14px;
    }
    .download-note{font-size:10.5px;color:#667B70;margin-top:3px}

    div[data-testid="stDataFrame"]{
        border:1px solid #D7E9DE;
        border-radius:12px;
        overflow:hidden;
    }
    div[data-testid="stMetric"]{
        background:#FFFFFF;
        border:1px solid #D7E9DE;
        border-radius:14px;
        padding:12px;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"]{
        color:#60756A!important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"]{
        color:#08743A!important;
    }

    /* ========================= KPI / ALERTAS EJECUTIVAS ========================= */
    .kpi{
        min-height:118px;
        padding:14px 15px 12px;
        display:flex;
        flex-direction:column;
        justify-content:flex-start;
        overflow:hidden;
    }
    .kpi .label{
        min-height:30px;
        display:flex;
        align-items:flex-start;
        line-height:1.28;
    }
    .kpi .value{
        max-width:100%;
        overflow:hidden;
        text-overflow:ellipsis;
        line-height:1.15;
        font-size:clamp(17px,1.45vw,24px);
        letter-spacing:-.35px;
    }
    .kpi .extra{
        line-height:1.3;
        min-height:16px;
    }
    .kpi[title]{cursor:help}

    .exec-strip{
        display:grid;
        grid-template-columns:1.15fr 1.15fr 1fr;
        gap:12px;
        margin:4px 0 20px;
    }
    .exec-item{
        background:#FFFFFF;
        border:1px solid #D7E9DE;
        border-radius:16px;
        padding:16px 17px;
        min-height:128px;
        box-shadow:var(--shadow);
        overflow:hidden;
    }
    .exec-item.danger{border-top:4px solid #E5484D}
    .exec-item.success{border-top:4px solid #159447}
    .exec-label{
        color:#6B8176;
        font-size:10px;
        font-weight:900;
        letter-spacing:.7px;
        margin-bottom:6px;
    }
    .exec-title{
        color:#173C2B;
        font-size:13px;
        line-height:1.35;
        font-weight:900;
        margin-bottom:8px;
    }
    .exec-value{
        color:#08743A;
        font-size:16px;
        line-height:1.35;
        font-weight:900;
        overflow-wrap:anywhere;
    }
    .exec-item.danger .exec-value{color:#B4232A}
    .exec-note{
        color:#667B70;
        font-size:11.5px;
        line-height:1.45;
        margin-top:5px;
    }

    .module-filter{
        background:#F5FCF7;
        border:1px solid #D5EBDD;
        border-radius:14px;
        padding:11px 13px 5px;
        margin:0 0 14px;
    }
    .module-filter-title{
        color:#08743A;
        font-size:11px;
        font-weight:900;
        letter-spacing:.5px;
        margin-bottom:7px;
    }
    .chart-filter-note{
        color:#71857B;
        font-size:10.5px;
        margin-top:3px;
    }

    @media(max-width:1100px){
        .hero-row{align-items:flex-start;flex-direction:column}
        .hero-meta{width:100%;justify-content:flex-start;flex-wrap:wrap}
        .topbar-desc{display:none}
    }
    @media(max-width:760px){
        .block-container{padding-left:.8rem!important;padding-right:.8rem!important}
        .hero h1{font-size:34px}
        .hero-meta{gap:15px}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONEXIÓN
# ============================================================

@st.cache_resource
def conectar_db():
    if not DB_PATH.exists():
        return None
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)


def consulta(sql, params=None):
    con = conectar_db()
    if con is None:
        return pd.DataFrame()

    try:
        return pd.read_sql_query(sql, con, params=params)
    except Exception:
        return pd.DataFrame()


# ============================================================
# UTILIDADES
# ============================================================

def moneda(x):
    try:
        return f"$ MXN {float(x):,.2f}"
    except Exception:
        return "$0.00"


def numero(x, dec=0):
    try:
        return f"{float(x):,.{dec}f}"
    except Exception:
        return f"{0:,.{dec}f}"


def pct(x, dec=1):
    try:
        return f"{float(x):.{dec}f}%"
    except Exception:
        return f"{0:.{dec}f}%"


def mes_es(x):
    try:
        return MONTHS_ES[int(x)]
    except Exception:
        return str(x)


def limpiar_numericos(df, columnas):
    df = df.copy()
    for col in columnas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def _abreviar_kpi(valor):
    """Abrevia cifras grandes para evitar que los KPI se recorten, conservando el valor completo en tooltip."""
    texto = str(valor)
    m = re.search(r"(-?\d[\d,]*(?:\.\d+)?)", texto)
    if not m:
        return texto
    try:
        numero_original = float(m.group(1).replace(",", ""))
    except Exception:
        return texto

    abs_num = abs(numero_original)
    if abs_num >= 1_000_000_000:
        corto = f"{numero_original / 1_000_000_000:.2f}B"
    elif abs_num >= 1_000_000:
        corto = f"{numero_original / 1_000_000:.2f}M"
    elif abs_num >= 1_000:
        corto = f"{numero_original / 1_000:.1f}K"
    else:
        corto = f"{numero_original:,.1f}".rstrip("0").rstrip(".")

    # Conserva prefijos/sufijos como "$ MXN", "L", "km", "min", "%", etc.
    prefijo = texto[:m.start()].strip()
    sufijo = texto[m.end():].strip()
    return " ".join(x for x in [prefijo, corto, sufijo] if x)


def kpi(icono, titulo, valor, extra=""):
    valor_completo = escape(str(valor))
    valor_corto = escape(_abreviar_kpi(valor))
    st.markdown(
        f"""
        <div class="kpi" title="{valor_completo}">
            <div class="icon">{icono}</div>
            <div class="label">{escape(str(titulo))}</div>
            <div class="value" title="{valor_completo}">{valor_corto}</div>
            <div class="extra">{escape(str(extra))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def seccion(titulo, subtitulo=""):
    st.markdown(f'<div class="section-title">{titulo}</div>', unsafe_allow_html=True)
    if subtitulo:
        st.markdown(
            f'<div class="section-subtitle">{subtitulo}</div>',
            unsafe_allow_html=True,
        )


def aviso(texto, tipo="info"):
    st.markdown(
        f'<div class="notice {tipo}">{texto}</div>',
        unsafe_allow_html=True,
    )



_WIDGET_COUNTER = 0

def _next_key(prefix="widget"):
    global _WIDGET_COUNTER
    _WIDGET_COUNTER += 1
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(prefix)).strip("_") or "widget"
    return f"siglog_{safe}_{_WIDGET_COUNTER}"

# Compatibilidad con módulos operativos antiguos.
_ORIGINAL_DOWNLOAD_BUTTON = st.download_button
_ORIGINAL_PLOTLY_CHART = st.plotly_chart

def _safe_download_button(*args, **kwargs):
    # Los módulos operativos antiguos pueden repetir labels/keys. Se fuerza una
    # clave única en cada render para eliminar StreamlitDuplicateElementId.
    label = args[0] if args else kwargs.get("label", "download")
    filename = args[2] if len(args) > 2 else kwargs.get("file_name", "")
    base_key = kwargs.get("key") or f"download_{label}_{filename}"
    kwargs["key"] = _next_key(base_key)
    return _ORIGINAL_DOWNLOAD_BUTTON(*args, **kwargs)

def _normalizar_titulo_modulo(fig):
    """Normaliza títulos de módulos operativos sin alterar sus datos ni cálculos."""
    try:
        modulo = st.session_state.get("seccion_actual_siglog", "")
        titulo = re.sub(r"<[^>]+>", "", str(fig.layout.title.text or "")).strip()
    except Exception:
        return fig

    t = titulo.lower()
    nuevo = None
    if modulo == "🛣️ Rutas":
        if any(k in t for k in ["retras", "tard", "costo"]):
            nuevo = "Rutas con mayor proporción de retrasos y costo"
        elif any(k in t for k in ["entrega", "envío", "envio", "frecuencia", "demanda"]):
            nuevo = "Rutas con mayor demanda y volumen"
    elif modulo == "⛽ Combustible":
        if "rendimiento" in t or "km/l" in t:
            nuevo = "Rendimiento de combustible por tipo de vehículo"
        elif any(k in t for k in ["consumo", "litro", "combustible"]):
            nuevo = "Consumo total de combustible por vehículo"

    if nuevo:
        try:
            fig.update_layout(title=nuevo)
        except Exception:
            pass
    return fig


def _safe_plotly_chart(*args, **kwargs):
    if args and hasattr(args[0], "update_layout"):
        args = (_normalizar_titulo_modulo(args[0]),) + tuple(args[1:])
    if "use_container_width" in kwargs:
        use = kwargs.pop("use_container_width")
        kwargs.setdefault("width", "stretch" if use else "content")
    return _ORIGINAL_PLOTLY_CHART(*args, **kwargs)

st.download_button = _safe_download_button
st.plotly_chart = _safe_plotly_chart

def base_fig(fig, height=390, showlegend=True):
    """Tema visual común para gráficas Plotly interactivas.

    Conserva el título original en metadata para que la tarjeta pueda mostrar
    un encabezado descriptivo sin duplicarlo dentro del lienzo de Plotly.
    """
    try:
        titulo_original = fig.layout.title.text
    except Exception:
        titulo_original = ""
    try:
        meta = dict(fig.layout.meta) if isinstance(fig.layout.meta, dict) else {}
    except Exception:
        meta = {}
    if titulo_original:
        meta["siglog_title"] = str(titulo_original)
        fig.update_layout(meta=meta)

    fig.update_layout(
        height=height, margin=dict(l=72,r=28,t=18,b=72),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, Arial, sans-serif", color=COLORS["negro"], size=12),
        showlegend=showlegend,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    bgcolor="rgba(255,255,255,0)", font=dict(color="#475569", size=11)),
        hoverlabel=dict(bgcolor="#0F172A", bordercolor="#334155",
                        font=dict(color="#FFFFFF", size=13, family="Inter, Arial, sans-serif"),
                        align="left"),
        hovermode="closest", dragmode="zoom",
    )
    fig.update_layout(title=None)
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#CBD5E1",
                     tickcolor="#CBD5E1", tickfont=dict(color="#475569", size=11),
                     title_font=dict(color="#334155", size=12), automargin=True, ticks="outside")
    fig.update_yaxes(showgrid=True, gridcolor="#E8EDF4", gridwidth=1, zeroline=False,
                     linecolor="#CBD5E1", tickcolor="#CBD5E1",
                     tickfont=dict(color="#475569", size=11),
                     title_font=dict(color="#334155", size=12), automargin=True, ticks="outside")
    return fig

def _titulo_figura(fig):
    """Obtiene el título original de Plotly sin etiquetas HTML."""
    try:
        titulo = fig.layout.title.text
    except Exception:
        titulo = ""
    if not titulo:
        try:
            meta = fig.layout.meta
            if isinstance(meta, dict):
                titulo = meta.get("siglog_title", "")
        except Exception:
            pass
    if titulo is None:
        return ""
    return re.sub(r"<[^>]+>", "", str(titulo)).strip()


def _periodo_dashboard():
    return st.session_state.get("periodo_dashboard", "periodo seleccionado")


def _periodo_titulo(periodo):
    """Convierte un rango dd/mm/yyyy en un periodo legible para títulos."""
    if not periodo:
        return "Periodo seleccionado"
    texto = str(periodo).strip()
    patron = r"(\d{1,2})/(\d{1,2})/(\d{4}).*?(\d{1,2})/(\d{1,2})/(\d{4})"
    m = re.search(patron, texto)
    if m:
        d1, m1, y1, d2, m2, y2 = map(int, m.groups())
        if y1 == y2:
            if m1 == m2:
                return f"{MONTHS_ES.get(m1, m1)} {y1}"
            return f"{MONTHS_ES.get(m1, m1)} - {MONTHS_ES.get(m2, m2)} {y1}"
        return f"{MONTHS_ES.get(m1, m1)} {y1} - {MONTHS_ES.get(m2, m2)} {y2}"
    return texto


def _descripcion_inteligente(titulo):
    t=(titulo or "").lower()
    reglas=[
        (("entregas se realizaron cada mes","entregas realizadas y entregas tardías"),
         "Distribución mensual del volumen total de entregas, separando las operaciones a tiempo de las que registraron retraso para localizar meses de mayor presión operativa."),
        (("estatus de entregas","estado de las entregas"),
         "Distribución de las entregas por estatus operativo; permite distinguir operaciones a tiempo, tardías, pendientes o canceladas y medir el peso de cada resultado."),
        (("retraso promedio","retraso promedio registrado"),
         "Compara los minutos promedio de demora por periodo. Los picos representan periodos en los que la puntualidad se deterioró respecto al resto del intervalo."),
        (("rutas más largas","distancia recorrida frente"),
         "Relaciona la distancia recorrida con el retraso registrado para observar si los recorridos más extensos presentan también mayores tiempos de demora."),
        (("rutas con mayor retraso","rutas presentan más retrasos","mayores retrasos"),
         "Ordena las rutas por demora y muestra el volumen de entregas asociado, permitiendo localizar corredores con mayor afectación de puntualidad."),
        (("vehículos con mayor costo","vehículos generan mayores costos","mayor costo"),
         "Compara el costo acumulado por vehículo y permite identificar las unidades que concentran la mayor parte del gasto registrado en el periodo."),
        (("vehículos con mayor proporción","proporción de entregas tardías"),
         "Muestra el porcentaje de entregas tardías dentro de cada vehículo; la comparación se realiza sobre las operaciones registradas para cada unidad."),
        (("consumo de combustible","vehículos consumen más combustible"),
         "Compara los litros de combustible registrados por vehículo y permite detectar unidades con consumo elevado frente al resto de la flota."),
        (("componentes que presentan","componentes que concentran"),
         "Distribuye las incidencias por componente para detectar piezas con recurrencia elevada y orientar la disponibilidad de refacciones y mantenimiento preventivo."),
        (("causas de las incidencias","causas principales de retraso","causas de retraso"),
         "Ordena las causas registradas por frecuencia y, cuando existe el dato, permite contrastarlas con costo, retraso u horas fuera de servicio."),
        (("tipo de mantenimiento",),
         "Compara la frecuencia y el costo de las intervenciones por tipo de mantenimiento para distinguir dónde se concentra la actividad preventiva, correctiva o de emergencia."),
        (("gravedad de la incidencia","gravedad de las incidencias"),
         "Distribuye las incidencias por severidad y muestra dónde se concentra el impacto operativo, económico o de indisponibilidad."),
        (("riesgo dentro de la flota","nivel de riesgo"),
         "Clasifica las unidades por nivel de riesgo a partir de los indicadores disponibles y permite localizar la parte de la flota que requiere seguimiento prioritario."),
        (("gasto histórico de mantenimiento",),
         "Compara el gasto acumulado de mantenimiento por vehículo y ayuda a detectar unidades cuyo costo histórico justifica una revisión de fallas recurrentes."),
        (("perfiles de comportamiento","proporción de las entregas pertenece","grupos de rutas similares"),
         "Agrupa rutas u operaciones con comportamientos parecidos y muestra la participación de cada grupo para distinguir patrones de demanda, retraso o costo."),
        (("factores que explican","variación explicada"),
         "Muestra la contribución relativa de los factores o componentes calculados por el modelo; una mayor contribución indica mayor peso estadístico, no causalidad por sí sola."),
        (("costo de una entrega frente",),
         "Relaciona el costo de cada entrega con su retraso para localizar operaciones que combinan alto impacto económico y bajo desempeño de puntualidad."),
        (("demanda",), "Muestra la evolución de la demanda de servicios durante el periodo y permite localizar los intervalos con mayor concentración de entregas."),
        (("saturación",), "Compara el volumen de operaciones por hora de salida para identificar las franjas de mayor saturación de la operación."),
        (("frecuencia",), "Ordena las entidades con mayor frecuencia de operación para localizar clientes, servicios o rutas que concentran recurrencia."),
    ]
    for claves,desc in reglas:
        if any(k in t for k in claves):
            return desc
    return "Compara la métrica representada entre las entidades o periodos del gráfico y permite localizar concentraciones, diferencias y valores extremos relevantes para la operación."


def _titulo_descriptivo(fig, titulo, periodo):
    """Construye un título específico de negocio cuando Plotly no trae uno útil."""
    base = re.sub(r"\s+", " ", str(titulo or "")).strip()
    genericos = {
        "", "visualización operativa", "visualizacion operativa",
        "análisis de operación", "analisis de operacion",
        "análisis operativo", "analisis operativo",
        "comportamiento de los indicadores operativos",
        "comportamiento de la operación",
    }

    if base.lower() in genericos:
        try:
            x_title = re.sub(
                r"<[^>]+>", "",
                str(getattr(fig.layout.xaxis.title, "text", "") or "")
            ).strip()
            y_title = re.sub(
                r"<[^>]+>", "",
                str(getattr(fig.layout.yaxis.title, "text", "") or "")
            ).strip()
        except Exception:
            x_title, y_title = "", ""

        x_low, y_low = x_title.lower(), y_title.lower()
        traces = list(getattr(fig, "data", []))
        nombres = [
            str(getattr(tr, "name", "") or "").strip()
            for tr in traces if getattr(tr, "name", "")
        ]

        # Reglas de negocio para los gráficos más comunes del dashboard.
        if "entregas" in y_low and ("mes" in x_low or "periodo" in x_low):
            base = "Entregas realizadas por periodo"
        elif "retraso" in y_low and ("mes" in x_low or "periodo" in x_low):
            base = "Retraso promedio de las entregas por periodo"
        elif "costo" in y_low and ("vehículo" in x_low or "numero" in x_low):
            base = "Costo logístico acumulado por vehículo"
        elif "combustible" in y_low and ("vehículo" in x_low or "numero" in x_low):
            base = "Consumo total de combustible por vehículo"
        elif "retraso" in y_low and ("ruta" in x_low or "ruta" in y_low):
            base = "Retraso promedio por ruta"
        elif "entregas" in y_low and ("ruta" in x_low or "ruta" in y_low):
            base = "Volumen de entregas por ruta"
        elif "entregas" in y_low and ("operador" in x_low or "operador" in y_low):
            base = "Entregas realizadas por operador"
        elif "variación explicada" in y_low or "varianza" in y_low:
            base = "Factores con mayor variación explicada"
        elif "participación" in y_low and ("perfil" in x_low or "grupo" in x_low):
            base = "Participación de los perfiles de operación"
        elif y_title and x_title:
            base = f"{y_title} por {x_title}"
        elif y_title:
            base = f"{y_title} por entidad"
        elif x_title:
            base = f"Indicadores operativos por {x_title}"
        elif nombres:
            base = "Comparación de " + ", ".join(nombres[:3])
        else:
            base = "Indicadores operativos del periodo seleccionado"

    return base


def _fig_filtrado_local(fig, filtros):
    """Aplica ajustes locales sin producir gráficas vacías por interpretar mal el eje."""
    try:
        figura = go.Figure(fig)
    except Exception:
        return fig

    serie = filtros.get("serie", "Todas")
    elemento = filtros.get("elemento", "Todos")

    for tr in figura.data:
        nombre = str(getattr(tr, "name", "") or "").strip()
        if serie != "Todas" and nombre != serie:
            tr.visible = False
            continue

        if elemento == "Todos":
            continue

        xs = _a_lista_plotly(getattr(tr, "x", None))
        ys = _a_lista_plotly(getattr(tr, "y", None))
        n = max(len(xs), len(ys))
        if n == 0:
            continue

        # La entidad puede estar en X (barras verticales/series temporales)
        # o en Y (barras horizontales). Se comprueban ambos ejes.
        keep = []
        for i in range(n):
            x = xs[i] if i < len(xs) else None
            y = ys[i] if i < len(ys) else None
            if str(x) == str(elemento) or str(y) == str(elemento):
                keep.append(i)

        # Si la selección no pertenece a este trace, simplemente lo ocultamos.
        if not keep:
            tr.visible = False
            continue

        try:
            tr.x = [xs[i] for i in keep] if xs else xs
            tr.y = [ys[i] for i in keep] if ys else ys
            if getattr(tr, "customdata", None) is not None:
                cd = _a_lista_plotly(tr.customdata)
                if len(cd) == n:
                    tr.customdata = [cd[i] for i in keep]
        except Exception:
            # Nunca romper una gráfica por un filtro de presentación.
            continue

    return figura


def _controles_locales_grafica(fig, titulo, datos=None):
    """Ajustes visuales exclusivos de una gráfica.

    El periodo global sigue siendo la fuente de verdad. Aquí sólo se permite
    enfocar la serie o entidad que se está visualizando; no se duplica el
    filtro global de año/mes/estado/tipo/riesgo.
    """
    traces = list(getattr(fig, "data", []))
    nombres = list(dict.fromkeys([
        str(getattr(tr, "name", "") or "").strip()
        for tr in traces if getattr(tr, "name", "")
    ]))

    xs, ys = [], []
    for tr in traces:
        xs.extend(_a_lista_plotly(getattr(tr, "x", None)))
        ys.extend(_a_lista_plotly(getattr(tr, "y", None)))

    opciones_elemento = list(dict.fromkeys([
        str(x) for x in (xs + ys)
        if x is not None and str(x).strip() not in ("", "nan", "None")
    ]))[:250]

    tiene_serie = len(nombres) > 1
    tiene_entidad = len(opciones_elemento) > 1
    filtros = {"serie": "Todas", "elemento": "Todos",
               "fecha_inicio": None, "fecha_fin": None}

    # No se muestra un expander vacío. El periodo ya está controlado arriba.
    if not tiene_serie and not tiene_entidad:
        return filtros

    with st.expander("🎛️ Ajustes de esta gráfica", expanded=False):
        st.caption(
            "Estos ajustes sólo cambian el enfoque de esta visualización. "
            "El periodo y los filtros de alcance se heredan de «Control del análisis»."
        )
        cols = st.columns(2)
        with cols[0]:
            if tiene_serie:
                filtros["serie"] = st.selectbox(
                    "Serie / métrica visible",
                    ["Todas"] + nombres,
                    key=_next_key(f"serie_{titulo}"),
                    help="Afecta únicamente esta gráfica."
                )
            else:
                st.caption("La gráfica contiene una sola métrica.")
        with cols[1]:
            if tiene_entidad:
                filtros["elemento"] = st.selectbox(
                    "Entidad / categoría",
                    ["Todos"] + opciones_elemento,
                    key=_next_key(f"elemento_{titulo}"),
                    help="Selecciona una categoría concreta sin modificar las demás gráficas."
                )
            else:
                st.caption("No hay categorías suficientes para un filtro local.")
        st.markdown(
            '<div class="chart-filter-note">Periodo heredado del filtro global: '
            f'{escape(_periodo_dashboard())}</div>',
            unsafe_allow_html=True,
        )
    return filtros


def _serie_numerica(datos, candidatos):
    if not isinstance(datos, pd.DataFrame) or datos.empty:
        return None, None
    for col in candidatos:
        if col in datos.columns:
            s = pd.to_numeric(datos[col], errors="coerce")
            if s.notna().any():
                return col, s
    return None, None


def _serie_numerica(datos, candidatos):
    if not isinstance(datos, pd.DataFrame) or datos.empty:
        return None, None
    for col in candidatos:
        if col in datos.columns:
            s = pd.to_numeric(datos[col], errors="coerce")
            if s.notna().any():
                return col, s
    return None, None


def _interpretacion_automatica(titulo, datos):
    t=(titulo or "").lower()
    if not isinstance(datos,pd.DataFrame) or datos.empty:
        return (
            f"En la selección actual no hay registros para la métrica «{titulo}». "
            "La gráfica se conserva visible para mostrar exactamente qué indicador "
            "está siendo consultado y evitar que una combinación de filtros oculte la pregunta."
        )
    total = len(datos)

    if "vehículos generan mayores costos" in t or "mayor costo" in t:
        col,s=_serie_numerica(datos,["costo_total","Costo total (MXN)","costo"])
        if col:
            i=s.idxmax()
            etiqueta=datos.loc[i,"numero_economico"] if "numero_economico" in datos.columns else datos.loc[i].get("Vehículo",i)
            total_costo = s.sum()
            pct = (s.loc[i]/total_costo*100) if total_costo else 0
            return f"El vehículo {etiqueta} registra el mayor costo del conjunto mostrado, con {moneda(s.loc[i])}, equivalente a {pct:.1f}% del costo representado. La diferencia frente al resto ayuda a detectar dónde conviene revisar combustible, mantenimiento y carga de trabajo."
    if "consumen más combustible" in t or "consumo de combustible" in t:
        col,s=_serie_numerica(datos,["litros","Combustible (L)","combustible_l"])
        if col:
            i=s.idxmax()
            etiqueta=datos.loc[i,"numero_economico"] if "numero_economico" in datos.columns else datos.loc[i].get("Vehículo",i)
            pct = (s.loc[i]/s.sum()*100) if s.sum() else 0
            return f"El vehículo {etiqueta} concentra el mayor consumo acumulado, con {s.loc[i]:,.1f} L ({pct:.1f}% de lo representado). Para saber si ese consumo es elevado por eficiencia y no sólo por uso, conviene contrastarlo con kilómetros y entregas."
    if "rutas son más utilizadas" in t or "mayor número de envíos" in t or "frecuencia" in t:
        col,s=_serie_numerica(datos,["entregas","envios","Entregas","Envíos"])
        if col:
            i=s.idxmax()
            etiqueta=datos.loc[i,"nombre_ruta"] if "nombre_ruta" in datos.columns else datos.loc[i].get("Ruta",i)
            pct=(s.loc[i]/s.sum()*100) if s.sum() else 0
            return f"La ruta {etiqueta} concentra {int(s.loc[i]):,} entregas, equivalente al {pct:.1f}% del volumen mostrado. Es el principal corredor para revisar capacidad, disponibilidad de vehículos y posibles cuellos de botella."
    if "operadores realizan más entregas" in t:
        col,s=_serie_numerica(datos,["entregas","Entregas"])
        if col:
            i=s.idxmax()
            etiqueta=datos.loc[i,"operador"] if "operador" in datos.columns else datos.loc[i].get("Operador",i)
            pct=(s.loc[i]/s.sum()*100) if s.sum() else 0
            return f"{etiqueta} realiza {int(s.loc[i]):,} entregas y concentra {pct:.1f}% de las operaciones representadas. La carga debe compararse con su puntualidad para distinguir productividad de sobreasignación."
    if "rutas presentan mayores retrasos" in t or "retraso" in t:
        col,s=_serie_numerica(datos,["tasa_retraso","Tasa tardía (%)","retraso","Retraso promedio (min)","retraso_promedio"])
        if col:
            i=s.idxmax()
            etiqueta=datos.loc[i,"nombre_ruta"] if "nombre_ruta" in datos.columns else datos.loc[i].get("Ruta",i)
            return f"La ruta {etiqueta} presenta el valor más alto del indicador de retraso ({s.loc[i]:.1f}). Es el punto con mayor desviación dentro de la selección y debe revisarse junto con horario, distancia y causa registrada."
    if "mantenimiento" in t:
        crit_cols=[c for c in ["criticas","críticas","emergencias","fallas_criticas","costo_total","horas_fuera_servicio"] if c in datos.columns]
        if crit_cols:
            score=pd.DataFrame(index=datos.index)
            for c in crit_cols:
                score[c]=pd.to_numeric(datos[c],errors="coerce").fillna(0)
            score["_prioridad"]=score.sum(axis=1)
            i=score["_prioridad"].idxmax()
            etiqueta=datos.loc[i,"numero_economico"] if "numero_economico" in datos.columns else datos.loc[i].get("Vehículo",i)
            detalles=", ".join(f"{c}: {score.loc[i,c]:,.0f}" for c in crit_cols if score.loc[i,c]>0)
            return f"La unidad {etiqueta} presenta la mayor combinación de indicadores de atención ({detalles}). Por eso aparece como prioridad operativa en esta vista; no se clasifica sólo por una variable aislada."
        return "La prioridad de mantenimiento se determina con los indicadores disponibles en esta vista. Revisa especialmente las unidades con mayor recurrencia de fallas, emergencias, costo u horas fuera de servicio."
    if "perfil" in t or "grupos de rutas" in t:
        col,s=_serie_numerica(datos,["Participación (%)","porcentaje_entregas","porcentaje"])
        if col:
            i=s.idxmax()
            etiqueta=datos.loc[i,"Grupo"] if "Grupo" in datos.columns else datos.loc[i].get("Perfil",i)
            return f"El grupo {etiqueta} representa {s.loc[i]:.1f}% de las observaciones mostradas. La agrupación sirve para separar rutas con comportamientos parecidos y enfocar estrategias distintas por perfil."
    if "factor" in t or "variación explicada" in t:
        col,s=_serie_numerica(datos,["Variación explicada (%)","varianza","porcentaje"])
        if col:
            i=s.idxmax()
            etiqueta=datos.loc[i,"Factor"] if "Factor" in datos.columns else datos.loc[i].get("Componente",i)
            return f"{etiqueta} aporta {s.loc[i]:.2f}% de la variación representada. Es el componente con mayor peso estadístico en esta gráfica; no debe interpretarse por sí solo como una causa."
    if "demanda" in t:
        col,s=_serie_numerica(datos,["entregas","Entregas"])
        if col:
            i=s.idxmax()
            etiqueta=datos.loc[i,"periodo"] if "periodo" in datos.columns else datos.loc[i].get("Periodo",i)
            return f"El periodo {etiqueta} concentra el mayor volumen, con {int(s.loc[i]):,} entregas. Esa concentración indica cuándo la capacidad de vehículos y operadores tiene mayor presión."
    if "saturación" in t:
        col,s=_serie_numerica(datos,["entregas","Entregas"])
        if col:
            i=s.idxmax()
            hora=datos.loc[i,"hora"] if "hora" in datos.columns else datos.loc[i].get("Hora",0)
            return f"La franja de {int(hora):02d}:00 concentra {int(s.loc[i]):,} operaciones, el máximo del periodo mostrado. Es el horario que primero conviene revisar para evitar saturación de salidas."
    if "estado de las entregas" in t or "estatus de entregas" in t:
        col,s=_serie_numerica(datos,["entregas","Entregas"])
        if col:
            i=s.idxmax()
            etiqueta=datos.loc[i,"estatus"] if "estatus" in datos.columns else datos.loc[i].get("Estado",i)
            pct=s.loc[i]/s.sum()*100 if s.sum() else 0
            return f"El estado {etiqueta} concentra {int(s.loc[i]):,} registros, equivalente al {pct:.1f}% del total representado. La participación de los estados permite dimensionar el nivel de cumplimiento del periodo."
    return f"La gráfica representa {total:,} registros y permite identificar el valor máximo ({_valor_maximo(datos)}) y su diferencia frente al resto. El hallazgo debe leerse con la métrica y la entidad que aparecen en los ejes."


def _valor_maximo(datos):
    if not isinstance(datos,pd.DataFrame) or datos.empty:
        return "sin datos"
    for c in datos.columns:
        s=pd.to_numeric(datos[c],errors="coerce")
        if s.notna().sum():
            try:
                return f"{s.max():,.2f}"
            except Exception:
                pass
    return "no disponible"


def _recomendacion_automatica(titulo, datos):
    t=(titulo or "").lower()
    if not isinstance(datos,pd.DataFrame) or datos.empty:
        return (
            "Con esta combinación de filtros no se deben tomar decisiones sobre "
            f"«{titulo}». Mantén el periodo como referencia y revisa el filtro local "
            "de la gráfica antes de comparar unidades, rutas, operadores o periodos."
        )
    if "costo" in t:
        return "Prioriza la unidad con mayor costo para revisar cuánto corresponde a combustible, mantenimiento y volumen de entregas. Si el costo alto está justificado por mayor utilización, evita penalizarla sólo por el monto acumulado."
    if "consumo" in t:
        return "Revisa primero las unidades con mayor consumo y calcula litros por kilómetro. Si también presentan un rendimiento bajo, programa inspección de eficiencia y compara contra vehículos equivalentes."
    if "retraso" in t:
        return "Ataca primero la ruta o periodo con mayor retraso: identifica la causa dominante, revisa la franja horaria y contrasta vehículo y distancia antes de modificar la asignación."
    if "mantenimiento" in t:
        return "Programa la atención de las unidades que combinan fallas críticas, emergencias, costo u horas fuera de servicio. La prioridad aumenta cuando varios indicadores apuntan a la misma unidad."
    if "operadores" in t:
        return "Compara carga de entregas contra puntualidad y retrasos. Si un operador concentra mucha operación pero también mucha demora, considera balancear asignaciones o revisar las rutas que recibe."
    if "perfil" in t or "grupo" in t:
        return "Asigna una estrategia por grupo: alta demanda requiere capacidad, alto retraso requiere revisión de causas y alto costo requiere análisis económico. Evita aplicar una misma acción a todos los perfiles."
    if "factor" in t:
        return "Usa los factores como resumen estadístico y revisa las variables originales antes de convertirlos en una acción. Un factor con mayor peso no implica causalidad."
    if "demanda" in t or "saturación" in t or "frecuencia" in t:
        return "Concentra capacidad de vehículos y operadores en los periodos, horarios y rutas con mayor volumen. Revisa si esa concentración coincide con retrasos para evitar crecer capacidad sin resolver el cuello de botella."
    return "Prioriza el valor más alto de la gráfica, identifica qué variable lo explica y contrasta el caso con el detalle operativo antes de realizar una reasignación o mantenimiento."


def _a_lista_plotly(valor):
    """Convierte listas/tuplas/arrays de Plotly o NumPy sin evaluar arrays como booleanos."""
    if valor is None:
        return []
    try:
        return list(valor)
    except Exception:
        return [valor]

def _fig_dataframe(fig):
    """Convierte los trazos visibles de Plotly en una tabla descargable."""
    filas = []
    for trace in getattr(fig, "data", []):
        xs = _a_lista_plotly(getattr(trace, "x", None))
        ys = _a_lista_plotly(getattr(trace, "y", None))
        n = max(len(xs), len(ys))
        if n == 0:
            continue
        nombre = getattr(trace, "name", "") or ""
        for i in range(n):
            filas.append({
                "Serie": str(nombre),
                "X": xs[i] if i < len(xs) else "",
                "Y": ys[i] if i < len(ys) else "",
            })
    return pd.DataFrame(filas)

def _pdf_grafica_bytes(titulo, pregunta, periodo, descripcion, interpretacion_texto,
                       recomendacion, fig, datos=None):
    """Genera un PDF ejecutivo individual para una gráfica."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    except Exception:
        return None

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=landscape(letter),
        rightMargin=1.1*cm, leftMargin=1.1*cm,
        topMargin=1.1*cm, bottomMargin=1.1*cm
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle("PDFTitle", parent=styles["Title"], fontName="Helvetica-Bold",
                           fontSize=20, textColor=colors.HexColor("#1E3A8A"), spaceAfter=7)
    body = ParagraphStyle("PDFBody", parent=styles["BodyText"], fontSize=9.2,
                          leading=12, textColor=colors.HexColor("#334155"))
    head = ParagraphStyle("PDFHead", parent=styles["Heading2"], fontSize=12.5,
                          textColor=colors.HexColor("#0F172A"), spaceBefore=7, spaceAfter=5)

    story = [
        Paragraph("SIG-LOG · Reporte individual de visualización", title),
        Paragraph(str(titulo), styles["Heading2"]),
        Paragraph(f"<b>Pregunta:</b> {pregunta}", body),
        Paragraph(f"<b>Periodo:</b> {periodo}", body),
        Spacer(1, .18*cm),
        Paragraph("<b>Interpretación de la gráfica</b>", head),
        Paragraph(str(descripcion), body),
        Paragraph(str(interpretacion_texto or "Consultar la tabla y el detalle interactivo para validar el hallazgo."), body),
        Spacer(1, .12*cm),
        Paragraph("<b>Recomendación</b>", head),
        Paragraph(str(recomendacion or "Revisar el hallazgo junto con los filtros y registros de detalle antes de tomar una decisión."), body),
        Spacer(1, .18*cm),
    ]

    # Inserta la gráfica real de Plotly en el PDF, no solamente la tabla X/Y.
    # Requiere kaleido para que Plotly pueda renderizar la imagen.
    try:
        chart_png = fig.to_image(format="png", width=1250, height=560, scale=1)
        chart_buffer = BytesIO(chart_png)
        chart_image = Image(chart_buffer, width=24.2*cm, height=10.8*cm)
        story += [
            Paragraph("<b>Gráfica</b>", head),
            chart_image,
            Spacer(1, .12*cm),
        ]
    except Exception:
        # El reporte no falla si Kaleido no está disponible.
        story += [
            Paragraph(
                "<b>Gráfica:</b> no fue posible incrustar la imagen en este entorno. "
                "El PDF conserva interpretación, recomendación y datos. Para incrustar la gráfica visual instala <b>kaleido</b>.",
                body
            ),
            Spacer(1, .12*cm),
        ]

    frame = datos.copy() if isinstance(datos, pd.DataFrame) and not datos.empty else _fig_dataframe(fig)
    if frame is not None and not frame.empty:
        view = frame.head(30).copy().fillna("")
        cols = list(view.columns[:8])
        values = [cols] + view[cols].astype(str).values.tolist()
        table = Table(values, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#2563EB")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("GRID",(0,0),(-1,-1),.25,colors.HexColor("#DDE5EF")),
            ("FONTSIZE",(0,0),(-1,-1),6.5),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F8FAFC")]),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
        ]))
        story += [Paragraph("<b>Datos representados</b>", head), table]

    story += [Spacer(1,.2*cm),
              Paragraph(f"Generado por SIG-LOG: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", body)]
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def _csv_grafica_bytes(fig, datos=None):
    frame = datos.copy() if isinstance(datos, pd.DataFrame) and not datos.empty else _fig_dataframe(fig)
    return frame.to_csv(index=False).encode("utf-8-sig") if frame is not None else b""

def _pregunta_desde_titulo(titulo):
    t = (titulo or "").lower()
    reglas = [
        ("ruta", "¿Qué rutas son más utilizadas?"),
        ("vehículos con mayor costo", "¿Qué vehículos generan mayores costos?"),
        ("vehículos con mayor consumo", "¿Qué vehículos consumen más combustible?"),
        ("consumo de combustible", "¿Qué vehículos consumen más combustible?"),
        ("operador", "¿Qué operadores realizan más entregas?"),
        ("retraso", "¿Qué rutas presentan mayores retrasos?"),
        ("mantenimiento", "¿Qué vehículos requieren mantenimiento?"),
        ("perfil", "¿Podemos identificar grupos de rutas similares?"),
        ("factor", "¿Qué factores explican el comportamiento?"),
        ("predic", "¿Es posible predecir si una entrega llegará tarde?"),
        ("causa", "¿Cuáles son las causas principales de retraso?"),
    ]
    for clave, pregunta in reglas:
        if clave in t:
            return pregunta
    return titulo or "¿Qué está ocurriendo en la operación?"

def mostrar_fig(fig, titulo=None, descripcion=None, periodo=None,
               pregunta=None, datos=None, interpretacion_texto=None,
               recomendacion=None):
    """Tarjeta de gráfica interactiva con filtros y reporte individual.

    Los filtros definidos aquí son exclusivos de esta gráfica. No modifican
    el filtro general del dashboard ni la lógica de los módulos existentes.
    """
    periodo = periodo or _periodo_dashboard()
    titulo_base = _titulo_descriptivo(fig, titulo or _titulo_figura(fig), periodo)
    titulo_visible = f"{titulo_base} ({_periodo_titulo(periodo)})"
    pregunta = pregunta or _pregunta_desde_titulo(titulo_base)
    descripcion_base = descripcion or _descripcion_inteligente(titulo_base)

    # Si no se recibió un DataFrame, utiliza los datos representados por Plotly
    # para que filtros, insights y reportes sigan funcionando.
    datos_base = datos.copy() if isinstance(datos, pd.DataFrame) else _fig_dataframe(fig)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)

    # Un solo encabezado por gráfica.
    st.markdown(
        f'<div class="chart-heading">{escape(titulo_visible)}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="chart-description"><b>{escape(pregunta)}</b><br>'
        f'{escape(descripcion_base)}</div>',
        unsafe_allow_html=True
    )

    # Filtros independientes de esta gráfica.
    filtros = _controles_locales_grafica(fig, titulo_base, datos_base)
    fig_local = _fig_filtrado_local(fig, filtros)

    datos_local = datos_base
    if isinstance(datos_base, pd.DataFrame) and not datos_base.empty:
        # Si el usuario seleccionó un elemento y existe una columna compatible,
        # también filtramos la tabla que alimenta el insight/PDF.
        elemento = filtros.get("elemento")
        if elemento and elemento != "Todos":
            posibles = ["nombre_ruta","Ruta","numero_economico","Vehículo",
                        "vehiculo","operador","Operador","periodo","Periodo",
                        "hora","Hora","estatus","Estado","Factor","Grupo","Perfil"]
            for col in posibles:
                if col in datos_local.columns:
                    mask = datos_local[col].astype(str).eq(str(elemento))
                    if mask.any():
                        datos_local = datos_local.loc[mask].copy()
                        break

    datos_fig_local = _fig_dataframe(fig_local)

    # Un ajuste local nunca debe convertir una gráfica válida en una tarjeta vacía.
    # Si la selección no es compatible con la estructura del trace, se conserva
    # la figura original y se informa de forma limpia.
    figura_tiene_trazos = bool(getattr(fig_local, "data", []))
    figura_tiene_datos = bool(datos_fig_local is not None and not datos_fig_local.empty)
    if not figura_tiene_trazos or not figura_tiene_datos:
        # Nunca dejamos desaparecer la visualización. Si el alcance global no
        # contiene filas para la métrica seleccionada, mostramos una figura de
        # estado en lugar de una tarjeta vacía. Esto conserva la estructura UX
        # de todas las preguntas de negocio y evita que el usuario crea que
        # falta una gráfica.
        fig_local = go.Figure()
        fig_local.add_trace(
            go.Bar(
                x=["Sin registros en la selección"],
                y=[0],
                marker=dict(color="#B9DCC6"),
                hovertemplate=(
                    "<b>Sin registros en la selección</b><br>"
                    "Valor representado: 0<extra></extra>"
                ),
                showlegend=False,
            )
        )
        fig_local.update_layout(
            annotations=[
                dict(
                    text="La métrica permanece visible aunque el filtro actual no tenga observaciones",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=13, color="#557064"),
                    align="center",
                )
            ],
            yaxis=dict(range=[0, 1]),
        )
        datos_fig_local = pd.DataFrame(
            {
                "Serie": [titulo_base],
                "X": ["Sin registros en la selección"],
                "Y": [0],
            }
        )

    # Plotly sigue siendo completamente interactivo.
    st.plotly_chart(
        fig_local,
        width="stretch",
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
        key=_next_key(f"plot_{titulo_base}")
    )

    datos_para_lectura = datos_local if isinstance(datos_local, pd.DataFrame) and not datos_local.empty else datos_fig_local
    if interpretacion_texto is None:
        interpretacion_texto = _interpretacion_automatica(titulo_base, datos_para_lectura)
    if recomendacion is None:
        recomendacion = _recomendacion_automatica(titulo_base, datos_para_lectura)

    st.markdown(
        f'<div class="interpretacion"><div class="interpretacion-title">'
        f'¿Qué significa?</div><div class="interpretacion-text">'
        f'{escape(str(interpretacion_texto))}</div></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="accion-card"><div class="accion-title">'
        f'¿Qué hacer?</div><div class="accion-text">'
        f'{escape(str(recomendacion))}</div></div>',
        unsafe_allow_html=True
    )

    pdf = _pdf_grafica_bytes(
        titulo_visible,
        pregunta,
        periodo,
        descripcion_base,
        interpretacion_texto,
        recomendacion,
        fig_local,
        datos_para_lectura
    )
    csv = _csv_grafica_bytes(fig_local, datos_para_lectura)

    safe = re.sub(r"[^A-Za-z0-9]+", "_", titulo_base).strip("_")[:70] or "grafica"

    c1, c2 = st.columns([1, 1])
    with c1:
        if pdf:
            st.download_button(
                "Descargar PDF de esta gráfica",
                pdf,
                f"SIGLOG_{safe}.pdf",
                "application/pdf",
                key=_next_key(f"pdf_{safe}"),
                width="stretch",
            )
        else:
            st.warning(
                "No se pudo generar el PDF. Instala ReportLab y, para incrustar "
                "la gráfica visual, Kaleido: pip install reportlab kaleido"
            )

    with c2:
        st.download_button(
            "Descargar datos CSV",
            csv,
            f"SIGLOG_{safe}.csv",
            "text/csv",
            key=_next_key(f"csv_{safe}"),
            width="stretch",
        )

    st.markdown("</div>", unsafe_allow_html=True)



def interpretacion(texto, tipo="info"):
    color = {
        "info": "#2563EB",
        "warning": "#EA580C",
        "danger": "#DC2626",
        "success": "#16A34A",
    }.get(tipo, "#2563EB")

    titulo = {
        "info": "¿Qué significa?",
        "warning": "⚠️ Punto de atención",
        "danger": "🚨 Prioridad",
        "success": "✅ Lectura positiva",
    }.get(tipo, "¿Qué significa?")

    st.markdown(
        f"""
        <div class="interpretacion" style="border-left-color:{color};">
            <div class="interpretacion-title">{titulo}</div>
            <div class="interpretacion-text">{texto}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def accion(titulo, texto):
    st.markdown(
        f"""
        <div class="accion-card">
            <div class="accion-title">{titulo}</div>
            <div class="accion-text">{texto}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def periodo_texto(df):
    if df.empty or "fecha_salida" not in df.columns:
        return "Sin periodo disponible"

    fechas = df["fecha_salida"].dropna()
    if fechas.empty:
        return "Sin periodo disponible"

    inicio = fechas.min()
    fin = fechas.max()

    if inicio.year == fin.year and inicio.month == fin.month:
        return f"{mes_es(inicio.month)} de {inicio.year}"

    return f"{inicio.strftime('%d/%m/%Y')} — {fin.strftime('%d/%m/%Y')}"


# ============================================================
# CARGA DE ENTREGAS
# ============================================================

@st.cache_data
def cargar_entregas():
    sql = """
    SELECT
        f.id_entrega,
        f.id_cliente,
        f.id_vehiculo,
        f.id_operador,
        f.id_ruta,
        f.id_tiempo,

        f.fecha_salida,
        f.hora_salida,
        f.fecha_entrega_programada,
        f.hora_entrega_programada,
        f.fecha_entrega_real,
        f.hora_entrega_real,

        f.peso_carga,
        f.cantidad_paquetes,
        f.distancia_real_km,

        f.tipo_combustible,
        f.combustible_consumido_litros,
        f.precio_combustible,

        f.costo_envio,
        f.costo_combustible,
        f.costo_total,

        f.minutos_retraso,
        f.entrega_tardia,
        f.estatus,
        f.observaciones,

        r.codigo_ruta,
        r.origen,
        r.destino,
        r.distancia_km AS distancia_ruta_km,
        r.tiempo_estimado_minutos,
        r.tipo_ruta,
        r.nivel_trafico,
        r.peajes,
        r.costo_peaje,
        r.numero_entregas AS entregas_programadas_ruta,
        r.tiempo_promedio_minutos,
        r.retraso_promedio_minutos,
        r.porcentaje_retraso AS porcentaje_retraso_ruta,
        r.consumo_promedio AS consumo_promedio_ruta,
        r.costo_promedio AS costo_promedio_ruta,
        r.estatus AS estatus_ruta,

        v.numero_economico,
        v.placas,
        v.marca,
        v.modelo,
        v.anio,
        v.tipo AS tipo_vehiculo,
        v.tipo_combustible AS combustible_vehiculo,
        v.capacidad_toneladas,
        v.kilometraje_actual,
        v.rendimiento_km_l,
        v.nivel_riesgo,
        v.estatus AS estatus_vehiculo

    FROM fact_entregas f
    LEFT JOIN dim_ruta r ON f.id_ruta = r.id_ruta
    LEFT JOIN dim_vehiculo v ON f.id_vehiculo = v.id_vehiculo
    """

    df = consulta(sql)
    if df.empty:
        return pd.DataFrame()

    df["fecha_salida"] = pd.to_datetime(df["fecha_salida"], errors="coerce")
    df["anio"] = df["fecha_salida"].dt.year
    df["mes"] = df["fecha_salida"].dt.month
    df["nombre_mes"] = df["mes"].map(MONTHS_ES)
    df["periodo"] = df["fecha_salida"].dt.to_period("M").astype(str)

    df["nombre_ruta"] = (
        df["codigo_ruta"].fillna("RUTA-" + df["id_ruta"].astype(str)).astype(str)
        + " · "
        + df["origen"].fillna("Origen").astype(str)
        + " → "
        + df["destino"].fillna("Destino").astype(str)
    )

    columnas = [
        "peso_carga", "cantidad_paquetes", "distancia_real_km",
        "combustible_consumido_litros", "precio_combustible",
        "costo_envio", "costo_combustible", "costo_total",
        "minutos_retraso", "entrega_tardia", "distancia_ruta_km",
        "tiempo_estimado_minutos", "peajes", "costo_peaje",
    ]

    return limpiar_numericos(df, columnas)


# ============================================================
# MANTENIMIENTO
# ============================================================

@st.cache_data
def cargar_mantenimiento():
    sql = """
    SELECT
        f.id_mantenimiento,
        f.id_vehiculo,
        f.id_componente,
        f.fecha_mantenimiento,
        f.tipo_mantenimiento,
        f.causa_falla,
        f.severidad,
        f.costo_repuesto,
        f.costo_mano_obra,
        f.costo_total,
        f.horas_fuera_servicio,

        c.nombre AS componente,
        c.categoria AS categoria_componente,
        c.vida_util_km,
        c.costo_reemplazo,

        v.numero_economico,
        v.placas,
        v.marca,
        v.modelo,
        v.nivel_riesgo

    FROM fact_mantenimiento f
    LEFT JOIN dim_componente c ON f.id_componente = c.id_componente
    LEFT JOIN dim_vehiculo v ON f.id_vehiculo = v.id_vehiculo
    """

    df = consulta(sql)
    if df.empty:
        return pd.DataFrame()

    df["fecha_mantenimiento"] = pd.to_datetime(
        df["fecha_mantenimiento"], errors="coerce"
    )

    columnas = [
        "costo_repuesto", "costo_mano_obra", "costo_total",
        "horas_fuera_servicio", "vida_util_km", "costo_reemplazo",
    ]

    return limpiar_numericos(df, columnas)


# ============================================================
# VEHÍCULOS
# ============================================================

@st.cache_data
def cargar_vehiculos():
    sql = """
    SELECT
        id_vehiculo,
        numero_economico,
        placas,
        marca,
        modelo,
        anio,
        tipo,
        tipo_combustible,
        capacidad_toneladas,
        kilometraje_actual,
        rendimiento_km_l,
        consumo_promedio,
        fecha_adquisicion,
        ultima_revision,
        proxima_revision,
        nivel_riesgo,
        estatus
    FROM dim_vehiculo
    """

    df = consulta(sql)

    columnas = [
        "anio", "capacidad_toneladas", "kilometraje_actual",
        "rendimiento_km_l", "consumo_promedio",
    ]

    return limpiar_numericos(df, columnas)


# ============================================================
# RIESGOS
# ============================================================

@st.cache_data
def cargar_riesgos():
    sql = """
    SELECT
        v.id_vehiculo,
        v.numero_economico,
        v.placas,
        v.marca,
        v.modelo,
        v.nivel_riesgo,
        v.estatus,

        COUNT(m.id_mantenimiento) AS mantenimientos,

        SUM(
            CASE WHEN m.tipo_mantenimiento IN ('Correctivo','Emergencia')
            THEN 1 ELSE 0 END
        ) AS fallas,

        SUM(
            CASE WHEN m.tipo_mantenimiento = 'Emergencia'
            THEN 1 ELSE 0 END
        ) AS emergencias,

        SUM(
            CASE WHEN m.severidad IN ('Crítica','Critica')
            THEN 1 ELSE 0 END
        ) AS fallas_criticas,

        COALESCE(SUM(m.costo_total),0) AS costo_mantenimiento,
        COALESCE(SUM(m.horas_fuera_servicio),0) AS horas_fuera_servicio

    FROM dim_vehiculo v
    LEFT JOIN fact_mantenimiento m ON v.id_vehiculo = m.id_vehiculo

    GROUP BY
        v.id_vehiculo,
        v.numero_economico,
        v.placas,
        v.marca,
        v.modelo,
        v.nivel_riesgo,
        v.estatus

    ORDER BY costo_mantenimiento DESC
    """

    df = consulta(sql)

    columnas = [
        "mantenimientos", "fallas", "emergencias",
        "fallas_criticas", "costo_mantenimiento",
        "horas_fuera_servicio",
    ]

    return limpiar_numericos(df, columnas)


# ============================================================
# CARGA
# ============================================================

if not DB_PATH.exists():
    st.error("❌ No se encontró el Data Warehouse.")
    st.code(str(DB_PATH))
    st.stop()

entregas = cargar_entregas()
mantenimiento = cargar_mantenimiento()
vehiculos = cargar_vehiculos()
riesgos = cargar_riesgos()

if entregas.empty:
    st.error(
        "❌ No se pudieron cargar las entregas. "
        "Verifica que la base de datos exista y que fact_entregas "
        "tenga información."
    )
    st.stop()


# ============================================================

# SIDEBAR: navegación; los filtros globales están en la zona principal.
st.sidebar.markdown("## 🚚 SIG-LOG")
st.sidebar.caption("Centro de control operativo")
seccion_actual=st.sidebar.radio("Módulo",[
    "🏠 Resumen",
    "📦 Entregas",
    "👥 Clientes",
    "🚚 Vehículos",
    "👨‍✈️ Operadores",
    "🛣️ Rutas",
    "⛽ Combustible",
    "📌 Asignaciones",
    "🔧 Mantenimiento",
    "⚠️ Riesgos y alertas",
    "🔎 Preguntas de negocio",
    "📈 Patrones de operación",
    "🤖 Análisis avanzado",
    "🗺️ Mapa operativo",
    "📊 Reportes",
    "📋 Datos",
])
st.session_state["seccion_actual_siglog"] = seccion_actual
st.sidebar.markdown("---")
st.sidebar.info(
    "Usa los filtros superiores para cambiar el periodo y el alcance del análisis. "
    "Las gráficas conservan interacción, zoom y detalle al pasar el cursor."
)

df_filtrado=entregas.copy()
anios=sorted(entregas["anio"].dropna().astype(int).unique().tolist())
meses=sorted(entregas["mes"].dropna().astype(int).unique().tolist())
estatus=sorted(entregas["estatus"].dropna().astype(str).unique().tolist())
tipos=sorted(entregas["tipo_vehiculo"].dropna().astype(str).unique().tolist())
niveles=sorted(entregas["nivel_riesgo"].dropna().astype(str).unique().tolist())

# Barra superior estilo aplicación ejecutiva.
st.markdown(
    f"""
    <div class="topbar">
      <div class="topbar-brand">
        <div class="topbar-logo">🚚</div>
        <div class="topbar-name">SIG-LOG</div>
        <div class="topbar-desc">Sistema Integral de Gestión Logística · Centro de control ejecutivo</div>
      </div>
      <div class="topbar-right">
        <div class="topbar-date">📅 {datetime.now().strftime('%d/%m/%Y')}</div>
        <div class="topbar-exit">↪ Salir</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Hero principal arriba de los filtros, como en la referencia visual.
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-row">
        <div class="hero-brand">
          <div class="hero-icon">🚚</div>
          <div>
            <h1>SIG-LOG</h1>
            <p>Sistema Integral de Gestión Logística · Centro de control ejecutivo</p>
          </div>
        </div>
        <div class="hero-meta">
          <div class="hero-meta-item">
            <div class="hero-meta-icon">🗓️</div>
            <div>
              <div class="hero-meta-label">Periodo disponible</div>
              <div class="hero-meta-value">{escape(periodo_texto(entregas))}</div>
            </div>
          </div>
          <div class="hero-meta-item">
            <div class="hero-meta-icon">🗃️</div>
            <div>
              <div class="hero-meta-label">Registros totales</div>
              <div class="hero-meta-value">{len(entregas):,}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Control de análisis: los filtros permanecen en la zona principal y ahora
# tienen suficiente espacio para mostrar sus etiquetas completas.
st.markdown(
"""
<div class="filter-title">
  <div class="filter-icon">🔎</div>
  <div>
    <div class="filter-kicker">CONTROL DEL ANÁLISIS</div>
    <div class="filter-main">Filtra la información que quieres estudiar</div>
    <div class="filter-help">Los cambios se aplican a indicadores, gráficas y recomendaciones.</div>
  </div>
</div>
<div class="filter-box">
""",
unsafe_allow_html=True
)

f1,f2,f3,f4,f5,fr=st.columns([.9,1.45,1.2,1.35,1.25,.78], gap="small")
with f1:
    anios_sel=st.multiselect("Año",anios,default=anios,key="f_anios",placeholder="Todos")
with f2:
    if len(anios)>1:
        opciones_mes=sorted(
            entregas[["anio","mes"]].dropna().drop_duplicates().assign(
                etiqueta=lambda x:x["mes"].astype(int).map(MONTHS_ES)+" "+x["anio"].astype(int).astype(str)
            ).to_dict("records"),
            key=lambda x:(x["anio"],x["mes"])
        )
        mapa_mes={x["etiqueta"]:(int(x["anio"]),int(x["mes"])) for x in opciones_mes}
        opciones=list(mapa_mes.keys())
        meses_sel=st.multiselect(
            "Mes",
            opciones,
            default=opciones,
            key="f_meses",
            placeholder="Todos"
        )
    else:
        opciones=[MONTHS_ES[m] for m in meses]
        meses_sel=st.multiselect(
            "Mes",
            opciones,
            default=opciones,
            key="f_meses",
            placeholder="Todos"
        )
with f3:
    estatus_sel=st.multiselect(
        "Estado de entrega",estatus,default=estatus,key="f_estatus",placeholder="Todos"
    )
with f4:
    tipos_sel=st.multiselect(
        "Tipo de vehículo",tipos,default=tipos,key="f_tipos",placeholder="Todos"
    )
with f5:
    niveles_sel=st.multiselect(
        "Nivel de riesgo",niveles,default=niveles,key="f_riesgos",placeholder="Todos"
    )
with fr:
    if st.button(
        "↺ Limpiar",
        width='stretch',
        help="Restablecer todos los filtros",
        key="btn_limpiar_filtros"
    ):
        for key in ["f_anios","f_meses","f_estatus","f_tipos","f_riesgos"]:
            st.session_state.pop(key,None)
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

if anios_sel: df_filtrado=df_filtrado[df_filtrado["anio"].isin(anios_sel)]
if meses_sel:
    if len(anios)>1:
        pares=[mapa_mes[x] for x in meses_sel]
        df_filtrado=df_filtrado[
            df_filtrado.apply(
                lambda r:(int(r["anio"]),int(r["mes"])) in pares,
                axis=1
            )
        ]
    else:
        meses_num=[m for m in meses if MONTHS_ES[m] in meses_sel]
        df_filtrado=df_filtrado[df_filtrado["mes"].isin(meses_num)]
if estatus_sel: df_filtrado=df_filtrado[df_filtrado["estatus"].isin(estatus_sel)]
if tipos_sel: df_filtrado=df_filtrado[df_filtrado["tipo_vehiculo"].isin(tipos_sel)]
if niveles_sel: df_filtrado=df_filtrado[df_filtrado["nivel_riesgo"].isin(niveles_sel)]

periodo_actual=periodo_texto(df_filtrado)
st.session_state["periodo_dashboard"]=periodo_actual
st.markdown(
    f'<div class="filter-summary"><span>Periodo visible</span><strong>{escape(periodo_actual)}</strong>'
    f'<span class="dot">•</span><span>Registros</span><strong>{len(df_filtrado):,}</strong>'
    f'<span class="dot">•</span><span>Filtros activos</span>'
    f'<strong>{len(anios_sel) if anios_sel else 0} años · '
    f'{len(estatus_sel) if estatus_sel else 0} estados · '
    f'{len(tipos_sel) if tipos_sel else 0} tipos · '
    f'{len(niveles_sel) if niveles_sel else 0} niveles</strong></div>',
    unsafe_allow_html=True,
)

# ============================================================
# PREGUNTAS DE NEGOCIO Y PATRONES
# ============================================================

def _filtro_local(df, key_prefix, incluir_vehiculo=True, incluir_ruta=True):
    """Filtros independientes para cada pregunta de negocio."""
    if df.empty:
        return df
    with st.container():
        st.markdown('<div class="business-filter">', unsafe_allow_html=True)
        cols = st.columns(4)
        out = df.copy()
        with cols[0]:
            years = sorted(out["anio"].dropna().astype(int).unique().tolist()) if "anio" in out else []
            ys = st.multiselect("Año", years, default=years, key=f"{key_prefix}_year")
            if ys: out = out[out["anio"].isin(ys)]
        with cols[1]:
            routes = sorted(out["nombre_ruta"].dropna().astype(str).unique().tolist()) if incluir_ruta and "nombre_ruta" in out else []
            rs = st.multiselect("Ruta", routes, default=[], key=f"{key_prefix}_route", placeholder="Todas")
            if rs: out = out[out["nombre_ruta"].isin(rs)]
        with cols[2]:
            vehicles = sorted(out["numero_economico"].dropna().astype(str).unique().tolist()) if incluir_vehiculo and "numero_economico" in out else []
            vs = st.multiselect("Vehículo", vehicles, default=[], key=f"{key_prefix}_vehicle", placeholder="Todos")
            if vs: out = out[out["numero_economico"].astype(str).isin(vs)]
        with cols[3]:
            st.caption(f"Registros resultantes: {len(out):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    return out

def _pregunta_header(titulo, descripcion):
    seccion(titulo, descripcion)

def _respuesta_texto(texto, tipo="info"):
    interpretacion(texto, tipo)

def _calcular_prioridad_mantenimiento(riesgos_df):
    """Indicador complementario: no reemplaza nivel_riesgo del Data Warehouse."""
    r=riesgos_df.copy()
    for c in ["fallas_criticas","emergencias","costo_mantenimiento","horas_fuera_servicio"]:
        if c not in r.columns: r[c]=0
        r[c]=pd.to_numeric(r[c],errors="coerce").fillna(0)
    def rank100(col):
        if len(r)<=1: return pd.Series(100.0,index=r.index)
        return r[col].rank(method="average",pct=True)*100
    r["score_criticas"]=rank100("fallas_criticas")
    r["score_emergencias"]=rank100("emergencias")
    r["score_costo"]=rank100("costo_mantenimiento")
    r["score_horas"]=rank100("horas_fuera_servicio")
    r["prioridad_score"]=(
        r["score_criticas"]*.40+r["score_emergencias"]*.30+
        r["score_costo"]*.20+r["score_horas"]*.10
    )
    riesgo=r["nivel_riesgo"].astype(str).str.upper() if "nivel_riesgo" in r else pd.Series("",index=r.index)
    maskc=riesgo.isin(["CRÍTICO","CRITICO"])
    maska=riesgo.eq("ALTO")
    r.loc[maskc,"prioridad_score"]=np.maximum(r.loc[maskc,"prioridad_score"],75)
    r.loc[maska,"prioridad_score"]=np.maximum(r.loc[maska,"prioridad_score"],55)
    r["prioridad_score"]=r["prioridad_score"].clip(0,100)
    r["prioridad_nivel"]=pd.cut(
        r["prioridad_score"],bins=[-0.01,25,50,75,100],
        labels=["BAJA","MEDIA","ALTA","CRÍTICA"]
    ).astype(str)
    return r

def modulo_preguntas_negocio(df, mantenimiento, riesgos):
    _pregunta_header(
        "🔎 Preguntas clave del negocio",
        "Respuestas directas a las preguntas solicitadas para la defensa del proyecto. Cada análisis conserva filtros, interacción y descarga individual."
    )
    if df.empty:
        aviso("No hay entregas disponibles para responder las preguntas con los filtros actuales.", "warning")
        return

    # 1 RUTAS
    st.markdown("### 🚚 ¿Qué rutas son más utilizadas?")
    d = _filtro_local(df, "q_rutas")
    r = d.groupby("nombre_ruta", as_index=False).agg(
        entregas=("id_entrega","count"), tardias=("entrega_tardia","sum"),
        costo=("costo_total","sum"), distancia=("distancia_real_km","sum")
    ).sort_values("entregas", ascending=False)
    if not r.empty:
        r["participacion"] = r["entregas"]/r["entregas"].sum()*100
        top=r.head(10).sort_values("entregas")
        fig=px.bar(top,x="entregas",y="nombre_ruta",orientation="h",text="entregas",
                   labels={"entregas":"Entregas","nombre_ruta":"Ruta"},
                   title="Top 10 rutas por número de envíos")
        fig.update_traces(marker_color=COLORS["azul"],texttemplate="%{text:,}",textposition="outside",
                          customdata=top[["participacion","tardias","costo"]],
                          hovertemplate="<b>%{y}</b><br>Entregas: %{x:,}<br>Participación: %{customdata[0]:.1f}%<br>Tardías: %{customdata[1]:,}<br>Costo: $ MXN %{customdata[2]:,.2f}<extra></extra>")
        mostrar_fig(base_fig(fig,480,False), "¿Qué rutas son más utilizadas?", datos=top,
                    pregunta="¿Qué rutas son más utilizadas?",
                    interpretacion_texto=f"La ruta con mayor demanda es {r.iloc[0]['nombre_ruta']}, con {int(r.iloc[0]['entregas']):,} entregas ({r.iloc[0]['participacion']:.1f}% del total).",
                    recomendacion="Revisar capacidad y disponibilidad de vehículos en las rutas de mayor frecuencia.")
        st.dataframe(r.rename(columns={"nombre_ruta":"Ruta","entregas":"Entregas","participacion":"Participación (%)","tardias":"Tardías","costo":"Costo (MXN)","distancia":"Distancia (km)"}),width='stretch',hide_index=True)

    # 2 VEHICULOS COSTO
    st.markdown("### 💰 ¿Qué vehículos generan mayores costos?")
    d = _filtro_local(df, "q_costos")
    if d.empty:
        aviso("No hay entregas para calcular costos con los filtros actuales.", "warning")
    else:
        d=d.copy()
        for c in ["costo_total","costo_envio","costo_combustible","distancia_real_km"]:
            if c in d.columns:
                d[c]=pd.to_numeric(d[c],errors="coerce").fillna(0)
        base_cost=d["costo_total"] if "costo_total" in d else pd.Series(0.0,index=d.index)
        d["costo_total_analizado"] = base_cost if float(base_cost.abs().sum())>0 else (
            d.get("costo_envio",0)+d.get("costo_combustible",0)
        )
        if "numero_economico" not in d.columns:
            d["numero_economico"]=d["id_vehiculo"].astype(str)
        d["numero_economico"]=d["numero_economico"].fillna("").astype(str)
        d.loc[d["numero_economico"].str.strip().eq(""),"numero_economico"]="Vehículo "+d["id_vehiculo"].astype(str)

        v=d.groupby(["id_vehiculo","numero_economico"],as_index=False).agg(
            entregas=("id_entrega","count"), costo_total=("costo_total_analizado","sum"),
            combustible=("costo_combustible","sum"), distancia=("distancia_real_km","sum")
        )
        if not mantenimiento.empty:
            mc=mantenimiento.groupby("id_vehiculo",as_index=False).agg(costo_mantenimiento=("costo_total","sum"))
            v=v.merge(mc,on="id_vehiculo",how="left")
        else:
            v["costo_mantenimiento"]=0
        v["costo_mantenimiento"]=pd.to_numeric(v["costo_mantenimiento"],errors="coerce").fillna(0)
        v["costo_promedio_entrega"]=np.where(v["entregas"]>0,v["costo_total"]/v["entregas"],0)
        v["participacion_costo"]=np.where(v["costo_total"].sum()>0,v["costo_total"]/v["costo_total"].sum()*100,0)
        top=v.sort_values("costo_total",ascending=False).head(10).sort_values("costo_total")
        if top.empty or float(top["costo_total"].abs().sum())==0:
            aviso("No hay costos registrados para los vehículos seleccionados.", "warning")
        else:
            fig=px.bar(top,x="costo_total",y="numero_economico",orientation="h",text="costo_total",
                       title="Vehículos con mayor costo logístico",
                       labels={"costo_total":"Costo total (MXN)","numero_economico":"Vehículo"})
            fig.update_traces(marker_color=COLORS["azul_oscuro"],texttemplate="$%{text:,.0f}",textposition="outside",
                              hovertemplate="<b>%{y}</b><br>Costo: $ MXN %{x:,.2f}<br>Entregas: %{customdata[0]:,}<br>Combustible: $ MXN %{customdata[1]:,.2f}<br>Mantenimiento: $ MXN %{customdata[2]:,.2f}<extra></extra>",
                              customdata=top[["entregas","combustible","costo_mantenimiento"]])
            mostrar_fig(base_fig(fig,480,False),"¿Qué vehículos generan mayores costos?",datos=top,
                        pregunta="¿Qué vehículos generan mayores costos?",
                        interpretacion_texto=f"{top.iloc[-1]['numero_economico']} concentra el mayor costo acumulado: {_money_local(top.iloc[-1]['costo_total'])}. Si costo_total estaba vacío/cero en el DW, se utilizó costo_envio + costo_combustible para evitar una visualización engañosa.",
                        recomendacion="Comparar costo, volumen de entregas, combustible y mantenimiento antes de decidir sustitución o reasignación.")
            st.dataframe(top.rename(columns={"numero_economico":"Vehículo","costo_total":"Costo total (MXN)","combustible":"Combustible (MXN)","costo_mantenimiento":"Mantenimiento (MXN)","costo_promedio_entrega":"Costo promedio/entrega","participacion_costo":"Participación costo (%)"}),width='stretch',hide_index=True)

    # 3 OPERADORES
    st.markdown("### 👨‍✈️ ¿Qué operadores realizan más entregas?")
    d = _filtro_local(df, "q_operadores", incluir_vehiculo=False, incluir_ruta=True)
    op=d.groupby("id_operador",as_index=False).agg(entregas=("id_entrega","count"),tardias=("entrega_tardia","sum"),retraso=("minutos_retraso","mean"),costo=("costo_total","sum"))
    op["puntualidad"]=np.where(op["entregas"]>0,(op["entregas"]-op["tardias"])/op["entregas"]*100,0)
    top=op.sort_values("entregas",ascending=False).head(12).sort_values("entregas")
    top["operador"]="Operador "+top["id_operador"].astype(str)
    fig=px.bar(top,x="entregas",y="operador",orientation="h",text="entregas",title="Operadores con mayor número de entregas",labels={"entregas":"Entregas","operador":"Operador"})
    fig.update_traces(marker_color=COLORS["morado"],texttemplate="%{text:,}",textposition="outside",customdata=top[["tardias","puntualidad","retraso"]],hovertemplate="<b>%{y}</b><br>Entregas: %{x:,}<br>Tardías: %{customdata[0]:,}<br>Puntualidad: %{customdata[1]:.1f}%<br>Retraso: %{customdata[2]:.1f} min<extra></extra>")
    mostrar_fig(base_fig(fig,500,False),"¿Qué operadores realizan más entregas?",datos=top,pregunta="¿Qué operadores realizan más entregas?",interpretacion_texto=f"{top.iloc[-1]['operador']} concentra la mayor carga del grupo mostrado.",recomendacion="Comparar carga y puntualidad para equilibrar servicios sin penalizar la dificultad de las rutas.")
    st.dataframe(top.rename(columns={"operador":"Operador","entregas":"Entregas","tardias":"Tardías","puntualidad":"Puntualidad (%)","retraso":"Retraso promedio (min)"}),width='stretch',hide_index=True)

    # 4 RUTAS RETRASO
    st.markdown("### ⏱️ ¿Qué rutas presentan mayores retrasos?")
    d = _filtro_local(df, "q_retrasos", incluir_vehiculo=False)
    rr=d.groupby("nombre_ruta",as_index=False).agg(entregas=("id_entrega","count"),tardias=("entrega_tardia","sum"),retraso=("minutos_retraso","mean"))
    rr["tasa_retraso"]=np.where(rr["entregas"]>0,rr["tardias"]/rr["entregas"]*100,0)
    top=rr.sort_values(["tasa_retraso","retraso"],ascending=False).head(12).sort_values("tasa_retraso")
    fig=px.bar(top,x="tasa_retraso",y="nombre_ruta",orientation="h",text="tasa_retraso",title="Rutas con mayor proporción de entregas tardías",labels={"tasa_retraso":"Tasa tardía (%)","nombre_ruta":"Ruta"})
    fig.update_traces(marker_color=COLORS["rojo"],texttemplate="%{text:.1f}%",textposition="outside",customdata=top[["entregas","tardias","retraso"]],hovertemplate="<b>%{y}</b><br>Tasa: %{x:.1f}%<br>Entregas: %{customdata[0]:,}<br>Tardías: %{customdata[1]:,}<br>Retraso promedio: %{customdata[2]:.1f} min<extra></extra>")
    mostrar_fig(base_fig(fig,500,False),"¿Qué rutas presentan mayores retrasos?",datos=top,pregunta="¿Qué rutas presentan mayores retrasos?",interpretacion_texto=f"La ruta {top.iloc[-1]['nombre_ruta']} presenta la mayor tasa tardía dentro del conjunto filtrado.",recomendacion="Revisar tráfico, horarios, distancia y asignación de unidades antes de cambiar la ruta.")
    st.dataframe(top.rename(columns={"nombre_ruta":"Ruta","entregas":"Entregas","tardias":"Tardías","tasa_retraso":"Tasa tardía (%)","retraso":"Retraso promedio (min)"}),width='stretch',hide_index=True)

    # 5 COMBUSTIBLE
    st.markdown("### ⛽ ¿Qué vehículos consumen más combustible?")
    d = _filtro_local(df, "q_combustible", incluir_ruta=False)
    cb=d.groupby(["id_vehiculo","numero_economico"],as_index=False).agg(litros=("combustible_consumido_litros","sum"),km=("distancia_real_km","sum"),costo=("costo_combustible","sum"),entregas=("id_entrega","count"))
    cb["km_l"]=np.where(cb["litros"]>0,cb["km"]/cb["litros"],0)
    top=cb.sort_values("litros",ascending=False).head(12)
    fig=px.bar(top,x="numero_economico",y="litros",text="litros",title="Vehículos con mayor consumo de combustible",labels={"numero_economico":"Vehículo","litros":"Litros"})
    fig.update_traces(marker_color=COLORS["amarillo"],texttemplate="%{text:,.0f} L",textposition="outside",customdata=top[["km","km_l","costo","entregas"]],hovertemplate="<b>%{x}</b><br>Litros: %{y:,.1f}<br>Km: %{customdata[0]:,.1f}<br>Rendimiento: %{customdata[1]:.2f} km/L<br>Costo: $ MXN %{customdata[2]:,.2f}<br>Entregas: %{customdata[3]:,}<extra></extra>")
    mostrar_fig(base_fig(fig,470,False),"¿Qué vehículos consumen más combustible?",datos=top,pregunta="¿Qué vehículos consumen más combustible?",interpretacion_texto=f"{top.iloc[0]['numero_economico']} registra el mayor consumo acumulado del grupo mostrado; debe compararse con sus kilómetros recorridos.",recomendacion="Evaluar rendimiento km/L y litros/100 km antes de concluir que una unidad es ineficiente.")
    st.dataframe(top.rename(columns={"numero_economico":"Vehículo","litros":"Combustible (L)","km":"Distancia (km)","km_l":"Rendimiento (km/L)","costo":"Costo combustible (MXN)","entregas":"Entregas"}),width='stretch',hide_index=True)

    # 6 CAUSAS
    st.markdown("### ⚠️ ¿Cuáles son las causas principales de retraso?")
    texto=d["observaciones"].fillna("").astype(str).str.lower() if "observaciones" in d else pd.Series("",index=d.index)
    causas_map={"Tráfico":"traf","Tráfico":"tráfico","Falla mecánica":"mecán","Clima":"clima","Accidente":"accident","Documentación":"document","Carga/descarga":"carga","Ruta":"ruta"}
    conteos=[]
    for causa, clave in causas_map.items():
        conteos.append((causa,int(texto.str.contains(clave,regex=False).sum())))
    causas_df=pd.DataFrame(conteos,columns=["Causa","Casos"]).sort_values("Casos",ascending=False)
    if causas_df["Casos"].sum()==0:
        aviso("El Data Warehouse no contiene una causa explícita de retraso en fact_entregas; se evita inventar causas. Se muestran las causas de mantenimiento en su módulo correspondiente.","warning")
    else:
        causas_df["Porcentaje"]=causas_df["Casos"]/causas_df["Casos"].sum()*100
        fig=px.bar(causas_df.sort_values("Casos"),x="Casos",y="Causa",orientation="h",text="Casos",title="Causas de retraso identificables en observaciones")
        fig.update_traces(marker_color=COLORS["naranja"],textposition="outside",customdata=causas_df.sort_values("Casos")[["Porcentaje"]],hovertemplate="<b>%{y}</b><br>Casos: %{x:,}<br>Participación: %{customdata[0]:.1f}%<extra></extra>")
        mostrar_fig(base_fig(fig,430,False),"¿Cuáles son las causas principales de retraso?",datos=causas_df,pregunta="¿Cuáles son las causas principales de retraso?",interpretacion_texto="Las causas se clasifican únicamente cuando aparecen en el texto de observaciones; no se agregan categorías no respaldadas por los datos.",recomendacion="Mejorar el registro estructurado de causa de retraso para permitir análisis más confiables.")

    # 7 MANTENIMIENTO
    st.markdown("### 🔧 ¿Qué vehículos requieren mantenimiento?")
    if not riesgos.empty:
        top=_calcular_prioridad_mantenimiento(riesgos).sort_values(
            ["prioridad_score","fallas_criticas","emergencias","costo_mantenimiento"],
            ascending=False
        ).head(15)
        graf_top=top.sort_values("prioridad_score")
        fig=px.bar(
            graf_top,x="prioridad_score",y="numero_economico",orientation="h",
            color="prioridad_nivel",
            color_discrete_map={"CRÍTICA":COLORS["rojo"],"ALTA":COLORS["naranja"],"MEDIA":COLORS["amarillo"],"BAJA":COLORS["verde"]},
            text="prioridad_score",
            title="Vehículos que requieren mayor atención de mantenimiento",
            labels={"prioridad_score":"Prioridad de mantenimiento (0–100)","numero_economico":"Vehículo","prioridad_nivel":"Prioridad"}
        )
        fig.update_traces(
            texttemplate="%{text:.0f}",textposition="outside",
            customdata=graf_top[["nivel_riesgo","fallas_criticas","emergencias","costo_mantenimiento","horas_fuera_servicio"]],
            hovertemplate="<b>%{y}</b><br>Prioridad: %{x:.0f}/100<br>Riesgo DW: %{customdata[0]}<br>Fallas críticas: %{customdata[1]:,}<br>Emergencias: %{customdata[2]:,}<br>Costo: $ MXN %{customdata[3]:,.2f}<br>Horas fuera: %{customdata[4]:,.1f}<extra></extra>"
        )
        p=top.iloc[0]
        mostrar_fig(
            base_fig(fig,540,True),"¿Qué vehículos requieren mantenimiento?",datos=top,
            pregunta="¿Qué vehículos requieren mantenimiento?",
            interpretacion_texto=f"{p['numero_economico']} es el caso de mayor prioridad con {p['prioridad_score']:.0f}/100 ({p['prioridad_nivel']}). La prioridad usa fallas críticas, emergencias, costo acumulado y horas fuera de servicio; el riesgo original del DW se conserva.",
            recomendacion="Atender primero las prioridades críticas y altas. Si una unidad MEDIO del DW tiene más fallas críticas, emergencias y costo que otras, su prioridad complementaria puede subir."
        )
        st.dataframe(top.rename(columns={"numero_economico":"Vehículo","nivel_riesgo":"Riesgo DW","prioridad_nivel":"Prioridad mantenimiento","prioridad_score":"Prioridad (0-100)","fallas":"Fallas","emergencias":"Emergencias","fallas_criticas":"Fallas críticas","costo_mantenimiento":"Costo mantenimiento (MXN)","horas_fuera_servicio":"Horas fuera de servicio"}),width="stretch",hide_index=True)
    else:
        aviso("No existen resultados de riesgo/mantenimiento disponibles.", "warning")

    # 8 PREDICCIÓN
    st.markdown("### 🔮 ¿Es posible predecir si una entrega llegará tarde?")
    modelos=list(MODELOS_DIR.glob("*")) if MODELOS_DIR.exists() else []
    reportes=list(REPORTES_MODELOS_DIR.glob("*")) if REPORTES_MODELOS_DIR.exists() else []
    pred=[p for p in modelos+reportes if any(k in p.name.lower() for k in ["clasif","retras","tard","predict","modelo"])]
    if pred:
        aviso(f"Sí. El proyecto contiene artefactos relacionados con modelos predictivos ({len(pred)} archivo(s)). La predicción debe presentarse como probabilidad/riesgo y no como certeza.", "success")
        st.dataframe(pd.DataFrame({"Archivo": [p.name for p in pred], "Tipo": [p.suffix for p in pred]}),width='stretch',hide_index=True)
    else:
        aviso("La interfaz puede responder la pregunta metodológicamente, pero no se encontró en las carpetas esperadas un modelo/reportes predictivos que permita mostrar una probabilidad individual sin inventarla.", "warning")

    # 9 CLUSTERS
    st.markdown("### 🧩 ¿Podemos identificar grupos de rutas similares?")
    perfil=KMEANS_DIR/"perfil_clusters.csv"
    if perfil.exists():
        cl=pd.read_csv(perfil)
        st.dataframe(cl,width='stretch',hide_index=True)
        pctcol=next((c for c in cl.columns if "porcentaje" in c.lower() or "pct" in c.lower()),None)
        if "cluster" in cl.columns and pctcol:
            graf=cl[["cluster",pctcol]].copy(); graf.columns=["Grupo","Participación (%)"]
            fig=px.bar(graf,x="Grupo",y="Participación (%)",text="Participación (%)",title="Grupos de rutas/operación con comportamiento similar")
            fig.update_traces(marker_color=COLORS["morado"],texttemplate="%{text:.1f}%",textposition="outside")
            mostrar_fig(base_fig(fig,430,False),"¿Podemos identificar grupos de rutas similares?",datos=graf,pregunta="¿Podemos identificar grupos de rutas similares?",interpretacion_texto="Los grupos se toman de los resultados de agrupamiento existentes en el proyecto; representan observaciones con características operativas similares.",recomendacion="Definir estrategias diferenciadas por grupo en lugar de aplicar la misma política a todas las rutas.")
    else:
        aviso("No se encontró perfil_clusters.csv; no se mostrará una segmentación inventada.", "warning")

def _money_local(x):
    try: return f"$ MXN {float(x):,.2f}"
    except Exception: return "$ MXN 0.00"

def modulo_patrones(df):
    _pregunta_header("📈 Patrones de operación detectados","Explora demanda, saturación horaria, frecuencia y concentración de envíos.")
    if df.empty:
        aviso("No hay datos para detectar patrones.","warning"); return
    # Demanda mensual
    d_demanda=_filtro_local(df,"pat_demanda")
    mensual=d_demanda.groupby(["anio","mes","nombre_mes"],as_index=False).agg(entregas=("id_entrega","count")).sort_values(["anio","mes"])
    mensual["periodo"]=mensual["nombre_mes"]+" "+mensual["anio"].astype(int).astype(str)
    fig=px.bar(mensual,x="periodo",y="entregas",text="entregas",title="Demanda de servicios por periodo",labels={"periodo":"Periodo","entregas":"Entregas"})
    fig.update_traces(marker_color=COLORS["verde"],textposition="outside",hovertemplate="<b>%{x}</b><br>Entregas: %{y:,}<extra></extra>")
    mostrar_fig(base_fig(fig,430,False),"Demanda de servicios",datos=mensual,pregunta="¿Cómo se comporta la demanda de servicios?",interpretacion_texto="Los picos muestran periodos de mayor carga de trabajo.",recomendacion="Preparar capacidad de vehículos y operadores para los periodos de mayor demanda.")
    # Horas
    h=_filtro_local(df,"pat_horas").copy()
    if "hora_salida" in h:
        _hora_raw = h["hora_salida"].astype(str)
        try:
            _hora_dt = pd.to_datetime(_hora_raw, format="mixed", errors="coerce")
        except Exception:
            _hora_dt = pd.to_datetime(_hora_raw, errors="coerce")
        h["hora"] = _hora_dt.dt.hour
    else:
        h["hora"] = np.nan
    if h["hora"].isna().all() and "hora_salida" in h:
        h["hora"]=pd.to_numeric(h["hora_salida"].astype(str).str[:2],errors="coerce")
    horas=h.dropna(subset=["hora"]).groupby("hora",as_index=False).agg(entregas=("id_entrega","count"),tardias=("entrega_tardia","sum"))
    if not horas.empty:
        horas["tasa_tardia"]=horas["tardias"]/horas["entregas"]*100
        fig=px.bar(horas,x="hora",y="entregas",text="entregas",title="Horarios de mayor saturación",labels={"hora":"Hora de salida","entregas":"Entregas"})
        fig.update_traces(marker_color=COLORS["turquesa"],textposition="outside",customdata=horas[["tardias","tasa_tardia"]],hovertemplate="<b>%{x}:00</b><br>Entregas: %{y:,}<br>Tardías: %{customdata[0]:,}<br>Tasa tardía: %{customdata[1]:.1f}%<extra></extra>")
        mostrar_fig(base_fig(fig,430,False),"Horarios de mayor saturación",datos=horas,pregunta="¿Cuáles son los horarios de mayor saturación?",interpretacion_texto=f"La franja de {int(horas.iloc[horas['entregas'].argmax()]['hora']):02d}:00 concentra el mayor volumen registrado.",recomendacion="Revisar capacidad y ventanas de salida en las horas de mayor concentración.")
    # frecuencia y rutas
    d_freq=_filtro_local(df,"pat_frecuencia",incluir_vehiculo=False)
    rutas=d_freq.groupby("nombre_ruta",as_index=False).size().rename(columns={"size":"envios"}).sort_values("envios",ascending=False).head(15)
    fig=px.bar(rutas.sort_values("envios"),x="envios",y="nombre_ruta",orientation="h",text="envios",title="Rutas con mayor número de envíos",labels={"envios":"Envíos","nombre_ruta":"Ruta"})
    fig.update_traces(marker_color=COLORS["azul"],textposition="outside")
    mostrar_fig(base_fig(fig,500,False),"Rutas con mayor número de envíos",datos=rutas,pregunta="¿Qué rutas concentran mayor frecuencia?",interpretacion_texto=f"La ruta {rutas.iloc[0]['nombre_ruta']} es la de mayor frecuencia en el conjunto mostrado.",recomendacion="Priorizar capacidad y monitoreo de las rutas de mayor frecuencia.")

def _filtro_contextual_modulo(df, modulo):
    """Filtro contextual opcional para módulos que necesitan foco adicional.

    El filtro global sigue siendo obligatorio y ya redujo el DataFrame.
    Estos controles sólo acotan la vista del módulo actual y no cambian
    el periodo global ni el resto del dashboard.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    if modulo not in {"👨‍✈️ Operadores", "🛣️ Rutas", "⛽ Combustible"}:
        return df

    campos = []
    if modulo == "👨‍✈️ Operadores":
        if "id_operador" in df.columns:
            campos.append(("Operador", "id_operador"))
        if "nombre_ruta" in df.columns:
            campos.append(("Ruta", "nombre_ruta"))
    elif modulo == "🛣️ Rutas":
        if "nombre_ruta" in df.columns:
            campos.append(("Ruta", "nombre_ruta"))
        if "nivel_trafico" in df.columns:
            campos.append(("Nivel de tráfico", "nivel_trafico"))
    elif modulo == "⛽ Combustible":
        if "numero_economico" in df.columns:
            campos.append(("Vehículo", "numero_economico"))
        if "tipo_combustible" in df.columns:
            campos.append(("Combustible", "tipo_combustible"))

    if not campos:
        return df

    st.markdown(
        '<div class="module-filter">'
        '<div class="module-filter-title">🎛️ ENFOQUE DEL MÓDULO</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(len(campos), gap="small")
    out = df.copy()
    for idx, (label, col) in enumerate(campos):
        opciones = sorted(out[col].dropna().astype(str).unique().tolist())
        with cols[idx]:
            seleccion = st.multiselect(
                label,
                opciones,
                default=[],
                placeholder="Todos",
                key=_next_key(f"modulo_{modulo}_{col}"),
                help="Este filtro sólo afecta al módulo actual; los filtros globales siguen definiendo el periodo y alcance."
            )
            if seleccion:
                out = out[out[col].astype(str).isin(seleccion)]
    st.caption(
        f"Vista contextual: {len(out):,} registros · periodo global {periodo_actual}"
    )
    return out


# ============================================================
# RESUMEN
# ============================================================

if seccion_actual == "🏠 Resumen":

    seccion(
        "Resumen ejecutivo",
        f"Panorama general del periodo seleccionado: {periodo_texto(df_filtrado)}.",
    )

    if df_filtrado.empty:
        aviso("No hay registros con los filtros seleccionados.", "warning")
        st.stop()

    total = len(df_filtrado)
    tardias = int(df_filtrado["entrega_tardia"].sum())
    tasa_tardia = tardias / total * 100 if total else 0
    retraso = df_filtrado["minutos_retraso"].mean()
    costo = df_filtrado["costo_total"].sum()
    combustible = df_filtrado["combustible_consumido_litros"].sum()
    distancia = df_filtrado["distancia_real_km"].sum()
    vehs = df_filtrado["id_vehiculo"].nunique()

    # Distribución 4 + 3: evita que cantidades largas como el costo total
    # queden comprimidas en siete columnas estrechas.
    c = st.columns(4, gap="medium")
    with c[0]:
        kpi("📦", "Entregas", numero(total), "Registros seleccionados")
    with c[1]:
        kpi("⏰", "Entregas tardías", numero(tardias), pct(tasa_tardia))
    with c[2]:
        kpi("⏱️", "Retraso promedio", f"{retraso:.1f} min", "Por entrega")
    with c[3]:
        kpi("💰", "Costo logístico", moneda(costo), "Envío + combustible")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    c = st.columns(3, gap="medium")
    with c[0]:
        kpi("⛽", "Combustible", f"{combustible:,.1f} L", "Consumo acumulado")
    with c[1]:
        kpi("📍", "Distancia", f"{distancia:,.1f} km", "Recorrido registrado")
    with c[2]:
        kpi("🚚", "Vehículos", numero(vehs), "Unidades involucradas")


    st.markdown("")
    try:
        top_ruta=(df_filtrado.groupby("nombre_ruta",as_index=False)
                  .agg(entregas=("id_entrega","count"),tardias=("entrega_tardia","sum"),
                       retraso=("minutos_retraso","mean")))
        top_ruta["tasa_tardia"]=np.where(top_ruta["entregas"]>0,top_ruta["tardias"]/top_ruta["entregas"]*100,0)
        top_ruta=top_ruta.sort_values(["tasa_tardia","entregas"],ascending=False).head(1)
        top_veh=(df_filtrado.groupby("numero_economico",as_index=False)
                 .agg(costo=("costo_total","sum"),entregas=("id_entrega","count"))
                 .sort_values("costo",ascending=False).head(1))
        if not top_ruta.empty and not top_veh.empty:
            r=top_ruta.iloc[0]; v=top_veh.iloc[0]
            st.markdown(f"""
            <div class="exec-strip">
              <div class="exec-item danger">
                <div class="exec-label">PUNTO DE ATENCIÓN</div>
                <div class="exec-title">Ruta con mayor proporción de retrasos</div>
                <div class="exec-value">{escape(str(r["nombre_ruta"]))} · {r["tasa_tardia"]:.1f}%</div>
                <div class="exec-note">{int(r["tardias"]):,} tardías de {int(r["entregas"]):,} entregas.</div>
              </div>
              <div class="exec-item">
                <div class="exec-label">MAYOR IMPACTO ECONÓMICO</div>
                <div class="exec-title">Vehículo con mayor costo acumulado</div>
                <div class="exec-value">{escape(str(v["numero_economico"]))} · $ MXN {v["costo"]:,.2f}</div>
                <div class="exec-note">Comparar el gasto con su volumen de trabajo antes de intervenir.</div>
              </div>
              <div class="exec-item success">
                <div class="exec-label">DECISIÓN SUGERIDA</div>
                <div class="exec-title">Investigar antes de modificar recursos</div>
                <div class="exec-note">Validar horario, distancia, tráfico, vehículo y antecedentes registrados.</div>
              </div>
            </div>
            """,unsafe_allow_html=True)
    except Exception:
        pass

    # ---- Entregas por mes
    mensual = (
        df_filtrado.groupby(["anio", "mes", "nombre_mes"], as_index=False)
        .agg(
            entregas=("id_entrega", "count"),
            tardias=("entrega_tardia", "sum"),
            retraso_promedio=("minutos_retraso", "mean"),
            costo=("costo_total", "sum"),
        )
        .sort_values(["anio", "mes"])
    )

    mensual["periodo_label"] = (
        mensual["nombre_mes"] + " " + mensual["anio"].astype(int).astype(str)
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            mensual,
            x="periodo_label",
            y="entregas",
            text="entregas",
            title="¿Cuántas entregas se realizaron cada mes?",
            labels={
                "periodo_label": "Mes",
                "entregas": "Número de entregas",
            },
        )
        fig.update_traces(
            marker_color=COLORS["azul"],
            texttemplate="%{text:,}",
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Entregas: %{y:,}<extra></extra>",
        )
        mostrar_fig(base_fig(fig, 380, False))

    with col2:
        fig = px.line(
            mensual,
            x="periodo_label",
            y="retraso_promedio",
            markers=True,
            title="¿Cómo evolucionó el retraso promedio?",
            labels={
                "periodo_label": "Mes",
                "retraso_promedio": "Retraso promedio (minutos)",
            },
        )
        fig.update_traces(
            line=dict(color=COLORS["naranja"], width=3),
            marker=dict(size=9),
            hovertemplate="<b>%{x}</b><br>Retraso: %{y:.1f} min<extra></extra>",
        )
        mostrar_fig(base_fig(fig, 380, False))

    st.markdown("")

    # ---- Estatus
    col1, col2 = st.columns(2)

    with col1:
        est = (
            df_filtrado.groupby("estatus")
            .size()
            .reset_index(name="entregas")
            .sort_values("entregas", ascending=False)
        )

        est["porcentaje"] = est["entregas"] / est["entregas"].sum() * 100

        fig = px.pie(
            est,
            names="estatus",
            values="entregas",
            hole=.58,
            title="¿Cuál es el estado de las entregas?",
            color="estatus",
            color_discrete_map=STATUS_COLORS,
        )

        fig.update_traces(
            textinfo="percent",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Entregas: %{value:,}<br>"
                "Participación: %{percent}<extra></extra>"
            ),
        )
        mostrar_fig(base_fig(fig, 390, True))

    with col2:
        fig = px.scatter(
            df_filtrado,
            x="distancia_real_km",
            y="minutos_retraso",
            color="entrega_tardia",
            hover_data=[
                "id_entrega",
                "nombre_ruta",
                "numero_economico",
                "fecha_salida",
                "costo_total",
            ],
            color_discrete_map={
                0: COLORS["verde"],
                1: COLORS["rojo"],
            },
            title="¿Las rutas más largas presentan más retrasos?",
            labels={
                "distancia_real_km": "Distancia recorrida (km)",
                "minutos_retraso": "Retraso (minutos)",
                "entrega_tardia": "Entrega tardía",
            },
        )
        fig.update_traces(
            marker=dict(size=7, opacity=.55),
            hovertemplate=(
                "<b>Entrega %{customdata[0]}</b><br>"
                "Ruta: %{customdata[1]}<br>"
                "Vehículo: %{customdata[2]}<br>"
                "Fecha: %{customdata[3]}<br>"
                "Distancia: %{x:.1f} km<br>"
                "Retraso: %{y:.1f} min<br>"
                "Costo: $ MXN %{customdata[4]:,.2f}<extra></extra>"
            ),
        )
        mostrar_fig(base_fig(fig, 390, True))

    st.markdown("")
    seccion(
        "Lectura ejecutiva",
        "Una lectura rápida de los resultados para convertir los datos en decisiones.",
    )

    tasa_tardia = tardias / total * 100 if total else 0
    if tasa_tardia >= 30:
        interpretacion(
            f"El {tasa_tardia:.1f}% de las entregas terminó tarde. La puntualidad debe considerarse una prioridad operativa en el periodo seleccionado.",
            "danger",
        )
    elif tasa_tardia >= 15:
        interpretacion(
            f"El {tasa_tardia:.1f}% de las entregas terminó tarde. Conviene revisar rutas y vehículos que concentran los mayores retrasos.",
            "warning",
        )
    else:
        interpretacion(
            f"El {tasa_tardia:.1f}% de las entregas terminó tarde. El comportamiento general es favorable, aunque conviene monitorear los casos extremos.",
            "success",
        )

    ra, rb, rc = st.columns(3)
    with ra:
        accion(
            "🚚 Revisar unidades con mayor retraso",
            "Utilizar el ranking de vehículos para priorizar revisiones operativas y detectar patrones repetitivos.",
        )
    with rb:
        accion(
            "📍 Analizar rutas problemáticas",
            "Revisar las rutas con mayor retraso promedio antes de modificar recursos o tiempos programados.",
        )
    with rc:
        accion(
            "📅 Preparar capacidad para meses de alta demanda",
            "Los periodos con mayor volumen pueden requerir mayor disponibilidad de vehículos y capacidad operativa.",
        )

    st.markdown("")
    seccion(
        "Indicadores de operación",
        "Valores acumulados del periodo seleccionado para apoyar decisiones de costo y capacidad.",
    )

    a, b, c = st.columns(3)
    with a:
        st.metric("📍 Distancia total", f"{distancia:,.1f} km")
    with b:
        st.metric("💵 Costo de envíos", moneda(df_filtrado["costo_envio"].sum()))
    with c:
        st.metric(
            "💰 Costo promedio por entrega",
            moneda(costo / total if total else 0),
        )


# ============================================================
# ENTREGAS
# ============================================================

elif seccion_actual == "📦 Entregas":

    seccion(
        "Entregas",
        f"Análisis del cumplimiento de entregas durante {periodo_texto(df_filtrado)}.",
    )

    if df_filtrado.empty:
        aviso("No hay entregas para los filtros seleccionados.", "warning")
        st.stop()

    total = len(df_filtrado)
    tardias = int(df_filtrado["entrega_tardia"].sum())

    c = st.columns(5)
    with c[0]:
        kpi("📦", "Entregas", numero(total))
    with c[1]:
        kpi("⏰", "Tardías", numero(tardias), pct(tardias / total * 100))
    with c[2]:
        kpi("⏱️", "Retraso promedio", f"{df_filtrado['minutos_retraso'].mean():.1f} min")
    with c[3]:
        kpi("📍", "Distancia", f"{df_filtrado['distancia_real_km'].sum():,.1f} km")
    with c[4]:
        kpi("💰", "Costo total", moneda(df_filtrado["costo_total"].sum()))

    st.markdown("")

    mensual = (
        df_filtrado.groupby(["anio", "mes", "nombre_mes"], as_index=False)
        .agg(
            entregas=("id_entrega", "count"),
            tardias=("entrega_tardia", "sum"),
            retraso=("minutos_retraso", "mean"),
        )
        .sort_values(["anio", "mes"])
    )

    mensual["periodo"] = (
        mensual["nombre_mes"] + " " + mensual["anio"].astype(int).astype(str)
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            mensual,
            x="periodo",
            y=["entregas", "tardias"],
            barmode="group",
            title="Entregas realizadas y entregas tardías por mes",
            labels={
                "periodo": "Mes",
                "value": "Número de entregas",
                "variable": "Indicador",
            },
            color_discrete_map={
                "entregas": COLORS["azul"],
                "tardias": COLORS["rojo"],
            },
        )
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:,}<extra></extra>"
        )
        mostrar_fig(base_fig(fig, 410, True))

    with col2:
        fig = px.line(
            mensual,
            x="periodo",
            y="retraso",
            markers=True,
            title="Retraso promedio registrado en cada mes",
            labels={
                "periodo": "Mes",
                "retraso": "Retraso promedio (minutos)",
            },
        )
        fig.update_traces(
            line=dict(color=COLORS["naranja"], width=3),
            marker=dict(size=9),
            hovertemplate="<b>%{x}</b><br>Retraso promedio: %{y:.1f} min<extra></extra>",
        )
        mostrar_fig(base_fig(fig, 410, False))

    st.markdown("---")

    rutas = (
        df_filtrado.groupby(["id_ruta", "nombre_ruta"], as_index=False)
        .agg(
            entregas=("id_entrega", "count"),
            retraso=("minutos_retraso", "mean"),
            tardias=("entrega_tardia", "sum"),
        )
        .sort_values("retraso", ascending=False)
        .head(10)
        .sort_values("retraso")
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            rutas,
            x="retraso",
            y="nombre_ruta",
            orientation="h",
            title="Las 10 rutas con mayor retraso promedio",
            labels={
                "retraso": "Retraso promedio (minutos)",
                "nombre_ruta": "Ruta",
            },
            text="retraso",
        )
        fig.update_traces(
            marker_color=COLORS["rojo"],
            texttemplate="%{text:.1f} min",
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Retraso: %{x:.1f} min<br>"
                "Entregas: %{customdata[0]:,}<br>"
                "Tardías: %{customdata[1]:,}<extra></extra>"
            ),
            customdata=rutas[["entregas", "tardias"]],
        )
        mostrar_fig(base_fig(fig, 500, False))

    veh = (
        df_filtrado.groupby(["id_vehiculo", "numero_economico"], as_index=False)
        .agg(
            entregas=("id_entrega", "count"),
            retraso=("minutos_retraso", "mean"),
            tardias=("entrega_tardia", "sum"),
        )
        .sort_values("retraso", ascending=False)
        .head(10)
        .sort_values("retraso")
    )

    with col2:
        fig = px.bar(
            veh,
            x="retraso",
            y="numero_economico",
            orientation="h",
            title="Los 10 vehículos con mayor retraso promedio",
            labels={
                "retraso": "Retraso promedio (minutos)",
                "numero_economico": "Vehículo",
            },
            text="retraso",
        )
        fig.update_traces(
            marker_color=COLORS["naranja"],
            texttemplate="%{text:.1f} min",
            textposition="outside",
            customdata=veh[["entregas", "tardias"]],
            hovertemplate=(
                "<b>Vehículo %{y}</b><br>"
                "Retraso: %{x:.1f} min<br>"
                "Entregas: %{customdata[0]:,}<br>"
                "Tardías: %{customdata[1]:,}<extra></extra>"
            ),
        )
        mostrar_fig(base_fig(fig, 500, False))

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            df_filtrado,
            x="distancia_real_km",
            y="minutos_retraso",
            size="cantidad_paquetes",
            color="nivel_trafico",
            hover_data=[
                "fecha_salida",
                "nombre_ruta",
                "numero_economico",
                "estatus",
                "costo_total",
            ],
            title="Distancia recorrida frente al tiempo de retraso",
            labels={
                "distancia_real_km": "Distancia recorrida (km)",
                "minutos_retraso": "Retraso (minutos)",
                "nivel_trafico": "Nivel de tráfico",
                "cantidad_paquetes": "Paquetes",
            },
        )
        fig.update_traces(
            marker=dict(opacity=.6),
        )
        mostrar_fig(base_fig(fig, 460, True))

    with col2:
        fig = px.scatter(
            df_filtrado,
            x="costo_total",
            y="minutos_retraso",
            size="distancia_real_km",
            color="estatus",
            color_discrete_map=STATUS_COLORS,
            hover_data=[
                "fecha_salida",
                "nombre_ruta",
                "numero_economico",
                "distancia_real_km",
            ],
            title="Costo de una entrega frente a su retraso",
            labels={
                "costo_total": "Costo total (MXN)",
                "minutos_retraso": "Retraso (minutos)",
                "estatus": "Estado",
                "distancia_real_km": "Distancia (km)",
            },
        )
        fig.update_traces(marker=dict(opacity=.6))
        mostrar_fig(base_fig(fig, 460, True))

    st.markdown("---")
    seccion(
        "¿Qué conviene revisar?",
        "Estas acciones se desprenden directamente de los indicadores mostrados en esta sección.",
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        accion(
            "⏰ Puntualidad",
            "Revisar los vehículos y rutas que aparecen repetidamente entre los valores más altos de retraso.",
        )
    with col_b:
        accion(
            "📍 Distancia",
            "Comparar las entregas de mayor distancia con su retraso para identificar posibles relaciones operativas.",
        )
    with col_c:
        accion(
            "💰 Costo",
            "Identificar entregas de alto costo que además presenten retrasos para priorizar su análisis.",
        )

    st.markdown("---")
    seccion(
        "Detalle de entregas",
        "Pasa el cursor sobre las gráficas para consultar información adicional o utiliza la tabla para revisar cada registro.",
    )

    columnas = [
        "id_entrega", "fecha_salida", "estatus", "nombre_ruta",
        "numero_economico", "tipo_vehiculo", "distancia_real_km",
        "minutos_retraso", "entrega_tardia",
        "combustible_consumido_litros", "costo_total",
    ]

    columnas = [x for x in columnas if x in df_filtrado.columns]

    st.dataframe(
        df_filtrado[columnas].sort_values(
            "minutos_retraso", ascending=False
        ),
        width='stretch',
        height=500,
        hide_index=True,
    )


# ============================================================
# VEHÍCULOS
# ============================================================

elif seccion_actual == "🚚 Vehículos":

    seccion(
        "Vehículos",
        "Comparación del desempeño de la flota según entregas, costos, combustible y puntualidad.",
    )

    resumen = (
        df_filtrado.groupby(
            [
                "id_vehiculo",
                "numero_economico",
                "marca",
                "modelo",
                "nivel_riesgo",
            ],
            as_index=False,
        )
        .agg(
            entregas=("id_entrega", "count"),
            entregas_tardias=("entrega_tardia", "sum"),
            retraso_promedio=("minutos_retraso", "mean"),
            distancia_km=("distancia_real_km", "sum"),
            combustible_l=("combustible_consumido_litros", "sum"),
            costo_total=("costo_total", "sum"),
        )
    )

    if resumen.empty:
        aviso("No existen vehículos para los filtros actuales.", "warning")
        st.stop()

    resumen["porcentaje_tardias"] = np.where(
        resumen["entregas"] > 0,
        resumen["entregas_tardias"] / resumen["entregas"] * 100,
        0,
    )

    c = st.columns(4)
    with c[0]:
        kpi("🚚", "Vehículos", numero(len(resumen)))
    with c[1]:
        kpi("💰", "Costo acumulado", moneda(resumen["costo_total"].sum()))
    with c[2]:
        kpi("⏰", "Tasa tardía promedio", pct(resumen["porcentaje_tardias"].mean()))
    with c[3]:
        kpi("⛽", "Combustible", f"{resumen['combustible_l'].sum():,.1f} L")

    st.markdown("")

    col1, col2 = st.columns(2)

    top_costo = (
        resumen.sort_values("costo_total", ascending=False)
        .head(15)
        .sort_values("costo_total")
    )

    with col1:
        fig = px.bar(
            top_costo,
            x="costo_total",
            y="numero_economico",
            orientation="h",
            text="costo_total",
            title="Vehículos con mayor costo logístico acumulado",
            labels={
                "costo_total": "Costo acumulado (MXN)",
                "numero_economico": "Número económico",
            },
        )
        fig.update_traces(
            marker_color=COLORS["azul"],
            texttemplate="$%{text:,.0f}",
            textposition="outside",
            hovertemplate=(
                "<b>Vehículo %{y}</b><br>"
                "Costo: $ MXN %{x:,.2f}<br>"
                "Entregas: %{customdata[0]:,}<br>"
                "Retraso promedio: %{customdata[1]:.1f} min<extra></extra>"
            ),
            customdata=top_costo[["entregas", "retraso_promedio"]],
        )
        mostrar_fig(base_fig(fig, 520, False))

    top_tardias = (
        resumen.sort_values("porcentaje_tardias", ascending=False)
        .head(15)
        .sort_values("porcentaje_tardias")
    )

    with col2:
        fig = px.bar(
            top_tardias,
            x="porcentaje_tardias",
            y="numero_economico",
            orientation="h",
            text="porcentaje_tardias",
            title="Vehículos con mayor proporción de entregas tardías",
            labels={
                "porcentaje_tardias": "Entregas tardías (%)",
                "numero_economico": "Número económico",
            },
        )
        fig.update_traces(
            marker_color=COLORS["rojo"],
            texttemplate="%{text:.1f}%",
            textposition="outside",
            hovertemplate=(
                "<b>Vehículo %{y}</b><br>"
                "Tardías: %{x:.1f}%<br>"
                "Entregas: %{customdata[0]:,}<br>"
                "Tardías: %{customdata[1]:,}<extra></extra>"
            ),
            customdata=top_tardias[["entregas", "entregas_tardias"]],
        )
        mostrar_fig(base_fig(fig, 520, False))

    st.markdown("---")

    col1, col2 = st.columns(2)

    consumo = resumen.sort_values("combustible_l", ascending=False).head(15)

    with col1:
        fig = px.bar(
            consumo,
            x="numero_economico",
            y="combustible_l",
            text="combustible_l",
            title="Vehículos con mayor consumo de combustible",
            labels={
                "numero_economico": "Número económico",
                "combustible_l": "Combustible consumido (L)",
            },
        )
        fig.update_traces(
            marker_color=COLORS["amarillo"],
            texttemplate="%{text:,.0f} L",
            textposition="outside",
            hovertemplate="<b>Vehículo %{x}</b><br>Consumo: %{y:,.1f} L<extra></extra>",
        )
        mostrar_fig(base_fig(fig, 450, False))

    retraso = resumen.sort_values(
        "retraso_promedio", ascending=False
    ).head(15)

    with col2:
        fig = px.bar(
            retraso,
            x="numero_economico",
            y="retraso_promedio",
            text="retraso_promedio",
            title="Vehículos con mayor retraso promedio",
            labels={
                "numero_economico": "Número económico",
                "retraso_promedio": "Retraso promedio (minutos)",
            },
        )
        fig.update_traces(
            marker_color=COLORS["naranja"],
            texttemplate="%{text:.1f} min",
            textposition="outside",
            hovertemplate="<b>Vehículo %{x}</b><br>Retraso: %{y:.1f} min<extra></extra>",
        )
        mostrar_fig(base_fig(fig, 450, False))

    st.markdown("---")
    seccion(
        "Decisiones sugeridas",
        "La comparación permite priorizar revisiones sin depender únicamente del costo.",
    )

    mejor_tardias = resumen.sort_values("porcentaje_tardias", ascending=False).head(1)
    mejor_costo = resumen.sort_values("costo_total", ascending=False).head(1)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        accion(
            "⏰ Priorizar puntualidad",
            "Dar seguimiento a los vehículos con mayor proporción de entregas tardías y verificar si el problema es recurrente.",
        )
    with col_b:
        accion(
            "💰 Revisar gasto acumulado",
            "Los vehículos con mayor costo logístico deben compararse contra su volumen de entregas para evitar conclusiones basadas solo en el monto.",
        )
    with col_c:
        accion(
            "⛽ Vigilar consumo",
            "Comparar consumo, distancia y volumen de trabajo antes de decidir acciones de eficiencia.",
        )

    st.markdown("---")
    seccion("Ranking de la flota", "Ordena y compara rápidamente el desempeño de cada vehículo.")

    st.dataframe(
        resumen.sort_values("costo_total", ascending=False),
        width='stretch',
        height=480,
        hide_index=True,
    )


# ============================================================
# MANTENIMIENTO
# ============================================================

elif seccion_actual == "🔧 Mantenimiento":

    seccion(
        "Mantenimiento",
        "Costos, frecuencia de intervenciones, fallas y tiempo que los vehículos permanecieron fuera de servicio.",
    )

    if mantenimiento.empty:
        aviso("No existen registros de mantenimiento.", "warning")
        st.stop()

    total_mant = len(mantenimiento)
    costo_mant = mantenimiento["costo_total"].sum()
    horas = mantenimiento["horas_fuera_servicio"].sum()
    veh_afectados = mantenimiento["id_vehiculo"].nunique()

    c = st.columns(4)
    with c[0]:
        kpi("🔧", "Mantenimientos", numero(total_mant))
    with c[1]:
        kpi("💰", "Costo total", moneda(costo_mant))
    with c[2]:
        kpi("🚧", "Horas fuera de servicio", numero(horas))
    with c[3]:
        kpi("🚚", "Vehículos afectados", numero(veh_afectados))

    st.markdown("")

    col1, col2 = st.columns(2)

    tipo = (
        mantenimiento.groupby("tipo_mantenimiento")
        .size()
        .reset_index(name="cantidad")
    )

    with col1:
        fig = px.pie(
            tipo,
            names="tipo_mantenimiento",
            values="cantidad",
            hole=.58,
            title="¿Qué tipo de mantenimiento se realiza con mayor frecuencia?",
            color="tipo_mantenimiento",
            color_discrete_map=MAINT_COLORS,
        )
        fig.update_traces(
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>Registros: %{value:,}<br>Participación: %{percent}<extra></extra>",
        )
        mostrar_fig(base_fig(fig, 400, True))

    costos_tipo = (
        mantenimiento.groupby("tipo_mantenimiento", as_index=False)
        .agg(costo=("costo_total", "sum"))
        .sort_values("costo", ascending=False)
    )

    with col2:
        fig = px.bar(
            costos_tipo,
            x="tipo_mantenimiento",
            y="costo",
            text="costo",
            title="¿En qué tipo de mantenimiento se concentra el gasto?",
            labels={
                "tipo_mantenimiento": "Tipo de mantenimiento",
                "costo": "Costo total (MXN)",
            },
            color="tipo_mantenimiento",
            color_discrete_map=MAINT_COLORS,
        )
        fig.update_traces(
            texttemplate="$%{text:,.0f}",
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Costo: $ MXN %{y:,.2f}<extra></extra>",
        )
        mostrar_fig(base_fig(fig, 400, False))

    st.markdown("---")

    col1, col2 = st.columns(2)

    componentes = (
        mantenimiento.groupby("componente")
        .agg(
            fallas=("id_mantenimiento", "count"),
            costo=("costo_total", "sum"),
        )
        .reset_index()
        .sort_values("fallas", ascending=False)
        .head(12)
        .sort_values("fallas")
    )

    with col1:
        fig = px.bar(
            componentes,
            x="fallas",
            y="componente",
            orientation="h",
            text="fallas",
            title="Componentes que presentan más incidencias",
            labels={
                "fallas": "Número de incidencias",
                "componente": "Componente",
            },
        )
        fig.update_traces(
            marker_color=COLORS["rojo"],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Incidencias: %{x:,}<extra></extra>",
        )
        mostrar_fig(base_fig(fig, 500, False))

    componentes_costo = (
        mantenimiento.groupby("componente")
        .agg(costo=("costo_total", "sum"))
        .reset_index()
        .sort_values("costo", ascending=False)
        .head(12)
        .sort_values("costo")
    )

    with col2:
        fig = px.bar(
            componentes_costo,
            x="costo",
            y="componente",
            orientation="h",
            text="costo",
            title="Componentes que concentran mayor gasto",
            labels={
                "costo": "Costo acumulado (MXN)",
                "componente": "Componente",
            },
        )
        fig.update_traces(
            marker_color=COLORS["morado"],
            texttemplate="$%{text:,.0f}",
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Costo: $ MXN %{x:,.2f}<extra></extra>",
        )
        mostrar_fig(base_fig(fig, 500, False))

    st.markdown("---")

    causas = (
        mantenimiento.groupby("causa_falla")
        .agg(
            cantidad=("id_mantenimiento", "count"),
            costo=("costo_total", "sum"),
            horas=("horas_fuera_servicio", "sum"),
        )
        .reset_index()
        .sort_values("cantidad", ascending=False)
        .head(12)
        .sort_values("cantidad")
    )

    fig = px.bar(
        causas,
        x="cantidad",
        y="causa_falla",
        orientation="h",
        text="cantidad",
        title="¿Cuáles son las principales causas de las incidencias?",
        labels={
            "cantidad": "Número de registros",
            "causa_falla": "Causa",
        },
        hover_data=["costo", "horas"],
    )
    fig.update_traces(
        marker_color=COLORS["naranja"],
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Registros: %{x:,}<br>"
            "Costo: $ MXN %{customdata[0]:,.2f}<br>"
            "Horas fuera de servicio: %{customdata[1]:,.1f}<extra></extra>"
        ),
    )
    mostrar_fig(base_fig(fig, 540, False))

    st.markdown("---")

    col1, col2 = st.columns(2)

    sev = (
        mantenimiento.groupby("severidad")
        .size()
        .reset_index(name="cantidad")
    )

    with col1:
        fig = px.pie(
            sev,
            names="severidad",
            values="cantidad",
            hole=.58,
            title="¿Qué tan graves son las incidencias registradas?",
        )
        fig.update_traces(
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>Registros: %{value:,}<br>Participación: %{percent}<extra></extra>",
        )
        mostrar_fig(base_fig(fig, 400, True))

    sev_costo = (
        mantenimiento.groupby("severidad", as_index=False)
        .agg(costo=("costo_total", "sum"))
        .sort_values("costo", ascending=False)
    )

    with col2:
        fig = px.bar(
            sev_costo,
            x="severidad",
            y="costo",
            text="costo",
            title="Costo acumulado según la gravedad de la incidencia",
            labels={
                "severidad": "Nivel de gravedad",
                "costo": "Costo total (MXN)",
            },
        )
        fig.update_traces(
            texttemplate="$%{text:,.0f}",
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Costo: $ MXN %{y:,.2f}<extra></extra>",
        )
        mostrar_fig(base_fig(fig, 400, False))

    st.markdown("---")
    seccion(
        "Recomendaciones de mantenimiento",
        "La información permite priorizar intervenciones por frecuencia, costo y gravedad.",
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        accion(
            "🔩 Atender componentes frecuentes",
            "Revisar los componentes con más incidencias para identificar oportunidades de mantenimiento preventivo.",
        )
    with col_b:
        accion(
            "💰 Controlar los mayores costos",
            "Analizar los componentes y tipos de mantenimiento que concentran mayor gasto acumulado.",
        )
    with col_c:
        accion(
            "🚨 Priorizar incidencias graves",
            "Las intervenciones de mayor severidad deben recibir seguimiento y documentación de causa.",
        )

    st.markdown("---")
    seccion("Historial de mantenimiento", "Consulta detallada de las intervenciones registradas.")

    columnas = [
        "id_mantenimiento", "fecha_mantenimiento", "numero_economico",
        "componente", "categoria_componente", "tipo_mantenimiento",
        "causa_falla", "severidad", "costo_repuesto",
        "costo_mano_obra", "costo_total", "horas_fuera_servicio",
    ]
    columnas = [x for x in columnas if x in mantenimiento.columns]

    st.dataframe(
        mantenimiento[columnas].sort_values(
            "costo_total", ascending=False
        ),
        width='stretch',
        height=520,
        hide_index=True,
    )


# ============================================================
# RIESGOS Y ALERTAS
# ============================================================

elif seccion_actual == "⚠️ Riesgos y alertas":

    seccion(
        "Riesgos y alertas",
        "Separa el nivel de riesgo registrado en el Data Warehouse de una prioridad complementaria basada en historial de mantenimiento.",
    )
    if riesgos.empty:
        aviso("No existen datos de riesgo.", "warning")
        st.stop()

    rp=_calcular_prioridad_mantenimiento(riesgos)
    nivel=rp["nivel_riesgo"].astype(str).str.upper()
    criticos=nivel.isin(["CRÍTICO","CRITICO"]).sum()
    altos=(nivel=="ALTO").sum()

    c=st.columns(4)
    with c[0]: kpi("🚨","Riesgo crítico (DW)",numero(criticos),"Etiqueta original")
    with c[1]: kpi("⚠️","Riesgo alto (DW)",numero(altos),"Etiqueta original")
    with c[2]: kpi("🔧","Fallas críticas",numero(rp["fallas_criticas"].sum()))
    with c[3]: kpi("🚨","Emergencias",numero(rp["emergencias"].sum()))

    aviso(
        "El <b>nivel_riesgo</b> es el valor original de la base. La <b>prioridad de mantenimiento</b> "
        "se calcula con 40% fallas críticas + 30% emergencias + 20% costo + 10% horas fuera de servicio. "
        "Por eso un vehículo MEDIO en la base puede tener prioridad ALTA si su historial es más severo.",
        "info",
    )

    dist=rp.groupby("nivel_riesgo").size().reset_index(name="vehiculos")
    dist["nivel"]=dist["nivel_riesgo"].astype(str).str.upper()
    dist["porcentaje"]=dist["vehiculos"]/dist["vehiculos"].sum()*100
    fig=px.bar(
        dist,x="nivel_riesgo",y="vehiculos",text="vehiculos",color="nivel",color_discrete_map=RISK_COLORS,
        title="¿Cómo está distribuido el riesgo registrado en la flota?",
        labels={"nivel_riesgo":"Nivel de riesgo registrado","vehiculos":"Número de vehículos","nivel":"Nivel"}
    )
    fig.update_traces(textposition="outside",customdata=dist["porcentaje"],
                      hovertemplate="<b>%{x}</b><br>Vehículos: %{y:,}<br>Participación: %{customdata:.1f}%<extra></extra>")
    mostrar_fig(
        base_fig(fig,420,True),"Distribución del riesgo registrado",datos=dist,
        pregunta="¿Cómo está distribuido el riesgo dentro de la flota?",
        interpretacion_texto="Esta gráfica muestra únicamente la etiqueta nivel_riesgo almacenada en el Data Warehouse.",
        recomendacion="Usar la prioridad de mantenimiento para ordenar la atención y el nivel_riesgo para conservar trazabilidad con la fuente."
    )

    st.markdown("---")
    seccion("Vehículos que requieren atención","Ordenados por prioridad calculada con severidad, emergencias, costo y horas fuera de servicio.")
    prioritarios=rp.sort_values(["prioridad_score","fallas_criticas","emergencias","costo_mantenimiento"],ascending=False).copy()
    st.dataframe(
        prioritarios.rename(columns={"numero_economico":"Vehículo","nivel_riesgo":"Riesgo DW","prioridad_nivel":"Prioridad mantenimiento","prioridad_score":"Prioridad (0-100)"}),
        width="stretch",height=430,hide_index=True
    )

    top=prioritarios.head(15).sort_values("prioridad_score").copy()
    fig=px.bar(
        top,x="prioridad_score",y="numero_economico",orientation="h",
        color="prioridad_nivel",
        color_discrete_map={"CRÍTICA":COLORS["rojo"],"ALTA":COLORS["naranja"],"MEDIA":COLORS["amarillo"],"BAJA":COLORS["verde"]},
        text="prioridad_score",
        title="Prioridad de mantenimiento por vehículo",
        labels={"prioridad_score":"Prioridad (0–100)","numero_economico":"Número económico","prioridad_nivel":"Prioridad"}
    )
    fig.update_traces(
        texttemplate="%{text:.0f}",textposition="outside",
        customdata=top[["nivel_riesgo","fallas_criticas","emergencias","costo_mantenimiento","horas_fuera_servicio"]],
        hovertemplate="<b>Vehículo %{y}</b><br>Prioridad: %{x:.0f}/100<br>Riesgo DW: %{customdata[0]}<br>Fallas críticas: %{customdata[1]:,}<br>Emergencias: %{customdata[2]:,}<br>Costo: $ MXN %{customdata[3]:,.2f}<br>Horas fuera: %{customdata[4]:,.1f}<extra></extra>"
    )
    mostrar_fig(
        base_fig(fig,540,True),"Prioridad de mantenimiento por vehículo",datos=top,
        pregunta="¿Qué vehículos requieren mantenimiento?",
        interpretacion_texto=f"{top.iloc[-1]['numero_economico']} es el caso de mayor prioridad dentro de los 15 mostrados ({top.iloc[-1]['prioridad_score']:.0f}/100).",
        recomendacion="Atender primero prioridades críticas y altas y verificar físicamente la unidad antes de asignarla a una ruta exigente."
    )

    st.markdown("---")
    seccion("Plan de atención","La prioridad calculada ordena el trabajo; el nivel de riesgo del DW conserva la referencia original.")
    col_a,col_b,col_c=st.columns(3)
    with col_a: accion("🔴 Prioridad crítica","Programar atención prioritaria y revisar fallas críticas, emergencias y antecedentes.")
    with col_b: accion("🟠 Prioridad alta","Programar inspección y seguimiento antes de que aumente la recurrencia.")
    with col_c: accion("🟢 Prioridad media o baja","Mantener monitoreo preventivo y revisar tendencias.")

    alertas_path=MANT_DIR/"alertas_mantenimiento.csv"
    if alertas_path.exists():
        st.markdown("---")
        seccion("Alertas generadas por mantenimiento","Avisos provenientes del análisis almacenado en el proyecto.")
        try:
            alertas=pd.read_csv(alertas_path)
            st.dataframe(alertas,width="stretch",height=420,hide_index=True)
        except Exception as exc:
            aviso(f"No se pudo leer el archivo de alertas: {exc}","warning")


# ============================================================
# ANÁLISIS AVANZADO
# ============================================================

elif seccion_actual == "🤖 Análisis avanzado":

    seccion(
        "Análisis avanzado",
        "Resultados analíticos del proyecto presentados con nombres comprensibles para consulta ejecutiva.",
    )

    aviso(
        "<b>Cómo leer 0 y 1:</b> cuando una variable binaria como <b>entrega_tardia</b> aparece en una gráfica, "
        "<b>0 = entrega no tardía / cumplida</b> y <b>1 = entrega tardía</b>. No son niveles de riesgo ni porcentajes.",
        "info",
    )
    tab1, tab2, tab3 = st.tabs(
        ["👥 Perfiles de operación", "📈 Factores principales", "🤖 Predicciones"]
    )

    # --------------------------------------------------------
    # PERFILES
    # --------------------------------------------------------
    with tab1:
        aviso(
            "Esta vista agrupa entregas con características similares para facilitar la identificación de patrones de comportamiento.",
            "info",
        )
        perfil = KMEANS_DIR / "perfil_clusters.csv"
        clusters = KMEANS_DIR / "entregas_clusters.csv"

        if perfil.exists():
            df = pd.read_csv(perfil)

            seccion(
                "Perfiles de comportamiento",
                "Agrupa entregas con características parecidas para facilitar la identificación de patrones operativos.",
            )

            st.dataframe(df, width='stretch', hide_index=True)

            if "cluster" in df.columns:
                pct_col = next(
                    (
                        c for c in df.columns
                        if c.lower() in [
                            "porcentaje_entregas",
                            "porcentaje",
                            "pct_entregas",
                        ]
                    ),
                    None,
                )

                if pct_col:
                    valores = pd.to_numeric(
                        df[pct_col], errors="coerce"
                    ).fillna(0)

                    graf = pd.DataFrame({
                        "Perfil": df["cluster"].astype(str),
                        "Participación (%)": valores,
                    })

                    fig = px.bar(
                        graf,
                        x="Perfil",
                        y="Participación (%)",
                        text="Participación (%)",
                        title="¿Qué proporción de las entregas pertenece a cada perfil?",
                        labels={
                            "Perfil": "Perfil de operación",
                            "Participación (%)": "Participación (%)",
                        },
                    )
                    fig.update_traces(
                        marker_color=COLORS["morado"],
                        texttemplate="%{text:.1f}%",
                        textposition="outside",
                        hovertemplate="<b>%{x}</b><br>Participación: %{y:.1f}%<extra></extra>",
                    )
                    mostrar_fig(base_fig(fig, 420, False))
        else:
            aviso(
                "Todavía no se encontró el archivo de perfiles de operación.",
                "warning",
            )

        if clusters.exists():
            with st.expander("Ver registros agrupados"):
                st.dataframe(
                    pd.read_csv(clusters).head(500),
                    width='stretch',
                    height=450,
                    hide_index=True,
                )

    # --------------------------------------------------------
    # FACTORES
    # --------------------------------------------------------
    with tab2:
        aviso(
            "Esta vista resume qué factores concentran una mayor parte de la variación observada en los datos.",
            "info",
        )
        varianza = PCA_DIR / "varianza_pca.csv"
        cargas = PCA_DIR / "cargas_componentes.csv"
        interpretacion = PCA_DIR / "interpretacion_componentes.csv"

        if varianza.exists():
            dfv = pd.read_csv(varianza)

            seccion(
                "Factores que explican el comportamiento",
                "Muestra qué proporción de la variación observada puede resumirse en cada factor principal.",
            )

            comp_col = next(
                (
                    c for c in dfv.columns
                    if "componente" in c.lower()
                ),
                dfv.columns[0],
            )

            pct_col = next(
                (
                    c for c in dfv.columns
                    if "porcentaje" in c.lower()
                    or "varianza" in c.lower()
                ),
                None,
            )

            if pct_col:
                valores = pd.to_numeric(
                    dfv[pct_col], errors="coerce"
                ).fillna(0)

                graf = pd.DataFrame({
                    "Factor": dfv[comp_col].astype(str),
                    "Variación explicada (%)": valores,
                })

                fig = px.bar(
                    graf,
                    x="Factor",
                    y="Variación explicada (%)",
                    text="Variación explicada (%)",
                    title="¿Qué factores explican una mayor parte de la información?",
                    labels={
                        "Factor": "Factor",
                        "Variación explicada (%)": "Variación explicada (%)",
                    },
                )
                fig.update_traces(
                    marker_color=COLORS["turquesa"],
                    texttemplate="%{text:.2f}%",
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Variación explicada: %{y:.2f}%<extra></extra>",
                )
                mostrar_fig(base_fig(fig, 430, False))

            st.dataframe(dfv, width='stretch', hide_index=True)

        if cargas.exists():
            with st.expander("Ver relación de variables con los factores"):
                st.dataframe(
                    pd.read_csv(cargas),
                    width='stretch',
                    hide_index=True,
                )

        if interpretacion.exists():
            with st.expander("Ver interpretación detallada"):
                st.dataframe(
                    pd.read_csv(interpretacion),
                    width='stretch',
                    hide_index=True,
                )

    # --------------------------------------------------------
    # MODELOS
    # --------------------------------------------------------
    with tab3:
        aviso(
            "Aquí se presentan los modelos entrenados y sus reportes. Los resultados pueden utilizarse como apoyo a la toma de decisiones, no como sustituto del criterio operativo.",
            "info",
        )
        seccion(
            "Predicciones y apoyo a decisiones",
            "Archivos de modelos y reportes de evaluación disponibles en el proyecto.",
        )

        if MODELOS_DIR.exists():
            archivos = sorted(
                [p.name for p in MODELOS_DIR.iterdir() if p.is_file()]
            )
            if archivos:
                for nombre in archivos:
                    st.write(f"✓ {nombre}")
            else:
                aviso("La carpeta existe pero no contiene modelos.", "info")
        else:
            aviso("No existe todavía la carpeta de modelos entrenados.", "warning")

        st.markdown("---")

        if REPORTES_MODELOS_DIR.exists():
            reportes = sorted(
                list(REPORTES_MODELOS_DIR.glob("*.csv"))
                + list(REPORTES_MODELOS_DIR.glob("*.json"))
            )

            if reportes:
                for archivo in reportes:
                    with st.expander(f"📄 {archivo.name}"):
                        try:
                            if archivo.suffix.lower() == ".csv":
                                st.dataframe(
                                    pd.read_csv(archivo),
                                    width='stretch',
                                    hide_index=True,
                                )
                            else:
                                with open(
                                    archivo,
                                    "r",
                                    encoding="utf-8"
                                ) as f:
                                    st.json(json.load(f))
                        except Exception as exc:
                            aviso(
                                f"No se pudo leer {archivo.name}: {exc}",
                                "warning",
                            )
            else:
                aviso("No hay reportes CSV o JSON disponibles.", "info")



# ============================================================
# MÓDULOS OPERATIVOS ADICIONALES
# ============================================================

elif seccion_actual in [
    "👥 Clientes","👨‍✈️ Operadores","🛣️ Rutas","⛽ Combustible",
    "📌 Asignaciones","🗺️ Mapa operativo","📊 Reportes"
]:
    if render_modulo is None:
        aviso("No se pudo cargar modulos_operativos_siglog.py.", "danger")
    else:
        df_modulo = _filtro_contextual_modulo(df_filtrado, seccion_actual)
        if df_modulo.empty:
            aviso(
                "No hay registros para el enfoque seleccionado dentro del periodo y filtros globales activos. "
                "Quita el filtro contextual para volver a ver todo el módulo.",
                "warning",
            )
        else:
            render_modulo(
                seccion_actual, df_modulo, consulta, BASE_DIR,
                mantenimiento=mantenimiento, vehiculos=vehiculos,
                riesgos=riesgos, periodo=periodo_actual,
            )

elif seccion_actual == "🔎 Preguntas de negocio":
    modulo_preguntas_negocio(df_filtrado, mantenimiento, riesgos)

elif seccion_actual == "📈 Patrones de operación":
    modulo_patrones(df_filtrado)

# ============================================================
# DATOS
# ============================================================

elif seccion_actual == "📋 Datos":

    seccion(
        "Exploración de datos",
        "Consulta directa de los registros utilizados por el dashboard.",
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📦 Entregas", "🔧 Mantenimiento", "🚚 Vehículos", "⚠️ Riesgos"]
    )

    with tab1:
        st.write(f"Registros visibles: **{len(df_filtrado):,}**")
        st.dataframe(
            df_filtrado,
            width='stretch',
            height=600,
            hide_index=True,
        )

        csv = df_filtrado.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar entregas filtradas",
            csv,
            "entregas_filtradas_siglog.csv",
            "text/csv",
            key=_next_key("entregas_filtradas_csv"),
        )

    with tab2:
        st.write(f"Registros de mantenimiento: **{len(mantenimiento):,}**")
        st.dataframe(
            mantenimiento,
            width='stretch',
            height=600,
            hide_index=True,
        )

        csv = mantenimiento.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar mantenimiento",
            csv,
            "mantenimiento_siglog.csv",
            "text/csv",
            key=_next_key("mantenimiento_csv"),
        )

    with tab3:
        st.write(f"Vehículos: **{len(vehiculos):,}**")
        st.dataframe(
            vehiculos,
            width='stretch',
            height=600,
            hide_index=True,
        )

    with tab4:
        st.write(f"Vehículos analizados: **{len(riesgos):,}**")
        st.dataframe(
            riesgos,
            width='stretch',
            height=600,
            hide_index=True,
        )


# ============================================================
# PIE
# ============================================================

st.markdown(
    """
    <div class="notice info">
        <strong>📌 Cómo aprovechar SIG-LOG:</strong>
        empieza con el alcance global para definir qué operación quieres estudiar.
        Después usa los ajustes locales de cada gráfica para enfocar una ruta,
        vehículo, operador o métrica sin alterar las demás visualizaciones.
        Pasa el cursor para consultar valores exactos y descarga el PDF cuando
        necesites conservar la evidencia, interpretación y recomendación de una gráfica.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

st.caption(
    "SIG-LOG · Sistema Integral de Gestión Logística · "
    "Data Warehouse + Analítica + Visualización"
)

st.caption(f"Fuente de datos: {DB_PATH}")