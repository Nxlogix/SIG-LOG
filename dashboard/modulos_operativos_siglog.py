
"""
Módulos operativos adicionales para SIG-LOG.

Este archivo NO reemplaza dashboard.py.
Se integra al dashboard actual para agregar:
- Clientes
- Operadores
- Rutas
- Combustible
- Asignaciones
- Reportes PDF / Excel / CSV
- Mapa operativo

La asignación se guarda en reportes/asignaciones_operativas.csv para no modificar
el Data Warehouse histórico durante la demostración académica.
"""

from pathlib import Path
from io import BytesIO
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIGURACIÓN
# ============================================================

def _paths(base_dir):
    base_dir = Path(base_dir)
    return {
        "base": base_dir,
        "reportes": base_dir / "reportes",
        "asignaciones": base_dir / "reportes" / "asignaciones_operativas.csv",
    }


# ============================================================
# UTILIDADES VISUALES
# ============================================================

COLORS = {
    "azul": "#2563EB",
    "azul_oscuro": "#1E3A8A",
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

def _num(x, dec=0):
    try:
        return f"{float(x):,.{dec}f}"
    except Exception:
        return f"{0:,.{dec}f}"

def _money(x):
    try:
        return f"$ MXN {float(x):,.2f}"
    except Exception:
        return "$ MXN 0.00"

def _pct(x, dec=1):
    try:
        return f"{float(x):.{dec}f}%"
    except Exception:
        return f"{0:.{dec}f}%"

def _card(title, value, note="", icon="•"):
    st.markdown(
        f"""
        <div style="
            background:#FFFFFF;border:1px solid #E2E8F0;border-radius:16px;
            padding:15px 16px;box-shadow:0 7px 20px rgba(15,23,42,.045);
            min-height:105px;">
            <div style="font-size:19px">{icon}</div>
            <div style="font-size:11px;font-weight:800;color:#64748B;margin-top:4px">
                {title}
            </div>
            <div style="font-size:23px;font-weight:850;color:#0F172A;margin-top:2px">
                {value}
            </div>
            <div style="font-size:10px;color:#94A3B8;margin-top:3px">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _section(title, subtitle):
    st.markdown(
        f"""
        <div style="margin:10px 0 16px">
            <div style="font-size:26px;font-weight:850;color:#0F172A">{title}</div>
            <div style="font-size:13px;line-height:1.5;color:#64748B">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _box(title, text, kind="info"):
    border = {
        "info": "#2563EB",
        "success": "#16A34A",
        "warning": "#EA580C",
        "danger": "#DC2626",
    }.get(kind, "#2563EB")
    bg = {
        "info": "#F8FBFF",
        "success": "#F8FCF9",
        "warning": "#FFFAF5",
        "danger": "#FFF8F8",
    }.get(kind, "#F8FBFF")
    st.markdown(
        f"""
        <div style="
            background:{bg};border:1px solid #E2E8F0;border-left:4px solid {border};
            border-radius:13px;padding:12px 14px;margin:6px 0 16px;">
            <div style="font-size:12px;font-weight:850;color:#0F172A">{title}</div>
            <div style="font-size:11.5px;line-height:1.5;color:#475569;margin-top:3px">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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

def _pdf_visual(titulo, fig, interpretacion="", recomendacion=""):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import landscape, letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    except Exception:
        return None
    b=BytesIO()
    doc=SimpleDocTemplate(b,pagesize=landscape(letter),rightMargin=1*cm,leftMargin=1*cm,topMargin=1*cm,bottomMargin=1*cm)
    s=getSampleStyleSheet()
    title=ParagraphStyle("T",parent=s["Title"],fontSize=19,textColor=colors.HexColor("#1E3A8A"))
    body=ParagraphStyle("B",parent=s["BodyText"],fontSize=9,leading=12,textColor=colors.HexColor("#334155"))
    story=[Paragraph("SIG-LOG · Reporte de gráfica",title),Paragraph(str(titulo),s["Heading2"]),
           Paragraph(f"<b>Periodo:</b> {st.session_state.get('periodo_dashboard','Periodo seleccionado')}",body),
           Spacer(1,.15*cm),Paragraph("<b>Interpretación</b>",s["Heading3"]),Paragraph(str(interpretacion or "Consultar los valores destacados y validar el detalle de los registros."),body),
           Paragraph("<b>Recomendación</b>",s["Heading3"]),Paragraph(str(recomendacion or "Revisar el hallazgo con las condiciones operativas antes de actuar."),body),
           Spacer(1,.15*cm)]
    df=_fig_dataframe(fig)
    if not df.empty:
        df=df.head(30).fillna("")
        vals=[list(df.columns)]+df.astype(str).values.tolist()
        tab=Table(vals,repeatRows=1)
        tab.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#2563EB")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("GRID",(0,0),(-1,-1),.25,colors.HexColor("#DDE5EF")),("FONTSIZE",(0,0),(-1,-1),6.5),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F8FAFC")])]))
        story += [Paragraph("<b>Datos representados</b>",s["Heading3"]),tab]
    story += [Spacer(1,.15*cm),Paragraph(f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",body)]
    doc.build(story); b.seek(0); return b.getvalue()

def _fig(fig, height=420, interpretacion="", recomendacion=""):
    titulo=""
    try:
        titulo=str(fig.layout.title.text or "Análisis operativo")
    except Exception:
        titulo="Análisis operativo"
    fig.update_layout(
        height=height,paper_bgcolor="#FFFFFF",plot_bgcolor="#FFFFFF",
        margin=dict(l=60,r=24,t=22,b=55),
        font=dict(family="Inter, Arial, sans-serif",color="#0F172A"),
        hoverlabel=dict(bgcolor="#0F172A",bordercolor="#334155",font=dict(color="#FFFFFF",size=13)),
        hovermode="closest")
    fig.update_xaxes(showgrid=False,zeroline=False,linecolor="#CBD5E1",tickcolor="#CBD5E1",tickfont=dict(color="#475569",size=11),automargin=True)
    fig.update_yaxes(showgrid=True,gridcolor="#E8EDF4",zeroline=False,linecolor="#CBD5E1",tickcolor="#CBD5E1",tickfont=dict(color="#475569",size=11),automargin=True)
    st.plotly_chart(fig,width='stretch',config={"displaylogo":False,"responsive":True,"scrollZoom":True,"displayModeBar":True,"modeBarButtonsToRemove":["lasso2d","select2d"]})
    pdf=_pdf_visual(titulo,fig,interpretacion,recomendacion)
    c1,c2=st.columns(2)
    with c1:
        if pdf:
            st.download_button("📄 Descargar PDF de esta gráfica",pdf,
                f"SIGLOG_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf","application/pdf",
                width='stretch')
        else: st.caption("PDF: instala reportlab.")
    with c2:
        st.download_button("⬇️ Descargar datos CSV",_fig_dataframe(fig).to_csv(index=False).encode("utf-8-sig"),
                           f"SIGLOG_grafica_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv","text/csv",
                           width='stretch')


# ============================================================
# CARGA DE DIMENSIONES
# ============================================================

def _consulta(consulta_fn, sql):
    try:
        df = consulta_fn(sql)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def cargar_clientes(consulta_fn):
    return _consulta(
        consulta_fn,
        """
        SELECT *
        FROM dim_cliente
        """
    )

def cargar_operadores(consulta_fn):
    return _consulta(
        consulta_fn,
        """
        SELECT *
        FROM dim_operador
        """
    )

def cargar_rutas(consulta_fn):
    return _consulta(
        consulta_fn,
        """
        SELECT *
        FROM dim_ruta
        """
    )


# ============================================================
# CLIENTES
# ============================================================

def modulo_clientes(df, consulta_fn):
    _section(
        "Clientes",
        "Consulta quién concentra la demanda, qué rutas utiliza y qué nivel de cumplimiento "
        "presentan sus servicios durante el periodo seleccionado.",
    )

    base = df.copy()

    if base.empty:
        st.warning("No hay entregas disponibles para analizar clientes.")
        return

    # Intentar recuperar catálogo de clientes.
    clientes = cargar_clientes(consulta_fn)

    # Resolver el nombre con las columnas que realmente existan.
    id_col = next((c for c in ["id_cliente", "cliente_id"] if c in base.columns), None)
    nombre_col = next(
        (c for c in ["nombre_cliente", "cliente", "nombre", "razon_social"] if c in clientes.columns),
        None,
    )

    if clientes.empty or id_col is None:
        resumen = (
            base.groupby("id_cliente", as_index=False)
            .agg(
                entregas=("id_entrega", "count"),
                tardias=("entrega_tardia", "sum"),
                costo=("costo_total", "sum"),
                retraso_promedio=("minutos_retraso", "mean"),
            )
        )
        resumen["cliente"] = resumen["id_cliente"].astype(str)
    else:
        resumen = (
            base.groupby("id_cliente", as_index=False)
            .agg(
                entregas=("id_entrega", "count"),
                tardias=("entrega_tardia", "sum"),
                costo=("costo_total", "sum"),
                retraso_promedio=("minutos_retraso", "mean"),
            )
        )
        if nombre_col and "id_cliente" in clientes.columns:
            mapa = clientes[["id_cliente", nombre_col]].drop_duplicates("id_cliente")
            resumen = resumen.merge(mapa, on="id_cliente", how="left")
            resumen["cliente"] = resumen[nombre_col].fillna(resumen["id_cliente"].astype(str))
        else:
            resumen["cliente"] = resumen["id_cliente"].astype(str)

    resumen["tasa_tardia"] = np.where(
        resumen["entregas"] > 0,
        resumen["tardias"] / resumen["entregas"] * 100,
        0,
    )
    resumen = resumen.sort_values("entregas", ascending=False)

    c = st.columns(4)
    with c[0]:
        _card("Clientes atendidos", _num(len(resumen)), "Con al menos una entrega", "👥")
    with c[1]:
        _card("Servicios registrados", _num(resumen["entregas"].sum()), "Periodo seleccionado", "📦")
    with c[2]:
        _card("Clientes con retrasos", _num((resumen["tardias"] > 0).sum()), "Presentaron al menos una tardía", "⚠️")
    with c[3]:
        _card("Costo de servicios", _money(resumen["costo"].sum()), "Envío + combustible + otros", "💰")

    st.markdown("")

    top = resumen.head(12).sort_values("entregas")
    fig = px.bar(
        top,
        x="entregas",
        y="cliente",
        orientation="h",
        text="entregas",
        title="Clientes con mayor número de entregas · periodo seleccionado",
        labels={"entregas": "Número de entregas", "cliente": "Cliente"},
    )
    fig.update_traces(
        marker_color=COLORS["azul"],
        texttemplate="%{text:,}",
        textposition="outside",
        customdata=top[["tardias", "tasa_tardia", "costo"]],
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Entregas: %{x:,}<br>"
            "Entregas tardías: %{customdata[0]:,}<br>"
            "Tasa tardía: %{customdata[1]:.1f}%<br>"
            "Costo: $ MXN %{customdata[2]:,.2f}<extra></extra>"
        ),
    )
    _fig(fig, 520)

    _box(
        "💡 ¿Qué significa?",
        "Una barra más larga representa un cliente con mayor volumen de servicios. "
        "El volumen por sí solo no significa mejor desempeño: también debes revisar el porcentaje de entregas tardías.",
    )

    st.dataframe(
        resumen[
            ["cliente", "entregas", "tardias", "tasa_tardia", "retraso_promedio", "costo"]
        ].rename(
            columns={
                "cliente": "Cliente",
                "entregas": "Entregas",
                "tardias": "Tardías",
                "tasa_tardia": "Tasa tardía (%)",
                "retraso_promedio": "Retraso promedio (min)",
                "costo": "Costo total (MXN)",
            }
        ),
        width='stretch',
        hide_index=True,
    )


# ============================================================
# OPERADORES
# ============================================================

def modulo_operadores(df, consulta_fn):
    _section(
        "Operadores",
        "Mide la carga de trabajo y puntualidad de cada operador para detectar concentración "
        "de servicios y apoyar una distribución equilibrada de la operación.",
    )

    if df.empty:
        st.warning("No hay entregas disponibles para analizar operadores.")
        return

    if "id_operador" not in df.columns:
        _box(
            "⚠️ Falta el identificador del operador",
            "La tabla fact_entregas de esta versión no expone id_operador. "
            "El archivo entregas sí contiene ese campo en la estructura del proyecto, "
            "por lo que debe conservarse al reconstruir el Data Warehouse.",
            "warning",
        )
        return

    operadores = cargar_operadores(consulta_fn)

    resumen = (
        df.groupby("id_operador", as_index=False)
        .agg(
            entregas=("id_entrega", "count"),
            tardias=("entrega_tardia", "sum"),
            retraso_promedio=("minutos_retraso", "mean"),
            distancia_km=("distancia_real_km", "sum"),
            costo=("costo_total", "sum"),
        )
    )

    nombre_col = next(
        (c for c in ["nombre_operador", "nombre", "operador", "nombre_completo"] if c in operadores.columns),
        None,
    )

    if not operadores.empty and nombre_col and "id_operador" in operadores.columns:
        mapa = operadores[["id_operador", nombre_col]].drop_duplicates("id_operador")
        resumen = resumen.merge(mapa, on="id_operador", how="left")
        resumen["operador"] = resumen[nombre_col].fillna(
            "Operador " + resumen["id_operador"].astype(str)
        )
    else:
        resumen["operador"] = "Operador " + resumen["id_operador"].astype(str)

    resumen["tasa_tardia"] = np.where(
        resumen["entregas"] > 0,
        resumen["tardias"] / resumen["entregas"] * 100,
        0,
    )
    resumen = resumen.sort_values("entregas", ascending=False)

    c = st.columns(4)
    with c[0]:
        _card("Operadores activos", _num(len(resumen)), "Con servicios en el periodo", "👨‍✈️")
    with c[1]:
        _card("Entregas", _num(resumen["entregas"].sum()), "Servicios asignados", "📦")
    with c[2]:
        _card("Tasa tardía media", _pct(resumen["tasa_tardia"].mean()), "Promedio simple por operador", "⏰")
    with c[3]:
        _card("Distancia recorrida", f"{resumen['distancia_km'].sum():,.1f} km", "Km registrados", "📍")

    st.markdown("")

    top = resumen.head(12).sort_values("entregas")
    fig = px.bar(
        top,
        x="entregas",
        y="operador",
        orientation="h",
        text="entregas",
        title="Entregas realizadas por operador · periodo seleccionado",
        labels={"entregas": "Número de entregas", "operador": "Operador"},
    )
    fig.update_traces(
        marker_color=COLORS["morado"],
        texttemplate="%{text:,}",
        textposition="outside",
        customdata=top[["tardias", "tasa_tardia", "costo"]],
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Entregas: %{x:,}<br>"
            "Tardías: %{customdata[0]:,}<br>"
            "Tasa tardía: %{customdata[1]:.1f}%<br>"
            "Costo asociado: $ MXN %{customdata[2]:,.2f}<extra></extra>"
        ),
    )
    _fig(fig, 520)

    _box(
        "🎯 ¿Qué decisión apoya?",
        "Si un operador concentra demasiados servicios o presenta una tasa tardía elevada, "
        "revisa la dificultad de sus rutas y la carga asignada antes de concluir que el problema es individual.",
        "success",
    )

    st.dataframe(
        resumen[
            ["operador", "entregas", "tardias", "tasa_tardia", "retraso_promedio", "distancia_km", "costo"]
        ].rename(
            columns={
                "operador": "Operador",
                "entregas": "Entregas",
                "tardias": "Tardías",
                "tasa_tardia": "Tasa tardía (%)",
                "retraso_promedio": "Retraso promedio (min)",
                "distancia_km": "Distancia (km)",
                "costo": "Costo asociado (MXN)",
            }
        ),
        width='stretch',
        hide_index=True,
    )


# ============================================================
# RUTAS
# ============================================================

def modulo_rutas(df, consulta_fn):
    _section(
        "Rutas",
        "Identifica las rutas con mayor demanda, retraso y consumo para decidir dónde "
        "conviene revisar horarios, capacidad, tráfico o asignación de unidades.",
    )

    if df.empty:
        st.warning("No hay entregas disponibles para analizar rutas.")
        return

    resumen = (
        df.groupby(["id_ruta", "nombre_ruta"], as_index=False)
        .agg(
            entregas=("id_entrega", "count"),
            tardias=("entrega_tardia", "sum"),
            retraso_promedio=("minutos_retraso", "mean"),
            distancia_promedio=("distancia_real_km", "mean"),
            combustible=("combustible_consumido_litros", "sum"),
            costo=("costo_total", "sum"),
        )
    )

    resumen["tasa_tardia"] = np.where(
        resumen["entregas"] > 0,
        resumen["tardias"] / resumen["entregas"] * 100,
        0,
    )

    c = st.columns(4)
    with c[0]:
        _card("Rutas utilizadas", _num(len(resumen)), "Con entregas en el periodo", "🛣️")
    with c[1]:
        r = resumen.sort_values("entregas", ascending=False).iloc[0]
        _card("Ruta más utilizada", str(r["nombre_ruta"]), f"{int(r['entregas']):,} entregas", "🏆")
    with c[2]:
        r = resumen.sort_values("tasa_tardia", ascending=False).iloc[0]
        _card("Mayor tasa tardía", _pct(r["tasa_tardia"]), str(r["nombre_ruta"]), "⚠️")
    with c[3]:
        _card("Costo de rutas", _money(resumen["costo"].sum()), "Costo total registrado", "💰")

    st.markdown("")

    a, b = st.columns(2)

    with a:
        top = resumen.sort_values("entregas", ascending=False).head(12).sort_values("entregas")
        fig = px.bar(
            top,
            x="entregas",
            y="nombre_ruta",
            orientation="h",
            text="entregas",
            title="Rutas con mayor número de entregas · periodo seleccionado",
            labels={"entregas": "Número de entregas", "nombre_ruta": "Ruta"},
        )
        fig.update_traces(
            marker_color=COLORS["azul"],
            texttemplate="%{text:,}",
            textposition="outside",
            customdata=top[["tardias", "tasa_tardia", "retraso_promedio"]],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Entregas: %{x:,}<br>"
                "Tardías: %{customdata[0]:,}<br>"
                "Tasa tardía: %{customdata[1]:.1f}%<br>"
                "Retraso promedio: %{customdata[2]:.1f} min<extra></extra>"
            ),
        )
        _fig(fig, 520)

    with b:
        top = resumen.sort_values("tasa_tardia", ascending=False).head(12).sort_values("tasa_tardia")
        fig = px.bar(
            top,
            x="tasa_tardia",
            y="nombre_ruta",
            orientation="h",
            text="tasa_tardia",
            title="Rutas con mayor porcentaje de entregas tardías · periodo seleccionado",
            labels={"tasa_tardia": "Entregas tardías (%)", "nombre_ruta": "Ruta"},
        )
        fig.update_traces(
            marker_color=COLORS["rojo"],
            texttemplate="%{text:.1f}%",
            textposition="outside",
            customdata=top[["entregas", "tardias", "retraso_promedio"]],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Tasa tardía: %{x:.1f}%<br>"
                "Entregas: %{customdata[0]:,}<br>"
                "Tardías: %{customdata[1]:,}<br>"
                "Retraso promedio: %{customdata[2]:.1f} min<extra></extra>"
            ),
        )
        _fig(fig, 520)

    _box(
        "💡 Regla para no equivocarse",
        "No elijas una ruta solo porque tiene más retrasos absolutos. "
        "Compara también cuántas entregas realizó: una ruta con 50 tardías de 1,000 servicios "
        "puede ser menos problemática que otra con 20 tardías de 80 servicios.",
    )

    st.dataframe(
        resumen.sort_values(["tasa_tardia", "entregas"], ascending=[False, False]).rename(
            columns={
                "nombre_ruta": "Ruta",
                "entregas": "Entregas",
                "tardias": "Tardías",
                "tasa_tardia": "Tasa tardía (%)",
                "retraso_promedio": "Retraso promedio (min)",
                "distancia_promedio": "Distancia promedio (km)",
                "combustible": "Combustible (L)",
                "costo": "Costo (MXN)",
            }
        ),
        width='stretch',
        hide_index=True,
    )


# ============================================================
# COMBUSTIBLE
# ============================================================

def modulo_combustible(df):
    _section(
        "Combustible",
        "Compara litros consumidos, distancia y rendimiento para detectar unidades "
        "con consumo elevado sin confundir mayor trabajo con ineficiencia.",
    )

    if df.empty:
        st.warning("No hay entregas disponibles para analizar combustible.")
        return

    resumen = (
        df.groupby(["id_vehiculo", "numero_economico"], as_index=False)
        .agg(
            litros=("combustible_consumido_litros", "sum"),
            km=("distancia_real_km", "sum"),
            entregas=("id_entrega", "count"),
            costo_combustible=("costo_combustible", "sum"),
        )
    )

    resumen["km_l"] = np.where(resumen["litros"] > 0, resumen["km"] / resumen["litros"], 0)
    resumen["litros_100km"] = np.where(
        resumen["km"] > 0, resumen["litros"] / resumen["km"] * 100, 0
    )

    c = st.columns(4)
    with c[0]:
        _card("Combustible consumido", f"{resumen['litros'].sum():,.1f} L", "Periodo seleccionado", "⛽")
    with c[1]:
        _card("Costo de combustible", _money(resumen["costo_combustible"].sum()), "Gasto registrado", "💰")
    with c[2]:
        _card("Distancia recorrida", f"{resumen['km'].sum():,.1f} km", "Recorrido registrado", "📍")
    with c[3]:
        _card("Rendimiento global", f"{resumen['km'].sum()/resumen['litros'].sum():.2f} km/L" if resumen["litros"].sum() else "0.00 km/L", "Km por litro", "📈")

    st.markdown("")

    a, b = st.columns(2)

    with a:
        top = resumen.sort_values("litros", ascending=False).head(15)
        fig = px.bar(
            top,
            x="numero_economico",
            y="litros",
            text="litros",
            title="Vehículos con mayor consumo de combustible · periodo seleccionado",
            labels={"numero_economico": "Vehículo", "litros": "Litros consumidos (L)"},
        )
        fig.update_traces(
            marker_color=COLORS["amarillo"],
            texttemplate="%{text:,.0f} L",
            textposition="outside",
            customdata=top[["km", "km_l", "costo_combustible", "entregas"]],
            hovertemplate=(
                "<b>Vehículo %{x}</b><br>"
                "Consumo: %{y:,.1f} L<br>"
                "Distancia: %{customdata[0]:,.1f} km<br>"
                "Rendimiento: %{customdata[1]:.2f} km/L<br>"
                "Costo: $ MXN %{customdata[2]:,.2f}<br>"
                "Entregas: %{customdata[3]:,}<extra></extra>"
            ),
        )
        _fig(fig, 460)

    with b:
        plot = resumen.replace([np.inf, -np.inf], np.nan).dropna(subset=["km_l"])
        fig = px.scatter(
            plot,
            x="km",
            y="litros",
            size="entregas",
            color="km_l",
            hover_name="numero_economico",
            labels={
                "km": "Distancia recorrida (km)",
                "litros": "Combustible consumido (L)",
                "km_l": "Rendimiento (km/L)",
                "entregas": "Entregas",
            },
            title="Distancia recorrida frente al consumo · eficiencia de la flota",
        )
        fig.update_traces(
            marker=dict(opacity=.75),
            hovertemplate=(
                "<b>Vehículo %{hovertext}</b><br>"
                "Distancia: %{x:,.1f} km<br>"
                "Combustible: %{y:,.1f} L<br>"
                "Rendimiento: %{marker.color:.2f} km/L<extra></extra>"
            ),
        )
        _fig(fig, 460)

    _box(
        "🎯 ¿Qué decisión apoya?",
        "Una unidad que consume muchos litros no necesariamente es ineficiente si también recorre más kilómetros. "
        "La señal importante es un rendimiento inferior al de vehículos con uso comparable.",
        "success",
    )

    st.dataframe(
        resumen.sort_values("litros", ascending=False).rename(
            columns={
                "numero_economico": "Vehículo",
                "litros": "Combustible (L)",
                "km": "Distancia (km)",
                "km_l": "Rendimiento (km/L)",
                "litros_100km": "Litros/100 km",
                "entregas": "Entregas",
                "costo_combustible": "Costo combustible (MXN)",
            }
        ),
        width='stretch',
        hide_index=True,
    )


# ============================================================
# ASIGNACIONES
# ============================================================

def _leer_asignaciones(path):
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "id_asignacion", "fecha", "id_entrega", "ruta",
                "vehiculo", "operador", "prioridad", "estado", "observaciones"
            ]
        )
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def modulo_asignaciones(df, consulta_fn, base_dir):
    _section(
        "Asignaciones operativas",
        "Permite preparar la asignación de una entrega a una ruta, vehículo y operador. "
        "Las asignaciones se guardan en un archivo operativo separado para no alterar el histórico del Data Warehouse.",
    )

    paths = _paths(base_dir)
    paths["reportes"].mkdir(parents=True, exist_ok=True)

    asignaciones = _leer_asignaciones(paths["asignaciones"])

    if df.empty:
        st.warning("No hay entregas disponibles para crear una asignación.")
        return

    vehiculos = sorted(
        df["numero_economico"].dropna().astype(str).unique().tolist()
    ) if "numero_economico" in df.columns else []

    operadores = []
    if "id_operador" in df.columns:
        op_df = cargar_operadores(consulta_fn)
        name_col = next(
            (c for c in ["nombre_operador", "nombre", "operador", "nombre_completo"] if c in op_df.columns),
            None,
        )
        if name_col and "id_operador" in op_df.columns:
            operadores = (
                op_df["id_operador"].astype(str) + " · " + op_df[name_col].astype(str)
            ).drop_duplicates().tolist()
        else:
            operadores = sorted(df["id_operador"].dropna().astype(str).unique().tolist())

    rutas = sorted(df["nombre_ruta"].dropna().astype(str).unique().tolist())
    entregas = df["id_entrega"].dropna().astype(str).tolist()

    if not operadores:
        _box(
            "⚠️ No hay operadores disponibles",
            "La estructura actual de fact_entregas debe conservar id_operador para habilitar la asignación por operador.",
            "warning",
        )
        return

    st.markdown("### Nueva asignación")

    c1, c2, c3 = st.columns(3)
    with c1:
        entrega = st.selectbox("Entrega", entregas, key="asig_entrega")
        ruta = st.selectbox("Ruta", rutas, key="asig_ruta")
    with c2:
        vehiculo = st.selectbox("Vehículo", vehiculos, key="asig_vehiculo")
        operador = st.selectbox("Operador", operadores, key="asig_operador")
    with c3:
        prioridad = st.selectbox("Prioridad", ["Normal", "Alta", "Urgente"], key="asig_prioridad")
        estado = st.selectbox("Estado", ["Programada", "Asignada", "En tránsito", "Completada"], key="asig_estado")

    observaciones = st.text_input(
        "Observaciones de la asignación",
        placeholder="Ej. Entrega prioritaria; revisar documentación antes de salida.",
        key="asig_obs",
    )

    if st.button("✅ Guardar asignación", type="primary", width='stretch'):
        nuevo = pd.DataFrame(
            [{
                "id_asignacion": datetime.now().strftime("%Y%m%d%H%M%S"),
                "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "id_entrega": entrega,
                "ruta": ruta,
                "vehiculo": vehiculo,
                "operador": operador,
                "prioridad": prioridad,
                "estado": estado,
                "observaciones": observaciones,
            }]
        )
        asignaciones = pd.concat([asignaciones, nuevo], ignore_index=True)
        asignaciones.to_csv(paths["asignaciones"], index=False, encoding="utf-8-sig")
        st.success("Asignación guardada correctamente.")
        st.rerun()

    st.markdown("---")
    st.markdown("### Asignaciones registradas")

    if asignaciones.empty:
        st.info("Todavía no existen asignaciones operativas guardadas.")
    else:
        st.dataframe(asignaciones, width='stretch', hide_index=True)
        st.download_button(
            "⬇️ Descargar asignaciones CSV",
            asignaciones.to_csv(index=False).encode("utf-8-sig"),
            "asignaciones_operativas_siglog.csv",
            "text/csv",
            width='stretch',
        )

    _box(
        "ℹ️ Importante",
        "Esta primera versión guarda la asignación como registro operativo independiente. "
        "No modifica fact_entregas ni el histórico del Data Warehouse, evitando alterar la evidencia analítica del proyecto.",
    )


# ============================================================
# MAPA OPERATIVO
# ============================================================

def modulo_mapa(df):
    _section(
        "Mapa operativo",
        "Visualiza la concentración de rutas y puntos de operación. Si el dataset contiene "
        "latitud/longitud, se utiliza un mapa geográfico; si no, se presenta un mapa esquemático de conexiones.",
    )

    if df.empty:
        st.warning("No hay datos para construir el mapa.")
        return

    lat_col = next((c for c in ["latitud", "latitude", "lat"] if c in df.columns), None)
    lon_col = next((c for c in ["longitud", "longitude", "lon", "lng"] if c in df.columns), None)

    if lat_col and lon_col:
        mapa = (
            df.dropna(subset=[lat_col, lon_col])
            .groupby([lat_col, lon_col], as_index=False)
            .agg(
                entregas=("id_entrega", "count"),
                tardias=("entrega_tardia", "sum"),
                costo=("costo_total", "sum"),
            )
        )
        mapa["tasa_tardia"] = np.where(
            mapa["entregas"] > 0,
            mapa["tardias"] / mapa["entregas"] * 100,
            0,
        )

        fig = px.scatter_map(
            mapa,
            lat=lat_col,
            lon=lon_col,
            size="entregas",
            color="tasa_tardia",
            hover_data=["entregas", "tardias", "costo"],
            zoom=5,
            height=650,
            labels={
                "entregas": "Entregas",
                "tardias": "Tardías",
                "tasa_tardia": "Tasa tardía (%)",
                "costo": "Costo (MXN)",
            },
            title="Concentración geográfica de entregas y retrasos",
        )
        fig.update_layout(map_style="open-street-map")
        _fig(fig, 650)
        return

    # Mapa esquemático sin inventar coordenadas geográficas.
    rutas = (
        df.groupby(["origen", "destino"], as_index=False)
        .agg(
            entregas=("id_entrega", "count"),
            tardias=("entrega_tardia", "sum"),
            costo=("costo_total", "sum"),
        )
        .sort_values("entregas", ascending=False)
        .head(25)
    )

    if "origen" not in df.columns or "destino" not in df.columns:
        _box(
            "⚠️ No hay coordenadas ni origen/destino",
            "Para mostrar un mapa geográfico real se deben agregar latitud/longitud al dataset de rutas. "
            "No se generan coordenadas artificiales porque podrían representar ubicaciones incorrectas.",
            "warning",
        )
        return

    nodos = sorted(set(rutas["origen"].astype(str)) | set(rutas["destino"].astype(str)))
    pos = {n: (i % 5, -(i // 5)) for i, n in enumerate(nodos)}

    fig = go.Figure()

    for _, r in rutas.iterrows():
        o = str(r["origen"])
        d = str(r["destino"])
        x1, y1 = pos[o]
        x2, y2 = pos[d]
        tasa = (r["tardias"] / r["entregas"] * 100) if r["entregas"] else 0
        color = COLORS["rojo"] if tasa >= 20 else COLORS["amarillo"] if tasa >= 10 else COLORS["azul"]

        fig.add_trace(
            go.Scatter(
                x=[x1, x2],
                y=[y1, y2],
                mode="lines",
                line=dict(color=color, width=max(1.5, min(8, 1 + r["entregas"] / 100))),
                hovertemplate=(
                    f"<b>{o} → {d}</b><br>"
                    f"Entregas: {int(r['entregas']):,}<br>"
                    f"Tardías: {int(r['tardias']):,}<br>"
                    f"Tasa tardía: {tasa:.1f}%<br>"
                    f"Costo: $ MXN {r['costo']:,.2f}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[pos[n][0] for n in nodos],
            y=[pos[n][1] for n in nodos],
            mode="markers+text",
            text=nodos,
            textposition="top center",
            marker=dict(size=18, color=COLORS["azul_oscuro"]),
            hovertemplate="<b>%{text}</b><extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        title="Mapa operativo esquemático · conexiones con mayor actividad",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor="#F8FAFC",
    )
    _fig(fig, 620)

    _box(
        "🗺️ ¿Por qué es esquemático?",
        "El archivo actual contiene nombres de origen y destino, pero no coordenadas geográficas. "
        "Por eso se muestran las conexiones sin inventar ubicaciones. Para un mapa real basta con agregar "
        "latitud y longitud a rutas.csv o dim_ruta.",
    )


# ============================================================
# REPORTES
# ============================================================

def _excel_bytes(sheets):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            safe_name = str(name)[:31] or "Datos"
            frame.to_excel(writer, index=False, sheet_name=safe_name)
    buffer.seek(0)
    return buffer.getvalue()

def _pdf_bytes(titulo, periodo, resumen, tablas):
    """
    PDF ejecutivo sencillo y reproducible.
    Requiere reportlab.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        )
    except Exception:
        return None

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        rightMargin=1.2 * cm,
        leftMargin=1.2 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "SigTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        textColor=colors.HexColor("#1E3A8A"),
        spaceAfter=8,
    )
    sub_style = ParagraphStyle(
        "SigSub",
        parent=styles["BodyText"],
        fontSize=9,
        textColor=colors.HexColor("#64748B"),
        leading=12,
    )
    head_style = ParagraphStyle(
        "SigHead",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=8,
        spaceAfter=6,
    )

    story = [
        Paragraph("SIG-LOG · Reporte ejecutivo", title_style),
        Paragraph(f"<b>{titulo}</b>", styles["Heading2"]),
        Paragraph(f"Periodo: {periodo}", sub_style),
        Spacer(1, 0.35 * cm),
    ]

    if resumen:
        data = [["Indicador", "Valor"]]
        for k, v in resumen.items():
            data.append([str(k), str(v)])

        table = Table(data, colWidths=[7 * cm, 6 * cm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#DDE5EF")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story += [table, Spacer(1, 0.45 * cm)]

    for nombre, frame in tablas.items():
        story.append(Paragraph(str(nombre), head_style))
        if frame is None or frame.empty:
            story.append(Paragraph("Sin registros disponibles.", sub_style))
            continue

        view = frame.head(15).copy()
        cols = list(view.columns[:7])
        view = view[cols].fillna("")
        data = [cols] + view.astype(str).values.tolist()
        table = Table(data, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#DDE5EF")),
                    ("FONTSIZE", (0, 0), (-1, -1), 6.5),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story += [table, Spacer(1, 0.35 * cm)]

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def modulo_reportes(df, mantenimiento, vehiculos, riesgos, base_dir, periodo):
    _section(
        "Reportes",
        "Descarga evidencia del análisis filtrado en CSV, Excel o PDF para presentar resultados, "
        "conservar evidencia y facilitar la toma de decisiones.",
    )

    paths = _paths(base_dir)
    paths["reportes"].mkdir(parents=True, exist_ok=True)

    total = len(df)
    tardias = int(df["entrega_tardia"].sum()) if "entrega_tardia" in df.columns else 0
    tasa = (tardias / total * 100) if total else 0
    costo = df["costo_total"].sum() if "costo_total" in df.columns else 0

    c = st.columns(4)
    with c[0]:
        _card("Registros", _num(total), "Universo filtrado", "📊")
    with c[1]:
        _card("Tardías", _num(tardias), _pct(tasa), "⏰")
    with c[2]:
        _card("Costo", _money(costo), "Periodo seleccionado", "💰")
    with c[3]:
        _card("Periodo", str(periodo), "Contexto del reporte", "📅")

    st.markdown("")
    _box(
        "📥 Qué puedes descargar",
        "CSV conserva el detalle completo. Excel organiza la información en hojas. "
        "PDF genera un reporte ejecutivo con indicadores y tablas resumidas.",
    )

    resumen = {
        "Entregas": f"{total:,}",
        "Entregas tardías": f"{tardias:,}",
        "Tasa tardía": f"{tasa:.1f}%",
        "Costo logístico": _money(costo),
        "Combustible": f"{df['combustible_consumido_litros'].sum():,.1f} L" if "combustible_consumido_litros" in df.columns else "0 L",
        "Distancia": f"{df['distancia_real_km'].sum():,.1f} km" if "distancia_real_km" in df.columns else "0 km",
    }

    tablas = {
        "Entregas": df,
        "Mantenimiento": mantenimiento,
        "Vehículos": vehiculos,
        "Riesgos": riesgos,
    }

    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Descargar entregas filtradas · CSV",
        csv_bytes,
        "siglog_entregas_filtradas.csv",
        "text/csv",
        width='stretch',
    )

    excel_bytes = _excel_bytes(tablas)
    st.download_button(
        "📊 Descargar reporte completo · Excel",
        excel_bytes,
        "SIG_LOG_reporte_completo.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width='stretch',
    )

    pdf_bytes = _pdf_bytes(
        "Reporte ejecutivo de operación logística",
        str(periodo),
        resumen,
        tablas,
    )

    if pdf_bytes:
        st.download_button(
            "📄 Descargar reporte ejecutivo · PDF",
            pdf_bytes,
            "SIG_LOG_reporte_ejecutivo.pdf",
            "application/pdf",
            width='stretch',
        )
    else:
        st.info("Para habilitar PDF instala reportlab: pip install reportlab")

    st.markdown("---")
    st.markdown("### Vista previa del reporte")
    st.dataframe(
        df.head(100),
        width='stretch',
        height=420,
        hide_index=True,
    )


# ============================================================
# DISPATCHER
# ============================================================

def render_modulo(
    modulo,
    df_filtrado,
    consulta_fn,
    base_dir,
    mantenimiento=None,
    vehiculos=None,
    riesgos=None,
    periodo="Periodo seleccionado",
):
    """
    Dispatcher principal.

    modulo:
        Clientes | Operadores | Rutas | Combustible | Asignaciones |
        Reportes | Mapa operativo
    """
    mantenimiento = mantenimiento if isinstance(mantenimiento, pd.DataFrame) else pd.DataFrame()
    vehiculos = vehiculos if isinstance(vehiculos, pd.DataFrame) else pd.DataFrame()
    riesgos = riesgos if isinstance(riesgos, pd.DataFrame) else pd.DataFrame()

    if modulo == "👥 Clientes":
        modulo_clientes(df_filtrado, consulta_fn)
    elif modulo == "👨‍✈️ Operadores":
        modulo_operadores(df_filtrado, consulta_fn)
    elif modulo == "🛣️ Rutas":
        modulo_rutas(df_filtrado, consulta_fn)
    elif modulo == "⛽ Combustible":
        modulo_combustible(df_filtrado)
    elif modulo == "📌 Asignaciones":
        modulo_asignaciones(df_filtrado, consulta_fn, base_dir)
    elif modulo == "🗺️ Mapa operativo":
        modulo_mapa(df_filtrado)
    elif modulo == "📊 Reportes":
        modulo_reportes(
            df_filtrado,
            mantenimiento,
            vehiculos,
            riesgos,
            base_dir,
            periodo,
        )
    else:
        st.info("Módulo no reconocido.")
