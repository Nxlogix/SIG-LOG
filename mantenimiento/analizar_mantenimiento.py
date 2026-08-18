import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# SIG-LOG - ANÁLISIS DE MANTENIMIENTO
# ============================================================

print("=" * 75)
print("             SIG-LOG - ANÁLISIS DE MANTENIMIENTO")
print("=" * 75)

BASE_DIR = Path(__file__).resolve().parent.parent

MANTENIMIENTO = BASE_DIR / "datos" / "limpios" / "mantenimiento.csv"
VEHICULOS = BASE_DIR / "datos" / "limpios" / "vehiculos.csv"
COMPONENTES = BASE_DIR / "datos" / "limpios" / "componentes.csv"

SALIDA_DIR = BASE_DIR / "mantenimiento" / "resultados"
SALIDA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. CARGAR DATOS
# ============================================================

print("\n1. CARGANDO DATOS")
print("-" * 75)

df = pd.read_csv(MANTENIMIENTO)
vehiculos = pd.read_csv(VEHICULOS)
componentes = pd.read_csv(COMPONENTES)

print(f"Mantenimientos : {len(df):,}")
print(f"Vehículos      : {len(vehiculos):,}")
print(f"Componentes    : {len(componentes):,}")

# ============================================================
# 2. PREPARAR DATOS
# ============================================================

df["fecha_mantenimiento"] = pd.to_datetime(
    df["fecha_mantenimiento"],
    errors="coerce"
)

# ============================================================
# 3. RESUMEN GENERAL
# ============================================================

print("\n2. RESUMEN GENERAL")
print("-" * 75)

total_mantenimientos = len(df)
total_costo = df["costo_total"].sum()
total_horas = df["horas_fuera_servicio"].sum()
total_vehiculos = df["id_vehiculo"].nunique()
total_componentes = df["id_componente"].nunique()

resumen = pd.DataFrame([{
    "total_mantenimientos": total_mantenimientos,
    "total_vehiculos": total_vehiculos,
    "total_componentes": total_componentes,
    "costo_total": round(total_costo, 2),
    "horas_fuera_servicio": round(total_horas, 2),
    "costo_promedio_mantenimiento": round(
        total_costo / total_mantenimientos, 2
    ),
    "horas_promedio_fuera_servicio": round(
        total_horas / total_mantenimientos, 2
    )
}])

resumen.to_csv(
    SALIDA_DIR / "resumen_mantenimiento.csv",
    index=False,
    encoding="utf-8-sig"
)

print(f"Mantenimientos          : {total_mantenimientos:,}")
print(f"Vehículos               : {total_vehiculos:,}")
print(f"Componentes             : {total_componentes:,}")
print(f"Costo total             : ${total_costo:,.2f}")
print(f"Horas fuera de servicio : {total_horas:,.0f}")

# ============================================================
# 4. FALLAS POR VEHÍCULO
# ============================================================

print("\n3. FALLAS POR VEHÍCULO")
print("-" * 75)

fallas_vehiculo = (
    df.groupby("id_vehiculo")
    .agg(
        mantenimientos=("id_mantenimiento", "count"),
        fallas=("tipo_mantenimiento", lambda x:
                (x != "Preventivo").sum()),
        emergencias=("tipo_mantenimiento", lambda x:
                     (x == "Emergencia").sum()),
        costo_total=("costo_total", "sum"),
        horas_fuera_servicio=("horas_fuera_servicio", "sum"),
        retrasos_por_mantenimiento=("horas_fuera_servicio", "mean"),
        severidad_critica=("severidad", lambda x:
                           (x == "Crítica").sum())
    )
    .reset_index()
)

fallas_vehiculo["porcentaje_costo_total"] = (
    fallas_vehiculo["costo_total"] /
    total_costo * 100
)

fallas_vehiculo["porcentaje_mantenimientos"] = (
    fallas_vehiculo["mantenimientos"] /
    total_mantenimientos * 100
)

fallas_vehiculo["indice_riesgo"] = (
    fallas_vehiculo["fallas"] * 2 +
    fallas_vehiculo["emergencias"] * 3 +
    fallas_vehiculo["severidad_critica"] * 4
)

def clasificar_riesgo(valor):
    if valor >= 30:
        return "CRÍTICO"
    elif valor >= 20:
        return "ALTO"
    elif valor >= 10:
        return "MODERADO"
    return "BAJO"

fallas_vehiculo["nivel_riesgo"] = (
    fallas_vehiculo["indice_riesgo"]
    .apply(clasificar_riesgo)
)

fallas_vehiculo = fallas_vehiculo.sort_values(
    "indice_riesgo",
    ascending=False
)

fallas_vehiculo.to_csv(
    SALIDA_DIR / "fallas_por_vehiculo.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    fallas_vehiculo[
        [
            "id_vehiculo",
            "mantenimientos",
            "fallas",
            "emergencias",
            "costo_total",
            "horas_fuera_servicio",
            "nivel_riesgo"
        ]
    ].head(10).to_string(index=False)
)

# ============================================================
# 5. FALLAS POR COMPONENTE
# ============================================================

print("\n4. FALLAS POR COMPONENTE")
print("-" * 75)

fallas_componente = (
    df.groupby(
        [
            "id_componente",
            "componente",
            "categoria_componente"
        ]
    )
    .agg(
        mantenimientos=("id_mantenimiento", "count"),
        fallas=("tipo_mantenimiento", lambda x:
                (x != "Preventivo").sum()),
        emergencias=("tipo_mantenimiento", lambda x:
                     (x == "Emergencia").sum()),
        vehiculos_afectados=("id_vehiculo", "nunique"),
        costo_total=("costo_total", "sum"),
        horas_fuera_servicio=("horas_fuera_servicio", "sum")
    )
    .reset_index()
)

fallas_componente["porcentaje_fallas"] = (
    fallas_componente["fallas"] /
    max(fallas_componente["fallas"].sum(), 1) * 100
)

fallas_componente["porcentaje_costo"] = (
    fallas_componente["costo_total"] /
    total_costo * 100
)

fallas_componente = fallas_componente.sort_values(
    "fallas",
    ascending=False
)

fallas_componente.to_csv(
    SALIDA_DIR / "fallas_por_componente.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    fallas_componente[
        [
            "componente",
            "categoria_componente",
            "fallas",
            "porcentaje_fallas",
            "vehiculos_afectados",
            "costo_total"
        ]
    ].head(10).to_string(index=False)
)

# ============================================================
# 6. CAUSAS DE FALLA
# ============================================================

print("\n5. CAUSAS DE FALLA")
print("-" * 75)

causas = (
    df[df["tipo_mantenimiento"] != "Preventivo"]
    .groupby("causa_falla")
    .agg(
        cantidad=("id_mantenimiento", "count"),
        vehiculos_afectados=("id_vehiculo", "nunique"),
        costo_total=("costo_total", "sum"),
        horas_fuera_servicio=("horas_fuera_servicio", "sum")
    )
    .reset_index()
)

total_fallas = causas["cantidad"].sum()

causas["porcentaje"] = (
    causas["cantidad"] /
    max(total_fallas, 1) * 100
)

causas = causas.sort_values(
    "cantidad",
    ascending=False
)

causas.to_csv(
    SALIDA_DIR / "causas_falla.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    causas.head(10).to_string(index=False)
)

# ============================================================
# 7. MANTENIMIENTO POR TIPO
# ============================================================

print("\n6. MANTENIMIENTO POR TIPO")
print("-" * 75)

tipo = (
    df.groupby("tipo_mantenimiento")
    .agg(
        cantidad=("id_mantenimiento", "count"),
        costo_total=("costo_total", "sum"),
        horas_fuera_servicio=("horas_fuera_servicio", "sum")
    )
    .reset_index()
)

tipo["porcentaje"] = (
    tipo["cantidad"] /
    total_mantenimientos * 100
)

tipo.to_csv(
    SALIDA_DIR / "mantenimiento_por_tipo.csv",
    index=False,
    encoding="utf-8-sig"
)

print(tipo.to_string(index=False))

# ============================================================
# 8. COSTOS POR VEHÍCULO
# ============================================================

print("\n7. COSTOS POR VEHÍCULO")
print("-" * 75)

costos = (
    df.groupby("id_vehiculo")
    .agg(
        mantenimientos=("id_mantenimiento", "count"),
        costo_repuesto=("costo_repuesto", "sum"),
        costo_mano_obra=("costo_mano_obra", "sum"),
        costo_total=("costo_total", "sum")
    )
    .reset_index()
)

costos["porcentaje_costo"] = (
    costos["costo_total"] /
    total_costo * 100
)

costos = costos.sort_values(
    "costo_total",
    ascending=False
)

costos.to_csv(
    SALIDA_DIR / "costos_por_vehiculo.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    costos.head(10).to_string(index=False)
)

# ============================================================
# 9. SEVERIDAD
# ============================================================

print("\n8. SEVERIDAD DE FALLAS")
print("-" * 75)

severidad = (
    df.groupby("severidad")
    .agg(
        cantidad=("id_mantenimiento", "count"),
        costo_total=("costo_total", "sum"),
        horas_fuera_servicio=("horas_fuera_servicio", "sum")
    )
    .reset_index()
)

severidad["porcentaje"] = (
    severidad["cantidad"] /
    total_mantenimientos * 100
)

severidad.to_csv(
    SALIDA_DIR / "severidad_fallas.csv",
    index=False,
    encoding="utf-8-sig"
)

print(severidad.to_string(index=False))

# ============================================================
# 10. ALERTAS DE MANTENIMIENTO
# ============================================================

print("\n9. GENERANDO ALERTAS")
print("-" * 75)

alertas = []

for _, row in fallas_vehiculo.iterrows():

    vehiculo = row["id_vehiculo"]

    if row["nivel_riesgo"] == "CRÍTICO":
        prioridad = "CRÍTICA"
        recomendacion = (
            "Programar inspección inmediata y revisar historial "
            "completo de componentes."
        )

    elif row["nivel_riesgo"] == "ALTO":
        prioridad = "ALTA"
        recomendacion = (
            "Programar mantenimiento preventivo prioritario."
        )

    elif row["nivel_riesgo"] == "MODERADO":
        prioridad = "MEDIA"
        recomendacion = (
            "Monitorear comportamiento y programar revisión."
        )

    else:
        prioridad = "BAJA"
        recomendacion = (
            "Continuar con mantenimiento preventivo programado."
        )

    alertas.append({
        "id_vehiculo": vehiculo,
        "nivel_riesgo": row["nivel_riesgo"],
        "prioridad": prioridad,
        "mantenimientos": row["mantenimientos"],
        "fallas": row["fallas"],
        "emergencias": row["emergencias"],
        "fallas_criticas": row["severidad_critica"],
        "costo_total": round(row["costo_total"], 2),
        "horas_fuera_servicio": round(
            row["horas_fuera_servicio"], 2
        ),
        "recomendacion": recomendacion
    })

alertas = pd.DataFrame(alertas)

orden_prioridad = {
    "CRÍTICA": 0,
    "ALTA": 1,
    "MEDIA": 2,
    "BAJA": 3
}

alertas["orden"] = alertas["prioridad"].map(
    orden_prioridad
)

alertas = alertas.sort_values(
    ["orden", "fallas"],
    ascending=[True, False]
)

alertas = alertas.drop(columns=["orden"])

alertas.to_csv(
    SALIDA_DIR / "alertas_mantenimiento.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    alertas.head(15).to_string(index=False)
)

# ============================================================
# 11. RESUMEN FINAL
# ============================================================

print("\n" + "=" * 75)
print("             ANÁLISIS DE MANTENIMIENTO COMPLETADO")
print("=" * 75)

print("\nArchivos generados:")

archivos = [
    "resumen_mantenimiento.csv",
    "fallas_por_vehiculo.csv",
    "fallas_por_componente.csv",
    "causas_falla.csv",
    "mantenimiento_por_tipo.csv",
    "costos_por_vehiculo.csv",
    "severidad_fallas.csv",
    "alertas_mantenimiento.csv"
]

for archivo in archivos:
    print(f"✓ {SALIDA_DIR / archivo}")

print("\n" + "=" * 75)