# ==============================================================
# SIG-LOG - VISUALIZACIÓN DE MANTENIMIENTO
# ==============================================================
# Unidad V: Gestión y análisis de mantenimiento
# ==============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTADOS_DIR = os.path.join(
    BASE_DIR,
    "mantenimiento",
    "resultados"
)

GRAFICAS_DIR = os.path.join(
    BASE_DIR,
    "graficas",
    "mantenimiento"
)

os.makedirs(GRAFICAS_DIR, exist_ok=True)


print("=" * 70)
print("          SIG-LOG - VISUALIZACIÓN DE MANTENIMIENTO")
print("=" * 70)


# ==============================================================
# 1. CARGANDO RESULTADOS
# ==============================================================

print("\n1. CARGANDO RESULTADOS")
print("-" * 70)

resumen = pd.read_csv(
    os.path.join(RESULTADOS_DIR, "resumen_mantenimiento.csv")
)

fallas_vehiculo = pd.read_csv(
    os.path.join(RESULTADOS_DIR, "fallas_por_vehiculo.csv")
)

fallas_componente = pd.read_csv(
    os.path.join(RESULTADOS_DIR, "fallas_por_componente.csv")
)

causas = pd.read_csv(
    os.path.join(RESULTADOS_DIR, "causas_falla.csv")
)

mantenimiento_tipo = pd.read_csv(
    os.path.join(RESULTADOS_DIR, "mantenimiento_por_tipo.csv")
)

costos_vehiculo = pd.read_csv(
    os.path.join(RESULTADOS_DIR, "costos_por_vehiculo.csv")
)

severidad = pd.read_csv(
    os.path.join(RESULTADOS_DIR, "severidad_fallas.csv")
)

alertas = pd.read_csv(
    os.path.join(RESULTADOS_DIR, "alertas_mantenimiento.csv")
)

print("✓ Resultados cargados")


# ==============================================================
# FUNCIÓN PARA GUARDAR GRÁFICAS
# ==============================================================

def guardar_grafica(nombre):
    ruta = os.path.join(GRAFICAS_DIR, nombre)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {ruta}")


# ==============================================================
# 2. MANTENIMIENTOS POR TIPO
# ==============================================================

print("\n2. MANTENIMIENTOS POR TIPO")
print("-" * 70)

plt.figure(figsize=(9, 6))

plt.bar(
    mantenimiento_tipo["tipo_mantenimiento"],
    mantenimiento_tipo["cantidad"]
)

plt.title("Cantidad de mantenimientos por tipo")
plt.xlabel("Tipo de mantenimiento")
plt.ylabel("Cantidad")

guardar_grafica("01_mantenimientos_por_tipo.png")


# ==============================================================
# 3. COSTO POR TIPO DE MANTENIMIENTO
# ==============================================================

print("\n3. COSTO POR TIPO DE MANTENIMIENTO")
print("-" * 70)

plt.figure(figsize=(9, 6))

plt.bar(
    mantenimiento_tipo["tipo_mantenimiento"],
    mantenimiento_tipo["costo_total"]
)

plt.title("Costo total por tipo de mantenimiento")
plt.xlabel("Tipo de mantenimiento")
plt.ylabel("Costo total ($)")

guardar_grafica("02_costos_por_tipo.png")


# ==============================================================
# 4. HORAS FUERA DE SERVICIO
# ==============================================================

print("\n4. HORAS FUERA DE SERVICIO")
print("-" * 70)

plt.figure(figsize=(9, 6))

plt.bar(
    mantenimiento_tipo["tipo_mantenimiento"],
    mantenimiento_tipo["horas_fuera_servicio"]
)

plt.title("Horas fuera de servicio por tipo de mantenimiento")
plt.xlabel("Tipo de mantenimiento")
plt.ylabel("Horas")

guardar_grafica("03_horas_fuera_servicio.png")


# ==============================================================
# 5. FALLAS POR COMPONENTE
# ==============================================================

print("\n5. FALLAS POR COMPONENTE")
print("-" * 70)

top_componentes = (
    fallas_componente
    .sort_values("fallas", ascending=False)
    .head(10)
    .sort_values("fallas")
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_componentes["componente"],
    top_componentes["fallas"]
)

plt.title("Top 10 componentes con más fallas")
plt.xlabel("Número de fallas")
plt.ylabel("Componente")

guardar_grafica("04_fallas_por_componente.png")


# ==============================================================
# 6. COSTOS POR COMPONENTE
# ==============================================================

print("\n6. COSTOS POR COMPONENTE")
print("-" * 70)

top_costos_componentes = (
    fallas_componente
    .sort_values("costo_total", ascending=False)
    .head(10)
    .sort_values("costo_total")
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_costos_componentes["componente"],
    top_costos_componentes["costo_total"]
)

plt.title("Top 10 componentes por costo")
plt.xlabel("Costo total ($)")
plt.ylabel("Componente")

guardar_grafica("05_costos_por_componente.png")


# ==============================================================
# 7. PRINCIPALES CAUSAS DE FALLA
# ==============================================================

print("\n7. PRINCIPALES CAUSAS DE FALLA")
print("-" * 70)

top_causas = (
    causas
    .sort_values("cantidad", ascending=False)
    .head(10)
    .sort_values("cantidad")
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_causas["causa_falla"],
    top_causas["cantidad"]
)

plt.title("Principales causas de falla")
plt.xlabel("Cantidad")
plt.ylabel("Causa")

guardar_grafica("06_causas_falla.png")


# ==============================================================
# 8. COSTOS POR VEHÍCULO
# ==============================================================

print("\n8. COSTOS POR VEHÍCULO")
print("-" * 70)

top_vehiculos = (
    costos_vehiculo
    .sort_values("costo_total", ascending=False)
    .head(10)
    .sort_values("costo_total")
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_vehiculos["id_vehiculo"].astype(str),
    top_vehiculos["costo_total"]
)

plt.title("Top 10 vehículos por costo de mantenimiento")
plt.xlabel("Costo total ($)")
plt.ylabel("ID vehículo")

guardar_grafica("07_costos_por_vehiculo.png")


# ==============================================================
# 9. FALLAS POR VEHÍCULO
# ==============================================================

print("\n9. FALLAS POR VEHÍCULO")
print("-" * 70)

top_fallas_vehiculo = (
    fallas_vehiculo
    .sort_values("fallas", ascending=False)
    .head(10)
    .sort_values("fallas")
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_fallas_vehiculo["id_vehiculo"].astype(str),
    top_fallas_vehiculo["fallas"]
)

plt.title("Top 10 vehículos con más fallas")
plt.xlabel("Número de fallas")
plt.ylabel("ID vehículo")

guardar_grafica("08_fallas_por_vehiculo.png")


# ==============================================================
# 10. SEVERIDAD
# ==============================================================

print("\n10. SEVERIDAD DE FALLAS")
print("-" * 70)

plt.figure(figsize=(9, 6))

plt.bar(
    severidad["severidad"],
    severidad["cantidad"]
)

plt.title("Distribución de fallas por severidad")
plt.xlabel("Severidad")
plt.ylabel("Cantidad")

guardar_grafica("09_severidad_fallas.png")


# ==============================================================
# 11. COSTO POR SEVERIDAD
# ==============================================================

print("\n11. COSTO POR SEVERIDAD")
print("-" * 70)

plt.figure(figsize=(9, 6))

plt.bar(
    severidad["severidad"],
    severidad["costo_total"]
)

plt.title("Costo de mantenimiento por severidad")
plt.xlabel("Severidad")
plt.ylabel("Costo total ($)")

guardar_grafica("10_costos_por_severidad.png")


# ==============================================================
# 12. HORAS FUERA DE SERVICIO POR SEVERIDAD
# ==============================================================

print("\n12. HORAS FUERA DE SERVICIO POR SEVERIDAD")
print("-" * 70)

plt.figure(figsize=(9, 6))

plt.bar(
    severidad["severidad"],
    severidad["horas_fuera_servicio"]
)

plt.title("Horas fuera de servicio por severidad")
plt.xlabel("Severidad")
plt.ylabel("Horas")

guardar_grafica("11_horas_por_severidad.png")


# ==============================================================
# 13. VEHÍCULOS CON ALERTA
# ==============================================================

print("\n13. VEHÍCULOS CON ALERTA")
print("-" * 70)

alertas_plot = (
    alertas["nivel_riesgo"]
    .value_counts()
)

plt.figure(figsize=(9, 6))

plt.bar(
    alertas_plot.index,
    alertas_plot.values
)

plt.title("Vehículos clasificados por nivel de riesgo")
plt.xlabel("Nivel de riesgo")
plt.ylabel("Cantidad de vehículos")

guardar_grafica("12_riesgo_vehiculos.png")


# ==============================================================
# 14. RESUMEN DE COSTOS
# ==============================================================

print("\n14. RESUMEN DE COSTOS")
print("-" * 70)

if "costo_total" in resumen.columns:

    plt.figure(figsize=(8, 6))

    plt.bar(
        ["Costo histórico"],
        [resumen["costo_total"].iloc[0]]
    )

    plt.title("Costo histórico total de mantenimiento")
    plt.ylabel("Costo ($)")

    guardar_grafica("13_costo_historico_total.png")


# ==============================================================
# FINAL
# ==============================================================

print("\n")
print("=" * 70)
print("       VISUALIZACIÓN DE MANTENIMIENTO COMPLETADA")
print("=" * 70)

print("\nGráficas generadas en:")
print(GRAFICAS_DIR)

print("\nArchivos principales:")

for archivo in sorted(os.listdir(GRAFICAS_DIR)):
    if archivo.endswith(".png"):
        print(f"✓ {archivo}")

print("\n" + "=" * 70)