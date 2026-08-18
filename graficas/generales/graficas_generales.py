import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

sys.path.append(str(BASE_DIR))

from data_warehouse.consultas_warehouse import (
    resumen_general,
    entregas_por_estatus,
    entregas_tardias,
    entregas_por_mes,
    retrasos_por_ruta,
    desempeno_vehiculos,
    costos_por_vehiculo,
    combustible_por_vehiculo,
    costo_vs_retraso,
    distancia_vs_retraso,
    mantenimiento_por_tipo,
    fallas_por_componente,
)


GRAFICAS_DIR = BASE_DIR / "graficas" / "generales"
GRAFICAS_DIR.mkdir(parents=True, exist_ok=True)


def guardar_grafica(nombre):
    ruta = GRAFICAS_DIR / nombre
    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {ruta}")


# ============================================================
# ENCABEZADO
# ============================================================

print("=" * 70)
print("             SIG-LOG - GRÁFICAS GENERALES")
print("=" * 70)


# ============================================================
# 1. RESUMEN GENERAL
# ============================================================

print("\n1. RESUMEN GENERAL")
print("-" * 70)

resumen = resumen_general()

print(resumen.to_string(index=False))


# ============================================================
# 2. ENTREGAS POR ESTATUS
# ============================================================

print("\n2. ENTREGAS POR ESTATUS")
print("-" * 70)

df = entregas_por_estatus()

plt.figure(figsize=(9, 6))

plt.bar(
    df["estatus"].astype(str),
    df["cantidad"]
)

plt.title("Entregas por estatus")
plt.xlabel("Estatus")
plt.ylabel("Cantidad de entregas")
plt.xticks(rotation=30)

guardar_grafica("01_entregas_por_estatus.png")


# ============================================================
# 3. ENTREGAS TARDÍAS
# ============================================================

print("\n3. ENTREGAS TARDÍAS")
print("-" * 70)

df = entregas_tardias()

labels = ["A tiempo", "Tardía"]

plt.figure(figsize=(8, 6))

plt.bar(
    labels,
    df["cantidad"]
)

plt.title("Entregas a tiempo vs tardías")
plt.xlabel("Tipo de entrega")
plt.ylabel("Cantidad")

guardar_grafica("02_entregas_tardias.png")


# ============================================================
# 4. ENTREGAS POR MES
# ============================================================

print("\n4. ENTREGAS POR MES")
print("-" * 70)

df = entregas_por_mes()

plt.figure(figsize=(12, 6))

plt.plot(
    df["nombre_mes"],
    df["cantidad_entregas"],
    marker="o"
)

plt.title("Cantidad de entregas por mes")
plt.xlabel("Mes")
plt.ylabel("Entregas")
plt.xticks(rotation=45)

guardar_grafica("03_entregas_por_mes.png")


# ============================================================
# 5. RETRASO PROMEDIO POR MES
# ============================================================

print("\n5. RETRASO PROMEDIO POR MES")
print("-" * 70)

plt.figure(figsize=(12, 6))

plt.plot(
    df["nombre_mes"],
    df["retraso_promedio"],
    marker="o"
)

plt.title("Retraso promedio por mes")
plt.xlabel("Mes")
plt.ylabel("Minutos de retraso")
plt.xticks(rotation=45)

guardar_grafica("04_retraso_por_mes.png")


# ============================================================
# 6. RUTAS CON MAYOR RETRASO
# ============================================================

print("\n6. RUTAS CON MAYOR RETRASO")
print("-" * 70)

df = retrasos_por_ruta().head(10)

plt.figure(figsize=(12, 7))

plt.barh(
    df["nombre_ruta"].astype(str),
    df["retraso_promedio"]
)

plt.title("Top 10 rutas con mayor retraso promedio")
plt.xlabel("Minutos de retraso")
plt.ylabel("Ruta")

plt.gca().invert_yaxis()

guardar_grafica("05_retrasos_por_ruta.png")


# ============================================================
# 7. VEHÍCULOS CON MAYOR RETRASO
# ============================================================

print("\n7. VEHÍCULOS CON MAYOR RETRASO")
print("-" * 70)

df = desempeno_vehiculos().head(10)

plt.figure(figsize=(12, 7))

plt.barh(
    df["numero_economico"].astype(str),
    df["retraso_promedio"]
)

plt.title("Top 10 vehículos con mayor retraso promedio")
plt.xlabel("Minutos de retraso")
plt.ylabel("Vehículo")

plt.gca().invert_yaxis()

guardar_grafica("06_retrasos_por_vehiculo.png")


# ============================================================
# 8. COSTOS POR VEHÍCULO
# ============================================================

print("\n8. COSTOS POR VEHÍCULO")
print("-" * 70)

df = costos_por_vehiculo().head(10)

plt.figure(figsize=(12, 7))

plt.barh(
    df["numero_economico"].astype(str),
    df["costo_total"]
)

plt.title("Top 10 vehículos por costo total de entregas")
plt.xlabel("Costo total")
plt.ylabel("Vehículo")

plt.gca().invert_yaxis()

guardar_grafica("07_costos_por_vehiculo.png")


# ============================================================
# 9. CONSUMO DE COMBUSTIBLE
# ============================================================

print("\n9. CONSUMO DE COMBUSTIBLE")
print("-" * 70)

df = combustible_por_vehiculo().head(10)

plt.figure(figsize=(12, 7))

plt.barh(
    df["numero_economico"].astype(str),
    df["combustible_total"]
)

plt.title("Top 10 vehículos por consumo de combustible")
plt.xlabel("Litros consumidos")
plt.ylabel("Vehículo")

plt.gca().invert_yaxis()

guardar_grafica("08_combustible_por_vehiculo.png")


# ============================================================
# 10. DISTANCIA VS RETRASO
# ============================================================

print("\n10. DISTANCIA VS RETRASO")
print("-" * 70)

df = distancia_vs_retraso()

plt.figure(figsize=(10, 7))

plt.scatter(
    df["distancia_real_km"],
    df["minutos_retraso"],
    alpha=0.35
)

plt.title("Distancia real vs minutos de retraso")
plt.xlabel("Distancia real (km)")
plt.ylabel("Minutos de retraso")

guardar_grafica("09_distancia_vs_retraso.png")


# ============================================================
# 11. COSTO VS RETRASO
# ============================================================

print("\n11. COSTO VS RETRASO")
print("-" * 70)

df = costo_vs_retraso()

plt.figure(figsize=(10, 7))

plt.scatter(
    df["costo_total"],
    df["minutos_retraso"],
    alpha=0.35
)

plt.title("Costo total vs minutos de retraso")
plt.xlabel("Costo total")
plt.ylabel("Minutos de retraso")

guardar_grafica("10_costo_vs_retraso.png")


# ============================================================
# 12. MANTENIMIENTO POR TIPO
# ============================================================

print("\n12. MANTENIMIENTO POR TIPO")
print("-" * 70)

df = mantenimiento_por_tipo()

plt.figure(figsize=(9, 6))

plt.bar(
    df["tipo_mantenimiento"],
    df["cantidad"]
)

plt.title("Mantenimientos por tipo")
plt.xlabel("Tipo de mantenimiento")
plt.ylabel("Cantidad")

plt.xticks(rotation=20)

guardar_grafica("11_mantenimiento_por_tipo.png")


# ============================================================
# 13. FALLAS POR COMPONENTE
# ============================================================

print("\n13. FALLAS POR COMPONENTE")
print("-" * 70)

df = fallas_por_componente().head(10)

plt.figure(figsize=(12, 7))

plt.barh(
    df["nombre"].astype(str),
    df["fallas"]
)

plt.title("Top 10 componentes con más fallas")
plt.xlabel("Número de fallas")
plt.ylabel("Componente")

plt.gca().invert_yaxis()

guardar_grafica("12_fallas_por_componente.png")


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 70)
print("        GRÁFICAS GENERALES COMPLETADAS")
print("=" * 70)

print("\nGráficas generadas:")

for archivo in sorted(GRAFICAS_DIR.glob("*.png")):
    print(f"✓ {archivo.name}")

print(f"\nUbicación:")
print(GRAFICAS_DIR)

print("=" * 70)