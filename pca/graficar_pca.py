import os

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SIG-LOG - VISUALIZACIÓN PCA
# ============================================================

print("=" * 70)
print("           SIG-LOG - VISUALIZACIÓN PCA")
print("=" * 70)


# ============================================================
# CONFIGURACIÓN
# ============================================================

CARPETA = "pca/resultados"

ARCHIVO_PCA = f"{CARPETA}/entregas_pca.csv"
ARCHIVO_VARIANZA = f"{CARPETA}/varianza_pca.csv"
ARCHIVO_CARGAS = f"{CARPETA}/cargas_componentes.csv"

CARPETA_GRAFICAS = "graficas/pca"

os.makedirs(CARPETA_GRAFICAS, exist_ok=True)


# ============================================================
# 1. CARGAR RESULTADOS
# ============================================================

print("\n1. CARGANDO RESULTADOS")
print("-" * 70)

df = pd.read_csv(ARCHIVO_PCA)
varianza = pd.read_csv(ARCHIVO_VARIANZA)
cargas = pd.read_csv(
    ARCHIVO_CARGAS,
    index_col=0
)

print(f"Entregas: {len(df):,}")
print(f"Componentes disponibles: {len(varianza)}")


# ============================================================
# 2. GRÁFICA DE VARIANZA INDIVIDUAL
# ============================================================

print("\n2. GRÁFICA DE VARIANZA EXPLICADA")
print("-" * 70)

plt.figure(figsize=(10, 6))

plt.bar(
    varianza["componente"],
    varianza["varianza_explicada"]
)

plt.xlabel("Componente principal")
plt.ylabel("Varianza explicada (%)")
plt.title(
    "SIG-LOG - Varianza explicada por componente PCA"
)

for i, valor in enumerate(
    varianza["varianza_explicada"]
):

    plt.text(
        i,
        valor + 0.8,
        f"{valor:.2f}%",
        ha="center"
    )

plt.tight_layout()

archivo = (
    f"{CARPETA_GRAFICAS}/"
    "01_varianza_explicada.png"
)

plt.savefig(
    archivo,
    dpi=150
)

plt.close()

print(f"✓ {archivo}")


# ============================================================
# 3. VARIANZA ACUMULADA
# ============================================================

print("\n3. GRÁFICA DE VARIANZA ACUMULADA")
print("-" * 70)

plt.figure(figsize=(10, 6))

plt.plot(
    varianza["componente"],
    varianza["varianza_acumulada"],
    marker="o"
)

plt.axhline(
    y=80,
    linestyle="--",
    label="80%"
)

plt.axhline(
    y=90,
    linestyle="--",
    label="90%"
)

plt.xlabel("Componente principal")
plt.ylabel("Varianza acumulada (%)")

plt.title(
    "SIG-LOG - Varianza acumulada del PCA"
)

plt.legend()

plt.tight_layout()

archivo = (
    f"{CARPETA_GRAFICAS}/"
    "02_varianza_acumulada.png"
)

plt.savefig(
    archivo,
    dpi=150
)

plt.close()

print(f"✓ {archivo}")


# ============================================================
# 4. PC1 VS PC2
# ============================================================

print("\n4. GRÁFICA PC1 VS PC2")
print("-" * 70)

plt.figure(figsize=(10, 7))

plt.scatter(
    df["PC1"],
    df["PC2"],
    alpha=0.5
)

plt.xlabel("PC1 - Costo y consumo")
plt.ylabel("PC2 - Carga y operación")

plt.title(
    "SIG-LOG - PCA: PC1 vs PC2"
)

plt.tight_layout()

archivo = (
    f"{CARPETA_GRAFICAS}/"
    "03_pca_pc1_pc2.png"
)

plt.savefig(
    archivo,
    dpi=150
)

plt.close()

print(f"✓ {archivo}")


# ============================================================
# 5. PC1 VS PC4
# ============================================================

print("\n5. GRÁFICA PC1 VS PC4")
print("-" * 70)

plt.figure(figsize=(10, 7))

plt.scatter(
    df["PC1"],
    df["PC4"],
    alpha=0.5
)

plt.xlabel("PC1 - Costo y consumo")
plt.ylabel("PC4 - Retrasos")

plt.title(
    "SIG-LOG - PCA: costo/consumo vs retrasos"
)

plt.tight_layout()

archivo = (
    f"{CARPETA_GRAFICAS}/"
    "04_pca_pc1_pc4.png"
)

plt.savefig(
    archivo,
    dpi=150
)

plt.close()

print(f"✓ {archivo}")


# ============================================================
# 6. PC2 VS PC3
# ============================================================

print("\n6. GRÁFICA PC2 VS PC3")
print("-" * 70)

plt.figure(figsize=(10, 7))

plt.scatter(
    df["PC2"],
    df["PC3"],
    alpha=0.5
)

plt.xlabel("PC2 - Carga y operación")
plt.ylabel("PC3 - Volumen de paquetes")

plt.title(
    "SIG-LOG - PCA: PC2 vs PC3"
)

plt.tight_layout()

archivo = (
    f"{CARPETA_GRAFICAS}/"
    "05_pca_pc2_pc3.png"
)

plt.savefig(
    archivo,
    dpi=150
)

plt.close()

print(f"✓ {archivo}")


# ============================================================
# 7. MAPA DE CARGAS
# ============================================================

print("\n7. MAPA DE CARGAS DE VARIABLES")
print("-" * 70)

plt.figure(figsize=(12, 7))

plt.imshow(
    cargas.iloc[:, :4],
    aspect="auto"
)

plt.colorbar(
    label="Carga de la variable"
)

plt.xticks(
    range(4),
    cargas.columns[:4]
)

plt.yticks(
    range(len(cargas.index)),
    cargas.index
)

plt.xlabel("Componentes principales")
plt.ylabel("Variables")

plt.title(
    "SIG-LOG - Contribución de variables al PCA"
)

plt.tight_layout()

archivo = (
    f"{CARPETA_GRAFICAS}/"
    "06_cargas_pca.png"
)

plt.savefig(
    archivo,
    dpi=150
)

plt.close()

print(f"✓ {archivo}")


# ============================================================
# RESUMEN
# ============================================================

print("\n")
print("=" * 70)
print("          VISUALIZACIÓN PCA COMPLETADA")
print("=" * 70)

print("\nGráficas generadas:")

print("✓ 01_varianza_explicada.png")
print("✓ 02_varianza_acumulada.png")
print("✓ 03_pca_pc1_pc2.png")
print("✓ 04_pca_pc1_pc4.png")
print("✓ 05_pca_pc2_pc3.png")
print("✓ 06_cargas_pca.png")

print(
    f"\nUbicación: "
    f"{os.path.abspath(CARPETA_GRAFICAS)}"
)

print("=" * 70)