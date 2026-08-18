import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# SIG-LOG - PCA
# Análisis de Componentes Principales
# Unidad IV: Análisis no supervisado
# ============================================================

print("=" * 70)
print("              SIG-LOG - PCA")
print("=" * 70)
print("Unidad IV: Análisis no supervisado")
print("Objetivo: reducción de dimensionalidad")
print("=" * 70)


# ============================================================
# CONFIGURACIÓN
# ============================================================

ARCHIVO = "datos/limpios/entregas.csv"
CARPETA_RESULTADOS = "pca/resultados"

os.makedirs(CARPETA_RESULTADOS, exist_ok=True)


# ============================================================
# 1. CARGAR DATOS
# ============================================================

print("\n1. CARGANDO DATOS")
print("-" * 70)

df = pd.read_csv(ARCHIVO)

print(f"Registros cargados: {len(df):,}")
print(f"Columnas disponibles: {len(df.columns)}")


# ============================================================
# 2. VARIABLES PARA PCA
# ============================================================

print("\n2. SELECCIONANDO VARIABLES")
print("-" * 70)

variables = [
    "peso_carga",
    "cantidad_paquetes",
    "distancia_real_km",
    "combustible_consumido_litros",
    "precio_combustible",
    "costo_envio",
    "costo_combustible",
    "costo_total",
    "minutos_retraso"
]

faltantes = [col for col in variables if col not in df.columns]

if faltantes:
    print("ERROR: faltan columnas:")
    for col in faltantes:
        print(f"  - {col}")
    raise SystemExit(1)

df_pca = df[variables].copy()

print("Variables seleccionadas:")
for variable in variables:
    print(f"  ✓ {variable}")


# ============================================================
# 3. LIMPIEZA
# ============================================================

print("\n3. PREPARANDO DATOS")
print("-" * 70)

for col in variables:
    df_pca[col] = pd.to_numeric(
        df_pca[col],
        errors="coerce"
    )

registros_antes = len(df_pca)

df_pca = df_pca.replace(
    [np.inf, -np.inf],
    np.nan
)

df_pca = df_pca.dropna()

registros_despues = len(df_pca)

print(f"Registros antes : {registros_antes:,}")
print(f"Registros válidos: {registros_despues:,}")
print(f"Registros eliminados: {registros_antes - registros_despues:,}")


# ============================================================
# 4. ESTANDARIZACIÓN
# ============================================================

print("\n4. ESTANDARIZANDO VARIABLES")
print("-" * 70)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(df_pca)

print("✓ Variables estandarizadas")
print("Media aproximada: 0")
print("Desviación estándar aproximada: 1")


# ============================================================
# 5. PCA
# ============================================================

print("\n5. EJECUTANDO PCA")
print("-" * 70)

pca_completo = PCA()

X_pca_completo = pca_completo.fit_transform(X_scaled)

varianza = pca_completo.explained_variance_ratio_
varianza_acumulada = np.cumsum(varianza)


# ============================================================
# 6. VARIANZA EXPLICADA
# ============================================================

print("\n6. VARIANZA EXPLICADA")
print("-" * 70)

tabla_varianza = pd.DataFrame({
    "componente": [
        f"PC{i + 1}"
        for i in range(len(varianza))
    ],
    "varianza_explicada": varianza * 100,
    "varianza_acumulada": varianza_acumulada * 100
})

for _, fila in tabla_varianza.iterrows():

    print(
        f"{fila['componente']:5} | "
        f"Individual: {fila['varianza_explicada']:6.2f}% | "
        f"Acumulada: {fila['varianza_acumulada']:6.2f}%"
    )


# ============================================================
# 7. DETERMINAR COMPONENTES
# ============================================================

print("\n7. SELECCIÓN DE COMPONENTES")
print("-" * 70)

umbral = 0.80

componentes_80 = np.argmax(
    varianza_acumulada >= umbral
) + 1

print(
    f"Componentes necesarios para explicar "
    f"al menos 80%: {componentes_80}"
)

componentes_90 = np.argmax(
    varianza_acumulada >= 0.90
) + 1

print(
    f"Componentes necesarios para explicar "
    f"al menos 90%: {componentes_90}"
)


# ============================================================
# 8. PCA FINAL
# ============================================================

print("\n8. GENERANDO PCA FINAL")
print("-" * 70)

pca = PCA(
    n_components=componentes_80
)

X_pca = pca.fit_transform(X_scaled)

print(
    f"✓ PCA generado con "
    f"{componentes_80} componentes"
)


# ============================================================
# 9. CARGAS DE LOS COMPONENTES
# ============================================================

print("\n9. CONTRIBUCIÓN DE VARIABLES")
print("-" * 70)

cargas = pd.DataFrame(
    pca.components_.T,
    index=variables,
    columns=[
        f"PC{i + 1}"
        for i in range(componentes_80)
    ]
)

for componente in cargas.columns:

    print(f"\n{componente}")

    orden = (
        cargas[componente]
        .abs()
        .sort_values(
            ascending=False
        )
    )

    for variable in orden.head(5).index:

        valor = cargas.loc[
            variable,
            componente
        ]

        print(
            f"  {variable:35} "
            f"{valor: .4f}"
        )


# ============================================================
# 10. INTERPRETACIÓN AUTOMÁTICA
# ============================================================

print("\n10. INTERPRETACIÓN DE COMPONENTES")
print("-" * 70)

interpretaciones = []

for componente in cargas.columns:

    valores = cargas[componente].abs()

    principales = valores.nlargest(3).index.tolist()

    descripcion = ", ".join(principales)

    interpretaciones.append({
        "componente": componente,
        "variables_principales": descripcion
    })

    print(
        f"{componente}: "
        f"{descripcion}"
    )


# ============================================================
# 11. GENERAR DATASET PCA
# ============================================================

print("\n11. GENERANDO DATASET PCA")
print("-" * 70)

resultado_pca = df.loc[
    df_pca.index
].copy()

for i in range(componentes_80):

    resultado_pca[
        f"PC{i + 1}"
    ] = X_pca[:, i]

print(
    f"✓ Registros con coordenadas PCA: "
    f"{len(resultado_pca):,}"
)


# ============================================================
# 12. GUARDAR RESULTADOS
# ============================================================

print("\n12. GUARDANDO RESULTADOS")
print("-" * 70)

archivo_varianza = (
    f"{CARPETA_RESULTADOS}/"
    "varianza_pca.csv"
)

archivo_cargas = (
    f"{CARPETA_RESULTADOS}/"
    "cargas_componentes.csv"
)

archivo_pca = (
    f"{CARPETA_RESULTADOS}/"
    "entregas_pca.csv"
)

archivo_interpretacion = (
    f"{CARPETA_RESULTADOS}/"
    "interpretacion_componentes.csv"
)

tabla_varianza.to_csv(
    archivo_varianza,
    index=False
)

cargas.to_csv(
    archivo_cargas
)

resultado_pca.to_csv(
    archivo_pca,
    index=False
)

pd.DataFrame(
    interpretaciones
).to_csv(
    archivo_interpretacion,
    index=False
)

print(f"✓ {os.path.abspath(archivo_varianza)}")
print(f"✓ {os.path.abspath(archivo_cargas)}")
print(f"✓ {os.path.abspath(archivo_pca)}")
print(f"✓ {os.path.abspath(archivo_interpretacion)}")


# ============================================================
# 13. RESUMEN
# ============================================================

print("\n")
print("=" * 70)
print("                 PCA COMPLETADO")
print("=" * 70)

print(
    f"\nRegistros analizados: "
    f"{len(resultado_pca):,}"
)

print(
    f"Variables originales: "
    f"{len(variables)}"
)

print(
    f"Componentes seleccionados: "
    f"{componentes_80}"
)

print(
    f"Varianza acumulada: "
    f"{varianza_acumulada[componentes_80 - 1] * 100:.2f}%"
)

print("\nArchivos generados:")

print("✓ varianza_pca.csv")
print("✓ cargas_componentes.csv")
print("✓ entregas_pca.csv")
print("✓ interpretacion_componentes.csv")

print("\n")
print("=" * 70)