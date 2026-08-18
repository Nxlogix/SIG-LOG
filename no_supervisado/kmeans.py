import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# SIG-LOG - ANÁLISIS NO SUPERVISADO
# K-MEANS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATOS_DIR = os.path.join(BASE_DIR, "datos", "limpios")
RESULTADOS_DIR = os.path.join(BASE_DIR, "no_supervisado", "resultados")

os.makedirs(RESULTADOS_DIR, exist_ok=True)


print("=" * 70)
print("           SIG-LOG - K-MEANS")
print("=" * 70)

# ============================================================
# 1. CARGAR DATOS
# ============================================================

archivo = os.path.join(DATOS_DIR, "entregas.csv")

print("\n1. CARGANDO DATOS")
print("-" * 70)

df = pd.read_csv(archivo)

print(f"Registros cargados: {len(df):,}")
print(f"Columnas disponibles: {len(df.columns)}")


# ============================================================
# 2. PREPARACIÓN
# ============================================================

print("\n2. PREPARANDO VARIABLES")
print("-" * 70)

columnas = [
    "distancia_real_km",
    "peso_carga",
    "cantidad_paquetes",
    "minutos_retraso",
    "combustible_consumido_litros",
    "costo_envio",
    "costo_combustible",
    "costo_total"
]

faltantes = [col for col in columnas if col not in df.columns]

if faltantes:
    print("\nERROR: faltan columnas:")
    for col in faltantes:
        print(f"  - {col}")
    raise SystemExit

datos = df[columnas].copy()

datos = datos.replace([np.inf, -np.inf], np.nan)
datos = datos.dropna()

print(f"Registros válidos: {len(datos):,}")


# ============================================================
# 3. ESTANDARIZACIÓN
# ============================================================

print("\n3. ESTANDARIZANDO VARIABLES")
print("-" * 70)

scaler = StandardScaler()

X = scaler.fit_transform(datos)

print("✓ Variables estandarizadas")


# ============================================================
# 4. BÚSQUEDA DEL MEJOR NÚMERO DE CLUSTERS
# ============================================================

print("\n4. EVALUACIÓN DE CLUSTERS")
print("-" * 70)

resultados_k = []

for k in range(2, 7):

    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    etiquetas = modelo.fit_predict(X)

    silhouette = silhouette_score(X, etiquetas)

    resultados_k.append({
        "k": k,
        "inercia": modelo.inertia_,
        "silhouette": silhouette
    })

    print(
        f"K={k} | "
        f"Inercia={modelo.inertia_:.2f} | "
        f"Silhouette={silhouette:.4f}"
    )


evaluacion = pd.DataFrame(resultados_k)

archivo_evaluacion = os.path.join(
    RESULTADOS_DIR,
    "evaluacion_kmeans.csv"
)

evaluacion.to_csv(
    archivo_evaluacion,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 5. SELECCIONAR K
# ============================================================

mejor_k = int(
    evaluacion.loc[
        evaluacion["silhouette"].idxmax(),
        "k"
    ]
)

mejor_silhouette = evaluacion["silhouette"].max()

print("\nMEJOR CONFIGURACIÓN")
print("-" * 70)
print(f"Número de clusters: {mejor_k}")
print(f"Silhouette Score:   {mejor_silhouette:.4f}")


# ============================================================
# 6. ENTRENAR MODELO FINAL
# ============================================================

print("\n6. ENTRENANDO K-MEANS FINAL")
print("-" * 70)

modelo_final = KMeans(
    n_clusters=mejor_k,
    random_state=42,
    n_init=10
)

datos["cluster"] = modelo_final.fit_predict(X)

print("✓ Modelo entrenado")


# ============================================================
# 7. AGREGAR IDS
# ============================================================

datos_resultado = df.loc[
    datos.index
].copy()

datos_resultado["cluster"] = datos["cluster"]


# ============================================================
# 8. PERFIL DE LOS CLUSTERS
# ============================================================

print("\n7. PERFIL DE LOS CLUSTERS")
print("-" * 70)

perfil = (
    datos_resultado
    .groupby("cluster")
    .agg(
        entregas=("cluster", "size"),
        distancia_promedio_km=("distancia_real_km", "mean"),
        peso_promedio=("peso_carga", "mean"),
        paquetes_promedio=("cantidad_paquetes", "mean"),
        retraso_promedio_min=("minutos_retraso", "mean"),
        combustible_promedio_l=("combustible_consumido_litros", "mean"),
        costo_promedio=("costo_envio", "mean"),
        precio_combustible_promedio=(
            "precio_combustible",
            "mean"
        )
    )
    .reset_index()
)

perfil["porcentaje_entregas"] = (
    perfil["entregas"] /
    perfil["entregas"].sum() *
    100
)

perfil = perfil.sort_values(
    "porcentaje_entregas",
    ascending=False
)

print(
    perfil.to_string(
        index=False,
        formatters={
            "distancia_promedio_km": "{:.2f}".format,
            "peso_promedio": "{:.2f}".format,
            "paquetes_promedio": "{:.2f}".format,
            "retraso_promedio_min": "{:.2f}".format,
            "combustible_promedio_l": "{:.2f}".format,
            "costo_promedio": "{:.2f}".format,
            "precio_combustible_promedio": "{:.2f}".format,
            "porcentaje_entregas": "{:.2f}".format
        }
    )
)


# ============================================================
# 9. GUARDAR RESULTADOS
# ============================================================

archivo_clusters = os.path.join(
    RESULTADOS_DIR,
    "entregas_clusters.csv"
)

archivo_perfil = os.path.join(
    RESULTADOS_DIR,
    "perfil_clusters.csv"
)

datos_resultado.to_csv(
    archivo_clusters,
    index=False,
    encoding="utf-8-sig"
)

perfil.to_csv(
    archivo_perfil,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. INTERPRETACIÓN AUTOMÁTICA
# ============================================================

print("\n8. INTERPRETACIÓN DE CLUSTERS")
print("-" * 70)

for _, fila in perfil.iterrows():

    cluster = int(fila["cluster"])

    retraso = fila["retraso_promedio_min"]
    distancia = fila["distancia_promedio_km"]
    combustible = fila["combustible_promedio_l"]

    if retraso >= 30:
        riesgo = "ALTO RIESGO DE RETRASO"
    elif retraso >= 15:
        riesgo = "RIESGO MODERADO"
    else:
        riesgo = "BAJO RIESGO DE RETRASO"

    print(
        f"\nCluster {cluster}: "
        f"{fila['porcentaje_entregas']:.2f}% de las entregas"
    )

    print(
        f"  Distancia promedio: "
        f"{distancia:.2f} km"
    )

    print(
        f"  Retraso promedio: "
        f"{retraso:.2f} minutos"
    )

    print(
        f"  Combustible promedio: "
        f"{combustible:.2f} L"
    )

    print(
        f"  Clasificación operativa: {riesgo}"
    )


# ============================================================
# 11. RESUMEN
# ============================================================

print("\n" + "=" * 70)
print("                 K-MEANS COMPLETADO")
print("=" * 70)

print(f"\nClusters generados: {mejor_k}")
print(f"Silhouette Score:   {mejor_silhouette:.4f}")

print("\nArchivos generados:")

print(f"✓ {archivo_evaluacion}")
print(f"✓ {archivo_clusters}")
print(f"✓ {archivo_perfil}")

print("\n" + "=" * 70)