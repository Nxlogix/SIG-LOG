import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("\n" + "=" * 75)
print("             SIG-LOG - MODELOS DE REGRESIÓN")
print("=" * 75)
print("Unidad III: Análisis supervisado")
print("Objetivo: predecir los minutos de retraso de una entrega")
print("=" * 75)


# ============================================================
# 1. DIRECTORIOS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATOS_DIR = os.path.join(
    BASE_DIR,
    "datos",
    "limpios"
)

MODELOS_DIR = os.path.join(
    BASE_DIR,
    "modelos_entrenados"
)

REPORTES_DIR = os.path.join(
    BASE_DIR,
    "reportes_modelos"
)

os.makedirs(MODELOS_DIR, exist_ok=True)
os.makedirs(REPORTES_DIR, exist_ok=True)


# ============================================================
# 2. CARGA DEL DATASET ORIGINAL
# ============================================================

ARCHIVO = os.path.join(
    DATOS_DIR,
    "entregas.csv"
)

print("\n" + "-" * 75)
print("1. CARGA DEL DATASET")
print("-" * 75)

print(f"Archivo:")
print(ARCHIVO)

df = pd.read_csv(ARCHIVO)

print(f"\nRegistros cargados : {len(df):,}")
print(f"Columnas           : {len(df.columns)}")


# ============================================================
# 3. VALIDACIÓN
# ============================================================

print("\n" + "-" * 75)
print("2. VALIDACIÓN DEL DATASET")
print("-" * 75)

print(
    f"Valores nulos : {df.isnull().sum().sum():,}"
)

print(
    f"Duplicados    : {df.duplicated().sum():,}"
)


# ============================================================
# 4. VARIABLE OBJETIVO
# ============================================================

OBJETIVO = "minutos_retraso"

print("\n" + "-" * 75)
print("3. VARIABLE OBJETIVO")
print("-" * 75)

if OBJETIVO not in df.columns:
    raise ValueError(
        f"No existe la columna '{OBJETIVO}' en el dataset."
    )

print(f"Variable a predecir: {OBJETIVO}")


# ============================================================
# 5. CONVERSIÓN DE DATOS
# ============================================================

print("\n" + "-" * 75)
print("4. PREPARACIÓN DE VARIABLES")
print("-" * 75)

# Convertir fecha de salida
if "fecha_salida" in df.columns:

    df["fecha_salida"] = pd.to_datetime(
        df["fecha_salida"],
        errors="coerce"
    )

    df["fecha_salida_anio"] = (
        df["fecha_salida"].dt.year
    )

    df["fecha_salida_mes"] = (
        df["fecha_salida"].dt.month
    )

    df["fecha_salida_dia"] = (
        df["fecha_salida"].dt.day
    )

    df["fecha_salida_dia_semana"] = (
        df["fecha_salida"].dt.dayofweek
    )

    df["fecha_salida_fin_semana"] = (
        df["fecha_salida"].dt.dayofweek >= 5
    ).astype(int)

    df["fecha_salida_semana"] = (
        df["fecha_salida"].dt.isocalendar().week.astype(int)
    )

    df["fecha_salida_trimestre"] = (
        df["fecha_salida"].dt.quarter
    )

    print("✓ Características de fecha generadas")


# ============================================================
# 6. CONVERSIÓN DE HORA
# ============================================================

if "hora_salida" in df.columns:

    def convertir_hora(hora):

        try:

            texto = str(hora)

            partes = texto.split(":")

            horas = int(partes[0])
            minutos = int(partes[1])

            return horas + (minutos / 60)

        except:

            return np.nan


    df["hora_salida_num"] = (
        df["hora_salida"].apply(convertir_hora)
    )

    print("✓ Hora de salida convertida")


# ============================================================
# 7. VARIABLES PERMITIDAS
# ============================================================

print("\n" + "-" * 75)
print("5. SELECCIÓN DE VARIABLES")
print("-" * 75)


# Estas variables representan información disponible
# antes o durante la preparación/salida de la entrega.

variables_numericas = [
    "peso_carga",
    "cantidad_paquetes",
    "distancia_real_km",
    "precio_combustible",
    "costo_envio",

    "fecha_salida_anio",
    "fecha_salida_mes",
    "fecha_salida_dia",
    "fecha_salida_dia_semana",
    "fecha_salida_fin_semana",
    "fecha_salida_semana",
    "fecha_salida_trimestre",

    "hora_salida_num"
]


variables_categoricas = [
    "tipo_combustible"
]


# Mantener solamente las columnas existentes

variables_numericas = [
    col
    for col in variables_numericas
    if col in df.columns
]

variables_categoricas = [
    col
    for col in variables_categoricas
    if col in df.columns
]


print("\nVARIABLE OBJETIVO")
print(f"  ✓ {OBJETIVO}")


print("\nVARIABLES NUMÉRICAS")

for col in variables_numericas:

    print(f"  ✓ {col}")


print("\nVARIABLES CATEGÓRICAS")

for col in variables_categoricas:

    print(f"  ✓ {col}")


# ============================================================
# 8. ELIMINACIÓN DE REGISTROS SIN OBJETIVO
# ============================================================

df = df.dropna(
    subset=[OBJETIVO]
).copy()


# ============================================================
# 9. LIMPIEZA DEL OBJETIVO
# ============================================================

df[OBJETIVO] = pd.to_numeric(
    df[OBJETIVO],
    errors="coerce"
)

df = df.dropna(
    subset=[OBJETIVO]
).copy()


print("\nRegistros después de limpieza:")
print(f"{len(df):,}")


# ============================================================
# 10. VARIABLES X E Y
# ============================================================

X = df[
    variables_numericas +
    variables_categoricas
].copy()

y = df[
    OBJETIVO
].copy()


# ============================================================
# 11. INFORMACIÓN DE LA VARIABLE OBJETIVO
# ============================================================

print("\n" + "-" * 75)
print("6. ANÁLISIS DE LA VARIABLE OBJETIVO")
print("-" * 75)

print(
    f"Promedio de retraso : {y.mean():.2f} minutos"
)

print(
    f"Mediana              : {y.median():.2f} minutos"
)

print(
    f"Mínimo               : {y.min():.2f} minutos"
)

print(
    f"Máximo               : {y.max():.2f} minutos"
)

print(
    f"Desviación estándar  : {y.std():.2f} minutos"
)


# ============================================================
# 12. DIVISIÓN TRAIN / TEST
# ============================================================

from sklearn.model_selection import train_test_split


print("\n" + "-" * 75)
print("7. DIVISIÓN TRAIN / TEST")
print("-" * 75)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print(
    f"Registros totales : {len(X):,}"
)

print(
    f"Entrenamiento     : {len(X_train):,} (80%)"
)

print(
    f"Prueba            : {len(X_test):,} (20%)"
)


# ============================================================
# 13. PREPROCESAMIENTO
# ============================================================

print("\n" + "-" * 75)
print("8. CONSTRUCCIÓN DEL PIPELINE")
print("-" * 75)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)


pipeline_numerico = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


pipeline_categorico = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocesador = ColumnTransformer(
    transformers=[
        (
            "numericas",
            pipeline_numerico,
            variables_numericas
        ),
        (
            "categoricas",
            pipeline_categorico,
            variables_categoricas
        )
    ]
)


print("✓ Pipeline numérico")
print("  - Imputación por mediana")
print("  - StandardScaler")

print("\n✓ Pipeline categórico")
print("  - Imputación por moda")
print("  - OneHotEncoder")


# ============================================================
# 14. TRANSFORMACIÓN
# ============================================================

print("\n" + "-" * 75)
print("9. PREPROCESAMIENTO")
print("-" * 75)


X_train_procesado = preprocesador.fit_transform(
    X_train
)

X_test_procesado = preprocesador.transform(
    X_test
)


print(
    f"Variables originales : {X.shape[1]}"
)

print(
    f"Variables procesadas : {X_train_procesado.shape[1]}"
)

print(
    f"Train procesado      : {X_train_procesado.shape[0]:,}"
)

print(
    f"Test procesado       : {X_test_procesado.shape[0]:,}"
)


# ============================================================
# 15. GUARDAR PREPROCESADOR
# ============================================================

preprocesador_path = os.path.join(
    MODELOS_DIR,
    "preprocesador_regresion.pkl"
)

joblib.dump(
    preprocesador,
    preprocesador_path
)

print("\n✓ Preprocesador guardado:")
print(preprocesador_path)


# ============================================================
# 16. MODELOS
# ============================================================

print("\n" + "-" * 75)
print("10. CONFIGURACIÓN DE MODELOS")
print("-" * 75)


modelos = {

    "LinearRegression":
        LinearRegression(),

    "DecisionTreeRegressor":
        DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ),

    "RandomForestRegressor":
        RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
}


for nombre in modelos:

    print(f"  ✓ {nombre}")


# ============================================================
# 17. ENTRENAMIENTO
# ============================================================

print("\n" + "-" * 75)
print("11. ENTRENAMIENTO DE MODELOS")
print("-" * 75)


resultados = []

predicciones_modelos = {}


for nombre, modelo in modelos.items():

    print("\n")
    print("=" * 65)
    print(f"MODELO: {nombre}")
    print("=" * 65)

    print("Entrenando...")

    modelo.fit(
        X_train_procesado,
        y_train
    )

    print("✓ Entrenamiento completado")


    # --------------------------------------------------------
    # PREDICCIONES
    # --------------------------------------------------------

    pred = modelo.predict(
        X_test_procesado
    )


    # Evitar retrasos negativos
    pred = np.maximum(
        pred,
        0
    )


    predicciones_modelos[nombre] = pred


    # --------------------------------------------------------
    # MÉTRICAS
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        pred
    )

    mse = mean_squared_error(
        y_test,
        pred
    )

    rmse = np.sqrt(
        mse
    )

    r2 = r2_score(
        y_test,
        pred
    )


    resultados.append({

        "modelo": nombre,

        "MAE": mae,

        "MSE": mse,

        "RMSE": rmse,

        "R2": r2

    })


    # --------------------------------------------------------
    # RESULTADOS
    # --------------------------------------------------------

    print("\nRESULTADOS")

    print(
        f"MAE  : {mae:.4f} minutos"
    )

    print(
        f"MSE  : {mse:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f} minutos"
    )

    print(
        f"R²   : {r2:.4f}"
    )


    # --------------------------------------------------------
    # GUARDAR MODELO
    # --------------------------------------------------------

    modelo_path = os.path.join(
        MODELOS_DIR,
        f"{nombre}.pkl"
    )

    joblib.dump(
        modelo,
        modelo_path
    )

    print("\n✓ Modelo guardado:")
    print(modelo_path)


# ============================================================
# 18. COMPARACIÓN
# ============================================================

print("\n" + "=" * 75)
print("12. COMPARACIÓN DE MODELOS")
print("=" * 75)


df_resultados = pd.DataFrame(
    resultados
)


# Para regresión:
# menor MAE = mejor
# menor RMSE = mejor
# mayor R² = mejor

df_resultados = df_resultados.sort_values(
    by="RMSE",
    ascending=True
).reset_index(
    drop=True
)


print(
    df_resultados.to_string(
        index=False
    )
)


# ============================================================
# 19. GUARDAR COMPARACIÓN
# ============================================================

comparacion_path = os.path.join(
    REPORTES_DIR,
    "comparacion_regresion.csv"
)

df_resultados.to_csv(
    comparacion_path,
    index=False
)

print("\n✓ Comparación guardada:")
print(comparacion_path)


# ============================================================
# 20. SELECCIÓN DEL MEJOR MODELO
# ============================================================

mejor_modelo_nombre = (
    df_resultados.iloc[0]["modelo"]
)


mejor_modelo = modelos[
    mejor_modelo_nombre
]


mejor_pred = predicciones_modelos[
    mejor_modelo_nombre
]


mejor_modelo_path = os.path.join(
    MODELOS_DIR,
    "mejor_regresor.pkl"
)


joblib.dump(
    mejor_modelo,
    mejor_modelo_path
)


print("\n" + "=" * 75)
print("13. MEJOR MODELO")
print("=" * 75)

print(
    f"\nModelo seleccionado: "
    f"{mejor_modelo_nombre}"
)


mejores_metricas = df_resultados.iloc[0]


print(
    f"MAE  : {mejores_metricas['MAE']:.4f}"
)

print(
    f"RMSE : {mejores_metricas['RMSE']:.4f}"
)

print(
    f"R²   : {mejores_metricas['R2']:.4f}"
)


print("\n✓ Mejor modelo guardado:")
print(mejor_modelo_path)


# ============================================================
# 21. PREDICCIONES DETALLADAS
# ============================================================

print("\n" + "-" * 75)
print("14. GENERANDO PREDICCIONES DETALLADAS")
print("-" * 75)


predicciones = X_test.copy()


predicciones["minutos_retraso_real"] = (
    y_test.values
)


predicciones[
    "minutos_retraso_predicho"
] = mejor_pred


predicciones[
    "error_minutos"
] = (
    predicciones[
        "minutos_retraso_real"
    ]
    -
    predicciones[
        "minutos_retraso_predicho"
    ]
)


predicciones[
    "error_absoluto"
] = (
    predicciones[
        "error_minutos"
    ].abs()
)


# Clasificación adicional para facilitar
# interpretación en reportes.

predicciones[
    "nivel_retraso_real"
] = pd.cut(
    predicciones[
        "minutos_retraso_real"
    ],
    bins=[
        -0.01,
        0,
        15,
        30,
        60,
        np.inf
    ],
    labels=[
        "Sin retraso",
        "Retraso leve",
        "Retraso moderado",
        "Retraso alto",
        "Retraso crítico"
    ]
)


predicciones[
    "nivel_retraso_predicho"
] = pd.cut(
    predicciones[
        "minutos_retraso_predicho"
    ],
    bins=[
        -0.01,
        0,
        15,
        30,
        60,
        np.inf
    ],
    labels=[
        "Sin retraso",
        "Retraso leve",
        "Retraso moderado",
        "Retraso alto",
        "Retraso crítico"
    ]
)


predicciones_path = os.path.join(
    REPORTES_DIR,
    "predicciones_regresion.csv"
)


predicciones.to_csv(
    predicciones_path,
    index=False
)


print("✓ Predicciones guardadas:")
print(predicciones_path)


# ============================================================
# 22. REPORTE JSON
# ============================================================

print("\n" + "-" * 75)
print("15. GENERACIÓN DEL REPORTE")
print("-" * 75)


reporte = {

    "proyecto": "SIG-LOG",

    "unidad": "III. Análisis supervisado",

    "tipo_modelo": "Regresión",

    "objetivo":
        "Predecir los minutos de retraso de una entrega",

    "variable_objetivo":
        OBJETIVO,

    "registros_totales":
        int(len(df)),

    "registros_entrenamiento":
        int(len(X_train)),

    "registros_prueba":
        int(len(X_test)),

    "variables_predictoras":
        variables_numericas +
        variables_categoricas,

    "numero_variables_originales":
        int(X.shape[1]),

    "numero_variables_procesadas":
        int(X_train_procesado.shape[1]),

    "modelos_evaluados": [],

    "mejor_modelo":
        mejor_modelo_nombre,

    "criterio_seleccion":
        "Menor RMSE",

    "interpretacion":

        "El modelo estima la cantidad de minutos "
        "de retraso utilizando información disponible "
        "antes o durante la preparación de la entrega."

}


for _, fila in df_resultados.iterrows():

    reporte[
        "modelos_evaluados"
    ].append({

        "modelo":
            fila["modelo"],

        "MAE":
            float(fila["MAE"]),

        "MSE":
            float(fila["MSE"]),

        "RMSE":
            float(fila["RMSE"]),

        "R2":
            float(fila["R2"])

    })


reporte_path = os.path.join(
    REPORTES_DIR,
    "reporte_regresion.json"
)


with open(
    reporte_path,
    "w",
    encoding="utf-8"
) as archivo:

    json.dump(
        reporte,
        archivo,
        indent=4,
        ensure_ascii=False
    )


print("✓ Reporte JSON guardado:")
print(reporte_path)


# ============================================================
# 23. RESUMEN FINAL
# ============================================================

print("\n")
print("=" * 75)
print("             REGRESIÓN COMPLETADA")
print("=" * 75)


print(
    f"\nModelo seleccionado:"
)

print(
    f"  {mejor_modelo_nombre}"
)


print(
    f"\nMétricas:"
)

print(
    f"  MAE  : {mejores_metricas['MAE']:.4f} minutos"
)

print(
    f"  RMSE : {mejores_metricas['RMSE']:.4f} minutos"
)

print(
    f"  R²   : {mejores_metricas['R2']:.4f}"
)


print("\nArchivos generados:")

print(
    "✓ preprocesador_regresion.pkl"
)

print(
    "✓ LinearRegression.pkl"
)

print(
    "✓ DecisionTreeRegressor.pkl"
)

print(
    "✓ RandomForestRegressor.pkl"
)

print(
    "✓ mejor_regresor.pkl"
)

print(
    "✓ comparacion_regresion.csv"
)

print(
    "✓ predicciones_regresion.csv"
)

print(
    "✓ reporte_regresion.json"
)


print("\n" + "=" * 75)