import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

print("\n" + "=" * 78)
print("              SIG-LOG - OPTIMIZACIÓN DE MODELOS")
print("=" * 78)
print("Unidad III: Análisis supervisado")
print("Clasificación + Regresión")
print("=" * 78)


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODELOS_DIR = os.path.join(
    BASE_DIR,
    "modelos_entrenados"
)

REPORTES_DIR = os.path.join(
    BASE_DIR,
    "reportes_modelos"
)

DATOS_DIR = os.path.join(
    BASE_DIR,
    "datos",
    "limpios"
)

os.makedirs(MODELOS_DIR, exist_ok=True)
os.makedirs(REPORTES_DIR, exist_ok=True)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def imprimir_separador():
    print("\n" + "-" * 78)


def rmse(y_real, y_pred):
    return np.sqrt(
        mean_squared_error(
            y_real,
            y_pred
        )
    )


def guardar_json(datos, ruta):
    with open(
        ruta,
        "w",
        encoding="utf-8"
    ) as archivo:

        json.dump(
            datos,
            archivo,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# PARTE I
# OPTIMIZACIÓN DE CLASIFICACIÓN
# ============================================================

print("\n")
print("=" * 78)
print("                 PARTE I - CLASIFICACIÓN")
print("=" * 78)

print("""
Objetivo:
Predecir si una entrega llegará tarde.

Variable objetivo:
entrega_tardia

0 = A tiempo
1 = Tardía
""")


# ------------------------------------------------------------
# 1. CARGA DE DATOS PROCESADOS
# ------------------------------------------------------------

print("\n1. CARGANDO DATOS DE CLASIFICACIÓN")
print("-" * 78)

X_train = joblib.load(
    os.path.join(
        MODELOS_DIR,
        "X_train.pkl"
    )
)

X_test = joblib.load(
    os.path.join(
        MODELOS_DIR,
        "X_test.pkl"
    )
)

y_train = joblib.load(
    os.path.join(
        MODELOS_DIR,
        "y_train.pkl"
    )
)

y_test = joblib.load(
    os.path.join(
        MODELOS_DIR,
        "y_test.pkl"
    )
)

print(f"X_train: {X_train.shape}")
print(f"X_test : {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test : {y_test.shape}")


# ------------------------------------------------------------
# 2. DISTRIBUCIÓN
# ------------------------------------------------------------

print("\n2. DISTRIBUCIÓN DE CLASES")
print("-" * 78)

conteo_train = pd.Series(y_train).value_counts().sort_index()

total_train = len(y_train)

for clase, cantidad in conteo_train.items():

    porcentaje = (
        cantidad /
        total_train
    ) * 100

    if clase == 0:
        nombre = "A tiempo"
    else:
        nombre = "Tardía"

    print(
        f"{clase} = {nombre:<12} "
        f"{cantidad:>5} "
        f"({porcentaje:6.2f}%)"
    )


# ------------------------------------------------------------
# 3. MODELOS BASE
# ------------------------------------------------------------

print("\n3. ENTRENANDO MODELOS BASE")
print("-" * 78)

modelos_base_clasificacion = {

    "LogisticRegression_Base":
        LogisticRegression(
            max_iter=5000,
            random_state=42
        ),

    "DecisionTree_Base":
        DecisionTreeClassifier(
            max_depth=10,
            random_state=42
        ),

    "RandomForest_Base":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
}


resultados_clasificacion = []


for nombre, modelo in modelos_base_clasificacion.items():

    print(f"\nModelo: {nombre}")

    modelo.fit(
        X_train,
        y_train
    )

    pred = modelo.predict(
        X_test
    )

    if hasattr(
        modelo,
        "predict_proba"
    ):

        prob = modelo.predict_proba(
            X_test
        )[:, 1]

    else:

        prob = pred

    accuracy = accuracy_score(
        y_test,
        pred
    )

    precision = precision_score(
        y_test,
        pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        pred,
        zero_division=0
    )

    roc = roc_auc_score(
        y_test,
        prob
    )

    resultados_clasificacion.append({

        "modelo": nombre,
        "tipo": "Base",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc

    })

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC AUC  : {roc:.4f}"
    )


# ------------------------------------------------------------
# 4. OPTIMIZACIÓN LOGISTIC REGRESSION
# ------------------------------------------------------------

print("\n")
print("=" * 78)
print("4. OPTIMIZANDO LOGISTIC REGRESSION")
print("=" * 78)

parametros_lr = {

    "C": [
        0.01,
        0.1,
        0.5,
        1,
        2,
        5,
        10
    ],

    "class_weight": [
        None,
        "balanced"
    ],

    "solver": [
        "liblinear",
        "lbfgs"
    ]

}


cv_clasificacion = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


grid_lr = GridSearchCV(

    estimator=LogisticRegression(
        max_iter=5000,
        random_state=42
    ),

    param_grid=parametros_lr,

    scoring="f1",

    cv=cv_clasificacion,

    n_jobs=-1,

    verbose=0
)


print("Buscando mejores hiperparámetros...")

grid_lr.fit(
    X_train,
    y_train
)

mejor_lr = grid_lr.best_estimator_

print("\n✓ Optimización terminada")

print(
    "Mejores parámetros:"
)

print(
    grid_lr.best_params_
)

print(
    f"Mejor F1 CV: "
    f"{grid_lr.best_score_:.4f}"
)


pred_lr = mejor_lr.predict(
    X_test
)

prob_lr = mejor_lr.predict_proba(
    X_test
)[:, 1]


resultados_clasificacion.append({

    "modelo":
        "LogisticRegression_Optimizada",

    "tipo":
        "Optimizado",

    "accuracy":
        accuracy_score(
            y_test,
            pred_lr
        ),

    "precision":
        precision_score(
            y_test,
            pred_lr,
            zero_division=0
        ),

    "recall":
        recall_score(
            y_test,
            pred_lr,
            zero_division=0
        ),

    "f1_score":
        f1_score(
            y_test,
            pred_lr,
            zero_division=0
        ),

    "roc_auc":
        roc_auc_score(
            y_test,
            prob_lr
        )

})


joblib.dump(

    mejor_lr,

    os.path.join(
        MODELOS_DIR,
        "LogisticRegression_Optimizada.pkl"
    )

)


# ------------------------------------------------------------
# 5. OPTIMIZACIÓN RANDOM FOREST CLASSIFIER
# ------------------------------------------------------------

print("\n")
print("=" * 78)
print("5. OPTIMIZANDO RANDOM FOREST")
print("=" * 78)


parametros_rf = {

    "n_estimators": [
        200,
        300
    ],

    "max_depth": [
        None,
        8,
        12,
        16
    ],

    "min_samples_split": [
        2,
        5,
        10
    ],

    "min_samples_leaf": [
        1,
        2,
        4
    ],

    "class_weight": [
        None,
        "balanced",
        "balanced_subsample"
    ]

}


grid_rf = GridSearchCV(

    estimator=RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    ),

    param_grid=parametros_rf,

    scoring="f1",

    cv=cv_clasificacion,

    n_jobs=-1,

    verbose=0
)


print(
    "Buscando mejores hiperparámetros..."
)

grid_rf.fit(
    X_train,
    y_train
)

mejor_rf = grid_rf.best_estimator_

print("\n✓ Optimización terminada")

print(
    "Mejores parámetros:"
)

print(
    grid_rf.best_params_
)

print(
    f"Mejor F1 CV: "
    f"{grid_rf.best_score_:.4f}"
)


pred_rf = mejor_rf.predict(
    X_test
)

prob_rf = mejor_rf.predict_proba(
    X_test
)[:, 1]


resultados_clasificacion.append({

    "modelo":
        "RandomForest_Optimizado",

    "tipo":
        "Optimizado",

    "accuracy":
        accuracy_score(
            y_test,
            pred_rf
        ),

    "precision":
        precision_score(
            y_test,
            pred_rf,
            zero_division=0
        ),

    "recall":
        recall_score(
            y_test,
            pred_rf,
            zero_division=0
        ),

    "f1_score":
        f1_score(
            y_test,
            pred_rf,
            zero_division=0
        ),

    "roc_auc":
        roc_auc_score(
            y_test,
            prob_rf
        )

})


joblib.dump(

    mejor_rf,

    os.path.join(
        MODELOS_DIR,
        "RandomForest_Optimizado.pkl"
    )

)


# ------------------------------------------------------------
# 6. MATRIZ DEL MEJOR MODELO OPTIMIZADO
# ------------------------------------------------------------

df_clasificacion = pd.DataFrame(
    resultados_clasificacion
)

df_clasificacion = (
    df_clasificacion
    .sort_values(
        "f1_score",
        ascending=False
    )
)


print("\n")
print("=" * 78)
print("6. RESULTADOS DE CLASIFICACIÓN")
print("=" * 78)

print(
    df_clasificacion.to_string(
        index=False
    )
)


mejor_clasificacion = (
    df_clasificacion.iloc[0]
)


print("\n")
print("MEJOR CLASIFICADOR:")
print(
    mejor_clasificacion[
        "modelo"
    ]
)

print(
    f"F1: "
    f"{mejor_clasificacion['f1_score']:.4f}"
)

print(
    f"Recall: "
    f"{mejor_clasificacion['recall']:.4f}"
)

print(
    f"ROC AUC: "
    f"{mejor_clasificacion['roc_auc']:.4f}"
)


# ------------------------------------------------------------
# 7. PREDICCIONES DETALLADAS
# ------------------------------------------------------------

print("\n7. GENERANDO PREDICCIONES")
print("-" * 78)

predicciones_clasificacion = pd.DataFrame({

    "real":
        y_test,

    "prediccion_logistic":
        pred_lr,

    "probabilidad_tardia_logistic":
        prob_lr,

    "prediccion_random_forest":
        pred_rf,

    "probabilidad_tardia_random_forest":
        prob_rf

})


predicciones_clasificacion[
    "resultado_logistic"
] = np.where(

    predicciones_clasificacion[
        "prediccion_logistic"
    ] == 1,

    "Tardía",

    "A tiempo"

)


predicciones_clasificacion[
    "nivel_riesgo_logistic"
] = pd.cut(

    predicciones_clasificacion[
        "probabilidad_tardia_logistic"
    ],

    bins=[
        -0.01,
        0.30,
        0.60,
        1.01
    ],

    labels=[
        "Bajo",
        "Medio",
        "Alto"
    ]

)


ruta_predicciones_clasificacion = os.path.join(

    REPORTES_DIR,

    "predicciones_clasificacion_optimizada.csv"

)


predicciones_clasificacion.to_csv(

    ruta_predicciones_clasificacion,

    index=False,

    encoding="utf-8-sig"

)


print(
    f"✓ {ruta_predicciones_clasificacion}"
)


# ============================================================
# PARTE II
# REGRESIÓN
# ============================================================

print("\n")
print("=" * 78)
print("                 PARTE II - REGRESIÓN")
print("=" * 78)

print("""
Objetivo:
Predecir los minutos de retraso de una entrega.

Variable objetivo:
minutos_retraso
""")


# ------------------------------------------------------------
# 8. CARGA DE DATASET PARA REGRESIÓN
# ------------------------------------------------------------

print("\n8. CARGANDO DATOS DE REGRESIÓN")
print("-" * 78)


archivo_entregas = os.path.join(

    DATOS_DIR,

    "entregas.csv"

)


df = pd.read_csv(
    archivo_entregas
)


print(
    f"Registros: {len(df):,}"
)

print(
    f"Columnas : {len(df.columns)}"
)


# ------------------------------------------------------------
# 9. FECHAS
# ------------------------------------------------------------

print("\n9. PREPARACIÓN DE VARIABLES")
print("-" * 78)


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
    df["fecha_salida_dia_semana"] >= 5
).astype(int)

df["fecha_salida_semana"] = (
    df["fecha_salida"].dt.isocalendar().week
    .astype(int)
)

df["fecha_salida_trimestre"] = (
    df["fecha_salida"].dt.quarter
)


def convertir_hora(valor):

    try:

        partes = str(valor).split(":")

        return (
            int(partes[0])
            +
            int(partes[1]) / 60
        )

    except:

        return np.nan


df["hora_salida_num"] = (

    df["hora_salida"]
    .apply(convertir_hora)

)


print(
    "✓ Características de fecha generadas"
)

print(
    "✓ Hora de salida convertida"
)


# ------------------------------------------------------------
# 10. VARIABLES DE REGRESIÓN
# ------------------------------------------------------------

objetivo_regresion = (
    "minutos_retraso"
)


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


X_reg = df[
    variables_numericas
    +
    variables_categoricas
].copy()


y_reg = pd.to_numeric(

    df[objetivo_regresion],

    errors="coerce"

)


mask = (

    X_reg.notna().any(axis=1)
    &
    y_reg.notna()

)


X_reg = X_reg.loc[mask]
y_reg = y_reg.loc[mask]


print(
    f"Variables utilizadas: "
    f"{len(X_reg.columns)}"
)

print(
    f"Registros válidos: "
    f"{len(X_reg):,}"
)


# ------------------------------------------------------------
# 11. DIVISIÓN REGRESIÓN
# ------------------------------------------------------------

from sklearn.model_selection import train_test_split

Xr_train, Xr_test, yr_train, yr_test = train_test_split(

    X_reg,

    y_reg,

    test_size=0.20,

    random_state=42

)


print("\nDivisión:")
print(
    f"Entrenamiento: {len(Xr_train):,}"
)

print(
    f"Prueba       : {len(Xr_test):,}"
)


# ------------------------------------------------------------
# 12. PREPROCESAMIENTO
# ------------------------------------------------------------

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


pipeline_numerico = Pipeline([

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

])


pipeline_categorico = Pipeline([

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

])


preprocesador_reg = ColumnTransformer([

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

])


Xr_train_proc = (
    preprocesador_reg.fit_transform(
        Xr_train
    )
)

Xr_test_proc = (
    preprocesador_reg.transform(
        Xr_test
    )
)


print(
    f"\nVariables procesadas: "
    f"{Xr_train_proc.shape[1]}"
)


joblib.dump(

    preprocesador_reg,

    os.path.join(
        MODELOS_DIR,
        "preprocesador_regresion_optimizado.pkl"
    )

)


# ------------------------------------------------------------
# 13. MODELOS BASE DE REGRESIÓN
# ------------------------------------------------------------

print("\n")
print("=" * 78)
print("13. MODELOS BASE DE REGRESIÓN")
print("=" * 78)


modelos_base_regresion = {

    "RandomForestRegressor_Base":
        RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),

    "DecisionTreeRegressor_Base":
        DecisionTreeRegressor(
            max_depth=10,
            random_state=42
        )

}


resultados_regresion = []


for nombre, modelo in modelos_base_regresion.items():

    print(
        f"\nModelo: {nombre}"
    )

    modelo.fit(
        Xr_train_proc,
        yr_train
    )

    pred = modelo.predict(
        Xr_test_proc
    )

    mae = mean_absolute_error(
        yr_test,
        pred
    )

    mse = mean_squared_error(
        yr_test,
        pred
    )

    rmse_val = np.sqrt(
        mse
    )

    r2 = r2_score(
        yr_test,
        pred
    )

    resultados_regresion.append({

        "modelo": nombre,

        "tipo": "Base",

        "MAE": mae,

        "MSE": mse,

        "RMSE": rmse_val,

        "R2": r2

    })

    print(
        f"MAE : {mae:.4f}"
    )

    print(
        f"RMSE: {rmse_val:.4f}"
    )

    print(
        f"R²  : {r2:.4f}"
    )


# ------------------------------------------------------------
# 14. OPTIMIZACIÓN RANDOM FOREST REGRESSOR
# ------------------------------------------------------------

print("\n")
print("=" * 78)
print("14. OPTIMIZANDO RANDOM FOREST REGRESSOR")
print("=" * 78)


parametros_rfr = {

    "n_estimators": [
        200,
        300
    ],

    "max_depth": [
        None,
        8,
        12,
        16
    ],

    "min_samples_split": [
        2,
        5,
        10
    ],

    "min_samples_leaf": [
        1,
        2,
        4
    ],

    "max_features": [
        "sqrt",
        1.0
    ]

}


cv_regresion = KFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)


grid_rfr = GridSearchCV(

    estimator=RandomForestRegressor(

        random_state=42,

        n_jobs=-1

    ),

    param_grid=parametros_rfr,

    scoring="neg_mean_absolute_error",

    cv=cv_regresion,

    n_jobs=-1,

    verbose=0

)


print(
    "Buscando mejores hiperparámetros..."
)

grid_rfr.fit(

    Xr_train_proc,

    yr_train

)


mejor_rfr = (
    grid_rfr.best_estimator_
)


print(
    "\n✓ Optimización terminada"
)

print(
    "Mejores parámetros:"
)

print(
    grid_rfr.best_params_
)

print(
    f"Mejor MAE CV: "
    f"{-grid_rfr.best_score_:.4f}"
)


pred_rfr = mejor_rfr.predict(

    Xr_test_proc

)


mae_rfr = mean_absolute_error(

    yr_test,

    pred_rfr

)

mse_rfr = mean_squared_error(

    yr_test,

    pred_rfr

)

rmse_rfr = np.sqrt(
    mse_rfr
)

r2_rfr = r2_score(

    yr_test,

    pred_rfr

)


resultados_regresion.append({

    "modelo":
        "RandomForestRegressor_Optimizado",

    "tipo":
        "Optimizado",

    "MAE":
        mae_rfr,

    "MSE":
        mse_rfr,

    "RMSE":
        rmse_rfr,

    "R2":
        r2_rfr

})


joblib.dump(

    mejor_rfr,

    os.path.join(

        MODELOS_DIR,

        "RandomForestRegressor_Optimizado.pkl"

    )

)


# ------------------------------------------------------------
# 15. RESULTADOS REGRESIÓN
# ------------------------------------------------------------

df_regresion = pd.DataFrame(

    resultados_regresion

)


df_regresion = (
    df_regresion
    .sort_values(
        "MAE",
        ascending=True
    )
)


print("\n")
print("=" * 78)
print("15. RESULTADOS DE REGRESIÓN")
print("=" * 78)

print(
    df_regresion.to_string(
        index=False
    )
)


mejor_regresion = (
    df_regresion.iloc[0]
)


print("\n")
print("MEJOR REGRESOR:")

print(
    mejor_regresion[
        "modelo"
    ]
)

print(
    f"MAE : "
    f"{mejor_regresion['MAE']:.4f}"
)

print(
    f"RMSE: "
    f"{mejor_regresion['RMSE']:.4f}"
)

print(
    f"R²  : "
    f"{mejor_regresion['R2']:.4f}"
)


# ------------------------------------------------------------
# 16. PREDICCIONES DETALLADAS
# ------------------------------------------------------------

print("\n16. GENERANDO PREDICCIONES DE REGRESIÓN")
print("-" * 78)


predicciones_regresion = pd.DataFrame({

    "retraso_real_minutos":
        yr_test.values,

    "retraso_predicho_minutos":
        pred_rfr

})


predicciones_regresion[
    "error_minutos"
] = (

    predicciones_regresion[
        "retraso_real_minutos"
    ]
    -
    predicciones_regresion[
        "retraso_predicho_minutos"
    ]

)


predicciones_regresion[
    "error_absoluto_minutos"
] = np.abs(

    predicciones_regresion[
        "error_minutos"
    ]

)


predicciones_regresion[
    "nivel_retraso_real"
] = pd.cut(

    predicciones_regresion[
        "retraso_real_minutos"
    ],

    bins=[

        -0.01,
        0,
        15,
        30,
        np.inf

    ],

    labels=[

        "Sin retraso",
        "Retraso leve",
        "Retraso medio",
        "Retraso alto"

    ]

)


ruta_pred_reg = os.path.join(

    REPORTES_DIR,

    "predicciones_regresion_optimizada.csv"

)


predicciones_regresion.to_csv(

    ruta_pred_reg,

    index=False,

    encoding="utf-8-sig"

)


print(
    f"✓ {ruta_pred_reg}"
)


# ------------------------------------------------------------
# 17. IMPORTANCIA DE VARIABLES
# ------------------------------------------------------------

print("\n17. IMPORTANCIA DE VARIABLES")
print("-" * 78)


try:

    nombres_variables = (
        preprocesador_reg
        .get_feature_names_out()
    )

    importancias = (
        mejor_rfr
        .feature_importances_
    )

    df_importancias = pd.DataFrame({

        "variable":
            nombres_variables,

        "importancia":
            importancias

    })

    df_importancias = (
        df_importancias
        .sort_values(
            "importancia",
            ascending=False
        )
    )

    ruta_importancias = os.path.join(

        REPORTES_DIR,

        "importancia_variables_regresion.csv"

    )

    df_importancias.to_csv(

        ruta_importancias,

        index=False,

        encoding="utf-8-sig"

    )

    print(
        f"✓ {ruta_importancias}"
    )

    print("\nTOP 15 VARIABLES:")

    print(
        df_importancias
        .head(15)
        .to_string(
            index=False
        )
    )

except Exception as error:

    print(
        "No fue posible calcular "
        "la importancia de variables."
    )

    print(error)


# ============================================================
# PARTE FINAL
# GUARDADO DE REPORTES
# ============================================================

print("\n")
print("=" * 78)
print("                  GUARDANDO REPORTES")
print("=" * 78)


# ------------------------------------------------------------
# 18. COMPARACIÓN CLASIFICACIÓN
# ------------------------------------------------------------

ruta_comparacion_clasificacion = os.path.join(

    REPORTES_DIR,

    "comparacion_clasificacion_optimizada.csv"

)


df_clasificacion.to_csv(

    ruta_comparacion_clasificacion,

    index=False,

    encoding="utf-8-sig"

)


print(
    f"✓ {ruta_comparacion_clasificacion}"
)


# ------------------------------------------------------------
# 19. COMPARACIÓN REGRESIÓN
# ------------------------------------------------------------

ruta_comparacion_regresion = os.path.join(

    REPORTES_DIR,

    "comparacion_regresion_optimizada.csv"

)


df_regresion.to_csv(

    ruta_comparacion_regresion,

    index=False,

    encoding="utf-8-sig"

)


print(
    f"✓ {ruta_comparacion_regresion}"
)


# ------------------------------------------------------------
# 20. REPORTE GENERAL JSON
# ------------------------------------------------------------

reporte_general = {

    "proyecto":
        "SIG-LOG",

    "unidad":
        "III. Análisis supervisado",

    "clasificacion": {

        "objetivo":
            "Predecir si una entrega llegará tarde",

        "mejor_modelo":
            str(
                mejor_clasificacion[
                    "modelo"
                ]
            ),

        "accuracy":
            float(
                mejor_clasificacion[
                    "accuracy"
                ]
            ),

        "precision":
            float(
                mejor_clasificacion[
                    "precision"
                ]
            ),

        "recall":
            float(
                mejor_clasificacion[
                    "recall"
                ]
            ),

        "f1_score":
            float(
                mejor_clasificacion[
                    "f1_score"
                ]
            ),

        "roc_auc":
            float(
                mejor_clasificacion[
                    "roc_auc"
                ]
            ),

        "mejores_parametros_logistic":
            grid_lr.best_params_,

        "mejores_parametros_random_forest":
            grid_rf.best_params_

    },

    "regresion": {

        "objetivo":
            "Predecir minutos de retraso",

        "mejor_modelo":
            str(
                mejor_regresion[
                    "modelo"
                ]
            ),

        "MAE":
            float(
                mejor_regresion[
                    "MAE"
                ]
            ),

        "MSE":
            float(
                mejor_regresion[
                    "MSE"
                ]
            ),

        "RMSE":
            float(
                mejor_regresion[
                    "RMSE"
                ]
            ),

        "R2":
            float(
                mejor_regresion[
                    "R2"
                ]
            ),

        "mejores_parametros":
            grid_rfr.best_params_

    }

}


ruta_reporte_general = os.path.join(

    REPORTES_DIR,

    "reporte_optimizacion.json"

)


guardar_json(

    reporte_general,

    ruta_reporte_general

)


print(
    f"✓ {ruta_reporte_general}"
)


# ============================================================
# RESUMEN FINAL
# ============================================================

print("\n")
print("=" * 78)
print("                 OPTIMIZACIÓN COMPLETADA")
print("=" * 78)


print("\nCLASIFICACIÓN")

print(
    f"Modelo: "
    f"{mejor_clasificacion['modelo']}"
)

print(
    f"F1: "
    f"{mejor_clasificacion['f1_score']:.4f}"
)

print(
    f"Recall: "
    f"{mejor_clasificacion['recall']:.4f}"
)

print(
    f"ROC AUC: "
    f"{mejor_clasificacion['roc_auc']:.4f}"
)


print("\nREGRESIÓN")

print(
    f"Modelo: "
    f"{mejor_regresion['modelo']}"
)

print(
    f"MAE: "
    f"{mejor_regresion['MAE']:.4f} minutos"
)

print(
    f"RMSE: "
    f"{mejor_regresion['RMSE']:.4f} minutos"
)

print(
    f"R²: "
    f"{mejor_regresion['R2']:.4f}"
)


print("\nArchivos principales generados:")

print(
    "✓ LogisticRegression_Optimizada.pkl"
)

print(
    "✓ RandomForest_Optimizado.pkl"
)

print(
    "✓ RandomForestRegressor_Optimizado.pkl"
)

print(
    "✓ preprocesador_regresion_optimizado.pkl"
)

print(
    "✓ comparacion_clasificacion_optimizada.csv"
)

print(
    "✓ comparacion_regresion_optimizada.csv"
)

print(
    "✓ predicciones_clasificacion_optimizada.csv"
)

print(
    "✓ predicciones_regresion_optimizada.csv"
)

print(
    "✓ importancia_variables_regresion.csv"
)

print(
    "✓ reporte_optimizacion.json"
)

print("\n")
print("=" * 78)
print("                    SIG-LOG LISTO")
print("=" * 78)