import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

print("\n" + "=" * 75)
print("             SIG-LOG - CLASIFICACIÓN DE ENTREGAS")
print("=" * 75)
print("Unidad III: Análisis supervisado")
print("Objetivo: predecir si una entrega llegará tarde")
print("=" * 75)


# ============================================================
# 1. DIRECTORIOS
# ============================================================

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

os.makedirs(MODELOS_DIR, exist_ok=True)
os.makedirs(REPORTES_DIR, exist_ok=True)


# ============================================================
# 2. CARGAR DATOS
# ============================================================

print("\n" + "-" * 75)
print("1. CARGA DE DATOS")
print("-" * 75)

X_train_path = os.path.join(
    MODELOS_DIR,
    "X_train.pkl"
)

X_test_path = os.path.join(
    MODELOS_DIR,
    "X_test.pkl"
)

y_train_path = os.path.join(
    MODELOS_DIR,
    "y_train.pkl"
)

y_test_path = os.path.join(
    MODELOS_DIR,
    "y_test.pkl"
)


X_train = joblib.load(X_train_path)
X_test = joblib.load(X_test_path)

y_train = joblib.load(y_train_path)
y_test = joblib.load(y_test_path)


print(f"X_train: {X_train.shape}")
print(f"X_test : {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test : {y_test.shape}")


# ============================================================
# 3. DISTRIBUCIÓN DE CLASES
# ============================================================

print("\n" + "-" * 75)
print("2. DISTRIBUCIÓN DE LA VARIABLE OBJETIVO")
print("-" * 75)

train_counts = pd.Series(y_train).value_counts().sort_index()
test_counts = pd.Series(y_test).value_counts().sort_index()

print("\nENTRENAMIENTO:")

for clase, cantidad in train_counts.items():

    porcentaje = (
        cantidad /
        len(y_train)
    ) * 100

    nombre = (
        "A tiempo"
        if clase == 0
        else "Tardía"
    )

    print(
        f"  {clase} = {nombre:<12} "
        f"{cantidad:>5} "
        f"({porcentaje:6.2f}%)"
    )


print("\nPRUEBA:")

for clase, cantidad in test_counts.items():

    porcentaje = (
        cantidad /
        len(y_test)
    ) * 100

    nombre = (
        "A tiempo"
        if clase == 0
        else "Tardía"
    )

    print(
        f"  {clase} = {nombre:<12} "
        f"{cantidad:>5} "
        f"({porcentaje:6.2f}%)"
    )


# ============================================================
# 4. MODELOS
# ============================================================

print("\n" + "-" * 75)
print("3. CONFIGURACIÓN DE MODELOS")
print("-" * 75)

modelos = {

    "LogisticRegression":
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=42
        ),

    "DecisionTree":
        DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42
        ),

    "RandomForest":
        RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
}


for nombre in modelos:

    print(f"  ✓ {nombre}")


# ============================================================
# 5. ENTRENAMIENTO
# ============================================================

print("\n" + "-" * 75)
print("4. ENTRENAMIENTO DE MODELOS")
print("-" * 75)


resultados = []

predicciones = {}

probabilidades = {}

matrices = {}

reportes = {}


for nombre, modelo in modelos.items():

    print("\n" + "=" * 65)
    print(f"MODELO: {nombre}")
    print("=" * 65)

    print("Entrenando...")

    modelo.fit(
        X_train,
        y_train
    )

    print("✓ Entrenamiento completado")


    # --------------------------------------------------------
    # PREDICCIONES
    # --------------------------------------------------------

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


    predicciones[nombre] = pred
    probabilidades[nombre] = prob


    # --------------------------------------------------------
    # MÉTRICAS
    # --------------------------------------------------------

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

    roc_auc = roc_auc_score(
        y_test,
        prob
    )


    # --------------------------------------------------------
    # MATRIZ DE CONFUSIÓN
    # --------------------------------------------------------

    matriz = confusion_matrix(
        y_test,
        pred
    )

    matrices[nombre] = matriz


    # --------------------------------------------------------
    # REPORTE
    # --------------------------------------------------------

    reporte = classification_report(
        y_test,
        pred,
        target_names=[
            "A tiempo",
            "Tardía"
        ],
        zero_division=0
    )

    reportes[nombre] = reporte


    # --------------------------------------------------------
    # GUARDAR RESULTADOS
    # --------------------------------------------------------

    resultados.append({

        "modelo": nombre,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "roc_auc": roc_auc

    })


    # --------------------------------------------------------
    # MOSTRAR RESULTADOS
    # --------------------------------------------------------

    print("\nRESULTADOS")

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
        f"ROC AUC  : {roc_auc:.4f}"
    )


    # --------------------------------------------------------
    # MATRIZ
    # --------------------------------------------------------

    print("\nMATRIZ DE CONFUSIÓN")

    print(
        "                Predicción"
    )

    print(
        "              A tiempo  Tardía"
    )

    print(
        f"Real A tiempo   "
        f"{matriz[0,0]:7d}"
        f"{matriz[0,1]:8d}"
    )

    print(
        f"Real Tardía     "
        f"{matriz[1,0]:7d}"
        f"{matriz[1,1]:8d}"
    )


    # --------------------------------------------------------
    # INTERPRETACIÓN DE MATRIZ
    # --------------------------------------------------------

    tn = matriz[0, 0]
    fp = matriz[0, 1]
    fn = matriz[1, 0]
    tp = matriz[1, 1]


    print("\nINTERPRETACIÓN")

    print(
        f"  ✓ A tiempo correctamente: {tn}"
    )

    print(
        f"  ⚠ Falsas alarmas:         {fp}"
    )

    print(
        f"  ⚠ Tardías no detectadas:  {fn}"
    )

    print(
        f"  ✓ Tardías detectadas:      {tp}"
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

    print(
        f"\n✓ Modelo guardado:"
    )

    print(
        modelo_path
    )


# ============================================================
# 6. COMPARACIÓN DE MODELOS
# ============================================================

print("\n" + "=" * 75)
print("5. COMPARACIÓN DE MODELOS")
print("=" * 75)


df_resultados = pd.DataFrame(
    resultados
)


df_resultados = df_resultados.sort_values(
    by=[
        "f1_score",
        "recall",
        "roc_auc"
    ],
    ascending=False
)


print(
    df_resultados.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 7. SELECCIÓN DEL MEJOR MODELO
# ============================================================

mejor_modelo = (
    df_resultados
    .iloc[0]
    ["modelo"]
)


print("\n" + "=" * 75)
print("6. MEJOR MODELO")
print("=" * 75)

print(
    f"\nModelo seleccionado: {mejor_modelo}"
)


mejor_f1 = df_resultados.iloc[0]["f1_score"]
mejor_recall = df_resultados.iloc[0]["recall"]
mejor_auc = df_resultados.iloc[0]["roc_auc"]


print(
    f"F1 Score : {mejor_f1:.4f}"
)

print(
    f"Recall   : {mejor_recall:.4f}"
)

print(
    f"ROC AUC  : {mejor_auc:.4f}"
)


# ============================================================
# 8. GUARDAR MEJOR MODELO
# ============================================================

mejor_modelo_obj = modelos[
    mejor_modelo
]


mejor_modelo_path = os.path.join(
    MODELOS_DIR,
    "mejor_clasificador.pkl"
)


joblib.dump(
    mejor_modelo_obj,
    mejor_modelo_path
)


print(
    f"\n✓ Mejor modelo guardado:"
)

print(
    mejor_modelo_path
)


# ============================================================
# 9. PREDICCIONES DETALLADAS
# ============================================================

print("\n" + "-" * 75)
print("7. GENERANDO PREDICCIONES DETALLADAS")
print("-" * 75)


mejores_predicciones = predicciones[
    mejor_modelo
]

mejores_probabilidades = probabilidades[
    mejor_modelo
]


df_predicciones = pd.DataFrame({

    "real": np.array(
        y_test
    ),

    "prediccion": mejores_predicciones,

    "probabilidad_tardia":
        mejores_probabilidades

})


df_predicciones[
    "resultado"
] = np.where(

    df_predicciones[
        "prediccion"
    ] == 1,

    "Tardía",

    "A tiempo"

)


df_predicciones[
    "nivel_riesgo"
] = pd.cut(

    df_predicciones[
        "probabilidad_tardia"
    ],

    bins=[
        -0.01,
        0.30,
        0.60,
        0.80,
        1.01
    ],

    labels=[
        "Bajo",
        "Medio",
        "Alto",
        "Crítico"
    ]
)


predicciones_path = os.path.join(
    REPORTES_DIR,
    "predicciones_clasificacion.csv"
)


df_predicciones.to_csv(
    predicciones_path,
    index=False,
    encoding="utf-8-sig"
)


print(
    f"✓ Predicciones guardadas:"
)

print(
    predicciones_path
)


# ============================================================
# 10. REPORTE DE COMPARACIÓN
# ============================================================

comparacion_path = os.path.join(
    REPORTES_DIR,
    "comparacion_modelos.csv"
)


df_resultados.to_csv(
    comparacion_path,
    index=False,
    encoding="utf-8-sig"
)


print(
    f"\n✓ Comparación guardada:"
)

print(
    comparacion_path
)


# ============================================================
# 11. REPORTE JSON
# ============================================================

reporte_final = {

    "proyecto":
        "SIG-LOG",

    "objetivo":
        "Predicción de entregas tardías",

    "variable_objetivo":
        "entrega_tardia",

    "clases": {

        "0":
            "A tiempo",

        "1":
            "Tardía"

    },

    "registros_entrenamiento":
        int(len(y_train)),

    "registros_prueba":
        int(len(y_test)),

    "mejor_modelo":
        mejor_modelo,

    "metricas_mejor_modelo": {

        "accuracy":
            float(
                df_resultados.iloc[0]["accuracy"]
            ),

        "precision":
            float(
                df_resultados.iloc[0]["precision"]
            ),

        "recall":
            float(
                df_resultados.iloc[0]["recall"]
            ),

        "f1_score":
            float(
                df_resultados.iloc[0]["f1_score"]
            ),

        "roc_auc":
            float(
                df_resultados.iloc[0]["roc_auc"]
            )

    },

    "modelos_evaluados":
        df_resultados.to_dict(
            orient="records"
        )

}


json_path = os.path.join(
    REPORTES_DIR,
    "reporte_clasificacion.json"
)


with open(
    json_path,
    "w",
    encoding="utf-8"
) as archivo:

    json.dump(
        reporte_final,
        archivo,
        indent=4,
        ensure_ascii=False
    )


print(
    f"\n✓ Reporte JSON guardado:"
)

print(
    json_path
)


# ============================================================
# 12. RESUMEN FINAL
# ============================================================

print("\n" + "=" * 75)
print("                 CLASIFICACIÓN COMPLETADA")
print("=" * 75)

print(
    f"\nModelo seleccionado: {mejor_modelo}"
)

print(
    f"F1 Score: {mejor_f1:.4f}"
)

print(
    f"Recall:   {mejor_recall:.4f}"
)

print(
    f"ROC AUC:  {mejor_auc:.4f}"
)

print("\nArchivos generados:")

print(
    "✓ LogisticRegression.pkl"
)

print(
    "✓ DecisionTree.pkl"
)

print(
    "✓ RandomForest.pkl"
)

print(
    "✓ mejor_clasificador.pkl"
)

print(
    "✓ comparacion_modelos.csv"
)

print(
    "✓ predicciones_clasificacion.csv"
)

print(
    "✓ reporte_clasificacion.json"
)

print("\n" + "=" * 75)