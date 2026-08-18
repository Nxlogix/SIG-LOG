# ============================================================
# SIG-LOG
# Sistema Integral de Gestión Logística
#
# Archivo: entrenamiento.py
# Unidad III: Análisis supervisado
#
# OBJETIVO:
# Preparar los datos para modelos de Machine Learning.
#
# MODELO DE CLASIFICACIÓN:
# Predecir si una entrega llegará tarde.
#
# 0 = A tiempo
# 1 = Tardía
#
# ============================================================

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CONFIGURACIÓN
# ============================================================

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent

DATOS_DIR = BASE_DIR / "datos" / "limpios"

MODELOS_DIR = BASE_DIR / "modelos_entrenados"

REPORTES_DIR = BASE_DIR / "reportes_modelos"

MODELOS_DIR.mkdir(parents=True, exist_ok=True)
REPORTES_DIR.mkdir(parents=True, exist_ok=True)

ARCHIVO_ENTREGAS = DATOS_DIR / "entregas.csv"

RANDOM_STATE = 42

TEST_SIZE = 0.20


# ============================================================
# FUNCIONES DE PRESENTACIÓN
# ============================================================

def imprimir_titulo(texto):

    print()
    print("=" * 70)
    print(texto.center(70))
    print("=" * 70)


def imprimir_subtitulo(texto):

    print()
    print("-" * 70)
    print(texto)
    print("-" * 70)


# ============================================================
# 1. CARGAR DATOS
# ============================================================

def cargar_datos():

    imprimir_subtitulo("1. CARGA DEL DATASET")

    if not ARCHIVO_ENTREGAS.exists():

        raise FileNotFoundError(
            f"No se encontró el archivo:\n{ARCHIVO_ENTREGAS}"
        )

    df = pd.read_csv(
        ARCHIVO_ENTREGAS,
        encoding="utf-8-sig"
    )

    print(
        f"Archivo: {ARCHIVO_ENTREGAS}"
    )

    print(
        f"Registros cargados: {len(df):,}"
    )

    print(
        f"Columnas encontradas: {len(df.columns)}"
    )

    return df


# ============================================================
# 2. INFORMACIÓN GENERAL
# ============================================================

def mostrar_informacion(df):

    imprimir_subtitulo(
        "2. INFORMACIÓN GENERAL DEL DATASET"
    )

    print(
        f"Registros : {len(df):,}"
    )

    print(
        f"Columnas  : {len(df.columns)}"
    )

    print()

    print("Columnas disponibles:")

    for i, columna in enumerate(
        df.columns,
        start=1
    ):

        print(
            f"  {i:02d}. {columna}"
        )


# ============================================================
# 3. VALIDACIÓN
# ============================================================

def validar_dataset(df):

    imprimir_subtitulo(
        "3. VALIDACIÓN DEL DATASET"
    )

    nulos = int(
        df.isnull().sum().sum()
    )

    duplicados = int(
        df.duplicated().sum()
    )

    print(
        f"Valores nulos totales : {nulos}"
    )

    print(
        f"Duplicados            : {duplicados}"
    )

    if nulos > 0:

        print(
            "Advertencia: existen valores nulos."
        )

    if duplicados > 0:

        print(
            "Advertencia: existen registros duplicados."
        )


# ============================================================
# 4. INGENIERÍA DE CARACTERÍSTICAS
# ============================================================

def crear_caracteristicas_fecha_hora(df):

    imprimir_subtitulo(
        "4. INGENIERÍA DE CARACTERÍSTICAS"
    )

    df = df.copy()

    # --------------------------------------------------------
    # FECHA DE SALIDA
    # --------------------------------------------------------

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
            df["fecha_salida"].dt.isocalendar().week
            .astype(int)
        )

        df["fecha_salida_trimestre"] = (
            df["fecha_salida"].dt.quarter
        )

        print(
            "Fecha utilizada: fecha_salida"
        )

        print(
            "Características de fecha generadas correctamente."
        )

    # --------------------------------------------------------
    # HORA DE SALIDA
    # --------------------------------------------------------

    if "hora_salida" in df.columns:

        hora = df["hora_salida"].astype(str)

        # Intentar formato HH:MM
        hora_convertida = pd.to_datetime(
            hora,
            format="%H:%M",
            errors="coerce"
        )

        # Intentar formato HH:MM:SS
        if hora_convertida.isna().all():

            hora_convertida = pd.to_datetime(
                hora,
                format="%H:%M:%S",
                errors="coerce"
            )

        if not hora_convertida.isna().all():

            df["hora_salida_num"] = (
                hora_convertida.dt.hour
                +
                hora_convertida.dt.minute / 60
            )

            print(
                "Hora utilizada: hora_salida"
            )

            print(
                "Variable hora_salida_num generada."
            )

        else:

            numerica = pd.to_numeric(
                df["hora_salida"],
                errors="coerce"
            )

            df["hora_salida_num"] = numerica

            print(
                "Hora convertida a variable numérica."
            )

    return df


# ============================================================
# 5. CREAR VARIABLE OBJETIVO
# ============================================================

def crear_variable_objetivo(df):

    imprimir_subtitulo(
        "5. VARIABLE OBJETIVO DE CLASIFICACIÓN"
    )

    df = df.copy()

    if "entrega_tardia" not in df.columns:

        if "minutos_retraso" not in df.columns:

            raise ValueError(
                "No existe entrega_tardia ni minutos_retraso."
            )

        df["minutos_retraso"] = pd.to_numeric(
            df["minutos_retraso"],
            errors="coerce"
        )

        df["entrega_tardia"] = (
            df["minutos_retraso"] > 0
        ).astype(int)

        print(
            "Variable entrega_tardia creada."
        )

    else:

        df["entrega_tardia"] = pd.to_numeric(
            df["entrega_tardia"],
            errors="coerce"
        )

        df["entrega_tardia"] = (
            df["entrega_tardia"]
            .fillna(0)
            .astype(int)
        )

        print(
            "La variable 'entrega_tardia' ya existe."
        )

    return df


# ============================================================
# 6. DETECCIÓN DE VARIABLES
# ============================================================

def detectar_variables(df):

    imprimir_subtitulo(
        "6. DETECCIÓN DE VARIABLES"
    )

    # --------------------------------------------------------
    # VARIABLE OBJETIVO
    # --------------------------------------------------------

    objetivo = "entrega_tardia"

    # --------------------------------------------------------
    # VARIABLES QUE NO SE DEBEN USAR
    # --------------------------------------------------------

    columnas_excluir = {

        # Objetivo
        "entrega_tardia",

        # IDs
        "id_entrega",
        "id_cliente",
        "id_vehiculo",
        "id_operador",
        "id_ruta",
        "id_componente",

        # Fechas originales
        "fecha_salida",
        "fecha_entrega_programada",
        "fecha_entrega_real",

        # Horas originales
        "hora_salida",
        "hora_entrega_programada",
        "hora_entrega_real",

        # Datos que solamente conocemos después
        # de realizar la entrega
        "minutos_retraso",
        "estatus",
        "observaciones",

        # Datos de combustible que pueden depender
        # de la operación ya realizada
        "combustible_consumido_litros",
        "costo_combustible",
        "costo_total",

    }

    # --------------------------------------------------------
    # COLUMNAS QUE SÍ QUEREMOS UTILIZAR
    # --------------------------------------------------------

    variables_pre_entrega = [

        # Características de la carga
        "peso_carga",
        "cantidad_paquetes",

        # Características de la ruta
        "distancia_real_km",

        # Combustible planeado / tipo
        "tipo_combustible",
        "precio_combustible",

        # Costo planeado del servicio
        "costo_envio",

        # Variables derivadas de fecha
        "fecha_salida_anio",
        "fecha_salida_mes",
        "fecha_salida_dia",
        "fecha_salida_dia_semana",
        "fecha_salida_fin_semana",
        "fecha_salida_semana",
        "fecha_salida_trimestre",

        # Hora de salida convertida
        "hora_salida_num",
    ]

    # --------------------------------------------------------
    # SOLO USAR COLUMNAS EXISTENTES
    # --------------------------------------------------------

    columnas_predictoras = [

        columna
        for columna in variables_pre_entrega
        if columna in df.columns
    ]

    # --------------------------------------------------------
    # VALIDACIÓN
    # --------------------------------------------------------

    if len(columnas_predictoras) == 0:

        raise ValueError(
            "No se encontraron variables predictoras."
        )

    X = df[columnas_predictoras].copy()

    y = df[objetivo].copy()

    # --------------------------------------------------------
    # TIPOS
    # --------------------------------------------------------

    columnas_numericas = list(
        X.select_dtypes(
            include=[np.number]
        ).columns
    )

    columnas_categoricas = list(
        X.select_dtypes(
            include=[
                "object",
                "string",
                "category",
                "bool"
            ]
        ).columns
    )

    # --------------------------------------------------------
    # MOSTRAR INFORMACIÓN
    # --------------------------------------------------------

    print()

    print(
        "VARIABLE OBJETIVO"
    )

    print(
        "  entrega_tardia"
    )

    print()

    print(
        "VARIABLES PREDICTORAS"
    )

    print(
        f"  Total: {len(columnas_predictoras)}"
    )

    print()

    print(
        "Variables numéricas:"
    )

    for columna in columnas_numericas:

        print(
            f"  ✓ {columna}"
        )

    print()

    print(
        "Variables categóricas:"
    )

    for columna in columnas_categoricas:

        print(
            f"  ✓ {columna}"
        )

    print()

    print(
        "VARIABLES EXCLUIDAS"
    )

    for columna in sorted(columnas_excluir):

        if columna in df.columns:

            print(
                f"  - {columna}"
            )

    return (
        X,
        y,
        columnas_numericas,
        columnas_categoricas,
        columnas_excluir
    )


# ============================================================
# 7. DISTRIBUCIÓN DEL OBJETIVO
# ============================================================

def analizar_objetivo(y):

    imprimir_subtitulo(
        "7. DISTRIBUCIÓN DE LA VARIABLE OBJETIVO"
    )

    conteo = (
        y.value_counts()
        .sort_index()
    )

    total = len(y)

    for valor in [0, 1]:

        cantidad = int(
            conteo.get(valor, 0)
        )

        porcentaje = (
            cantidad / total * 100
            if total > 0
            else 0
        )

        nombre = (
            "A tiempo"
            if valor == 0
            else "Tardía"
        )

        print(
            f"{nombre:<15}: "
            f"{cantidad:>6,} "
            f"({porcentaje:>6.2f}%)"
        )


# ============================================================
# 8. DIVISIÓN TRAIN / TEST
# ============================================================

def dividir_datos(X, y):

    imprimir_subtitulo(
        "8. DIVISIÓN TRAIN / TEST"
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    print(
        f"Registros totales      : {len(X):,}"
    )

    print(
        f"Entrenamiento          : "
        f"{len(X_train):,} "
        f"({len(X_train)/len(X)*100:.2f}%)"
    )

    print(
        f"Prueba                 : "
        f"{len(X_test):,} "
        f"({len(X_test)/len(X)*100:.2f}%)"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# 9. CONSTRUIR PREPROCESADOR
# ============================================================

def construir_preprocesador(
    columnas_numericas,
    columnas_categoricas
):

    imprimir_subtitulo(
        "9. CONSTRUCCIÓN DEL PIPELINE"
    )

    # --------------------------------------------------------
    # PIPELINE NUMÉRICO
    # --------------------------------------------------------

    pipeline_numerico = Pipeline(
        steps=[

            (
                "imputador",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "escalador",
                StandardScaler()
            )

        ]
    )

    # --------------------------------------------------------
    # PIPELINE CATEGÓRICO
    # --------------------------------------------------------

    pipeline_categorico = Pipeline(
        steps=[

            (
                "imputador",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )

        ]
    )

    # --------------------------------------------------------
    # TRANSFORMADOR
    # --------------------------------------------------------

    transformadores = []

    if columnas_numericas:

        transformadores.append(
            (
                "numericas",
                pipeline_numerico,
                columnas_numericas
            )
        )

    if columnas_categoricas:

        transformadores.append(
            (
                "categoricas",
                pipeline_categorico,
                columnas_categoricas
            )
        )

    preprocesador = ColumnTransformer(
        transformers=transformadores,
        remainder="drop"
    )

    print(
        "Pipeline numérico:"
    )

    print(
        "  ✓ Imputación por mediana"
    )

    print(
        "  ✓ StandardScaler"
    )

    print()

    print(
        "Pipeline categórico:"
    )

    print(
        "  ✓ Imputación por moda"
    )

    print(
        "  ✓ OneHotEncoder"
    )

    print()

    print(
        "Pipeline construido correctamente."
    )

    return preprocesador


# ============================================================
# 10. PREPROCESAR
# ============================================================

def ajustar_preprocesador(
    preprocesador,
    X_train,
    X_test
):

    imprimir_subtitulo(
        "10. PREPROCESAMIENTO"
    )

    # IMPORTANTE:
    # Solo se ajusta con entrenamiento.

    X_train_procesado = (
        preprocesador.fit_transform(
            X_train
        )
    )

    X_test_procesado = (
        preprocesador.transform(
            X_test
        )
    )

    X_train_procesado = np.asarray(
        X_train_procesado,
        dtype=np.float64
    )

    X_test_procesado = np.asarray(
        X_test_procesado,
        dtype=np.float64
    )

    print(
        f"Dimensiones originales : "
        f"{X_train.shape[1]} variables"
    )

    print(
        f"Dimensiones procesadas : "
        f"{X_train_procesado.shape[1]} variables"
    )

    print(
        f"Train procesado        : "
        f"{X_train_procesado.shape[0]:,} registros"
    )

    print(
        f"Test procesado         : "
        f"{X_test_procesado.shape[0]:,} registros"
    )

    return (
        X_train_procesado,
        X_test_procesado
    )


# ============================================================
# 11. NOMBRES DE VARIABLES
# ============================================================

def obtener_nombres_variables(
    preprocesador
):

    try:

        nombres = (
            preprocesador
            .get_feature_names_out()
            .tolist()
        )

    except Exception:

        nombres = []

    return nombres


# ============================================================
# 12. GUARDAR ARTEFACTOS
# ============================================================

def guardar_objetos(
    preprocesador,
    X_train,
    X_test,
    y_train,
    y_test,
    nombres_variables,
    columnas_originales,
    columnas_numericas,
    columnas_categoricas
):

    imprimir_subtitulo(
        "11. GUARDADO DE ARTEFACTOS"
    )

    archivo_preprocesador = (
        MODELOS_DIR /
        "preprocesador_entregas.pkl"
    )

    joblib.dump(
        preprocesador,
        archivo_preprocesador
    )

    joblib.dump(
        X_train,
        MODELOS_DIR /
        "X_train.pkl"
    )

    joblib.dump(
        X_test,
        MODELOS_DIR /
        "X_test.pkl"
    )

    joblib.dump(
        y_train,
        MODELOS_DIR /
        "y_train.pkl"
    )

    joblib.dump(
        y_test,
        MODELOS_DIR /
        "y_test.pkl"
    )

    joblib.dump(
        nombres_variables,
        MODELOS_DIR /
        "nombres_variables.pkl"
    )

    metadata = {

        "proyecto":
            "SIG-LOG",

        "descripcion":
            "Sistema Integral de Gestión Logística",

        "modelo_objetivo":
            "clasificacion",

        "variable_objetivo":
            "entrega_tardia",

        "clases": {

            "0":
                "A tiempo",

            "1":
                "Tardía"

        },

        "random_state":
            RANDOM_STATE,

        "test_size":
            TEST_SIZE,

        "columnas_originales":
            columnas_originales,

        "columnas_numericas":
            columnas_numericas,

        "columnas_categoricas":
            columnas_categoricas,

        "nombres_variables_procesadas":
            nombres_variables,

        "total_variables_procesadas":
            len(nombres_variables),

        "archivos": [

            "preprocesador_entregas.pkl",

            "X_train.pkl",

            "X_test.pkl",

            "y_train.pkl",

            "y_test.pkl",

            "nombres_variables.pkl"

        ]

    }

    archivo_metadata = (
        MODELOS_DIR /
        "metadata_entrenamiento.json"
    )

    with open(
        archivo_metadata,
        "w",
        encoding="utf-8"
    ) as archivo:

        json.dump(
            metadata,
            archivo,
            ensure_ascii=False,
            indent=4
        )

    print(
        f"✓ {archivo_preprocesador}"
    )

    print(
        f"✓ {MODELOS_DIR / 'X_train.pkl'}"
    )

    print(
        f"✓ {MODELOS_DIR / 'X_test.pkl'}"
    )

    print(
        f"✓ {MODELOS_DIR / 'y_train.pkl'}"
    )

    print(
        f"✓ {MODELOS_DIR / 'y_test.pkl'}"
    )

    print(
        f"✓ {MODELOS_DIR / 'nombres_variables.pkl'}"
    )

    print(
        f"✓ {archivo_metadata}"
    )


# ============================================================
# 13. GUARDAR DATASETS PROCESADOS
# ============================================================

def guardar_dataset_preparado(
    X_train,
    X_test,
    y_train,
    y_test,
    nombres_variables
):

    imprimir_subtitulo(
        "12. GENERACIÓN DE DATASETS PROCESADOS"
    )

    columnas = nombres_variables

    df_train = pd.DataFrame(
        X_train,
        columns=columnas
    )

    df_train[
        "entrega_tardia"
    ] = np.asarray(
        y_train
    )

    df_test = pd.DataFrame(
        X_test,
        columns=columnas
    )

    df_test[
        "entrega_tardia"
    ] = np.asarray(
        y_test
    )

    archivo_train = (
        REPORTES_DIR /
        "dataset_entrenamiento.csv"
    )

    archivo_test = (
        REPORTES_DIR /
        "dataset_prueba.csv"
    )

    df_train.to_csv(
        archivo_train,
        index=False,
        encoding="utf-8-sig"
    )

    df_test.to_csv(
        archivo_test,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"✓ {archivo_train}"
    )

    print(
        f"✓ {archivo_test}"
    )


# ============================================================
# 14. REPORTE
# ============================================================

def generar_reporte(
    df,
    X,
    y,
    X_train,
    X_test,
    nombres_variables
):

    imprimir_subtitulo(
        "13. REPORTE DEL ENTRENAMIENTO"
    )

    reporte = {

        "proyecto":
            "SIG-LOG",

        "proceso":
            "Preparación de datos para aprendizaje supervisado",

        "registros_originales":
            int(len(df)),

        "registros_predictivos":
            int(len(X)),

        "variables_originales":
            int(len(df.columns)),

        "variables_predictoras":
            int(X.shape[1]),

        "variables_procesadas":
            int(len(nombres_variables)),

        "registros_entrenamiento":
            int(len(X_train)),

        "registros_prueba":
            int(len(X_test)),

        "variable_objetivo":
            "entrega_tardia",

        "clases": {

            "0":
                "A tiempo",

            "1":
                "Tardía"

        },

        "random_state":
            RANDOM_STATE,

        "test_size":
            TEST_SIZE

    }

    archivo_reporte = (
        REPORTES_DIR /
        "reporte_entrenamiento.json"
    )

    with open(
        archivo_reporte,
        "w",
        encoding="utf-8"
    ) as archivo:

        json.dump(
            reporte,
            archivo,
            ensure_ascii=False,
            indent=4
        )

    print(
        "✓ Reporte guardado:"
    )

    print(
        f"  {archivo_reporte}"
    )


# ============================================================
# 15. RESUMEN
# ============================================================

def mostrar_resumen(
    df,
    X,
    y,
    X_train,
    X_test,
    nombres_variables
):

    imprimir_titulo(
        "RESUMEN DEL PREPROCESAMIENTO"
    )

    print(
        f"Dataset original       : "
        f"{len(df):,} registros"
    )

    print(
        f"Variables predictoras  : "
        f"{X.shape[1]}"
    )

    print(
        f"Variables procesadas   : "
        f"{len(nombres_variables)}"
    )

    print(
        f"Datos entrenamiento    : "
        f"{len(X_train):,}"
    )

    print(
        f"Datos prueba           : "
        f"{len(X_test):,}"
    )

    print()

    print(
        "Variable objetivo:"
    )

    print(
        "  entrega_tardia"
    )

    print(
        "  0 = A tiempo"
    )

    print(
        "  1 = Tardía"
    )

    print()

    print(
        "Interpretación:"
    )

    print(
        "  El modelo utilizará únicamente"
    )

    print(
        "  información disponible antes/durante"
    )

    print(
        "  la preparación de una entrega."
    )

    print()

    print(
        "Archivos disponibles:"
    )

    print(
        f"  {MODELOS_DIR}"
    )

    print()

    print(
        "SIG-LOG: PREPROCESAMIENTO COMPLETADO"
    )


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main():

    imprimir_titulo(
        "SIG-LOG - PREPARACIÓN PARA MACHINE LEARNING"
    )

    print(
        "Unidad III: Análisis supervisado"
    )

    print(
        "Clasificación de entregas tardías"
    )

    print()

    print(
        f"Directorio del proyecto:"
    )

    print(
        f"{BASE_DIR}"
    )

    # --------------------------------------------------------
    # 1
    # --------------------------------------------------------

    df = cargar_datos()

    # --------------------------------------------------------
    # 2
    # --------------------------------------------------------

    mostrar_informacion(df)

    # --------------------------------------------------------
    # 3
    # --------------------------------------------------------

    validar_dataset(df)

    # --------------------------------------------------------
    # 4
    # --------------------------------------------------------

    df = crear_caracteristicas_fecha_hora(
        df
    )

    # --------------------------------------------------------
    # 5
    # --------------------------------------------------------

    df = crear_variable_objetivo(
        df
    )

    # --------------------------------------------------------
    # 6
    # --------------------------------------------------------

    (
        X,
        y,
        columnas_numericas,
        columnas_categoricas,
        columnas_excluir
    ) = detectar_variables(
        df
    )

    # --------------------------------------------------------
    # 7
    # --------------------------------------------------------

    analizar_objetivo(
        y
    )

    # --------------------------------------------------------
    # 8
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = dividir_datos(
        X,
        y
    )

    # --------------------------------------------------------
    # 9
    # --------------------------------------------------------

    preprocesador = construir_preprocesador(
        columnas_numericas,
        columnas_categoricas
    )

    # --------------------------------------------------------
    # 10
    # --------------------------------------------------------

    (
        X_train_procesado,
        X_test_procesado
    ) = ajustar_preprocesador(
        preprocesador,
        X_train,
        X_test
    )

    # --------------------------------------------------------
    # 11
    # --------------------------------------------------------

    nombres_variables = (
        obtener_nombres_variables(
            preprocesador
        )
    )

    # --------------------------------------------------------
    # GUARDAR ARTEFACTOS
    # --------------------------------------------------------

    guardar_objetos(
        preprocesador,
        X_train_procesado,
        X_test_procesado,
        y_train,
        y_test,
        nombres_variables,
        list(df.columns),
        columnas_numericas,
        columnas_categoricas
    )

    # --------------------------------------------------------
    # GUARDAR DATASETS
    # --------------------------------------------------------

    guardar_dataset_preparado(
        X_train_procesado,
        X_test_procesado,
        y_train,
        y_test,
        nombres_variables
    )

    # --------------------------------------------------------
    # REPORTE
    # --------------------------------------------------------

    generar_reporte(
        df,
        X,
        y,
        X_train,
        X_test,
        nombres_variables
    )

    # --------------------------------------------------------
    # RESUMEN
    # --------------------------------------------------------

    mostrar_resumen(
        df,
        X,
        y,
        X_train,
        X_test,
        nombres_variables
    )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    main()