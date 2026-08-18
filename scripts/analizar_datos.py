import pandas as pd
import os


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATOS_DIR = os.path.join(
    BASE_DIR,
    "datos"
)


# ============================================================
# CARGAR ARCHIVOS
# ============================================================

print("\n==========================================")
print("       SIG-LOG - ANÁLISIS DE DATOS")
print("==========================================")

clientes = pd.read_csv(
    os.path.join(
        DATOS_DIR,
        "clientes.csv"
    )
)

vehiculos = pd.read_csv(
    os.path.join(
        DATOS_DIR,
        "vehiculos.csv"
    )
)

operadores = pd.read_csv(
    os.path.join(
        DATOS_DIR,
        "operadores.csv"
    )
)

rutas = pd.read_csv(
    os.path.join(
        DATOS_DIR,
        "rutas.csv"
    )
)

componentes = pd.read_csv(
    os.path.join(
        DATOS_DIR,
        "componentes.csv"
    )
)

entregas = pd.read_csv(
    os.path.join(
        DATOS_DIR,
        "entregas.csv"
    )
)


# ============================================================
# CONVERTIR ENTREGA_TARDIA A BOOLEAN
# ============================================================

if entregas["entrega_tardia"].dtype != bool:

    entregas[
        "entrega_tardia"
    ] = (
        entregas[
            "entrega_tardia"
        ]
        .astype(str)
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False
        })
        .fillna(False)
    )


# ============================================================
# DATASETS
# ============================================================

datasets = {

    "clientes":
        clientes,

    "vehiculos":
        vehiculos,

    "operadores":
        operadores,

    "rutas":
        rutas,

    "componentes":
        componentes,

    "entregas":
        entregas
}


# ============================================================
# 1. INFORMACIÓN GENERAL
# ============================================================

print("\n==========================================")
print("1. REGISTROS POR DATASET")
print("==========================================")

for nombre, df in datasets.items():

    print(
        f"{nombre.capitalize():<15}"
        f"{len(df):,}"
    )


# ============================================================
# 2. VALORES NULOS
# ============================================================

print("\n==========================================")
print("2. VALORES NULOS")
print("==========================================")

for nombre, df in datasets.items():

    nulos = df.isnull().sum().sum()

    print(
        f"{nombre:<15}"
        f"{nulos:,} valores nulos"
    )


# ============================================================
# 3. REGISTROS DUPLICADOS
# ============================================================

print("\n==========================================")
print("3. REGISTROS DUPLICADOS")
print("==========================================")

for nombre, df in datasets.items():

    duplicados = df.duplicated().sum()

    print(
        f"{nombre:<15}"
        f"{duplicados:,} duplicados"
    )


# ============================================================
# 4. ANÁLISIS DE ENTREGAS
# ============================================================

print("\n==========================================")
print("4. ANÁLISIS DE ENTREGAS")
print("==========================================")

total_entregas = len(
    entregas
)

entregas_tardias = int(
    entregas[
        "entrega_tardia"
    ].sum()
)

entregas_tiempo = (
    total_entregas -
    entregas_tardias
)

porcentaje_tardias = (
    entregas_tardias /
    total_entregas *
    100
)

print(
    f"Total de entregas:     "
    f"{total_entregas:,}"
)

print(
    f"Entregas a tiempo:     "
    f"{entregas_tiempo:,}"
)

print(
    f"Entregas tardías:      "
    f"{entregas_tardias:,}"
)

print(
    f"Porcentaje tardías:    "
    f"{porcentaje_tardias:.2f}%"
)


# ============================================================
# 5. ANÁLISIS DE RETRASOS
# ============================================================

print("\n==========================================")
print("5. ANÁLISIS DE RETRASOS")
print("==========================================")

retraso_promedio_general = (
    entregas[
        "minutos_retraso"
    ].mean()
)

entregas_con_retraso = entregas[
    entregas[
        "entrega_tardia"
    ]
]

if len(entregas_con_retraso) > 0:

    retraso_promedio_tardias = (
        entregas_con_retraso[
            "minutos_retraso"
        ].mean()
    )

else:

    retraso_promedio_tardias = 0


print(
    f"Retraso promedio general:     "
    f"{retraso_promedio_general:.2f} minutos"
)

print(
    f"Retraso promedio tardías:     "
    f"{retraso_promedio_tardias:.2f} minutos"
)

print(
    f"Retraso máximo:               "
    f"{entregas['minutos_retraso'].max():.0f} minutos"
)

print(
    f"Retraso mínimo:               "
    f"{entregas['minutos_retraso'].min():.0f} minutos"
)

print(
    f"Mediana de retraso:           "
    f"{entregas['minutos_retraso'].median():.2f} minutos"
)


# ============================================================
# 6. ANÁLISIS DE DISTANCIAS
# ============================================================

print("\n==========================================")
print("6. ANÁLISIS DE DISTANCIAS")
print("==========================================")

distancia_total = (
    entregas[
        "distancia_real_km"
    ].sum()
)

distancia_promedio = (
    entregas[
        "distancia_real_km"
    ].mean()
)

distancia_maxima = (
    entregas[
        "distancia_real_km"
    ].max()
)

distancia_minima = (
    entregas[
        "distancia_real_km"
    ].min()
)

print(
    f"Distancia total:       "
    f"{distancia_total:,.2f} km"
)

print(
    f"Distancia promedio:    "
    f"{distancia_promedio:,.2f} km"
)

print(
    f"Distancia máxima:      "
    f"{distancia_maxima:,.2f} km"
)

print(
    f"Distancia mínima:      "
    f"{distancia_minima:,.2f} km"
)


# ============================================================
# 7. ANÁLISIS DE COMBUSTIBLE
# ============================================================

print("\n==========================================")
print("7. ANÁLISIS DE COMBUSTIBLE")
print("==========================================")

combustible_total = (
    entregas[
        "combustible_consumido_litros"
    ].sum()
)

combustible_promedio = (
    entregas[
        "combustible_consumido_litros"
    ].mean()
)

costo_combustible = (
    entregas[
        "costo_combustible"
    ].sum()
)

precio_promedio = (
    entregas[
        "precio_combustible"
    ].mean()
)

print(
    f"Combustible total:            "
    f"{combustible_total:,.2f} L"
)

print(
    f"Consumo promedio por entrega: "
    f"{combustible_promedio:,.2f} L"
)

print(
    f"Costo combustible:            "
    f"${costo_combustible:,.2f}"
)

print(
    f"Precio promedio combustible:  "
    f"${precio_promedio:,.2f}/L"
)


# ============================================================
# 8. ANÁLISIS DE COSTOS
# ============================================================

print("\n==========================================")
print("8. ANÁLISIS DE COSTOS")
print("==========================================")

costo_envios = (
    entregas[
        "costo_envio"
    ].sum()
)

costo_total = (
    entregas[
        "costo_total"
    ].sum()
)

costo_promedio = (
    entregas[
        "costo_total"
    ].mean()
)

porcentaje_combustible = (
    costo_combustible /
    costo_total *
    100
)

print(
    f"Costo de envíos:          "
    f"${costo_envios:,.2f}"
)

print(
    f"Costo combustible:        "
    f"${costo_combustible:,.2f}"
)

print(
    f"Costo total:              "
    f"${costo_total:,.2f}"
)

print(
    f"Costo promedio entrega:   "
    f"${costo_promedio:,.2f}"
)

print(
    f"Combustible / operación:  "
    f"{porcentaje_combustible:.2f}%"
)


# ============================================================
# 9. RUTAS CON MÁS ENTREGAS
# ============================================================

print("\n==========================================")
print("9. RUTAS CON MÁS ENTREGAS")
print("==========================================")

rutas_entregas = (
    entregas
    .groupby(
        "id_ruta"
    )
    .size()
    .reset_index(
        name="entregas"
    )
    .sort_values(
        "entregas",
        ascending=False
    )
    .head(10)
)

print(
    rutas_entregas.to_string(
        index=False
    )
)


# ============================================================
# 10. VEHÍCULOS CON MÁS ENTREGAS
# ============================================================

print("\n==========================================")
print("10. VEHÍCULOS CON MÁS ENTREGAS")
print("==========================================")

vehiculos_entregas = (
    entregas
    .groupby(
        "id_vehiculo"
    )
    .size()
    .reset_index(
        name="entregas"
    )
    .sort_values(
        "entregas",
        ascending=False
    )
    .head(10)
)

print(
    vehiculos_entregas.to_string(
        index=False
    )
)


# ============================================================
# 11. OPERADORES CON MÁS ENTREGAS
# ============================================================

print("\n==========================================")
print("11. OPERADORES CON MÁS ENTREGAS")
print("==========================================")

operadores_entregas = (
    entregas
    .groupby(
        "id_operador"
    )
    .size()
    .reset_index(
        name="entregas"
    )
    .sort_values(
        "entregas",
        ascending=False
    )
    .head(10)
)

print(
    operadores_entregas.to_string(
        index=False
    )
)


# ============================================================
# 12. CAUSAS DE RETRASO
# ============================================================

print("\n==========================================")
print("12. CAUSAS DE RETRASO")
print("==========================================")

causas = {}

for observacion in entregas_con_retraso[
    "observaciones"
]:

    if pd.isna(
        observacion
    ):
        continue

    partes = str(
        observacion
    ).split(",")

    for causa in partes:

        causa = causa.strip()

        if causa:

            causas[causa] = (
                causas.get(
                    causa,
                    0
                ) + 1
            )


if causas:

    causas_df = (
        pd.DataFrame(
            list(
                causas.items()
            ),
            columns=[
                "causa",
                "cantidad"
            ]
        )
        .sort_values(
            "cantidad",
            ascending=False
        )
    )

    print(
        causas_df.to_string(
            index=False
        )
    )

else:

    print(
        "No se encontraron causas de retraso."
    )


# ============================================================
# 13. COMBUSTIBLE POR TIPO
# ============================================================

print("\n==========================================")
print("13. COMBUSTIBLE POR TIPO")
print("==========================================")

combustible_tipo = (
    entregas
    .groupby(
        "tipo_combustible",
        as_index=False
    )
    .agg(

        entregas=(
            "id_entrega",
            "count"
        ),

        litros_consumidos=(
            "combustible_consumido_litros",
            "sum"
        ),

        costo_combustible=(
            "costo_combustible",
            "sum"
        ),

        precio_promedio=(
            "precio_combustible",
            "mean"
        )
    )
)

combustible_tipo[
    "litros_consumidos"
] = combustible_tipo[
    "litros_consumidos"
].round(2)

combustible_tipo[
    "costo_combustible"
] = combustible_tipo[
    "costo_combustible"
].round(2)

combustible_tipo[
    "precio_promedio"
] = combustible_tipo[
    "precio_promedio"
].round(2)

print(
    combustible_tipo.to_string(
        index=False
    )
)


# ============================================================
# 14. PARTICIPACIÓN DEL CONSUMO
# ============================================================

print("\n==========================================")
print("14. PARTICIPACIÓN DEL CONSUMO")
print("==========================================")

total_litros = (
    combustible_tipo[
        "litros_consumidos"
    ].sum()
)

combustible_tipo[
    "porcentaje_litros"
] = (
    combustible_tipo[
        "litros_consumidos"
    ]
    /
    total_litros
    *
    100
).round(2)

print(
    combustible_tipo[
        [
            "tipo_combustible",
            "porcentaje_litros"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 15. VEHÍCULOS CON MAYOR CONSUMO
# ============================================================

print("\n==========================================")
print("15. VEHÍCULOS CON MAYOR CONSUMO")
print("==========================================")

vehiculos_consumo = (
    entregas
    .groupby(
        "id_vehiculo"
    )
    .agg(

        entregas=(
            "id_entrega",
            "count"
        ),

        litros_consumidos=(
            "combustible_consumido_litros",
            "sum"
        ),

        costo_combustible=(
            "costo_combustible",
            "sum"
        ),

        kilometros=(
            "distancia_real_km",
            "sum"
        )
    )
    .reset_index()
    .sort_values(
        "litros_consumidos",
        ascending=False
    )
    .head(10)
)

print(
    vehiculos_consumo.to_string(
        index=False
    )
)


# ============================================================
# 16. RUTAS CON MÁS RETRASOS
# ============================================================

print("\n==========================================")
print("16. RUTAS CON MÁS RETRASOS")
print("==========================================")

rutas_retrasos = (
    entregas
    .groupby(
        "id_ruta"
    )
    .agg(

        entregas=(
            "id_entrega",
            "count"
        ),

        entregas_tardias=(
            "entrega_tardia",
            "sum"
        ),

        retraso_promedio=(
            "minutos_retraso",
            "mean"
        )
    )
    .reset_index()
)

rutas_retrasos[
    "porcentaje_retraso"
] = (
    rutas_retrasos[
        "entregas_tardias"
    ]
    /
    rutas_retrasos[
        "entregas"
    ]
    *
    100
)

rutas_retrasos[
    "retraso_promedio"
] = rutas_retrasos[
    "retraso_promedio"
].round(2)

rutas_retrasos[
    "porcentaje_retraso"
] = rutas_retrasos[
    "porcentaje_retraso"
].round(2)

rutas_retrasos = (
    rutas_retrasos
    .sort_values(
        "porcentaje_retraso",
        ascending=False
    )
    .head(10)
)

print(
    rutas_retrasos.to_string(
        index=False
    )
)


# ============================================================
# 17. OPERADORES CON MÁS RETRASOS
# ============================================================

print("\n==========================================")
print("17. OPERADORES CON MÁS RETRASOS")
print("==========================================")

operadores_retrasos = (
    entregas
    .groupby(
        "id_operador"
    )
    .agg(

        entregas=(
            "id_entrega",
            "count"
        ),

        entregas_tardias=(
            "entrega_tardia",
            "sum"
        ),

        retraso_promedio=(
            "minutos_retraso",
            "mean"
        )
    )
    .reset_index()
)

operadores_retrasos[
    "porcentaje_retraso"
] = (
    operadores_retrasos[
        "entregas_tardias"
    ]
    /
    operadores_retrasos[
        "entregas"
    ]
    *
    100
)

operadores_retrasos[
    "retraso_promedio"
] = operadores_retrasos[
    "retraso_promedio"
].round(2)

operadores_retrasos[
    "porcentaje_retraso"
] = operadores_retrasos[
    "porcentaje_retraso"
].round(2)

operadores_retrasos = (
    operadores_retrasos
    .sort_values(
        "porcentaje_retraso",
        ascending=False
    )
    .head(10)
)

print(
    operadores_retrasos.to_string(
        index=False
    )
)


# ============================================================
# 18. COSTOS POR TIPO DE COMBUSTIBLE
# ============================================================

print("\n==========================================")
print("18. COSTOS POR TIPO DE COMBUSTIBLE")
print("==========================================")

costos_combustible = (
    entregas
    .groupby(
        "tipo_combustible"
    )
    .agg(

        costo_combustible=(
            "costo_combustible",
            "sum"
        ),

        litros=(
            "combustible_consumido_litros",
            "sum"
        ),

        entregas=(
            "id_entrega",
            "count"
        )
    )
    .reset_index()
)

costos_combustible[
    "costo_promedio_entrega"
] = (
    costos_combustible[
        "costo_combustible"
    ]
    /
    costos_combustible[
        "entregas"
    ]
).round(2)

print(
    costos_combustible.to_string(
        index=False
    )
)


# ============================================================
# 19. RENDIMIENTO GENERAL
# ============================================================

print("\n==========================================")
print("19. RENDIMIENTO GENERAL")
print("==========================================")

rendimiento_general = (
    distancia_total /
    combustible_total
)

print(
    f"Rendimiento general de la operación: "
    f"{rendimiento_general:.2f} km/L"
)


# ============================================================
# 20. ANÁLISIS POR TIPO DE VEHÍCULO
# ============================================================

print("\n==========================================")
print("20. ANÁLISIS POR TIPO DE VEHÍCULO")
print("==========================================")

entregas_vehiculos = entregas.merge(
    vehiculos[
        [
            "id_vehiculo",
            "tipo",
            "marca",
            "capacidad_toneladas",
            "rendimiento_km_l"
        ]
    ],
    on="id_vehiculo",
    how="left"
)

analisis_tipo_vehiculo = (
    entregas_vehiculos
    .groupby(
        "tipo"
    )
    .agg(

        entregas=(
            "id_entrega",
            "count"
        ),

        entregas_tardias=(
            "entrega_tardia",
            "sum"
        ),

        distancia_km=(
            "distancia_real_km",
            "sum"
        ),

        combustible_litros=(
            "combustible_consumido_litros",
            "sum"
        ),

        costo_total=(
            "costo_total",
            "sum"
        )
    )
    .reset_index()
)

analisis_tipo_vehiculo[
    "porcentaje_retraso"
] = (
    analisis_tipo_vehiculo[
        "entregas_tardias"
    ]
    /
    analisis_tipo_vehiculo[
        "entregas"
    ]
    *
    100
).round(2)

print(
    analisis_tipo_vehiculo.to_string(
        index=False
    )
)


# ============================================================
# 21. ANÁLISIS MENSUAL
# ============================================================

print("\n==========================================")
print("21. ANÁLISIS MENSUAL")
print("==========================================")

entregas[
    "fecha_salida"
] = pd.to_datetime(
    entregas[
        "fecha_salida"
    ]
)

analisis_mensual = (
    entregas
    .groupby(
        entregas[
            "fecha_salida"
        ].dt.to_period(
            "M"
        )
    )
    .agg(

        entregas=(
            "id_entrega",
            "count"
        ),

        entregas_tardias=(
            "entrega_tardia",
            "sum"
        ),

        distancia_km=(
            "distancia_real_km",
            "sum"
        ),

        combustible_litros=(
            "combustible_consumido_litros",
            "sum"
        ),

        costo_total=(
            "costo_total",
            "sum"
        )
    )
    .reset_index()
)

analisis_mensual[
    "fecha_salida"
] = analisis_mensual[
    "fecha_salida"
].astype(str)

analisis_mensual[
    "porcentaje_retraso"
] = (
    analisis_mensual[
        "entregas_tardias"
    ]
    /
    analisis_mensual[
        "entregas"
    ]
    *
    100
).round(2)

print(
    analisis_mensual.to_string(
        index=False
    )
)


# ============================================================
# 22. TOP CLIENTES POR ENTREGAS
# ============================================================

print("\n==========================================")
print("22. CLIENTES CON MÁS ENTREGAS")
print("==========================================")

clientes_entregas = (
    entregas
    .groupby(
        "id_cliente"
    )
    .agg(
        entregas=(
            "id_entrega",
            "count"
        ),

        costo_total=(
            "costo_total",
            "sum"
        )
    )
    .reset_index()
    .sort_values(
        "entregas",
        ascending=False
    )
    .head(10)
)

print(
    clientes_entregas.to_string(
        index=False
    )
)


# ============================================================
# RESUMEN FINAL
# ============================================================

print("\n==========================================")
print("       RESUMEN DEL DATASET")
print("==========================================")

print(
    f"""
Registros analizados:          {total_entregas:,}

Entregas a tiempo:             {entregas_tiempo:,}

Entregas tardías:              {entregas_tardias:,}

Porcentaje de retrasos:        {porcentaje_tardias:.2f}%

Retraso promedio general:      {retraso_promedio_general:.2f} min

Retraso promedio tardías:      {retraso_promedio_tardias:.2f} min

Distancia recorrida:           {distancia_total:,.2f} km

Combustible consumido:         {combustible_total:,.2f} L

Rendimiento general:           {rendimiento_general:.2f} km/L

Costo de combustible:          ${costo_combustible:,.2f}

Costo de envíos:               ${costo_envios:,.2f}

Costo total de operación:      ${costo_total:,.2f}

Costo promedio por entrega:    ${costo_promedio:,.2f}
"""
)

print("==========================================")
print("       ANÁLISIS COMPLETADO")
print("==========================================")