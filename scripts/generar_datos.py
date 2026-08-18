import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import os


# ============================================================
# CONFIGURACIÓN
# ============================================================

SEED = 42

np.random.seed(SEED)
random.seed(SEED)


# ============================================================
# CARPETA BASE Y CARPETA DE DATOS
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

os.makedirs(
    DATOS_DIR,
    exist_ok=True
)


# ============================================================
# CANTIDADES DE REGISTROS
# ============================================================

NUM_CLIENTES = 100
NUM_VEHICULOS = 40
NUM_OPERADORES = 60
NUM_RUTAS = 50
NUM_ENTREGAS = 5000


# ============================================================
# CATÁLOGOS
# ============================================================

nombres = [
    "Carlos", "Ana", "Luis", "Sofía", "Daniel",
    "María", "Jorge", "Fernanda", "Miguel", "Andrea",
    "José", "Laura", "Ricardo", "Paola", "Diego"
]

apellidos = [
    "Hernández", "García", "Martínez", "López",
    "González", "Pérez", "Rodríguez", "Sánchez",
    "Ramírez", "Torres", "Flores", "Rivera",
    "Morales", "Cruz", "Reyes"
]

ciudades = [
    "Toluca",
    "Metepec",
    "Lerma",
    "CDMX",
    "Naucalpan",
    "Tlalnepantla",
    "Querétaro",
    "Puebla",
    "Cuernavaca",
    "Pachuca"
]

marcas = [
    "Ford",
    "Chevrolet",
    "Toyota",
    "Nissan",
    "Mercedes-Benz",
    "Volkswagen",
    "Isuzu",
    "Kenworth"
]

tipos_vehiculo = [
    "Camioneta",
    "Van",
    "Camión",
    "Torton",
    "Tráiler"
]

componentes_catalogo = [
    ("Motor", "Motor"),
    ("Alternador", "Sistema eléctrico"),
    ("Batería", "Sistema eléctrico"),
    ("Pastillas de freno", "Frenos"),
    ("Discos de freno", "Frenos"),
    ("Neumáticos", "Suspensión"),
    ("Amortiguadores", "Suspensión"),
    ("Radiador", "Motor"),
    ("Bomba de agua", "Motor"),
    ("Bomba de combustible", "Combustible"),
    ("Inyectores", "Combustible"),
    ("Filtro de aire", "Motor"),
    ("Filtro de combustible", "Combustible"),
    ("Clutch", "Transmisión"),
    ("Caja de cambios", "Transmisión"),
    ("Banda de distribución", "Motor"),
    ("Sensor de temperatura", "Sensores"),
    ("Sensor de presión", "Sensores"),
    ("Luces", "Sistema eléctrico"),
    ("Sistema de escape", "Escape")
]


# ============================================================
# 1. CLIENTES
# ============================================================

clientes = []

for i in range(1, NUM_CLIENTES + 1):

    nombre = random.choice(nombres)
    apellido = random.choice(apellidos)

    cliente = {
        "id_cliente": i,
        "nombre": nombre,
        "apellido": apellido,
        "empresa": f"Empresa {chr(65 + ((i - 1) % 26))}-{i:03d}",
        "telefono": f"722{random.randint(1000000, 9999999)}",
        "correo": f"cliente{i}@logistol.com",
        "direccion": f"Av. Principal #{random.randint(100, 999)}",
        "ciudad": random.choice(ciudades),
        "estado": "Estado de México",
        "tipo_cliente": random.choice([
            "Empresa",
            "Empresa",
            "Empresa",
            "Particular"
        ]),
        "fecha_registro": (
            datetime(2025, 1, 1)
            + timedelta(days=random.randint(0, 500))
        ).date(),
        "estatus": random.choice([
            "Activo",
            "Activo",
            "Activo",
            "Inactivo"
        ])
    }

    clientes.append(cliente)

df_clientes = pd.DataFrame(clientes)


# ============================================================
# 2. VEHÍCULOS
# ============================================================

vehiculos = []

for i in range(1, NUM_VEHICULOS + 1):

    tipo = random.choice(tipos_vehiculo)

    if tipo == "Camioneta":

        capacidad = round(
            np.random.uniform(0.8, 2.0),
            2
        )

    elif tipo == "Van":

        capacidad = round(
            np.random.uniform(1.0, 3.0),
            2
        )

    elif tipo == "Camión":

        capacidad = round(
            np.random.uniform(4.0, 8.0),
            2
        )

    elif tipo == "Torton":

        capacidad = round(
            np.random.uniform(8.0, 15.0),
            2
        )

    else:

        capacidad = round(
            np.random.uniform(15.0, 30.0),
            2
        )

    anio = random.randint(2018, 2025)

    if tipo in ["Camioneta", "Van"]:

        combustible = random.choice([
            "Gasolina",
            "Gasolina",
            "Diesel"
        ])

    else:

        combustible = "Diesel"

    vehiculo = {
        "id_vehiculo": i,

        "numero_economico":
            f"V-{i:03d}",

        "placas":
            f"ABC-{100+i}",

        "marca":
            random.choice(marcas),

        "modelo":
            str(anio),

        "anio":
            anio,

        "tipo":
            tipo,

        "tipo_combustible":
            combustible,

        "capacidad_toneladas":
            capacidad,

        "kilometraje_actual":
            round(
                np.random.uniform(
                    10000,
                    250000
                ),
                2
            ),

        "rendimiento_km_l":
            round(
                np.random.uniform(
                    4.5,
                    12.5
                ),
                2
            ),

        "consumo_promedio":
            round(
                np.random.uniform(
                    8,
                    35
                ),
                2
            ),

        "fecha_adquisicion":
            datetime(
                anio,
                1,
                1
            ).date(),

        "ultima_revision":
            (
                datetime(2026, 1, 1)
                + timedelta(
                    days=random.randint(
                        0,
                        200
                    )
                )
            ).date(),

        "proxima_revision":
            (
                datetime(2026, 7, 1)
                + timedelta(
                    days=random.randint(
                        0,
                        180
                    )
                )
            ).date(),

        "nivel_riesgo":
            random.choice([
                "Bajo",
                "Bajo",
                "Bajo",
                "Medio",
                "Alto"
            ]),

        "estatus":
            random.choice([
                "Disponible",
                "Disponible",
                "En ruta",
                "Mantenimiento"
            ])
    }

    vehiculos.append(vehiculo)

df_vehiculos = pd.DataFrame(
    vehiculos
)


# ============================================================
# 3. OPERADORES
# ============================================================

operadores = []

for i in range(1, NUM_OPERADORES + 1):

    nombre = random.choice(nombres)
    apellido = random.choice(apellidos)

    operador = {
        "id_operador": i,

        "numero_empleado":
            f"OP-{i:03d}",

        "nombre":
            nombre,

        "apellido":
            apellido,

        "licencia":
            f"LIC-{10000+i}",

        "tipo_licencia":
            random.choice([
                "B",
                "C",
                "E"
            ]),

        "fecha_ingreso":
            (
                datetime(2021, 1, 1)
                + timedelta(
                    days=random.randint(
                        0,
                        1800
                    )
                )
            ).date(),

        "telefono":
            f"722{random.randint(1000000, 9999999)}",

        "correo":
            f"operador{i}@logistol.com",

        "entregas_realizadas":
            0,

        "entregas_tardias":
            0,

        "porcentaje_puntualidad":
            0,

        "kilometros_recorridos":
            0,

        "estatus":
            "Activo"
    }

    operadores.append(operador)

df_operadores = pd.DataFrame(
    operadores
)


# ============================================================
# 4. RUTAS
# ============================================================

rutas = []

for i in range(1, NUM_RUTAS + 1):

    origen, destino = random.sample(
        ciudades,
        2
    )

    distancia = round(
        np.random.uniform(
            20,
            450
        ),
        2
    )

    tiempo_estimado = int(
        distancia *
        np.random.uniform(
            1.5,
            2.5
        )
    )

    nivel_trafico = random.choice([
        "Bajo",
        "Medio",
        "Medio",
        "Alto"
    ])

    peaje = random.choice([
        True,
        False
    ])

    costo_peaje = (
        round(
            np.random.uniform(
                50,
                600
            ),
            2
        )
        if peaje
        else 0
    )

    ruta = {
        "id_ruta": i,

        "codigo_ruta":
            f"R-{i:03d}",

        "origen":
            origen,

        "destino":
            destino,

        "distancia_km":
            distancia,

        "tiempo_estimado_minutos":
            tiempo_estimado,

        "tipo_ruta":
            random.choice([
                "Urbana",
                "Foránea",
                "Interestatal"
            ]),

        "nivel_trafico":
            nivel_trafico,

        "peajes":
            peaje,

        "costo_peaje":
            costo_peaje,

        "numero_entregas":
            0,

        "tiempo_promedio_minutos":
            0,

        "retraso_promedio_minutos":
            0,

        "porcentaje_retraso":
            0,

        "consumo_promedio":
            0,

        "costo_promedio":
            0,

        "estatus":
            "Activa"
    }

    rutas.append(ruta)

df_rutas = pd.DataFrame(
    rutas
)


# ============================================================
# 5. COMPONENTES
# ============================================================

componentes = []

for i, (nombre, categoria) in enumerate(
    componentes_catalogo,
    start=1
):

    componente = {
        "id_componente": i,

        "nombre":
            nombre,

        "categoria":
            categoria,

        "vida_util_km":
            random.randint(
                10000,
                150000
            ),

        "costo_reemplazo":
            round(
                np.random.uniform(
                    500,
                    25000
                ),
                2
            ),

        "descripcion":
            f"Componente correspondiente a {categoria}",

        "estatus":
            "Activo"
    }

    componentes.append(
        componente
    )

df_componentes = pd.DataFrame(
    componentes
)


# ============================================================
# MOSTRAR CATÁLOGOS
# ============================================================

print("\n==========================================")
print("       SIG-LOG - DATASET GENERADO")
print("==========================================")

print("\nClientes:")
print(df_clientes.head())

print("\nVehículos:")
print(df_vehiculos.head())

print("\nOperadores:")
print(df_operadores.head())

print("\nRutas:")
print(df_rutas.head())

print("\nComponentes:")
print(df_componentes.head())


# ============================================================
# GUARDAR CATÁLOGOS
# ============================================================

df_clientes.to_csv(
    os.path.join(
        DATOS_DIR,
        "clientes.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

df_vehiculos.to_csv(
    os.path.join(
        DATOS_DIR,
        "vehiculos.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

df_operadores.to_csv(
    os.path.join(
        DATOS_DIR,
        "operadores.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

df_rutas.to_csv(
    os.path.join(
        DATOS_DIR,
        "rutas.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

df_componentes.to_csv(
    os.path.join(
        DATOS_DIR,
        "componentes.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

print("\n==========================================")
print("Catálogos guardados correctamente en /datos")
print("==========================================")


# ============================================================
# 6. GENERAR ENTREGAS
# ============================================================

entregas = []

fecha_inicio = datetime(
    2026,
    1,
    1
)

fecha_fin = datetime(
    2026,
    8,
    15
)

dias_periodo = (
    fecha_fin -
    fecha_inicio
).days


for i in range(
    1,
    NUM_ENTREGAS + 1
):

    # --------------------------------------------------------
    # Seleccionar registros
    # --------------------------------------------------------

    cliente = random.choice(
        clientes
    )

    vehiculo = random.choice(
        vehiculos
    )

    operador = random.choice(
        operadores
    )

    ruta = random.choice(
        rutas
    )

    # --------------------------------------------------------
    # Fecha de salida
    # --------------------------------------------------------

    fecha_salida = (
        fecha_inicio +
        timedelta(
            days=random.randint(
                0,
                dias_periodo
            )
        )
    )

    # --------------------------------------------------------
    # Hora de salida
    # --------------------------------------------------------

    hora_salida = random.choice([
        5, 6, 7, 8, 8, 9, 9, 10,
        11, 12, 13, 14, 15, 16,
        17, 18, 19, 20
    ])

    minuto_salida = random.randint(
        0,
        59
    )

    hora_salida_obj = datetime(
        fecha_salida.year,
        fecha_salida.month,
        fecha_salida.day,
        hora_salida,
        minuto_salida
    )

    # --------------------------------------------------------
    # Peso de carga
    # --------------------------------------------------------

    peso_maximo = vehiculo[
        "capacidad_toneladas"
    ]

    peso_carga = round(
        np.random.uniform(
            0.2,
            peso_maximo * 0.95
        ),
        2
    )

    # --------------------------------------------------------
    # Cantidad de paquetes
    # --------------------------------------------------------

    cantidad_paquetes = random.randint(
        1,
        80
    )

    # --------------------------------------------------------
    # Distancia real
    # --------------------------------------------------------

    distancia_real = round(
        ruta["distancia_km"] *
        np.random.uniform(
            0.95,
            1.15
        ),
        2
    )

    # --------------------------------------------------------
    # Tiempo estimado
    # --------------------------------------------------------

    tiempo_estimado = ruta[
        "tiempo_estimado_minutos"
    ]

    # ========================================================
    # PROBABILIDAD DE RETRASO
    # ========================================================

    probabilidad_retraso = 0.10

    if ruta["nivel_trafico"] == "Alto":

        probabilidad_retraso += 0.20

    elif ruta["nivel_trafico"] == "Medio":

        probabilidad_retraso += 0.10

    if hora_salida in [
        7, 8, 9,
        17, 18, 19
    ]:

        probabilidad_retraso += 0.12

    if distancia_real > 300:

        probabilidad_retraso += 0.10

    if peso_carga > (
        peso_maximo * 0.80
    ):

        probabilidad_retraso += 0.08

    probabilidad_retraso = min(
        probabilidad_retraso,
        0.90
    )

    # ========================================================
    # DETERMINAR ENTREGA
    # ========================================================

    entrega_tardia = (
        random.random()
        <
        probabilidad_retraso
    )

    # ========================================================
    # CALCULAR RETRASO
    # ========================================================

    if entrega_tardia:

        retraso = random.randint(
            10,
            30
        )

        if ruta["nivel_trafico"] == "Alto":

            retraso += random.randint(
                10,
                30
            )

        if hora_salida in [
            7, 8, 9,
            17, 18, 19
        ]:

            retraso += random.randint(
                5,
                25
            )

        if distancia_real > 300:

            retraso += random.randint(
                5,
                20
            )

        if peso_carga > (
            peso_maximo * 0.80
        ):

            retraso += random.randint(
                5,
                15
            )

    else:

        retraso = 0

    # ========================================================
    # TIEMPO REAL
    # ========================================================

    tiempo_real = (
        tiempo_estimado +
        retraso
    )

    # ========================================================
    # ENTREGA PROGRAMADA
    # ========================================================

    fecha_entrega_programada = (
        fecha_salida.date()
    )

    hora_entrega_programada = (
        hora_salida_obj +
        timedelta(
            minutes=tiempo_estimado
        )
    ).time()

    # ========================================================
    # ENTREGA REAL
    # ========================================================

    hora_entrega_real_obj = (
        hora_salida_obj +
        timedelta(
            minutes=tiempo_real
        )
    )

    fecha_entrega_real = (
        hora_entrega_real_obj.date()
    )

    hora_entrega_real = (
        hora_entrega_real_obj.time()
    )

    # ========================================================
    # ESTATUS
    # ========================================================

    if entrega_tardia:

        estatus = "Tardía"

    else:

        estatus = "Entregada"

    # ========================================================
    # COMBUSTIBLE
    # ========================================================

    rendimiento = vehiculo[
        "rendimiento_km_l"
    ]

    combustible_consumido = round(
        distancia_real /
        rendimiento,
        2
    )

    # --------------------------------------------------------
    # Precio combustible
    # --------------------------------------------------------

    if vehiculo[
        "tipo_combustible"
    ] == "Diesel":

        precio_combustible = round(
            random.uniform(
                24,
                27
            ),
            2
        )

    else:

        precio_combustible = round(
            random.uniform(
                23,
                26
            ),
            2
        )

    # --------------------------------------------------------
    # Costo combustible
    # --------------------------------------------------------

    costo_combustible = round(
        combustible_consumido *
        precio_combustible,
        2
    )

    # ========================================================
    # COSTO DEL ENVÍO
    # ========================================================

    costo_envio = round(
        500 +
        (distancia_real * 5) +
        (peso_carga * 100) +
        (cantidad_paquetes * 8),
        2
    )

    # ========================================================
    # COSTO TOTAL
    # ========================================================

    costo_total = round(
        costo_envio +
        costo_combustible,
        2
    )

    # ========================================================
    # OBSERVACIONES
    # ========================================================

    if entrega_tardia:

        causas_retraso = []

        if ruta["nivel_trafico"] == "Alto":

            causas_retraso.append(
                "Tráfico alto"
            )

        if hora_salida in [
            7, 8, 9,
            17, 18, 19
        ]:

            causas_retraso.append(
                "Hora pico"
            )

        if distancia_real > 300:

            causas_retraso.append(
                "Distancia larga"
            )

        if peso_carga > (
            peso_maximo * 0.80
        ):

            causas_retraso.append(
                "Carga pesada"
            )

        if not causas_retraso:

            causas_retraso.append(
                "Incidencia operativa"
            )

        observaciones = ", ".join(
            causas_retraso
        )

    else:

        observaciones = (
            "Entrega realizada en tiempo"
        )

    # ========================================================
    # CREAR REGISTRO
    # ========================================================

    entrega = {
        "id_entrega":
            i,

        "id_cliente":
            cliente["id_cliente"],

        "id_vehiculo":
            vehiculo["id_vehiculo"],

        "id_operador":
            operador["id_operador"],

        "id_ruta":
            ruta["id_ruta"],

        "fecha_salida":
            fecha_salida.date(),

        "hora_salida":
            hora_salida_obj.time(),

        "fecha_entrega_programada":
            fecha_entrega_programada,

        "hora_entrega_programada":
            hora_entrega_programada,

        "fecha_entrega_real":
            fecha_entrega_real,

        "hora_entrega_real":
            hora_entrega_real,

        "peso_carga":
            peso_carga,

        "cantidad_paquetes":
            cantidad_paquetes,

        "distancia_real_km":
            distancia_real,

        "tipo_combustible":
            vehiculo["tipo_combustible"],

        "combustible_consumido_litros":
            combustible_consumido,

        "precio_combustible":
            precio_combustible,

        "costo_envio":
            costo_envio,

        "costo_combustible":
            costo_combustible,

        "costo_total":
            costo_total,

        "minutos_retraso":
            retraso,

        "entrega_tardia":
            entrega_tardia,

        "estatus":
            estatus,

        "observaciones":
            observaciones
    }

    entregas.append(
        entrega
    )


# ============================================================
# CREAR DATAFRAME DE ENTREGAS
# ============================================================

df_entregas = pd.DataFrame(
    entregas
)


# ============================================================
# GUARDAR ENTREGAS
# ============================================================

df_entregas.to_csv(
    os.path.join(
        DATOS_DIR,
        "entregas.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# MOSTRAR INFORMACIÓN
# ============================================================

print("\n==========================================")
print("           ENTREGAS GENERADAS")
print("==========================================")

print(
    f"Total de entregas: "
    f"{len(df_entregas):,}"
)

print(
    f"Entregas tardías: "
    f"{int(df_entregas['entrega_tardia'].sum()):,}"
)

print(
    f"Porcentaje de retrasos: "
    f"{df_entregas['entrega_tardia'].mean() * 100:.2f}%"
)

print(
    f"Distancia total: "
    f"{df_entregas['distancia_real_km'].sum():,.2f} km"
)

print(
    f"Combustible total: "
    f"{df_entregas['combustible_consumido_litros'].sum():,.2f} L"
)

print(
    f"Costo total: "
    f"${df_entregas['costo_total'].sum():,.2f}"
)


# ============================================================
# RESUMEN DE COMBUSTIBLE
# ============================================================

print("\n==========================================")
print("       RESUMEN DE COMBUSTIBLE")
print("==========================================")

resumen_combustible = (
    df_entregas
    .groupby("tipo_combustible")
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
        )
    )
    .reset_index()
)

resumen_combustible[
    "litros_consumidos"
] = resumen_combustible[
    "litros_consumidos"
].round(2)

resumen_combustible[
    "costo_combustible"
] = resumen_combustible[
    "costo_combustible"
].round(2)

print(
    resumen_combustible.to_string(
        index=False
    )
)


# ============================================================
# ACTUALIZAR ESTADÍSTICAS DE OPERADORES
# ============================================================

for operador in operadores:

    id_operador = operador[
        "id_operador"
    ]

    registros = df_entregas[
        df_entregas[
            "id_operador"
        ] == id_operador
    ]

    entregas_realizadas = len(
        registros
    )

    entregas_tardias = int(
        registros[
            "entrega_tardia"
        ].sum()
    )

    if entregas_realizadas > 0:

        puntualidad = round(
            (
                (
                    entregas_realizadas -
                    entregas_tardias
                )
                /
                entregas_realizadas
            ) * 100,
            2
        )

    else:

        puntualidad = 0

    kilometros = round(
        registros[
            "distancia_real_km"
        ].sum(),
        2
    )

    # --------------------------------------------------------
    # CORRECCIÓN IMPORTANTE:
    # Se modifica el diccionario operador,
    # NO la lista operadores.
    # --------------------------------------------------------

    operador[
        "entregas_realizadas"
    ] = entregas_realizadas

    operador[
        "entregas_tardias"
    ] = entregas_tardias

    operador[
        "porcentaje_puntualidad"
    ] = puntualidad

    operador[
        "kilometros_recorridos"
    ] = kilometros


# Convertir nuevamente la lista a DataFrame

df_operadores = pd.DataFrame(
    operadores
)


# Guardar operadores actualizados

df_operadores.to_csv(
    os.path.join(
        DATOS_DIR,
        "operadores.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# ACTUALIZAR ESTADÍSTICAS DE RUTAS
# ============================================================

for ruta in rutas:

    id_ruta = ruta[
        "id_ruta"
    ]

    registros = df_entregas[
        df_entregas[
            "id_ruta"
        ] == id_ruta
    ]

    numero_entregas = len(
        registros
    )

    if numero_entregas > 0:

        retraso_promedio = round(
            registros[
                "minutos_retraso"
            ].mean(),
            2
        )

        retrasos = int(
            registros[
                "entrega_tardia"
            ].sum()
        )

        porcentaje_retraso = round(
            (
                retrasos /
                numero_entregas
            ) * 100,
            2
        )

        consumo_promedio = round(
            registros[
                "combustible_consumido_litros"
            ].mean(),
            2
        )

        costo_promedio = round(
            registros[
                "costo_total"
            ].mean(),
            2
        )

    else:

        retraso_promedio = 0
        porcentaje_retraso = 0
        consumo_promedio = 0
        costo_promedio = 0

    ruta[
        "numero_entregas"
    ] = numero_entregas

    ruta[
        "tiempo_promedio_minutos"
    ] = round(
        registros[
            "minutos_retraso"
        ].mean(),
        2
    ) if numero_entregas > 0 else 0

    ruta[
        "retraso_promedio_minutos"
    ] = retraso_promedio

    ruta[
        "porcentaje_retraso"
    ] = porcentaje_retraso

    ruta[
        "consumo_promedio"
    ] = consumo_promedio

    ruta[
        "costo_promedio"
    ] = costo_promedio


# Convertir rutas a DataFrame

df_rutas = pd.DataFrame(
    rutas
)


# Guardar rutas actualizadas

df_rutas.to_csv(
    os.path.join(
        DATOS_DIR,
        "rutas.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# ARCHIVOS GENERADOS
# ============================================================

print("\n==========================================")
print("       ARCHIVOS GENERADOS")
print("==========================================")

print(
    f"✓ {os.path.join(DATOS_DIR, 'clientes.csv')}"
)

print(
    f"✓ {os.path.join(DATOS_DIR, 'vehiculos.csv')}"
)

print(
    f"✓ {os.path.join(DATOS_DIR, 'operadores.csv')}"
)

print(
    f"✓ {os.path.join(DATOS_DIR, 'rutas.csv')}"
)

print(
    f"✓ {os.path.join(DATOS_DIR, 'componentes.csv')}"
)

print(
    f"✓ {os.path.join(DATOS_DIR, 'entregas.csv')}"
)

print("\n==========================================")
print("       PROCESO COMPLETADO")
print("==========================================")