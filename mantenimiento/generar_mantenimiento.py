import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# SIG-LOG - GENERADOR DE HISTORIAL DE MANTENIMIENTO
# ============================================================

print("=" * 70)
print("          SIG-LOG - HISTORIAL DE MANTENIMIENTO")
print("=" * 70)

BASE_DIR = Path(__file__).resolve().parent.parent

VEHICULOS = BASE_DIR / "datos" / "limpios" / "vehiculos.csv"
COMPONENTES = BASE_DIR / "datos" / "limpios" / "componentes.csv"

SALIDA_DIR = BASE_DIR / "datos" / "limpios"
SALIDA_DIR.mkdir(parents=True, exist_ok=True)

SALIDA = SALIDA_DIR / "mantenimiento.csv"

np.random.seed(42)

# ============================================================
# 1. CARGAR DATOS
# ============================================================

print("\n1. CARGANDO DATOS")
print("-" * 70)

vehiculos = pd.read_csv(VEHICULOS)
componentes = pd.read_csv(COMPONENTES)

print(f"Vehículos disponibles   : {len(vehiculos):,}")
print(f"Componentes disponibles : {len(componentes):,}")

# ============================================================
# 2. PREPARAR FECHAS
# ============================================================

vehiculos["ultima_revision"] = pd.to_datetime(
    vehiculos["ultima_revision"],
    errors="coerce"
)

vehiculos["proxima_revision"] = pd.to_datetime(
    vehiculos["proxima_revision"],
    errors="coerce"
)

vehiculos["fecha_adquisicion"] = pd.to_datetime(
    vehiculos["fecha_adquisicion"],
    errors="coerce"
)

# ============================================================
# 3. CATÁLOGOS
# ============================================================

tipos_mantenimiento = [
    "Preventivo",
    "Preventivo",
    "Preventivo",
    "Correctivo",
    "Correctivo",
    "Emergencia"
]

severidades = [
    "Baja",
    "Baja",
    "Media",
    "Media",
    "Alta",
    "Crítica"
]

causas_por_categoria = {
    "Motor": [
        "Desgaste por kilometraje",
        "Falta de lubricación",
        "Sobrecalentamiento",
        "Desgaste interno",
        "Fuga de aceite"
    ],
    "Frenos": [
        "Desgaste de pastillas",
        "Desgaste de discos",
        "Pérdida de presión",
        "Fuga de líquido",
        "Desgaste por uso"
    ],
    "Suspensión": [
        "Desgaste por uso",
        "Fatiga del componente",
        "Golpe por irregularidad del camino",
        "Desgaste de bujes",
        "Holgura"
    ],
    "Transmisión": [
        "Desgaste por kilometraje",
        "Falta de lubricación",
        "Desgaste de engranajes",
        "Sobrecalentamiento",
        "Fuga de aceite"
    ],
    "Eléctrico": [
        "Falla eléctrica",
        "Desgaste de batería",
        "Corto circuito",
        "Conexión defectuosa",
        "Fusible dañado"
    ],
    "Llantas": [
        "Desgaste de banda",
        "Ponchadura",
        "Daño por objeto",
        "Desgaste irregular",
        "Baja presión"
    ],
    "Refrigeración": [
        "Fuga de refrigerante",
        "Sobrecalentamiento",
        "Desgaste de manguera",
        "Falla de bomba",
        "Obstrucción"
    ],
    "Combustible": [
        "Obstrucción",
        "Fuga",
        "Desgaste de bomba",
        "Falla de inyector",
        "Contaminación"
    ]
}

# ============================================================
# 4. GENERAR REGISTROS
# ============================================================

print("\n2. GENERANDO HISTORIAL")
print("-" * 70)

registros = []

contador = 1

for _, vehiculo in vehiculos.iterrows():

    # Entre 5 y 15 eventos por vehículo
    cantidad_eventos = np.random.randint(5, 16)

    fecha_inicio = vehiculo["fecha_adquisicion"]

    if pd.isna(fecha_inicio):
        fecha_inicio = pd.Timestamp("2024-01-01")

    fecha_fin = pd.Timestamp("2026-08-15")

    fechas = pd.to_datetime(
        np.random.randint(
            fecha_inicio.value // 10**9,
            fecha_fin.value // 10**9,
            cantidad_eventos
        ),
        unit="s"
    )

    fechas = sorted(fechas)

    for fecha in fechas:

        componente = componentes.sample(
            n=1,
            random_state=contador
        ).iloc[0]

        categoria = str(componente["categoria"])

        causas = causas_por_categoria.get(
            categoria,
            [
                "Desgaste por uso",
                "Falla mecánica",
                "Falla eléctrica",
                "Mantenimiento preventivo"
            ]
        )

        tipo = np.random.choice(tipos_mantenimiento)

        severidad = np.random.choice(severidades)

        causa = np.random.choice(causas)

        kilometraje_actual = float(
            vehiculo["kilometraje_actual"]
        )

        # Kilometraje aproximado en el momento del mantenimiento
        km_evento = max(
            0,
            kilometraje_actual - np.random.randint(0, 120000)
        )

        # Costo base del componente
        costo_componente = float(
            componente["costo_reemplazo"]
        )

        # Mano de obra
        mano_obra = np.random.randint(500, 3501)

        # Si es preventivo suele ser más económico
        if tipo == "Preventivo":
            factor = np.random.uniform(0.35, 0.80)
        elif tipo == "Correctivo":
            factor = np.random.uniform(0.80, 1.50)
        else:
            factor = np.random.uniform(1.20, 2.00)

        costo_repuesto = costo_componente * factor

        costo_total = costo_repuesto + mano_obra

        # Tiempo fuera de servicio
        if severidad == "Baja":
            horas = np.random.randint(1, 5)
        elif severidad == "Media":
            horas = np.random.randint(4, 13)
        elif severidad == "Alta":
            horas = np.random.randint(12, 31)
        else:
            horas = np.random.randint(24, 73)

        # Solución
        if tipo == "Preventivo":
            soluciones = [
                "Inspección y ajuste",
                "Lubricación",
                "Cambio preventivo",
                "Ajuste del componente",
                "Revisión general"
            ]
        else:
            soluciones = [
                "Reemplazo del componente",
                "Reparación del componente",
                "Ajuste y calibración",
                "Corrección de falla",
                "Cambio de pieza dañada"
            ]

        solucion = np.random.choice(soluciones)

        estado = np.random.choice(
            [
                "Completado",
                "Completado",
                "Completado",
                "Pendiente"
            ]
        )

        registros.append({
            "id_mantenimiento": f"M{contador:05d}",
            "id_vehiculo": vehiculo["id_vehiculo"],
            "id_componente": componente["id_componente"],
            "fecha_mantenimiento": fecha.strftime("%Y-%m-%d"),
            "kilometraje_mantenimiento": round(km_evento, 0),
            "tipo_mantenimiento": tipo,
            "categoria_componente": categoria,
            "componente": componente["nombre"],
            "severidad": severidad,
            "causa_falla": causa,
            "solucion": solucion,
            "horas_fuera_servicio": horas,
            "costo_repuesto": round(costo_repuesto, 2),
            "costo_mano_obra": mano_obra,
            "costo_total": round(costo_total, 2),
            "estado": estado
        })

        contador += 1

# ============================================================
# 5. DATAFRAME FINAL
# ============================================================

mantenimiento = pd.DataFrame(registros)

mantenimiento = mantenimiento.sort_values(
    ["id_vehiculo", "fecha_mantenimiento"]
).reset_index(drop=True)

# ============================================================
# 6. GUARDAR
# ============================================================

mantenimiento.to_csv(
    SALIDA,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 7. RESUMEN
# ============================================================

print(f"\nRegistros generados: {len(mantenimiento):,}")

print(
    f"Vehículos involucrados: "
    f"{mantenimiento['id_vehiculo'].nunique():,}"
)

print(
    f"Componentes involucrados: "
    f"{mantenimiento['id_componente'].nunique():,}"
)

print(
    f"Costo total histórico: "
    f"${mantenimiento['costo_total'].sum():,.2f}"
)

print(
    f"Tiempo fuera de servicio: "
    f"{mantenimiento['horas_fuera_servicio'].sum():,.0f} horas"
)

print("\nTipos de mantenimiento:")
print(
    mantenimiento["tipo_mantenimiento"]
    .value_counts()
)

print("\nSeveridad:")
print(
    mantenimiento["severidad"]
    .value_counts()
)

print("\nArchivo generado:")
print(f"✓ {SALIDA}")

print("\n" + "=" * 70)
print("              MANTENIMIENTO COMPLETADO")
print("=" * 70)