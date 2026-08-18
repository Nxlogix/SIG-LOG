import sqlite3
from pathlib import Path

# ============================================================
# SIG-LOG - CREACIÓN DEL DATA WAREHOUSE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DW_DIR = BASE_DIR / "data_warehouse"
DB_PATH = DW_DIR / "siglog_dw.db"

DW_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("              SIG-LOG - DATA WAREHOUSE")
print("=" * 70)

print("\n1. CREANDO BASE DE DATOS")
print("-" * 70)

conexion = sqlite3.connect(DB_PATH)
cursor = conexion.cursor()

# ============================================================
# DIMENSION CLIENTE
# ============================================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS dim_cliente (
    id_cliente INTEGER PRIMARY KEY,
    nombre TEXT,
    telefono TEXT,
    correo TEXT,
    direccion TEXT,
    ciudad TEXT,
    estado TEXT,
    codigo_postal TEXT,
    fecha_registro TEXT,
    estatus TEXT
)
""")

# ============================================================
# DIMENSION VEHICULO
# ============================================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS dim_vehiculo (
    id_vehiculo INTEGER PRIMARY KEY,
    numero_economico TEXT,
    placas TEXT,
    marca TEXT,
    modelo TEXT,
    anio INTEGER,
    tipo TEXT,
    tipo_combustible TEXT,
    capacidad_toneladas REAL,
    kilometraje_actual REAL,
    rendimiento_km_l REAL,
    consumo_promedio REAL,
    fecha_adquisicion TEXT,
    ultima_revision TEXT,
    proxima_revision TEXT,
    nivel_riesgo TEXT,
    estatus TEXT
)
""")

# ============================================================
# DIMENSION OPERADOR
# ============================================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS dim_operador (
    id_operador INTEGER PRIMARY KEY,
    nombre TEXT,
    apellido TEXT,
    licencia TEXT,
    tipo_licencia TEXT,
    telefono TEXT,
    fecha_ingreso TEXT,
    experiencia_anios REAL,
    calificacion REAL,
    estatus TEXT
)
""")

# ============================================================
# DIMENSION RUTA
# ============================================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS dim_ruta (
    id_ruta INTEGER PRIMARY KEY,
    nombre_ruta TEXT,
    origen TEXT,
    destino TEXT,
    distancia_km REAL,
    tiempo_estimado_min REAL,
    tipo_ruta TEXT,
    nivel_dificultad TEXT,
    peajes REAL,
    estatus TEXT
)
""")

# ============================================================
# DIMENSION COMPONENTE
# ============================================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS dim_componente (
    id_componente INTEGER PRIMARY KEY,
    nombre TEXT,
    categoria TEXT,
    vida_util_km REAL,
    costo_reemplazo REAL,
    descripcion TEXT,
    estatus TEXT
)
""")

# ============================================================
# DIMENSION TIEMPO
# ============================================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS dim_tiempo (
    id_tiempo INTEGER PRIMARY KEY,
    fecha TEXT UNIQUE,
    anio INTEGER,
    trimestre INTEGER,
    mes INTEGER,
    nombre_mes TEXT,
    semana INTEGER,
    dia INTEGER,
    dia_semana INTEGER,
    nombre_dia TEXT,
    fin_semana INTEGER
)
""")

# ============================================================
# TABLA DE HECHOS - ENTREGAS
# ============================================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS fact_entregas (
    id_entrega INTEGER PRIMARY KEY,
    id_cliente INTEGER,
    id_vehiculo INTEGER,
    id_operador INTEGER,
    id_ruta INTEGER,
    id_tiempo INTEGER,

    fecha_salida TEXT,
    hora_salida TEXT,

    fecha_entrega_programada TEXT,
    hora_entrega_programada TEXT,

    fecha_entrega_real TEXT,
    hora_entrega_real TEXT,

    peso_carga REAL,
    cantidad_paquetes REAL,
    distancia_real_km REAL,

    tipo_combustible TEXT,
    combustible_consumido_litros REAL,
    precio_combustible REAL,

    costo_envio REAL,
    costo_combustible REAL,
    costo_total REAL,

    minutos_retraso REAL,
    entrega_tardia INTEGER,

    estatus TEXT,
    observaciones TEXT,

    FOREIGN KEY (id_cliente)
        REFERENCES dim_cliente(id_cliente),

    FOREIGN KEY (id_vehiculo)
        REFERENCES dim_vehiculo(id_vehiculo),

    FOREIGN KEY (id_operador)
        REFERENCES dim_operador(id_operador),

    FOREIGN KEY (id_ruta)
        REFERENCES dim_ruta(id_ruta),

    FOREIGN KEY (id_tiempo)
        REFERENCES dim_tiempo(id_tiempo)
)
""")

# ============================================================
# TABLA DE HECHOS - MANTENIMIENTO
# ============================================================
#
# IMPORTANTE:
# id_mantenimiento se maneja como TEXT para evitar errores
# si el generador utiliza identificadores alfanuméricos.
#

cursor.execute("""
CREATE TABLE IF NOT EXISTS fact_mantenimiento (
    id_mantenimiento TEXT PRIMARY KEY,

    id_vehiculo INTEGER,
    id_componente INTEGER,

    fecha_mantenimiento TEXT,
    tipo_mantenimiento TEXT,
    causa_falla TEXT,
    severidad TEXT,

    kilometraje REAL,

    costo_repuesto REAL,
    costo_mano_obra REAL,
    costo_total REAL,

    horas_fuera_servicio REAL,

    descripcion TEXT,

    FOREIGN KEY (id_vehiculo)
        REFERENCES dim_vehiculo(id_vehiculo),

    FOREIGN KEY (id_componente)
        REFERENCES dim_componente(id_componente)
)
""")

# ============================================================
# INDICES
# ============================================================

print("\n2. CREANDO ÍNDICES")
print("-" * 70)

indices = [

    """
    CREATE INDEX IF NOT EXISTS idx_entregas_cliente
    ON fact_entregas(id_cliente)
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_entregas_vehiculo
    ON fact_entregas(id_vehiculo)
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_entregas_operador
    ON fact_entregas(id_operador)
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_entregas_ruta
    ON fact_entregas(id_ruta)
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_entregas_tiempo
    ON fact_entregas(id_tiempo)
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_mantenimiento_vehiculo
    ON fact_mantenimiento(id_vehiculo)
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_mantenimiento_componente
    ON fact_mantenimiento(id_componente)
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_mantenimiento_fecha
    ON fact_mantenimiento(fecha_mantenimiento)
    """
]

for indice in indices:
    cursor.execute(indice)

conexion.commit()

# ============================================================
# VERIFICACIÓN
# ============================================================

print("\n3. VERIFICANDO ESTRUCTURA")
print("-" * 70)

cursor.execute("""
SELECT name
FROM sqlite_master
WHERE type='table'
ORDER BY name
""")

tablas = cursor.fetchall()

for tabla in tablas:
    print(f"✓ {tabla[0]}")

conexion.close()

print("\n" + "=" * 70)
print("             DATA WAREHOUSE CREADO")
print("=" * 70)

print("\nBase de datos:")
print(f"✓ {DB_PATH}")

print("\nTablas creadas:")
for tabla in tablas:
    print(f"✓ {tabla[0]}")

print("\nSiguiente paso:")
print("Ejecutar cargar_warehouse.py")

print("=" * 70)