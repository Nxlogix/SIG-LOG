import sqlite3
from pathlib import Path

import pandas as pd


# ============================================================
# SIG-LOG - CARGA DEL DATA WAREHOUSE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATOS_DIR = BASE_DIR / "datos" / "limpios"
DW_DIR = BASE_DIR / "data_warehouse"
DB_PATH = DW_DIR / "siglog_dw.db"

print("=" * 70)
print("             SIG-LOG - CARGA DATA WAREHOUSE")
print("=" * 70)


# ============================================================
# RUTAS DE ARCHIVOS
# ============================================================

archivos = {
    "clientes": DATOS_DIR / "clientes.csv",
    "vehiculos": DATOS_DIR / "vehiculos.csv",
    "operadores": DATOS_DIR / "operadores.csv",
    "rutas": DATOS_DIR / "rutas.csv",
    "componentes": DATOS_DIR / "componentes.csv",
    "entregas": DATOS_DIR / "entregas.csv",
    "mantenimiento": DATOS_DIR / "mantenimiento.csv",
}


# ============================================================
# CARGAR CSV
# ============================================================

print("\n1. CARGANDO ARCHIVOS CSV")
print("-" * 70)

clientes = pd.read_csv(archivos["clientes"])
vehiculos = pd.read_csv(archivos["vehiculos"])
operadores = pd.read_csv(archivos["operadores"])
rutas = pd.read_csv(archivos["rutas"])
componentes = pd.read_csv(archivos["componentes"])
entregas = pd.read_csv(archivos["entregas"])
mantenimiento = pd.read_csv(archivos["mantenimiento"])

print(f"✓ Clientes      : {len(clientes):,}")
print(f"✓ Vehículos     : {len(vehiculos):,}")
print(f"✓ Operadores    : {len(operadores):,}")
print(f"✓ Rutas         : {len(rutas):,}")
print(f"✓ Componentes   : {len(componentes):,}")
print(f"✓ Entregas      : {len(entregas):,}")
print(f"✓ Mantenimiento : {len(mantenimiento):,}")


# ============================================================
# CONEXIÓN
# ============================================================

conexion = sqlite3.connect(DB_PATH)

# Activar claves foráneas
conexion.execute("PRAGMA foreign_keys = ON")


# ============================================================
# LIMPIEZA / CONVERSIÓN DE IDs
# ============================================================

def convertir_id_entero(df, columna):
    if columna in df.columns:
        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce"
        )

        df[columna] = df[columna].astype("Int64")


# Dimensiones
convertir_id_entero(clientes, "id_cliente")
convertir_id_entero(vehiculos, "id_vehiculo")
convertir_id_entero(operadores, "id_operador")
convertir_id_entero(rutas, "id_ruta")
convertir_id_entero(componentes, "id_componente")

# Entregas
for columna in [
    "id_entrega",
    "id_cliente",
    "id_vehiculo",
    "id_operador",
    "id_ruta"
]:
    convertir_id_entero(entregas, columna)

# Mantenimiento
convertir_id_entero(mantenimiento, "id_vehiculo")
convertir_id_entero(mantenimiento, "id_componente")


# ============================================================
# DIMENSIONES
# ============================================================

print("\n2. CARGANDO DIMENSIONES")
print("-" * 70)

# ------------------------------------------------------------
# CLIENTES
# ------------------------------------------------------------

clientes.to_sql(
    "dim_cliente",
    conexion,
    if_exists="replace",
    index=False
)

print(f"✓ dim_cliente: {len(clientes):,}")


# ------------------------------------------------------------
# VEHICULOS
# ------------------------------------------------------------

vehiculos.to_sql(
    "dim_vehiculo",
    conexion,
    if_exists="replace",
    index=False
)

print(f"✓ dim_vehiculo: {len(vehiculos):,}")


# ------------------------------------------------------------
# OPERADORES
# ------------------------------------------------------------

operadores.to_sql(
    "dim_operador",
    conexion,
    if_exists="replace",
    index=False
)

print(f"✓ dim_operador: {len(operadores):,}")


# ------------------------------------------------------------
# RUTAS
# ------------------------------------------------------------

rutas.to_sql(
    "dim_ruta",
    conexion,
    if_exists="replace",
    index=False
)

print(f"✓ dim_ruta: {len(rutas):,}")


# ------------------------------------------------------------
# COMPONENTES
# ------------------------------------------------------------

componentes.to_sql(
    "dim_componente",
    conexion,
    if_exists="replace",
    index=False
)

print(f"✓ dim_componente: {len(componentes):,}")


# ============================================================
# DIMENSION TIEMPO
# ============================================================

print("\n3. GENERANDO DIMENSIÓN TIEMPO")
print("-" * 70)

fechas = pd.concat(
    [
        entregas["fecha_salida"],
        entregas["fecha_entrega_programada"],
        entregas["fecha_entrega_real"],
        mantenimiento["fecha_mantenimiento"]
    ],
    ignore_index=True
)

fechas = pd.to_datetime(
    fechas,
    errors="coerce"
)

fechas = fechas.dropna().drop_duplicates().sort_values()

dim_tiempo = pd.DataFrame({
    "fecha": fechas.dt.strftime("%Y-%m-%d")
})

dim_tiempo.insert(
    0,
    "id_tiempo",
    range(1, len(dim_tiempo) + 1)
)

dim_tiempo["fecha_dt"] = pd.to_datetime(
    dim_tiempo["fecha"]
)

dim_tiempo["anio"] = dim_tiempo["fecha_dt"].dt.year

dim_tiempo["trimestre"] = (
    dim_tiempo["fecha_dt"].dt.quarter
)

dim_tiempo["mes"] = (
    dim_tiempo["fecha_dt"].dt.month
)

dim_tiempo["nombre_mes"] = (
    dim_tiempo["fecha_dt"].dt.month_name()
)

dim_tiempo["semana"] = (
    dim_tiempo["fecha_dt"].dt.isocalendar().week
    .astype(int)
)

dim_tiempo["dia"] = (
    dim_tiempo["fecha_dt"].dt.day
)

dim_tiempo["dia_semana"] = (
    dim_tiempo["fecha_dt"].dt.dayofweek
)

dim_tiempo["nombre_dia"] = (
    dim_tiempo["fecha_dt"].dt.day_name()
)

dim_tiempo["fin_semana"] = (
    dim_tiempo["fecha_dt"]
    .dt.dayofweek
    .isin([5, 6])
    .astype(int)
)

dim_tiempo = dim_tiempo.drop(
    columns=["fecha_dt"]
)

dim_tiempo.to_sql(
    "dim_tiempo",
    conexion,
    if_exists="replace",
    index=False
)

print(f"✓ dim_tiempo: {len(dim_tiempo):,}")


# ============================================================
# MAPA FECHA -> ID TIEMPO
# ============================================================

mapa_tiempo = dict(
    zip(
        dim_tiempo["fecha"],
        dim_tiempo["id_tiempo"]
    )
)


# ============================================================
# HECHOS DE ENTREGAS
# ============================================================

print("\n4. CARGANDO HECHOS DE ENTREGAS")
print("-" * 70)

entregas["id_tiempo"] = (
    pd.to_datetime(
        entregas["fecha_salida"],
        errors="coerce"
    )
    .dt.strftime("%Y-%m-%d")
    .map(mapa_tiempo)
)

entregas.to_sql(
    "fact_entregas",
    conexion,
    if_exists="replace",
    index=False
)

print(f"✓ fact_entregas: {len(entregas):,}")


# ============================================================
# HECHOS DE MANTENIMIENTO
# ============================================================

print("\n5. CARGANDO HECHOS DE MANTENIMIENTO")
print("-" * 70)


# ------------------------------------------------------------
# ASEGURAR ID DE MANTENIMIENTO
# ------------------------------------------------------------

if "id_mantenimiento" not in mantenimiento.columns:

    mantenimiento.insert(
        0,
        "id_mantenimiento",
        [
            f"MNT-{i:05d}"
            for i in range(
                1,
                len(mantenimiento) + 1
            )
        ]
    )

else:

    mantenimiento["id_mantenimiento"] = (
        mantenimiento["id_mantenimiento"]
        .astype(str)
        .str.strip()
    )


# ------------------------------------------------------------
# FECHA
# ------------------------------------------------------------

mantenimiento["fecha_mantenimiento"] = (
    pd.to_datetime(
        mantenimiento["fecha_mantenimiento"],
        errors="coerce"
    )
    .dt.strftime("%Y-%m-%d")
)


# ------------------------------------------------------------
# NUMERICOS
# ------------------------------------------------------------

columnas_numericas = [
    "id_vehiculo",
    "id_componente",
    "kilometraje",
    "costo_repuesto",
    "costo_mano_obra",
    "costo_total",
    "horas_fuera_servicio"
]

for columna in columnas_numericas:

    if columna in mantenimiento.columns:

        mantenimiento[columna] = pd.to_numeric(
            mantenimiento[columna],
            errors="coerce"
        )


# ------------------------------------------------------------
# ELIMINAR REGISTROS SIN CLAVES FORÁNEAS
# ------------------------------------------------------------

mantenimiento = mantenimiento.dropna(
    subset=[
        "id_vehiculo",
        "id_componente"
    ]
).copy()

mantenimiento["id_vehiculo"] = (
    mantenimiento["id_vehiculo"]
    .astype(int)
)

mantenimiento["id_componente"] = (
    mantenimiento["id_componente"]
    .astype(int)
)


# ------------------------------------------------------------
# ELIMINAR DUPLICADOS
# ------------------------------------------------------------

mantenimiento = mantenimiento.drop_duplicates(
    subset=["id_mantenimiento"]
)


# ------------------------------------------------------------
# ORDENAR COLUMNAS
# ------------------------------------------------------------

columnas_mantenimiento = [
    "id_mantenimiento",
    "id_vehiculo",
    "id_componente",
    "fecha_mantenimiento",
    "tipo_mantenimiento",
    "causa_falla",
    "severidad",
    "kilometraje",
    "costo_repuesto",
    "costo_mano_obra",
    "costo_total",
    "horas_fuera_servicio",
    "descripcion"
]

columnas_existentes = [
    columna
    for columna in columnas_mantenimiento
    if columna in mantenimiento.columns
]

mantenimiento = mantenimiento[
    columnas_existentes
]


# ------------------------------------------------------------
# CARGAR
# ------------------------------------------------------------

mantenimiento.to_sql(
    "fact_mantenimiento",
    conexion,
    if_exists="replace",
    index=False
)

print(
    f"✓ fact_mantenimiento: "
    f"{len(mantenimiento):,}"
)


# ============================================================
# VERIFICACIÓN FINAL
# ============================================================

print("\n6. VERIFICACIÓN FINAL")
print("-" * 70)

tablas = [
    "dim_cliente",
    "dim_vehiculo",
    "dim_operador",
    "dim_ruta",
    "dim_componente",
    "dim_tiempo",
    "fact_entregas",
    "fact_mantenimiento"
]

for tabla in tablas:

    resultado = conexion.execute(
        f"SELECT COUNT(*) FROM {tabla}"
    ).fetchone()

    cantidad = resultado[0]

    print(
        f"✓ {tabla:<25} {cantidad:>8,} registros"
    )


# ============================================================
# VALIDACIONES
# ============================================================

print("\n7. VALIDACIONES")
print("-" * 70)

# Entregas
entregas_dw = conexion.execute(
    "SELECT COUNT(*) FROM fact_entregas"
).fetchone()[0]

# Mantenimiento
mantenimiento_dw = conexion.execute(
    "SELECT COUNT(*) FROM fact_mantenimiento"
).fetchone()[0]

# Dimensiones
clientes_dw = conexion.execute(
    "SELECT COUNT(*) FROM dim_cliente"
).fetchone()[0]

vehiculos_dw = conexion.execute(
    "SELECT COUNT(*) FROM dim_vehiculo"
).fetchone()[0]

componentes_dw = conexion.execute(
    "SELECT COUNT(*) FROM dim_componente"
).fetchone()[0]


print(
    f"Entregas esperadas : {len(entregas):,}"
)

print(
    f"Entregas cargadas  : {entregas_dw:,}"
)

print(
    f"Mantenimiento esperado : {len(mantenimiento):,}"
)

print(
    f"Mantenimiento cargado  : {mantenimiento_dw:,}"
)


if (
    entregas_dw == len(entregas)
    and mantenimiento_dw == len(mantenimiento)
    and clientes_dw == len(clientes)
    and vehiculos_dw == len(vehiculos)
    and componentes_dw == len(componentes)
):

    print("\n✓ TODAS LAS VALIDACIONES CORRECTAS")

else:

    print("\n⚠ REVISAR ALGUNA CANTIDAD")


# ============================================================
# CERRAR
# ============================================================

conexion.commit()
conexion.close()


print("\n" + "=" * 70)
print("             DATA WAREHOUSE CARGADO")
print("=" * 70)

print("\nBase de datos:")
print(f"✓ {DB_PATH}")

print("\nProceso completado correctamente.")
print("=" * 70)