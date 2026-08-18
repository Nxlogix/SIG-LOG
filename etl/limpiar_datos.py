"""
=========================================================
SIG-LOG
Extracción del conocimiento en bases de datos

Proceso ETL
1. Limpieza de datos

Autor: Sofía
=========================================================
"""

import pandas as pd
from pathlib import Path

# =========================================================
# RUTAS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATOS = BASE_DIR / "datos"

SALIDA = DATOS / "limpios"

SALIDA.mkdir(exist_ok=True)

# =========================================================
# ARCHIVOS
# =========================================================

archivos = {
    "clientes": DATOS / "clientes.csv",
    "vehiculos": DATOS / "vehiculos.csv",
    "operadores": DATOS / "operadores.csv",
    "rutas": DATOS / "rutas.csv",
    "componentes": DATOS / "componentes.csv",
    "entregas": DATOS / "entregas.csv"
}

print("=" * 60)
print("       SIG-LOG - ETL: LIMPIEZA DE DATOS")
print("=" * 60)

# =========================================================
# LIMPIEZA
# =========================================================

for nombre, archivo in archivos.items():

    print(f"\nProcesando: {nombre.upper()}")

    df = pd.read_csv(archivo)

    registros_originales = len(df)

    # -----------------------------
    # Eliminar duplicados
    # -----------------------------

    duplicados = df.duplicated().sum()

    if duplicados > 0:
        df = df.drop_duplicates()

    # -----------------------------
    # Eliminar filas totalmente vacías
    # -----------------------------

    df = df.dropna(how="all")

    # -----------------------------
    # Limpiar espacios
    # -----------------------------

    for col in df.select_dtypes(include="object"):

        df[col] = df[col].astype(str).str.strip()

    # -----------------------------
    # Valores nulos
    # -----------------------------

    nulos = df.isnull().sum().sum()

    # -----------------------------
    # Guardar
    # -----------------------------

    salida = SALIDA / f"{nombre}.csv"

    df.to_csv(salida, index=False)

    print(f"Registros originales : {registros_originales}")
    print(f"Duplicados           : {duplicados}")
    print(f"Valores nulos        : {nulos}")
    print(f"Registros finales    : {len(df)}")
    print("Estado               : OK")

print("\n" + "=" * 60)
print("LIMPIEZA FINALIZADA")
print("=" * 60)

print("\nArchivos guardados en:")

for archivo in SALIDA.glob("*.csv"):
    print("✓", archivo)