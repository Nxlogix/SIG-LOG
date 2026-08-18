import sqlite3
from pathlib import Path
import pandas as pd


# ============================================================
# SIG-LOG - CONSULTAS DATA WAREHOUSE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data_warehouse" / "siglog_dw.db"


def conexion_dw():
    """Crea una conexión al Data Warehouse."""
    return sqlite3.connect(DB_PATH)


def ejecutar_consulta(sql, params=None):
    """Ejecuta una consulta y devuelve un DataFrame."""
    conexion = conexion_dw()

    try:
        if params:
            df = pd.read_sql_query(sql, conexion, params=params)
        else:
            df = pd.read_sql_query(sql, conexion)

        return df

    finally:
        conexion.close()


# ============================================================
# 1. RESUMEN GENERAL
# ============================================================

def resumen_general():

    sql = """
    SELECT
        COUNT(*) AS total_entregas,
        SUM(entrega_tardia) AS entregas_tardias,
        AVG(minutos_retraso) AS retraso_promedio,
        SUM(costo_total) AS costo_total,
        AVG(distancia_real_km) AS distancia_promedio,
        SUM(combustible_consumido_litros) AS combustible_total
    FROM fact_entregas
    """

    return ejecutar_consulta(sql)


# ============================================================
# 2. ENTREGAS POR ESTATUS
# ============================================================

def entregas_por_estatus():

    sql = """
    SELECT
        estatus,
        COUNT(*) AS cantidad
    FROM fact_entregas
    GROUP BY estatus
    ORDER BY cantidad DESC
    """

    return ejecutar_consulta(sql)


# ============================================================
# 3. ENTREGAS TARDÍAS
# ============================================================

def entregas_tardias():

    sql = """
    SELECT
        entrega_tardia,
        COUNT(*) AS cantidad
    FROM fact_entregas
    GROUP BY entrega_tardia
    ORDER BY entrega_tardia
    """

    return ejecutar_consulta(sql)


# ============================================================
# 4. ENTREGAS POR MES
# ============================================================

def entregas_por_mes():

    sql = """
    SELECT
        t.anio,
        t.mes,
        t.nombre_mes,
        COUNT(f.id_entrega) AS cantidad_entregas,
        SUM(f.entrega_tardia) AS entregas_tardias,
        AVG(f.minutos_retraso) AS retraso_promedio
    FROM fact_entregas f

    INNER JOIN dim_tiempo t
        ON f.id_tiempo = t.id_tiempo

    GROUP BY
        t.anio,
        t.mes,
        t.nombre_mes

    ORDER BY
        t.anio,
        t.mes
    """

    return ejecutar_consulta(sql)


# ============================================================
# 5. RETRASOS POR RUTA
# ============================================================

def retrasos_por_ruta():

    sql = """
    SELECT
        r.id_ruta,
        r.nombre_ruta,
        r.origen,
        r.destino,

        COUNT(f.id_entrega) AS entregas,

        AVG(f.minutos_retraso) AS retraso_promedio,

        SUM(f.entrega_tardia) AS entregas_tardias,

        AVG(f.distancia_real_km) AS distancia_promedio

    FROM fact_entregas f

    INNER JOIN dim_ruta r
        ON f.id_ruta = r.id_ruta

    GROUP BY
        r.id_ruta,
        r.nombre_ruta,
        r.origen,
        r.destino

    ORDER BY
        retraso_promedio DESC
    """

    return ejecutar_consulta(sql)


# ============================================================
# 6. DESEMPEÑO DE VEHÍCULOS
# ============================================================

def desempeno_vehiculos():

    sql = """
    SELECT
        v.id_vehiculo,
        v.numero_economico,
        v.marca,
        v.modelo,

        COUNT(f.id_entrega) AS entregas,

        SUM(f.entrega_tardia) AS entregas_tardias,

        AVG(f.minutos_retraso) AS retraso_promedio,

        AVG(f.distancia_real_km) AS distancia_promedio,

        SUM(f.combustible_consumido_litros) AS combustible_total,

        SUM(f.costo_total) AS costo_total

    FROM fact_entregas f

    INNER JOIN dim_vehiculo v
        ON f.id_vehiculo = v.id_vehiculo

    GROUP BY
        v.id_vehiculo,
        v.numero_economico,
        v.marca,
        v.modelo

    ORDER BY
        retraso_promedio DESC
    """

    return ejecutar_consulta(sql)


# ============================================================
# 7. COSTOS POR VEHÍCULO
# ============================================================

def costos_por_vehiculo():

    sql = """
    SELECT
        v.id_vehiculo,
        v.numero_economico,
        v.marca,
        v.modelo,

        COUNT(f.id_entrega) AS entregas,

        SUM(f.costo_envio) AS costo_envio,

        SUM(f.costo_combustible) AS costo_combustible,

        SUM(f.costo_total) AS costo_total

    FROM fact_entregas f

    INNER JOIN dim_vehiculo v
        ON f.id_vehiculo = v.id_vehiculo

    GROUP BY
        v.id_vehiculo,
        v.numero_economico,
        v.marca,
        v.modelo

    ORDER BY
        costo_total DESC
    """

    return ejecutar_consulta(sql)


# ============================================================
# 8. COMBUSTIBLE POR VEHÍCULO
# ============================================================

def combustible_por_vehiculo():

    sql = """
    SELECT
        v.id_vehiculo,
        v.numero_economico,
        v.marca,
        v.modelo,

        SUM(f.combustible_consumido_litros) AS combustible_total,

        AVG(f.combustible_consumido_litros) AS combustible_promedio,

        SUM(f.distancia_real_km) AS distancia_total

    FROM fact_entregas f

    INNER JOIN dim_vehiculo v
        ON f.id_vehiculo = v.id_vehiculo

    GROUP BY
        v.id_vehiculo,
        v.numero_economico,
        v.marca,
        v.modelo

    ORDER BY
        combustible_total DESC
    """

    return ejecutar_consulta(sql)


# ============================================================
# 9. COSTO VS RETRASO
# ============================================================

def costo_vs_retraso():

    sql = """
    SELECT
        id_entrega,
        costo_total,
        minutos_retraso,
        entrega_tardia
    FROM fact_entregas
    """

    return ejecutar_consulta(sql)


# ============================================================
# 10. DISTANCIA VS RETRASO
# ============================================================

def distancia_vs_retraso():

    sql = """
    SELECT
        id_entrega,
        distancia_real_km,
        minutos_retraso,
        entrega_tardia
    FROM fact_entregas
    """

    return ejecutar_consulta(sql)


# ============================================================
# 11. MANTENIMIENTO POR TIPO
# ============================================================

def mantenimiento_por_tipo():

    sql = """
    SELECT
        tipo_mantenimiento,
        COUNT(*) AS cantidad,
        SUM(costo_total) AS costo_total,
        SUM(horas_fuera_servicio) AS horas_fuera_servicio
    FROM fact_mantenimiento
    GROUP BY tipo_mantenimiento
    ORDER BY cantidad DESC
    """

    return ejecutar_consulta(sql)


# ============================================================
# 12. MANTENIMIENTO POR VEHÍCULO
# ============================================================

def mantenimiento_por_vehiculo():

    sql = """
    SELECT
        v.id_vehiculo,
        v.numero_economico,

        COUNT(m.id_mantenimiento) AS mantenimientos,

        SUM(
            CASE
                WHEN m.tipo_mantenimiento = 'Correctivo'
                THEN 1
                ELSE 0
            END
        ) AS fallas,

        SUM(
            CASE
                WHEN m.tipo_mantenimiento = 'Emergencia'
                THEN 1
                ELSE 0
            END
        ) AS emergencias,

        SUM(m.costo_total) AS costo_total,

        SUM(m.horas_fuera_servicio) AS horas_fuera_servicio

    FROM fact_mantenimiento m

    INNER JOIN dim_vehiculo v
        ON m.id_vehiculo = v.id_vehiculo

    GROUP BY
        v.id_vehiculo,
        v.numero_economico

    ORDER BY
        costo_total DESC
    """

    return ejecutar_consulta(sql)


# ============================================================
# 13. FALLAS POR COMPONENTE
# ============================================================

def fallas_por_componente():

    sql = """
    SELECT
        c.id_componente,
        c.nombre AS nombre_componente,
        c.categoria,

        COUNT(m.id_mantenimiento) AS fallas,

        COUNT(DISTINCT m.id_vehiculo) AS vehiculos_afectados,

        SUM(m.costo_total) AS costo_total

    FROM fact_mantenimiento m

    INNER JOIN dim_componente c
        ON m.id_componente = c.id_componente

    GROUP BY
        c.id_componente,
        c.nombre,
        c.categoria

    ORDER BY
        fallas DESC
    """

    return ejecutar_consulta(sql)


# ============================================================
# 14. VEHÍCULOS EN RIESGO
# ============================================================

def vehiculos_riesgo():

    sql = """
    SELECT
        id_vehiculo,
        numero_economico,
        marca,
        modelo,
        nivel_riesgo,
        estatus
    FROM dim_vehiculo

    ORDER BY
        nivel_riesgo
    """

    return ejecutar_consulta(sql)


# ============================================================
# PRUEBA DEL ARCHIVO
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("       SIG-LOG - PRUEBA DE CONSULTAS DATA WAREHOUSE")
    print("=" * 70)

    print("\n1. RESUMEN GENERAL")
    print("-" * 70)
    print(resumen_general().to_string(index=False))

    print("\n2. ENTREGAS POR ESTATUS")
    print("-" * 70)
    print(entregas_por_estatus().to_string(index=False))

    print("\n3. ENTREGAS POR MES")
    print("-" * 70)
    print(entregas_por_mes().to_string(index=False))

    print("\n4. MANTENIMIENTO POR TIPO")
    print("-" * 70)
    print(mantenimiento_por_tipo().to_string(index=False))

    print("\n" + "=" * 70)
    print("             CONSULTAS DISPONIBLES")
    print("=" * 70)

    funciones = [
        "resumen_general()",
        "entregas_por_estatus()",
        "entregas_tardias()",
        "entregas_por_mes()",
        "retrasos_por_ruta()",
        "desempeno_vehiculos()",
        "costos_por_vehiculo()",
        "combustible_por_vehiculo()",
        "costo_vs_retraso()",
        "distancia_vs_retraso()",
        "mantenimiento_por_tipo()",
        "mantenimiento_por_vehiculo()",
        "fallas_por_componente()",
        "vehiculos_riesgo()"
    ]

    for funcion in funciones:
        print(f"    {funcion}")

    print("\n" + "=" * 70)