# SIG-LOG
## Sistema Integral de Gestión Logística

Sistema integral para la gestión, análisis y optimización de operaciones logísticas mediante procesamiento de datos, análisis estadístico, aprendizaje automático y herramientas de visualización.

---

##  Descripción del proyecto

SIG-LOG es un sistema orientado al análisis y gestión de información logística. El proyecto integra diferentes procesos para transformar datos operativos en información útil para la toma de decisiones.

El sistema contempla:

- Gestión y análisis de clientes.
- Gestión y análisis de vehículos.
- Gestión de operadores.
- Análisis de rutas.
- Seguimiento de entregas.
- Análisis de combustible.
- Gestión y análisis de mantenimiento.
- Procesamiento y limpieza de datos.
- Data Warehouse.
- Análisis exploratorio.
- Análisis de componentes principales (PCA).
- Modelos de aprendizaje supervisado.
- Análisis no supervisado mediante clustering.
- Predicciones y evaluación de modelos.
- Dashboard de consulta y análisis.
- Generación de gráficas y reportes.

---


Tecnologías utilizadas

El proyecto utiliza principalmente:

Python
Pandas
NumPy
Scikit-learn
Matplotlib
SQLite
Streamlit
Jupyter / entorno Python
Git y GitHub


Flujo de procesamiento

El flujo general del sistema es:
Datos originales
       │
       ▼
Carga de datos
       │
       ▼
Limpieza
       │
       ▼
Transformación
       │
       ▼
Datos preparados
       │
       ▼
Data Warehouse
       │
       ├───────────────► Análisis exploratorio
       │
       ├───────────────► PCA
       │
       ├───────────────► Modelos supervisados
       │
       └───────────────► Clustering
                              │
                              ▼
                         Resultados
                              │
                              ▼
                          Dashboard



Análisis y modelos
Análisis supervisado

Se incluyen modelos para tareas de clasificación y regresión.

Entre los modelos entrenados se encuentran:

Decision Tree
Random Forest
Logistic Regression
Linear Regression

También se incluyen versiones optimizadas de algunos modelos.

Los modelos entrenados se almacenan en:

modelos_entrenados/


Análisis no supervisado

El proyecto utiliza técnicas de clustering para identificar grupos de entregas con características similares.

Los scripts correspondientes se encuentran en:

no_supervisado/

Los resultados generados se almacenan en:

no_supervisado/resultados/

Análisis PCA

El análisis de Componentes Principales permite estudiar la variabilidad de las variables utilizadas en el análisis de entregas.

Los scripts se encuentran en:

pca/

Los resultados se almacenan en:

pca/resultados/

Y las gráficas correspondientes se encuentran en:

graficas/pca/
Visualización

El proyecto contiene gráficas relacionadas con:

Entregas por estatus.
Entregas tardías.
Entregas por periodo.
Retrasos por ruta.
Retrasos por vehículo.
Costos.
Consumo de combustible.
Mantenimiento.
Fallas.
Severidad.
Riesgo de vehículos.
Análisis PCA.

Las gráficas se encuentran principalmente en:

graficas/

Dashboard

El sistema cuenta con un dashboard para consultar y visualizar los resultados del análisis.

El código principal se encuentra en:

dashboard/dashboard.py

Los módulos operativos se encuentran en:

dashboard/modulos_operativos_siglog.py

Data Warehouse

SIG-LOG cuenta con un Data Warehouse local basado en SQLite.

Los archivos relacionados se encuentran en:

data_warehouse/

Incluyendo:

crear_warehouse.py
cargar_warehouse.py
consultas_warehouse.py
siglog_dw.db

El Data Warehouse permite centralizar los datos procesados para facilitar las consultas y análisis.

Nota: XAMPP/MySQL corresponde a la base de datos operacional definida para el sistema. El análisis y los procesos de Data Warehouse del proyecto utilizan el almacenamiento configurado dentro de data_warehouse/.

 
 Base de datos operacional

La estructura operacional del sistema contempla entidades como:

Clientes
Vehículos
Operadores
Rutas
Entregas
Combustible
Componentes
Mantenimientos

Las relaciones entre estas entidades permiten conservar la integridad de la información logística.

La documentación de la estructura de datos puede consultarse en:

manuales/Diccionario_Datos_iniciales_SIG_LOG.pdf

 
 Documentación

El proyecto incluye documentación técnica y de usuario.

Manual técnico
manuales/Manual_Tecnico_SIG_LOG.pdf

Contiene información relacionada con:

Arquitectura.
Estructura del sistema.
Componentes.
Procesamiento de datos.
Modelos.
Data Warehouse.
Configuración.
Implementación.
Manual de usuario
manuales/Manual_de_Usuario_SIG_LOG.pdf

Contiene las instrucciones para utilizar el sistema y consultar sus principales funcionalidades.

Diccionario de datos
manuales/Diccionario_Datos_iniciales_SIG_LOG.pdf

Contiene la descripción de las estructuras y variables utilizadas en el sistema.

Guía de instalación

Para realizar la implementación desde cero se debe consultar:

README_INSTALACION_SIGLOG.md
 Instalación
1. Clonar el repositorio
git clone https://github.com/Nxlogix/SIG-LOG.git
cd SIG-LOG
2. Crear un entorno virtual

En Windows:

python -m venv venv

Activar:

venv\Scripts\activate

En caso de utilizar Anaconda, también puede utilizarse un entorno de Conda.

3. Instalar dependencias
pip install -r requirements.txt
4. Revisar la documentación

Antes de ejecutar el sistema se recomienda consultar:

README_INSTALACION_SIGLOG.md

y los manuales ubicados en:

manuales/

Ejecución

La ejecución depende del flujo que se desee utilizar.

Los principales componentes se encuentran en:

etl/
data_warehouse/
modelos/
pca/
no_supervisado/
dashboard/

Para la ejecución específica del dashboard y los procesos de análisis, consultar el manual técnico y la guía de instalación.


Resultados

Los resultados generados por el sistema se encuentran organizados en diferentes carpetas:

mantenimiento/resultados/
no_supervisado/resultados/
pca/resultados/
reportes_modelos/
graficas/

Estos archivos contienen resultados de análisis, evaluaciones, predicciones, agrupaciones y visualizaciones.


Proyecto académico

Proyecto: SIG-LOG
Nombre: Sistema Integral de Gestión Logística

El proyecto integra técnicas de:

Ingeniería y preparación de datos.
Análisis exploratorio.
Análisis estadístico.
Aprendizaje automático.
Reducción de dimensionalidad.
Clustering.
Visualización de datos.
Gestión de información logística.

Recomendación para la implementación

Para una implementación completa se recomienda seguir este orden:

1. Clonar repositorio
        ↓
2. Instalar Python y dependencias
        ↓
3. Revisar README_INSTALACION_SIGLOG.md
        ↓
4. Configurar la base de datos operacional
        ↓
5. Preparar / cargar datos
        ↓
6. Ejecutar procesos ETL
        ↓
7. Crear / cargar Data Warehouse
        ↓
8. Ejecutar análisis
        ↓
9. Consultar resultados
        ↓
10. Ejecutar Dashboard
🔗 Repositorio

Repositorio oficial del proyecto:

https://github.com/Nxlogix/SIG-LOG



### Después de pegarlo


Desde:


```powershell
C:\Users\sofia\Documents\SIG-LOG

ejecuta:

git add README.md
git commit -m "Agregar README principal del proyecto"
git push
