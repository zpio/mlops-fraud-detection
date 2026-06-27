# 🛡️ End-to-End MLOps System for Credit Card Fraud Detection

Este repositorio contiene un sistema de producción completo y automatizado para la detección de fraudes en transacciones de tarjetas de crédito financieras. El proyecto va más allá del modelado estadístico tradicional, implementando una arquitectura robusta de **MLOps** que cubre desde la ingesta de datos controlada por versiones hasta el despliegue en contenedores, la automatización CI/CD y el monitoreo continuo de la degradación del modelo.

## 💼 Impacto de Negocio y Desafío Técnico
En el sector bancario, el fraude representa pérdidas millonarias directas. Técnicamente, este escenario presenta un desafío crítico: **un desbalance masivo de clases (0.17% de transacciones fraudulentas)**. 

Un enfoque ingenuo optimizado para la *Exactitud (Accuracy)* daría un 99.8%, pero fallaría en detectar el fraude. Este sistema implementa un modelo basado en la optimización estricta de las métricas **Recall** (minimizar el fraude que se escapa) y **Precision** (minimizar el bloqueo erróneo a clientes legítimos), logrando un balance equilibrado de **~80.6% en ambas métricas**.

---

## 🏗️ Arquitectura del Sistema y Flujo de Datos

El sistema está diseñado bajo principios de ingeniería de software modular. A continuación se detalla cómo interactúan los componentes del pipeline:

```text
[Datos en OpenML] 
       │
       ▼ (Ejecución Única)
1. src/data_ingest.py   ──► [data/raw/credit_card_fraud.csv]
       │
       ▼ (Pipeline Semanal / Reentrenamiento)
2. src/preprocess.py    ──► [data/processed/train.csv & test.csv] ──► [models/scaler.joblib]
       │
       ├──────────────────────────────────────┐
       ▼ (Mapeo de Experimentos)              ▼ (Vigilancia Estadística)
3. src/train.py ──► [Servidor MLflow]     5. src/monitor.py ──► [drift_report.html]
       │
       ▼ (Artefacto Registrado)
[models/fraud_model.joblib]
       │
       ▼ (Empaquetado Inmutable)
4. src/api.py (FastAPI) ──► [Contenedor Docker] ──► Consumo por App del Banco
```

-----

## 📂 Utilidad de los Archivos Core


- **src/data_ingest.py**: Automatiza la extracción segura de los datos reales desde repositorios en la nube hacia el almacenamiento local. Incluye manejo defensivo de excepciones y un sistema de logs (logging) para auditorías de infraestructura.

- **src/preprocess.py**: Realiza la división estratificada de los datos (manteniendo la proporción de fraude) y escala las variables numéricas. Implementa ingeniería defensiva para evitar la Fuga de Datos (Data Leakage) calculando los parámetros estadísticos únicamente sobre el conjunto de entrenamiento.

- **src/train.py**: Entrena un clasificador de ensamble balanceado. Se conecta con el servidor local de MLflow para registrar de forma automática los parámetros, las curvas de rendimiento y el artefacto final del modelo.

- **src/api.py**: Construye una interfaz de programación de aplicaciones (API REST) de alto rendimiento utilizando FastAPI. Traduce el modelo matemático en un servicio listo para ser consumido por equipos de Frontend o aplicaciones móviles mediante el envío de solicitudes JSON.

- **src/monitor.py**: Mitiga la degradación del modelo en producción analizando el Data Drift (Deriva de Datos) mediante EvidentlyAI. Compara las distribuciones de las transacciones entrantes con la línea base de entrenamiento utilizando pruebas estadísticas avanzadas.

-----


## 🛠️ Stack Tecnológico Utilizado


- **Engine & Modelado**: Python 3.12, Pandas, Scikit-Learn, Joblib.

- **Versionamiento de Datos**: DVC (Data Version Control) para desvincular el almacenamiento de datos pesados del historial de Git.

- **Tracking de Experimentos**: MLflow (Métricas, Parámetros y Registro de Modelos).

- **Servicio Web**: FastAPI, Uvicorn, Pydantic (Validación estricta de esquemas de datos).

- **Contenerización**: Docker y Docker Compose para garantizar entornos inmutables y reproducibles.

- **CI/CD**: GitHub Actions para pruebas automáticas de calidad de código y linter en la nube.

- **Monitoreo**: EvidentlyAI (Análisis estadístico de Drift).

-----

## 🚦 Guía de Operación Local

**1. Clonar e Instalar Dependencias**

```bash
git clone [https://github.com/zpio/mlops-fraud-detection.git](https://github.com/tu-usuario/mlops-fraud-detection.git)

cd mlops-fraud-detection

python -m venv venv

source venv/Scripts/activate

pip install -r requirements.txt
```

**2. Ingesta Automática de Datos**

Por motivos de optimización y seguridad, los archivos pesados no se almacenan en el repositorio. El sistema está diseñado para auto-reconstruirse. Ejecuta el script de ingesta para descargar la data cruda original de forma programática desde OpenML:

```bash
python src/data_ingest.py
```

**3. Ejecución del Pipeline y Tracking (Modo Desarrollo)**

Para inicializar el servidor de auditoría y lanzar el entrenamiento:

```bash
# Terminal 1: Iniciar el servidor central de tracking
mlflow server --host 127.0.0.1 --port 5000

# Terminal 2: Ejecutar la secuencia del pipeline
python src/preprocess.py
python src/train.py
```

**4. Despliegue de Producción Contenerizado (Modo Infraestructura)**

Para levantar el servicio de predicciones en tiempo real de forma aislada, sin depender de la configuración local del sistema operativo, ejecute:

```bash
docker-compose up -d --build
```

Una vez activo el contenedor, puede interactuar con la API y enviar transacciones de prueba ingresando desde el navegador a: http://localhost:8000/docs

-----

**🤖 Red de Seguridad CI/CD (GitHub Actions)**

Cada confirmación de código (git push) desencadena una tarea automatizada en los servidores de GitHub configurada en .github/workflows/ci.yml. Este pipeline realiza un proceso de Linting mediante flake8 para auditar la calidad del software, asegurar el cumplimiento de las buenas prácticas de programación (PEP 8) y evitar el despliegue de código corrupto o inestable.