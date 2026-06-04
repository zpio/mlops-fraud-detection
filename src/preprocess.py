import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Configuración de logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

def preprocess_data():
    """
    Carga los datos crudos, realiza la división estratificada,
    escala las variables numéricas sin fuga de datos y guarda los artefactos.
    """
    raw_data_path = os.path.join("data", "raw", "credit_card_fraud.csv")
    processed_dir = os.path.join("data", "processed")
    models_dir = "models"

    # Asegurar que los directorios de salida existan
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(raw_data_path):
        logging.error(f"No se encontró el archivo {raw_data_path}. Ejecuta data_ingest.py primero.")
        return

    logging.info("Cargando datos crudos para preprocesamiento...")
    df = pd.read_csv(raw_data_path)

    # Separar características (X) y etiqueta objetivo (y)
    X = df.drop(columns=["target"])
    y = df["target"]

    logging.info("Realizando división estratificada (Train/Test split)...")
    # Stratify=y asegura la misma proporción de fraudes en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Buscamos dinámicamente las columnas numéricas que no sean las de PCA (V1-V28)
    # Esto previene caídas si la fuente de datos cambia ligeramente los nombres
    posibles_columnas = ['Time', 'Amount', 'time', 'amount']
    cols_to_scale = [col for col in posibles_columnas if col in X_train.columns]
    
    if not cols_to_scale:
        logging.warning("No se encontraron columnas 'Time' o 'Amount' para escalar.")
    else:
        logging.info(f"Escalando características encontradas: {cols_to_scale}...")
        scaler = StandardScaler()
        
        # IMPORTANTE: Ajustamos (fit) SOLO con X_train para evitar Data Leakage
        X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
        
        # Aplicamos la transformación a X_test
        X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    # Guardar los datasets procesados
    logging.info("Guardando datasets procesados en data/processed/...")
    
    # Unimos X e y temporalmente solo para guardarlos ordenados en CSV
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    train_data.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    test_data.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

    # Guardar el scaler para usarlo en producción (la API lo necesitará)
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    logging.info(f"Scaler guardado exitosamente en {scaler_path}")

    logging.info("Preprocesamiento completado con éxito.")

if __name__ == "__main__":
    logging.info("--- Iniciando Etapa de Preprocesamiento ---")
    preprocess_data()
    logging.info("--- Etapa de Preprocesamiento Finalizada ---")