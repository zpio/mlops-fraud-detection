import os
import logging
import pandas as pd
from sklearn.datasets import fetch_openml

# 1. Configuración del sistema de Logs (Auditoría)
# Esto guardará un registro tanto en la consola como en un archivo llamado 'pipeline.log'
logging.basicConfig(
    level=logging.INFO, # solo mensajes informativos (INFO)
    format="%(asctime)s [%(levelname)s] %(message)s", #Fecha/Hora exacta, [la Gravedad] y el Texto del mensaje
    handlers=[
        logging.FileHandler("pipeline.log"), # Historial guardado en archivo
        logging.StreamHandler() # mensajes en tiempo real terminal
    ]
)

def download_real_data():
    """
    Descarga el dataset real de fraude desde OpenML de forma automatizada
    y lo almacena en la ruta local de datos crudos (raw).
    """
    raw_data_path = os.path.join("data", "raw", "credit_card_fraud.csv")
    
    # Verificación de seguridad: si ya lo descargamos, no gastamos internet ni tiempo de cómputo de nuevo
    if os.path.exists(raw_data_path):
        logging.info(f"El dataset ya existe en {raw_data_path}. Omitiendo la descarga.")
        return

    logging.info("Iniciando la descarga del dataset real de fraude desde OpenML (esto puede tomar un minuto)...")
    
    try:
        # ID 1597 corresponde al dataset oficial de Credit Card Fraud Detection
        # as_frame=True nos devuelve directamente un DataFrame de Pandas
        mnist = fetch_openml(data_id=1597, as_frame=True, parser="auto")
        df = mnist.frame
        
        # Renombrar la columna objetivo a un estándar de ingeniería ('target')
        df = df.rename(columns={"Class": "target"})
        
        # Guardar el archivo en la carpeta local que Git ignora (pero DVC controlará después)
        df.to_csv(raw_data_path, index=False)
        logging.info(f"¡Éxito! Dataset descargado y guardado correctamente en: {raw_data_path}")
        logging.info(f"Dimensiones del dataset: {df.shape[0]} filas y {df.shape[1]} columnas.")
        
    except Exception as e:
        logging.error(f"Error crítico durante la descarga de datos: {str(e)}")
        raise e

if __name__ == "__main__":
    # Este bloque asegura que el script se pueda ejecutar directamente desde la terminal
    logging.info("--- Iniciando Etapa de Ingestión de Datos ---")
    download_real_data()
    logging.info("--- Etapa de Ingestión de Datos Finalizada ---")