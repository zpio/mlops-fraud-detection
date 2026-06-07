import pandas as pd
import os
import logging
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Configurar logs
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def generate_drift_report():
    """
    Compara los datos de referencia (entrenamiento) contra los actuales (prueba)
    para detectar cambios en la distribución estadística de las variables.
    """
    train_path = os.path.join("data", "processed", "train.csv")
    test_path = os.path.join("data", "processed", "test.csv")
    report_path = "drift_report.html"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logging.error("Faltan los datasets procesados. Ejecuta preprocess.py primero.")
        return

    logging.info("Cargando datos de referencia y actuales...")
    reference_data = pd.read_csv(train_path)
    current_data = pd.read_csv(test_path)

    # El monitoreo de Drift se hace sobre las variables independientes (X), no sobre la predicción
    features = [col for col in reference_data.columns if col != 'target']

    logging.info("Calculando métricas estadísticas de Data Drift...")
    # Inicializar el reporte usando el paquete predefinido de Evidently
    drift_report = Report(metrics=[DataDriftPreset()])
    
    # Ejecutar la comparación
    drift_report.run(
        reference_data=reference_data[features], 
        current_data=current_data[features]
    )
    
    # Guardar el resultado en un archivo interactivo
    drift_report.save_html(report_path)
    logging.info(f"¡Éxito! Reporte interactivo generado en: {report_path}")

if __name__ == "__main__":
    generate_drift_report()