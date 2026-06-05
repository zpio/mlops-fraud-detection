import os
import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def train_model():
    """Entrena el modelo de detección de fraude y registra métricas en MLflow."""
    train_path = os.path.join("data", "processed", "train.csv")
    test_path = os.path.join("data", "processed", "test.csv")
    models_dir = "models"
    
    if not os.path.exists(train_path):
        logging.error("No se encontraron los datos procesados. Ejecuta preprocess.py primero.")
        return

    logging.info("Cargando datos de entrenamiento y prueba...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_test = df_test.drop(columns=["target"])
    y_test = df_test["target"]

    # Definir hiperparámetros (idealmente vendrían de un archivo de config, pero los definimos aquí)
    n_estimators = 100
    max_depth = 10
    random_state = 42

    # Configurar la conexión con el servidor local de MLflow
    tracking_uri = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(tracking_uri)
    
    # Ingeniería Defensiva: Verificar si el servidor está vivo antes de continuar
    try:
        import requests
        response = requests.get(tracking_uri)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        logging.error(f"¡El servidor MLflow no responde en {tracking_uri}! Asegúrate de ejecutar 'mlflow server --host 127.0.0.1 --port 5000' en otra terminal.")
        return
    
    mlflow.set_experiment("Fraud_Detection_Experiment")

    logging.info("Iniciando ejecución en MLflow...")
    with mlflow.start_run():
        # 1. Registrar hiperparámetros
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        logging.info("Entrenando RandomForestClassifier (esto puede tomar un momento)...")
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state,
            n_jobs=-1, # Usa todos los núcleos del procesador
            class_weight="balanced" # Vital para datos desbalanceados
        )
        model.fit(X_train, y_train)

        logging.info("Evaluando el modelo...")
        predictions = model.predict(X_test)
        
        # 2. Calcular métricas clave para fraude
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        # 3. Registrar métricas en MLflow
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # 4. Registrar el modelo en sí (el artefacto)
        mlflow.sklearn.log_model(model, "random_forest_model")

        # Guardar localmente para nuestro contenedor de Docker más adelante
        model_path = os.path.join(models_dir, "fraud_model.joblib")
        joblib.dump(model, model_path)

        logging.info(f"¡Entrenamiento exitoso! Métricas -> Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        logging.info(f"Modelo guardado localmente en {model_path}")

if __name__ == "__main__":
    train_model()