import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Fraud Detection API", version="1.0.0")

# Cargar los artefactos
MODEL_PATH = os.path.join("models", "fraud_model.joblib")
SCALER_PATH = os.path.join("models", "scaler.joblib")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    raise RuntimeError("Faltan archivos en /models. Entrena el modelo primero.")

# Esquema de seguridad: QUITAMOS 'Time' porque el modelo no la conoce
class TransactionData(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float

# La puerta principal de la API
@app.get("/")
def home():
    return {"status": "API de Fraude Activa"}

# La puerta de predicción
@app.post("/predict")
def predict_fraud(transaction: TransactionData):
    try:
        # Transformar los datos de entrada
        df = pd.DataFrame([transaction.dict()])
        
        # Escalar dinámicamente SOLO las columnas que existan (Igual que en preprocess)
        # Esto previene errores si la API recibe un esquema ligeramente distinto
        posibles_columnas = ['Time', 'Amount', 'time', 'amount']
        cols_to_scale = [col for col in posibles_columnas if col in df.columns]
        
        if cols_to_scale:
             df[cols_to_scale] = scaler.transform(df[cols_to_scale])
        
        # Predecir
        prediction = model.predict(df)
        resultado = "Fraude" if prediction[0] == 1 else "Legítima"
        
        return {
            "prediccion_cruda": int(prediction[0]),
            "resultado": resultado
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))