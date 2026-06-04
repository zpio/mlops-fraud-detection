import os
import pandas as pd
import pytest

# Rutas de los archivos que esperamos que el pipeline haya creado
TRAIN_PATH = os.path.join("data", "processed", "train.csv")
TEST_PATH = os.path.join("data", "processed", "test.csv")

def test_processed_files_exist():
    """Verifica que el script de preprocesamiento realmente guardó los archivos."""
    assert os.path.exists(TRAIN_PATH), f"Falta el archivo: {TRAIN_PATH}"
    assert os.path.exists(TEST_PATH), f"Falta el archivo: {TEST_PATH}"

def test_target_column_present():
    """Asegura que no hayamos borrado por accidente la variable a predecir."""
    df_train = pd.read_csv(TRAIN_PATH)
    assert "target" in df_train.columns, "¡Alerta! La columna 'target' desapareció."

def test_no_null_values_introduced():
    """Valida que el escalado no haya introducido valores NaN (Not a Number)."""
    df_train = pd.read_csv(TRAIN_PATH)
    total_nulls = df_train.isnull().sum().sum()
    assert total_nulls == 0, f"Se encontraron {total_nulls} valores nulos en los datos procesados."

def test_stratified_split_ratio():
    """
    Verifica que la estratificación funcionó. 
    La proporción de fraudes debe ser casi idéntica en train y test.
    """
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    
    train_fraud_ratio = df_train["target"].mean()
    test_fraud_ratio = df_test["target"].mean()
    
    # Tolerancia matemática muy pequeña (1%)
    diferencia = abs(train_fraud_ratio - test_fraud_ratio)
    assert diferencia < 0.01, f"Fuga detectada: Desbalance en las proporciones de fraude. Dif: {diferencia}"