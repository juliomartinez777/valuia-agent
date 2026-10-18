"""Carga de datos y separación entrenamiento/prueba (la misma del notebook)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parents[1]
RUTA_CSV = RAIZ / "data" / "AmesHousing.csv"
RUTA_MODELO = RAIZ / "models" / "valuia.joblib"
RUTA_COLUMNAS = RAIZ / "models" / "columnas.json"


def cargar_datos():
    df = pd.read_csv(RUTA_CSV)
    df.columns = df.columns.str.replace(r"[ /]", "", regex=True)
    return df


def cargar_columnas():
    with open(RUTA_COLUMNAS) as f:
        c = json.load(f)
    return c["num"], c["cat"]


def separar(df):
    """Misma separación 80/20 del notebook (random_state=42)."""
    num, cat = cargar_columnas()
    X = df[num + cat]
    y = np.log(df["SalePrice"])
    return train_test_split(X, y, test_size=0.2, random_state=42)


if __name__ == "__main__":
    df = cargar_datos()
    X_train, X_test, y_train, y_test = separar(df)
    print("Entrenamiento:", len(X_train), "| Prueba:", len(X_test))
    print("Columnas:", cargar_columnas())