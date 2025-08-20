from __future__ import annotations
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import os

RANDOM_STATE = 42

class DelayModel:
    """
    Modelo de clasificación de retrasos con RandomForest.
    - One-hot para OPERA, TIPOVUELO, MES
    - Conserva el orden de features visto en fit (para predicción consistente)
    - Valida aerolíneas conocidas (known_airlines_)
    """
    def __init__(self):
        self._model: Optional[RandomForestClassifier] = None
        self._feature_order: Optional[List[str]] = None
        self.known_airlines_: Optional[List[str]] = None

    # ------------ Helpers de features ------------
    def _feature_columns_fit(self, df: pd.DataFrame) -> List[str]:
        self.known_airlines_ = sorted(df["OPERA"].dropna().unique().tolist())
        opera_cols = [f"OPERA_{v}" for v in self.known_airlines_]
        tipov_cols = ["TIPOVUELO_I", "TIPOVUELO_N"]
        mes_cols = [f"MES_{m}" for m in range(1, 13)]
        return opera_cols + tipov_cols + mes_cols

    def _to_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        # get_dummies
        X = pd.get_dummies(
            df[["OPERA", "TIPOVUELO", "MES"]],
            columns=["OPERA", "TIPOVUELO", "MES"],
            prefix=["OPERA", "TIPOVUELO", "MES"]
        )
        # agregar faltantes y ordenar
        for c in columns:
            if c not in X.columns:
                X[c] = 0
        X = X[columns]
        # asegurar tipo numérico
        return X.astype(np.float32)

    # ------------ API pública ------------
    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        fit_mode: bool = False,
    ):
        """
        Si fit_mode=True: define vocabulario (aerolíneas) y el orden de columnas.
        Si target_column no es None, devuelve (X, y); si no, solo X.
        """
        required = {"OPERA", "TIPOVUELO", "MES"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

        if fit_mode:
            cols = self._feature_columns_fit(data)
            self._feature_order = cols
        elif self._feature_order is None:
            raise RuntimeError("El modelo no ha sido ajustado: _feature_order es None.")

        X = self._to_features(data, self._feature_order)

        if target_column:
            if target_column not in data:
                raise ValueError(f"No se encontró la columna objetivo '{target_column}'.")
            y = data[target_column].astype(int).to_numpy()
            return X, y
        return X

    def fit(self, df: pd.DataFrame, target_column: str = "delay") -> dict:
        """
        Entrena el RandomForest y devuelve métricas simples en el set de validación.
        """
        # preparar features/labels
        X, y = self.preprocess(df, target_column=target_column, fit_mode=True)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

        self._model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
        )
        self._model.fit(X_train, y_train)

        y_pred = self._model.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        return {"f1_val": float(f1), "n_features": X.shape[1]}

    def predict(self, X: pd.DataFrame | np.ndarray) -> List[int]:
        if self._model is None:
            raise RuntimeError("El modelo no ha sido entrenado.")
        if isinstance(X, pd.DataFrame) and self._feature_order is not None:
            # garantiza el mismo orden
            for c in self._feature_order:
                if c not in X:
                    X[c] = 0
            X = X[self._feature_order]
        preds = self._model.predict(X)
        return [int(p) for p in preds]

    # ------------ Conveniencia ------------
    def _derive_delay_if_needed(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Si el dataset no trae 'delay', lo deriva a partir de 'min_diff' (>=15 → 1).
        """
        if "delay" in df.columns:
            return df
        if "min_diff" in df.columns:
            df = df.copy()
            df["delay"] = (df["min_diff"] >= 15).astype(int)
            return df
        raise ValueError(
            "No se encontró columna 'delay'. Si no existe, se requiere 'min_diff' para derivarla."
        )

    def fit_from_csv(self, path: str = "data/data.csv") -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No se encontró el dataset en {path}")
        df = pd.read_csv(path)
        df = self._derive_delay_if_needed(df)
        return self.fit(df, target_column="delay")
