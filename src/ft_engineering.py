"""Ingeniería de características (v1.0.1).

Primer componente del flujo: limpia los datos, arma el ColumnTransformer
y devuelve los conjuntos de entrenamiento y evaluación ya transformados.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from cargar_datos import cargarDatos

TARGET = "Pago_atiempo"

COLUMNAS_NUMERIC = [
    "capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente",
    "total_otros_prestamos", "cuota_pactada", "puntaje", "puntaje_datacredito",
    "cant_creditosvigentes", "huella_consulta", "saldo_mora", "saldo_total",
    "saldo_principal", "saldo_mora_codeudor", "creditos_sectorFinanciero",
    "creditos_sectorCooperativo", "creditos_sectorReal",
    "promedio_ingresos_datacredito",
]
COLUMNAS_CATEGORIC = ["tipo_laboral", "tendencia_ingresos"]
COLUMNAS_ORDINAL = ["tipo_credito"]
EXCLUIR_DE_X = ["fecha_prestamo"]

TENDENCIAS_VALIDAS = ["Creciente", "Decreciente", "Estable"]
TIPOS_CREDITO_RAROS = [7, 68]
EDAD_MAXIMA = 100


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica las correcciones de calidad detectadas en el EDA."""
    df = df.copy()
    df = df.replace(["NA", "N/A", "null", "NULL", "?", "Sin dato"], np.nan)
    # Edades imposibles -> NaN (el imputer les asigna la mediana del train)
    df.loc[df["edad_cliente"] > EDAD_MAXIMA, "edad_cliente"] = np.nan
    # Tipos de crédito casi sin uso
    df = df[~df["tipo_credito"].isin(TIPOS_CREDITO_RAROS)]
    # tendencia_ingresos con valores numéricos inválidos -> NaN
    df["tendencia_ingresos"] = df["tendencia_ingresos"].where(
        df["tendencia_ingresos"].isin(TENDENCIAS_VALIDAS), np.nan
    )
    return df.reset_index(drop=True)


def _get_feature_columns(df: pd.DataFrame, excluir: list[str] | None = None):
    excluir = excluir or []
    numeric = [c for c in COLUMNAS_NUMERIC if c in df.columns and c not in excluir]
    cat_ = [c for c in COLUMNAS_CATEGORIC if c in df.columns and c not in excluir]
    ord_ = [c for c in COLUMNAS_ORDINAL if c in df.columns and c not in excluir]
    return numeric, cat_, ord_


def build_preprocessor(numeric_cols, categorical_cols, ordinal_cols) -> ColumnTransformer:
    """ColumnTransformer según el diagrama del enunciado."""
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric_cols))
    if categorical_cols:
        transformers.append((
            "categoric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first",
                                         sparse_output=False)),
            ]),
            categorical_cols,
        ))
    if ordinal_cols:
        transformers.append((
            "ordinal",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value",
                                           unknown_value=-1)),
            ]),
            ordinal_cols,
        ))
    return ColumnTransformer(transformers=transformers, remainder="drop",
                             verbose_feature_names_out=False)


# "puntaje" separa perfectamente pagadores de no pagadores (corte ~63), lo que
# apunta a fuga de información. Se excluye por defecto; pasar excluir_features=[]
# para volver a incluirla.
EXCLUIR_POR_DEFECTO = ["puntaje"]


def ft_engineering(test_size: float = 0.2, random_state: int = 1234,
                   excluir_features: list[str] | None = EXCLUIR_POR_DEFECTO,
                   df: pd.DataFrame | None = None) -> dict:
    """Devuelve X_train, X_test, y_train, y_test (DataFrames transformados),
    el preprocesador ya ajustado y los nombres de las features.

    excluir_features: columnas a sacar de X. Por defecto ["puntaje"] (posible
    fuga de información). Usar [] para incluir todas las columnas.
    """
    df = limpiar_datos(cargarDatos() if df is None else df)
    numeric, cat_, ord_ = _get_feature_columns(df, excluir_features)

    X = df[numeric + cat_ + ord_]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor(numeric, cat_, ord_)
    # El ajuste se hace SOLO con train para no filtrar información del test
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    cols = list(preprocessor.get_feature_names_out())

    return {
        "X_train": pd.DataFrame(X_train_t, columns=cols, index=X_train.index),
        "X_test": pd.DataFrame(X_test_t, columns=cols, index=X_test.index),
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_names": cols,
    }

if __name__ == "__main__":
    out = ft_engineering()
    print("Train:", out["X_train"].shape, "| Test:", out["X_test"].shape)
    print("Nulos en train:", int(out["X_train"].isna().sum().sum()))
    print("% no pago -> train:", round((out["y_train"] == 0).mean() * 100, 2),
          "| test:", round((out["y_test"] == 0).mean() * 100, 2))
    print("Features:", out["feature_names"])
