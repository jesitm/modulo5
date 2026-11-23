import os
import pandas as pd


def cargarDatos(ruta: str | None = None) -> pd.DataFrame:
    """Lee Base_de_datos.xlsx (por defecto, en la misma carpeta que este script)."""
    if ruta is None:
        ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Base_de_datos.xlsx")
    return pd.read_excel(ruta)
