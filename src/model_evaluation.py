"""Evaluación y comparación de modelos (v1.1.2).

Genera: tabla resumen, comparación del umbral por defecto vs el ajustado,
gráficos comparativos train vs test, valor de negocio según el % de rechazo,
curvas ROC y matriz de confusión del mejor modelo. Todo se calcula a partir de
los modelos entrenados (nada pegado a mano).

Requiere en esta carpeta (src/): model_training.ipynb, ft_engineering.py y
cargar_datos.py. La base Base_de_datos.xlsx va en la raíz del proyecto.

Uso (desde cualquier carpeta):  python src/model_evaluation.py
Los resultados se guardan en <raíz del proyecto>/salidas/.
"""
import io
import json
import os
import sys
import types

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

# Carpeta de este archivo: ahí buscamos el notebook y los módulos .py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


def importar_notebook(nombre: str) -> types.ModuleType:
    """Importa un notebook .ipynb como si fuera un módulo.

    Ejecuta sus celdas de código en orden dentro de un módulo nuevo. Como el
    módulo no se llama "__main__", la celda final del notebook (protegida por
    `if __name__ == "__main__"`) no se ejecuta y no se vuelve a entrenar todo.
    Reemplaza a la librería `import_ipynb` y no necesita instalar nada.
    """
    ruta = os.path.join(BASE_DIR, f"{nombre}.ipynb")
    with open(ruta, encoding="utf-8") as f:
        notebook = json.load(f)

    modulo = types.ModuleType(nombre)
    modulo.__file__ = ruta
    sys.modules[nombre] = modulo

    for celda in notebook["cells"]:
        if celda["cell_type"] != "code":
            continue
        # Se descartan los comandos mágicos de Jupyter (%matplotlib, !pip...)
        lineas = [l for l in "".join(celda["source"]).splitlines()
                  if not l.lstrip().startswith(("%", "!"))]
        exec(compile("\n".join(lineas), ruta, "exec"), modulo.__dict__)
    return modulo


model_training = importar_notebook("model_training")

# Títulos en español para cada gráfico
TITULOS = {
    "accuracy": "Accuracy general",
    "precision": "Precisión (no paga a tiempo)",
    "recall": "Recall (no paga a tiempo)",
    "f1_score": "F1 (no paga a tiempo)",
    "roc_auc": "ROC AUC",
    "casosNoPagoAtiempo": "Casos predichos como no paga",
    "valor_negocio": "Valor de negocio (millones)",
    "pct_valor_capturado": "% del valor máximo capturado",
}


def tabla_resumen(results_df: pd.DataFrame, best: str | None = None,
                  umbrales: dict | None = None) -> pd.DataFrame:
    """Tabla modelo x métrica (test, con el umbral ajustado).

    Incluye la brecha de recall train - test y, si se entregan, los umbrales.
    """
    columnas = model_training.METRICAS + model_training.METRICAS_NEGOCIO
    pv = results_df.pivot_table(index=["Model", "Data Set"], columns="Metric",
                                values="Score")[columnas]
    test = pv.xs("test", level="Data Set").copy()
    train = pv.xs("train", level="Data Set")
    # Brecha grande = el modelo memoriza train y generaliza mal (sobreajuste)
    test["brecha_recall_train_test"] = train["recall"] - test["recall"]
    # El valor de negocio se muestra en millones para que sea legible
    test["valor_negocio"] = test["valor_negocio"] / 1e6
    test = test.rename(columns={"valor_negocio": "valor_negocio_M"})
    if umbrales is not None:
        test["umbral"] = [umbrales[m] for m in test.index]
    test = test.sort_values("valor_negocio_M", ascending=False)
    if best in test.index:  # marca el ganador con una estrella
        test.index = [f"{m} ★" if m == best else m for m in test.index]
    return test.round(3)


def tabla_umbral(res: dict) -> pd.DataFrame:
    """Compara, en test, la decisión por defecto de cada modelo con el umbral ajustado."""
    filas = []
    for nombre, r in res["resultados"].items():
        base, ajust = r["base_test"], r["test"]
        filas.append({
            "modelo": nombre,
            "valor_defecto_M": base["valor_negocio"] / 1e6,
            "valor_ajustado_M": ajust["valor_negocio"] / 1e6,
            "recall_defecto": base["recall"],
            "recall_ajustado": ajust["recall"],
            "precision_defecto": base["precision"],
            "precision_ajustada": ajust["precision"],
            "rechazados_defecto": base["casosNoPagoAtiempo"],
            "rechazados_ajustado": ajust["casosNoPagoAtiempo"],
        })
    return pd.DataFrame(filas).set_index("modelo").round(3)


def grafico_comparativo(results_df: pd.DataFrame) -> plt.Figure:
    """Barras agrupadas train vs test por métrica y modelo (con el umbral ajustado)."""
    metricas = model_training.METRICAS + model_training.METRICAS_NEGOCIO
    fig, axes = plt.subplots(4, 2, figsize=(16, 19))
    for ax, metrica in zip(axes.flatten(), metricas):
        d = results_df[results_df["Metric"] == metrica].copy()
        if metrica == "valor_negocio":
            d["Score"] = d["Score"] / 1e6  # millones
        sns.barplot(data=d, x="Model", y="Score", hue="Data Set", ax=ax,
                    palette="cividis")
        ax.set_title(TITULOS[metrica], fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("Puntuación")
        ax.tick_params(axis="x", rotation=30)
        # Escalas: el AUC parte de 0,45 para ver mejor las diferencias; las
        # métricas entre 0 y 1 van de 0 a 1; el resto no tiene tope fijo
        if metrica == "roc_auc":
            ax.set_ylim(0.45, 1.05)
        elif metrica in ("accuracy", "precision", "recall", "f1_score"):
            ax.set_ylim(0, 1.05)
        if metrica in model_training.METRICAS_NEGOCIO:
            ax.axhline(0, color="k", lw=0.8)  # 0 = aprobar a todos
    fig.suptitle("Comparación de modelos con umbral ajustado: train vs test",
                 fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def grafico_valor_umbral(res: dict) -> plt.Figure:
    """Valor de negocio (fuera de muestra) según el % de clientes rechazados.

    Cada punto marcado es el umbral elegido para ese modelo. Por encima de 0
    el modelo mejora a aprobar a todos.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    for nombre, r in res["resultados"].items():
        # Solo la zona útil (hasta 6 % de rechazo): más allá el valor se
        # desploma y aplastaría la escala del gráfico
        curva = r["curva"][r["curva"]["pct_rechazados"] <= 0.06]
        linea, = ax.plot(curva["pct_rechazados"] * 100, curva["valor_negocio"] / 1e6,
                         label=nombre)
        # Punto del umbral elegido
        elegido = curva.loc[curva["umbral"] == r["umbral"]].iloc[0]
        ax.scatter(elegido["pct_rechazados"] * 100, elegido["valor_negocio"] / 1e6,
                   color=linea.get_color(), zorder=3, s=60, edgecolor="k")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlim(0, 6)
    ax.set(xlabel="% de clientes rechazados (los más riesgosos)",
           ylabel="Valor de negocio fuera de muestra (millones)",
           title="Valor de negocio según el umbral (● = umbral elegido)")
    ax.legend()
    fig.tight_layout()
    return fig


def grafico_roc(res: dict) -> plt.Figure:
    """Curvas ROC en test (la clase positiva es 'no paga')."""
    data = res["data"]
    y_bin = (data["y_test"] == 0).astype(int)  # 1 = no paga
    fig, ax = plt.subplots(figsize=(7, 6))
    for nombre, modelo in res["models"].items():
        fpr, tpr, _ = roc_curve(
            y_bin, model_training.score_no_pago(modelo, data["X_test"]))
        ax.plot(fpr, tpr, label=nombre)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)  # referencia: azar
    ax.set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos",
           title="Curvas ROC en test (clase: no paga)")
    ax.legend()
    fig.tight_layout()
    return fig


def grafico_confusion(res: dict) -> plt.Figure:
    """Matriz de confusión del mejor modelo en test, con su umbral ajustado."""
    best, data = res["best"], res["data"]
    score = model_training.score_no_pago(res["models"][best], data["X_test"])
    pred = np.where(score >= res["umbrales"][best], 0, 1)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(
        data["y_test"], pred, ax=ax,
        display_labels=["No paga (0)", "Paga (1)"], cmap="Blues", colorbar=False)
    ax.set_title(f"Matriz de confusión en test: {best} (umbral ajustado)")
    fig.tight_layout()
    return fig


def evaluation(res: dict | None = None, carpeta: str | None = None) -> io.BytesIO:
    """Corre (o recibe) el entrenamiento, imprime las tablas y devuelve el PNG
    comparativo en memoria. Si se da `carpeta`, guarda además todos los gráficos
    y las tablas en CSV."""
    res = res or model_training.train_all()
    tabla = tabla_resumen(res["results_df"], res["best"], res["umbrales"])
    antes_despues = tabla_umbral(res)

    print("Resumen en test (umbral ajustado):")
    print(tabla.to_string())
    print("\nUmbral por defecto vs ajustado (test):")
    print(antes_despues.to_string())
    print(f"\nModelo seleccionado: {res['best']}")

    # Gráfico comparativo en memoria (útil, por ejemplo, para devolverlo en una API)
    fig = grafico_comparativo(res["results_df"])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    buf.seek(0)

    # Si se indica una carpeta, también se guardan los archivos
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)
        fig.savefig(f"{carpeta}/comparacion_modelos.png", dpi=110)
        for nombre, f in (("valor_vs_umbral", grafico_valor_umbral(res)),
                          ("curvas_roc", grafico_roc(res)),
                          ("matriz_confusion", grafico_confusion(res))):
            f.savefig(f"{carpeta}/{nombre}.png", dpi=110)
            plt.close(f)
        tabla.to_csv(f"{carpeta}/tabla_resumen.csv")
        antes_despues.to_csv(f"{carpeta}/comparacion_umbral.csv")
    plt.close(fig)
    return buf


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")  # sin ventana: solo guarda los archivos
    # La carpeta de salidas se ancla a la raíz del proyecto, sin importar
    # desde dónde se ejecute el script
    evaluation(carpeta=os.path.join(BASE_DIR, "..", "salidas"))
