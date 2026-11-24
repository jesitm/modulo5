"""Evaluación y comparación de modelos (v1.1.0).

Genera: tabla resumen, gráficos comparativos train vs test, curvas ROC y
matriz de confusión del mejor modelo. Todo se calcula a partir de los modelos
entrenados (nada pegado a mano).

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
}


def tabla_resumen(results_df: pd.DataFrame, best: str | None = None) -> pd.DataFrame:
    """Tabla modelo x métrica (test) con la brecha de recall train - test."""
    pv = results_df.pivot_table(index=["Model", "Data Set"], columns="Metric",
                                values="Score")[model_training.METRICAS]
    test = pv.xs("test", level="Data Set").copy()
    train = pv.xs("train", level="Data Set")
    # Brecha grande = el modelo memoriza train y generaliza mal (sobreajuste)
    test["brecha_recall_train_test"] = train["recall"] - test["recall"]
    test = test.sort_values("recall", ascending=False)
    if best in test.index:  # marca el ganador con una estrella
        test.index = [f"{m} ★" if m == best else m for m in test.index]
    return test.round(3)


def grafico_comparativo(results_df: pd.DataFrame) -> plt.Figure:
    """Barras agrupadas train vs test por métrica y modelo."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    for ax, metrica in zip(axes.flatten(), model_training.METRICAS):
        d = results_df[results_df["Metric"] == metrica]
        sns.barplot(data=d, x="Model", y="Score", hue="Data Set", ax=ax,
                    palette="cividis")
        ax.set_title(TITULOS[metrica], fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("Puntuación")
        ax.tick_params(axis="x", rotation=30)
        # Escalas: el AUC parte de 0,45 para ver mejor las diferencias; el
        # conteo de casos no tiene tope fijo; el resto va de 0 a 1
        if metrica == "roc_auc":
            ax.set_ylim(0.45, 1.05)
        elif metrica != "casosNoPagoAtiempo":
            ax.set_ylim(0, 1.05)
    fig.suptitle("Comparación de modelos: train vs test", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
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
    """Matriz de confusión del mejor modelo en test."""
    best, data = res["best"], res["data"]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_estimator(
        res["models"][best], data["X_test"], data["y_test"], ax=ax,
        display_labels=["No paga (0)", "Paga (1)"], cmap="Blues", colorbar=False)
    ax.set_title(f"Matriz de confusión en test: {best}")
    fig.tight_layout()
    return fig


def evaluation(res: dict | None = None, carpeta: str | None = None) -> io.BytesIO:
    """Corre (o recibe) el entrenamiento, imprime la tabla y devuelve el PNG
    comparativo en memoria. Si se da `carpeta`, guarda además todos los gráficos
    y la tabla resumen en CSV."""
    res = res or model_training.train_all()
    tabla = tabla_resumen(res["results_df"], res["best"])
    print(tabla.to_string())
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
        for nombre, f in (("curvas_roc", grafico_roc(res)),
                          ("matriz_confusion", grafico_confusion(res))):
            f.savefig(f"{carpeta}/{nombre}.png", dpi=110)
            plt.close(f)
        tabla.to_csv(f"{carpeta}/tabla_resumen.csv")
    plt.close(fig)
    return buf


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")  # sin ventana: solo guarda los archivos
    # La carpeta de salidas se ancla a la raíz del proyecto, sin importar
    # desde dónde se ejecute el script
    evaluation(carpeta=os.path.join(BASE_DIR, "..", "salidas"))
