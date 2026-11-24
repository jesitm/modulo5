# Módulo 5 · Predicción de pago a tiempo de créditos

Proyecto de *machine learning* supervisado que predice si un cliente va a **pagar su préstamo a tiempo** (`Pago_atiempo`: 1 = paga, 0 = no paga). Cubre la ingeniería de características, el entrenamiento de varios modelos de clasificación y su evaluación comparativa.

## Datos

`Base_de_datos.xlsx`: 10.763 préstamos y 23 columnas con datos del crédito (tipo, capital, plazo, cuota), del cliente (edad, salario, tipo laboral) y de su historial en centrales de riesgo (puntajes, saldos, mora, consultas, créditos por sector).

- **Target desbalanceado:** ~95 % paga a tiempo y ~5 % no paga.
- **Nulos:** `promedio_ingresos_datacredito` y `tendencia_ingresos` tienen ~27 %.

## Estructura del proyecto

```
modulo5/
├── Base_de_datos.xlsx          # datos de entrada
├── requeriments.txt            # dependencias
├── README.md
├── salidas/                    # se genera al ejecutar (gráficos, tablas, modelo)
└── src/
    ├── cargar_datos.py         # lectura de la base
    ├── comprension_eda.ipynb   # análisis exploratorio y calidad de datos
    ├── ft_engineering.py       # limpieza, split y preprocesamiento (v1.0.1)
    ├── model_training.ipynb    # entrenamiento y selección del modelo (v1.1.0)
    └── model_evaluation.py     # tabla resumen y gráficos comparativos (v1.1.0)
```

## Instalación

```bash
python -m venv venv
source venv/bin/activate        # en Windows: venv\Scripts\activate
pip install -r requeriments.txt
```

## Cómo ejecutarlo

| Qué querés hacer | Comando |
|---|---|
| Ver el análisis exploratorio | abrir `src/comprension_eda.ipynb` |
| Entrenar y ver la tabla de resultados | abrir y ejecutar `src/model_training.ipynb` |
| Generar tabla resumen y todos los gráficos | `python src/model_evaluation.py` |
| Probar solo el preprocesamiento | `python src/ft_engineering.py` |

`model_evaluation.py` importa el notebook de entrenamiento (ejecuta sus celdas de definición, no re-entrena dos veces) y guarda en `salidas/`:
`comparacion_modelos.png`, `curvas_roc.png`, `matriz_confusion.png` y `tabla_resumen.csv`.
Al ejecutar el notebook también se guardan el mejor modelo (`modelo_*.joblib`), el preprocesador y `resultados_modelos.csv`.

## Flujo de trabajo

1. **Carga** (`cargar_datos.py`): lee la base de Excel.
2. **EDA** (`comprension_eda.ipynb`): detecta problemas de calidad y relaciones con el target. Ejemplos: edades imposibles, tipos de crédito casi sin uso, tendencias de ingresos con valores inválidos, mayor mora entre independientes y con tendencia de ingresos decreciente.
3. **Ingeniería de características** (`ft_engineering.py`):
   - Limpieza según el EDA.
   - División 80/20 **estratificada** en train y test.
   - `ColumnTransformer`: numéricas con imputación por mediana; categóricas con imputación por moda más *OneHot*; ordinal con imputación más *OrdinalEncoder*.
   - El preprocesador se ajusta **solo con train**, para no filtrar información del test.
4. **Entrenamiento** (`model_training.ipynb`): regresión logística, SVC lineal, árbol de decisión, random forest y XGBoost (este último solo si está instalado). Todos manejan el desbalance de clases. Funciones reutilizables: `summarize_classification`, `build_model`, `select_best`, `train_all`.
5. **Evaluación** (`model_evaluation.py`): tabla resumen, comparación train vs test, curvas ROC y matriz de confusión del mejor modelo.

## Decisiones clave

- **`puntaje` se excluye de las features.** Separa perfectamente a pagadores de no pagadores (los que no pagan tienen `puntaje` ≤ 62,7 y los que pagan ≥ 63,8), lo que indica probable **fuga de información**. Para incluirla: `ft_engineering(excluir_features=[])`.
- **Se evalúa la clase "no paga" (0).** Precisión, recall y F1 se calculan con `pos_label=0`; el *accuracy* solo no sirve con un 95 % de una clase.
- **Criterio de selección:** ranking promedio de **recall** (clase 0) y **ROC AUC** en test. Usar solo el recall favorecería a un modelo que marcara a todos como morosos.
- **ROC AUC con probabilidades**, no con clases duras, para no subestimar la capacidad de discriminar.

## Resultados de referencia (test, sin `puntaje`)

Con los modelos de scikit-learn:

| Modelo | Recall (no paga) | Precisión | ROC AUC | Brecha recall train − test |
|---|---|---|---|---|
| **logistic** (seleccionado) | 0,62 | 0,08 | 0,68 | −0,02 |
| svc | 0,61 | 0,08 | 0,68 | −0,01 |
| decision_tree | 0,51 | 0,06 | 0,61 | 0,19 |
| random_forest | 0,40 | 0,10 | 0,69 | 0,44 |

- La regresión logística detecta ~6 de cada 10 clientes que no pagan y casi no sobreajusta.
- El random forest tiene buen AUC, pero memoriza train (recall 0,84 en train contra 0,40 en test).
- La precisión es baja (~8 %): para capturar morosos, el modelo marca como riesgosos a muchos buenos clientes.
- Si XGBoost está instalado, se suma una fila a la tabla y el ganador puede cambiar.

## Limitaciones y próximos pasos

- Ajustar hiperparámetros y el **umbral de decisión** para mejorar la precisión sin perder demasiado recall.
- Definir una métrica de negocio (costo de un moroso no detectado frente a un buen cliente rechazado).
- Probar un corte temporal con `fecha_prestamo` en lugar de una división aleatoria.
- Confirmar con el área de negocio qué representa `puntaje` antes de decidir si se puede usar.
