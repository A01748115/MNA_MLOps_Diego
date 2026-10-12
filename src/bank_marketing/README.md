# Bank Marketing: Logistic Regression vs XGBoost

Comparacion de un modelo de Logistic Regression contra un modelo XGBoost para
predecir si un cliente contratara un deposito a plazo fijo, usando el dataset
[Bank Marketing (UCI)](https://archive.ics.uci.edu/dataset/222/bank+marketing)
(version `bank-additional-full.csv`, 41,188 filas).

## Contenido de esta carpeta

| Archivo | Descripcion |
|---|---|
| `bank_marketing_train.py` | Script ejecutable: carga de datos, preprocesamiento, entrenamiento y evaluacion de ambos modelos, con tracking en MLflow. |
| `Notebook_Baseline_BecerrilDiego.ipynb` | Notebook de exploracion de datos (EDA) unicamente — no contiene entrenamiento ni tracking. |
| `requirements.txt` | Dependencias necesarias para correr el script, con versiones fijas. |

El dataset (`bank_marketing.csv`) vive en `data/bank_marketing/`, un nivel arriba
de `src/`.

## 1. Preparar el entorno

Requiere **Python 3.13**. Desde la raiz del repositorio:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r src/bank_marketing/requirements.txt
```

## 2. Ejecutar el script

**Importante:** el script siempre debe correrse desde la **raiz del repositorio**
(no desde dentro de `src/bank_marketing/`), para que pueda encontrar el dataset
en `data/bank_marketing/` y para que `mlflow.db` / `mlruns/` queden en la raiz,
junto con los demas ejercicios del curso.

```powershell
cd <raiz-del-repositorio>
python src/bank_marketing/bank_marketing_train.py
```

Sin argumentos, corre con los valores por defecto y reproduce los resultados
documentados en `Notebook_Baseline_BecerrilDiego.ipynb`.

### Argumentos disponibles

| Argumento | Tipo | Default | Descripcion |
|---|---|---|---|
| `--seed` | int | `42` | Semilla aleatoria (train/test split, validacion cruzada y ambos modelos). |
| `--lr-c` | float | `0.1` | Parametro `C` (inverso de la regularizacion) de Logistic Regression. |
| `--xgb-max-depth` | int | `4` | Profundidad maxima de los arboles de XGBoost. |
| `--xgb-n-estimators` | int | `300` | Numero de arboles (rondas de boosting) de XGBoost. |
| `--xgb-learning-rate` | float | `0.1` | Tasa de aprendizaje (`eta`) de XGBoost. |

Se puede ver la lista completa en cualquier momento con:

```powershell
python src/bank_marketing/bank_marketing_train.py --help
```

### Ejemplos

```powershell
python src/bank_marketing/bank_marketing_train.py
python src/bank_marketing/bank_marketing_train.py --seed 7
python src/bank_marketing/bank_marketing_train.py --xgb-max-depth 6 --xgb-n-estimators 500
python src/bank_marketing/bank_marketing_train.py --lr-c 1.0
python src/bank_marketing/bank_marketing_train.py --xgb-learning-rate 0.05 --seed 99
```

Cada corrida entrena y evalua **ambos** modelos (Logistic Regression y XGBoost)
y registra dos *runs* independientes en MLflow bajo el mismo experimento.

### Reproducibilidad

Correr el script dos veces con exactamente los mismos argumentos produce
resultados identicos: `--seed` controla el train/test split, la validacion
cruzada estratificada y la inicializacion de ambos modelos.

## 3. Ver los resultados en MLflow

Por defecto, el tracking es 100% local (sin servidor ni tunel externo),
usando un `mlflow.db`  en la raiz del repositorio. Para abrir la
interfaz:

```powershell
cd <raiz-del-repositorio>
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Luego abrir `http://localhost:5000` y buscar el experimento
**`bank_marketing_lr_vs_xgb`**. Cada corrida registra:

- **Parametros** (`log_param`): la semilla y los hiperparametros usados.
- **Metricas** (`log_metric`): ROC-AUC, F1 y accuracy, tanto de validacion
  cruzada (`cv_*`) como del set de prueba (`test_*`).
- **Artefacto** (`log_artifact`): una matriz de confusion en PNG.
- **Modelo entrenado** (`mlflow.sklearn.log_model`), nombrado segun el
  algoritmo (`logistic_regression` o `xgboost`) para distinguirlos en la
  vista de *Logged Models*.

Para usar un backend distinto (por ejemplo, un servidor MLflow remoto), se
puede sobreescribir con la variable de entorno `MLFLOW_TRACKING_URI` antes de
correr el script:

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
python src/bank_marketing/bank_marketing_train.py
```

## 4. Integracion continua (CI)

El workflow [`.github/workflows/bank_marketing.yml`](../../.github/workflows/bank_marketing.yml)
corre automaticamente en cada *push* a `main` que modifique archivos de este
ejercicio (`src/bank_marketing/**` o `data/bank_marketing/**`). Valida, desde
un checkout limpio, que las dependencias declaradas en `requirements.txt` son
suficientes y que el script corre sin errores con los valores por defecto.
