"""
Entrenamiento de Logistic Regression vs XGBoost sobre el dataset Bank Marketing (UCI).

Uso (ejecutar siempre desde la raiz del repo, para que mlruns/ y mlflow.db queden
juntos en la raiz, junto con los demas ejercicios del curso):
    python src/bank_marketing/bank_marketing_train.py
    python src/bank_marketing/bank_marketing_train.py --seed 7 --xgb-max-depth 6 --xgb-n-estimators 500

Sin argumentos, corre con los valores por defecto (los mismos usados en
Notebook_Baseline_BecerrilDiego.ipynb). Cada corrida (Logistic Regression y XGBoost)
se registra como un run separado en MLflow, bajo el mismo experimento, con:
hiperparametros (log_param), metricas de CV y de test (log_metric), una matriz de
confusion (log_artifact) y el modelo entrenado (mlflow.sklearn.log_model).
"""

import argparse
import os
import tempfile
from pathlib import Path

import matplotlib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline

matplotlib.use("Agg")  # no se necesita una ventana grafica para guardar el PNG
import matplotlib.pyplot as plt  # noqa: E402  (import tras matplotlib.use)

# Este script vive en <repo>/src/bank_marketing/; subimos dos niveles para llegar
# a la raiz del repo, donde conviven los demas ejercicios (data/, mlflow.db, etc.).
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "bank_marketing" / "bank_marketing.csv"
TARGET = "y"
LEAKY_COLS = ["duration"]  # fuga de datos: solo se conoce despues de la llamada

NUMERIC_FEATURES = [
    "age", "campaign", "pdays", "previous", "contactado_antes",
    "poutcome_success", "contactos_totales",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
]
CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome", "pdays_bucket", "age_bin",
]

# Tracking 100% local (sqlite + archivos), sin depender de un servidor/tunel externo.
# Se comparte el mismo mlflow.db de la raiz del repo (usado tambien por el ejemplo
# de Iris en src/train.ipynb), cada quien bajo su propio nombre de experimento.
# Para ver los resultados: mlflow ui --backend-store-uri sqlite:///mlflow.db (desde la raiz del repo)
# Se puede sobreescribir con la variable de entorno MLFLOW_TRACKING_URI si se quiere
# apuntar a otro backend (por ejemplo, un servidor mlflow remoto).
DEFAULT_TRACKING_URI = f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}"
MLFLOW_EXPERIMENT_NAME = "bank_marketing_lr_vs_xgb"


def parse_args():
    """Argumentos de linea de comandos: semilla aleatoria + hiperparametros de ambos modelos."""
    parser = argparse.ArgumentParser(
        description="Entrena y compara Logistic Regression vs XGBoost en Bank Marketing (UCI).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semilla aleatoria para train/test split, validacion cruzada y los modelos (default: 42).",
    )
    parser.add_argument(
        "--lr-c", type=float, default=0.1, dest="lr_c",
        help="Parametro C (inverso de la fuerza de regularizacion) de Logistic Regression (default: 0.1).",
    )
    parser.add_argument(
        "--xgb-max-depth", type=int, default=4,
        help="Profundidad maxima de los arboles de XGBoost (default: 4).",
    )
    parser.add_argument(
        "--xgb-n-estimators", type=int, default=300,
        help="Numero de arboles (rondas de boosting) de XGBoost (default: 300).",
    )
    parser.add_argument(
        "--xgb-learning-rate", type=float, default=0.1,
        help="Tasa de aprendizaje (eta) de XGBoost (default: 0.1).",
    )
    return parser.parse_args()


def cargar_datos(path=DATA_PATH):
    """Carga el dataset Bank Marketing (UCI), separado por ';'."""
    return pd.read_csv(path, sep=";")


def limpiar_bank(df):
    """Limpieza fila a fila (sin estadisticos aprendidos, por lo que no hay fuga de datos)."""
    df = df.drop(columns=LEAKY_COLS, errors="ignore").copy()

    # pdays=999 significa "nunca contactado antes": se separa en un flag binario ...
    df["contactado_antes"] = (df["pdays"] != 999).astype(int)
    # ... y en una version categorica con mas granularidad (sobre el valor original)
    df["pdays_bucket"] = (
        pd.cut(
            df["pdays"].where(df["pdays"] != 999, np.nan),
            bins=[-1, 7, 15, df["pdays"].max()],
            labels=["reciente", "medio", "lejano"],
        )
        .astype("object")
        .fillna("nunca")
    )
    df.loc[df["pdays"] == 999, "pdays"] = np.nan

    # poutcome=='success' aislado como bandera binaria (senal fuerte de contactos previos)
    df["poutcome_success"] = (df["poutcome"] == "success").astype(int)

    # age en bandas: la suscripcion suele tener forma de U respecto a la edad
    df["age_bin"] = pd.cut(
        df["age"], bins=[0, 25, 35, 45, 55, 65, 150],
        labels=["<=25", "26-35", "36-45", "46-55", "56-65", "65+"],
    ).astype("object")

    # intensidad total de contacto: esta campana + campanas anteriores
    df["contactos_totales"] = df["campaign"] + df["previous"]

    return df


def preparar_datos(bank_df, seed):
    """Limpia el dataset y separa en train/test estratificado."""
    bank_clean = limpiar_bank(bank_df)
    X = bank_clean.drop(columns=TARGET)
    y = (bank_clean[TARGET] == "yes").astype(int)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)


def construir_pipelines(seed, lr_c, xgb_max_depth, xgb_n_estimators, xgb_learning_rate):
    """Arma los pipelines de preprocesamiento + modelo para LR y XGBoost."""
    preprocess_lr = ColumnTransformer([
        ("num", ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("yeo_johnson", PowerTransformer(method="yeo-johnson", standardize=True)),
        ]), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])

    preprocess_xgb = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])

    pipe_lr = ImbPipeline([
        ("preprocess", preprocess_lr),
        ("model", LogisticRegression(
            max_iter=1000, class_weight="balanced", C=lr_c, penalty="l2", random_state=seed,
        )),
    ])

    # scale_pos_weight y reg_lambda quedan fijos (no son parte de los 3+ hiperparametros
    # expuestos por CLI), ajustados manualmente en Prueba.ipynb.
    pipe_xgb = ImbPipeline([
        ("preprocess", preprocess_xgb),
        ("model", XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            max_depth=xgb_max_depth, n_estimators=xgb_n_estimators, learning_rate=xgb_learning_rate,
            scale_pos_weight=7.5, reg_lambda=5, random_state=seed,
        )),
    ])

    return pipe_lr, pipe_xgb


def entrenar_y_registrar(nombre, pipe, X_train, y_train, X_test, y_test, cv, scoring, extra_params):
    """
    Corre validacion cruzada estratificada sobre el set de entrenamiento, ajusta el
    pipeline completo, lo evalua en el set de prueba y registra en MLflow:
    hiperparametros (log_param), metricas de CV y de test (log_metric), una matriz
    de confusion (log_artifact) y el modelo entrenado (mlflow.sklearn.log_model).
    """
    with mlflow.start_run(run_name=nombre):
        # Hiperparametros expuestos por CLI (semilla + los propios de este modelo)
        for key, value in extra_params.items():
            mlflow.log_param(key, value)

        # Validacion cruzada estratificada (solo en el set de entrenamiento)
        cv_resultados = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        for metric in scoring:
            valores = cv_resultados[f"test_{metric}"]
            mlflow.log_metric(f"cv_{metric}_mean", valores.mean())
            mlflow.log_metric(f"cv_{metric}_std", valores.std())
            print(f"CV {metric}: {valores.mean():.4f} +/- {valores.std():.4f}")

        # Ajuste final sobre todo el set de entrenamiento y evaluacion en el holdout
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        test_metrics = {
            "test_roc_auc": roc_auc_score(y_test, y_proba),
            "test_f1": f1_score(y_test, y_pred),
            "test_accuracy": accuracy_score(y_test, y_pred),
        }
        mlflow.log_metrics(test_metrics)
        print(classification_report(y_test, y_pred, target_names=["no", "yes"]))

        # Todos los hiperparametros del modelo (paso final del pipeline), ademas de
        # los explicitamente logueados arriba
        mlflow.log_params(pipe.named_steps["model"].get_params())

        # Artefacto: matriz de confusion como imagen PNG
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=["no", "yes"], ax=ax, colorbar=False,
        )
        ax.set_title(f"Matriz de confusion - {nombre}")
        with tempfile.TemporaryDirectory() as tmp_dir:
            fig_path = Path(tmp_dir) / f"confusion_matrix_{nombre}.png"
            fig.savefig(fig_path, bbox_inches="tight")
            mlflow.log_artifact(str(fig_path))
        plt.close(fig)

        # Modelo versionado con su firma de entrada/salida
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            pipe, name="model", signature=signature, input_example=X_test.head(),
            serialization_format="cloudpickle",
        )

    return pipe, cv_resultados, test_metrics


def main():
    args = parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")

    bank_df = cargar_datos()
    X_train, X_test, y_train, y_test = preparar_datos(bank_df, args.seed)
    pipe_lr, pipe_xgb = construir_pipelines(
        args.seed, args.lr_c, args.xgb_max_depth, args.xgb_n_estimators, args.xgb_learning_rate,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    scoring = ["roc_auc", "f1", "accuracy"]

    _, _, metricas_lr = entrenar_y_registrar(
        "logistic_regression", pipe_lr, X_train, y_train, X_test, y_test, cv, scoring,
        extra_params={"seed": args.seed, "lr_c": args.lr_c},
    )
    _, _, metricas_xgb = entrenar_y_registrar(
        "xgboost", pipe_xgb, X_train, y_train, X_test, y_test, cv, scoring,
        extra_params={
            "seed": args.seed,
            "xgb_max_depth": args.xgb_max_depth,
            "xgb_n_estimators": args.xgb_n_estimators,
            "xgb_learning_rate": args.xgb_learning_rate,
        },
    )

    resumen = pd.DataFrame({"Logistic Regression": metricas_lr, "XGBoost": metricas_xgb})
    print(resumen)
    return resumen


if __name__ == "__main__":
    main()
