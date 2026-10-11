"""Small, reproducible AutoML-style regression benchmark with MLflow tracking.

This experiment is intentionally separate from final_model.joblib and scaler.joblib.
It uses leakage-aware features and writes new artifacts; it never overwrites the
production model used by the current Streamlit application.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from estate_ml.data import load_training_data


def make_candidates(random_state: int):
    # Tree-based models do not require scaling; imputation is kept in the pipeline.
    return {
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=300, random_state=random_state, n_jobs=-1,
                min_samples_leaf=2
            )),
        ]),
        "extra_trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=300, random_state=random_state, n_jobs=-1,
                min_samples_leaf=2
            )),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(random_state=random_state)),
        ]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="bengaluru_house_prices.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", ""))
    args = parser.parse_args()

    X, y = load_training_data(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("EstateML-AutoML-Experiments")

    candidates = make_candidates(args.random_state)
    results = []
    best_name, best_model, best_rmse = None, None, float("inf")

    for name, model in candidates.items():
        with mlflow.start_run(run_name=name):
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=5,
                scoring="neg_root_mean_squared_error", n_jobs=1
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
            mae = float(mean_absolute_error(y_test, predictions))
            r2 = float(r2_score(y_test, predictions))
            cv_rmse = float(-np.mean(cv_scores))

            mlflow.log_param("model_name", name)
            mlflow.log_param("random_state", args.random_state)
            mlflow.log_param("test_size", args.test_size)
            mlflow.log_param("features", ",".join(X.columns))
            mlflow.log_metric("cv_rmse", cv_rmse)
            mlflow.log_metric("test_rmse", rmse)
            mlflow.log_metric("test_mae", mae)
            mlflow.log_metric("test_r2", r2)
            mlflow.sklearn.log_model(model, artifact_path="model", serialization_format="cloudpickle")

            results.append({
                "model": name, "cv_rmse": cv_rmse, "test_rmse": rmse,
                "test_mae": mae, "test_r2": r2
            })
            if rmse < best_rmse:
                best_name, best_model, best_rmse = name, model, rmse

    out = Path("artifacts")
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, out / "automl_model.joblib")
    (out / "automl_metrics.json").write_text(
        json.dumps({"selected_model": best_name, "results": results}, indent=2),
        encoding="utf-8",
    )
    (out / "automl_feature_columns.json").write_text(
        json.dumps(list(X.columns), indent=2), encoding="utf-8"
    )
    print(json.dumps({"selected_model": best_name, "results": results}, indent=2))
    print("\nIMPORTANT: this is an experimental model. Do not replace the live model "
          "until it has been reviewed and the inference feature mapping is migrated.")


if __name__ == "__main__":
    main()
