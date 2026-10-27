"""Tune four regressors on property listings with train-only GridSearchCV."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from estate_ml.property_model import (
    ARTIFACT_DIR, FEATURE_SCHEMA, SCHEMA_VERSION, TARGET, make_pipeline,
    prepare_property_data, save_property_artifacts,
)
from estate_ml.forecast import estimate_locality_growth

# ── Reproducible evaluation settings ───────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Fixed, intentionally small grids. Each grid includes that estimator's usual
# defaults so its best CV score can be compared with an untuned baseline.
CANDIDATES = {
    "LinearRegression": {
        "estimator": LinearRegression,
        "scale_numeric": True,
        "baseline": {"model__fit_intercept": True},
        "grid": {"model__fit_intercept": [True, False]},
    },
    "DecisionTreeRegressor": {
        "estimator": DecisionTreeRegressor,
        "scale_numeric": False,
        "fixed": {"model__random_state": RANDOM_SEED},
        "baseline": {
            "model__max_depth": None,
            "model__min_samples_split": 2,
            "model__min_samples_leaf": 1,
        },
        "grid": {
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2, 5],
        },
    },
    "RandomForestRegressor": {
        "estimator": RandomForestRegressor,
        "scale_numeric": False,
        "fixed": {"model__random_state": RANDOM_SEED, "model__n_jobs": 1},
        "baseline": {
            "model__n_estimators": 100,
            "model__max_depth": None,
            "model__min_samples_split": 2,
            "model__min_samples_leaf": 1,
        },
        "grid": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 16],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        },
    },
    "KNeighborsRegressor": {
        "estimator": KNeighborsRegressor,
        "scale_numeric": True,
        "baseline": {
            "model__n_neighbors": 5,
            "model__weights": "uniform",
            "model__p": 2,
        },
        "grid": {
            "model__n_neighbors": [5, 9, 15],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
    },
}


def _pipeline(name: str):
    """Build one candidate's preprocessing and estimator pipeline."""
    candidate = CANDIDATES[name]
    pipeline = make_pipeline(
        candidate["estimator"],
        scale_numeric=candidate["scale_numeric"],
    )
    if candidate.get("fixed"):
        pipeline.set_params(**candidate["fixed"])
    return pipeline


def _metrics(actual_prices, predicted_prices):
    """Return regression errors in lakhs plus relative-error coverage."""
    actual_prices = np.asarray(actual_prices, dtype=float)
    predicted_prices = np.asarray(predicted_prices, dtype=float)
    percentage_error = np.abs(predicted_prices - actual_prices) / np.abs(actual_prices)
    return {
        "rmse_lakhs": float(np.sqrt(mean_squared_error(actual_prices, predicted_prices))),
        "mae_lakhs": float(mean_absolute_error(actual_prices, predicted_prices)),
        "r2": float(r2_score(actual_prices, predicted_prices)),
        "within_10_percent": float(np.mean(percentage_error <= 0.10) * 100),
        "within_20_percent": float(np.mean(percentage_error <= 0.20) * 100),
    }


def _tune_candidate(name, training_features, training_prices, cv):
    """Tune one candidate on training rows only; no holdout enters this step."""
    candidate = CANDIDATES[name]
    baseline = _pipeline(name).set_params(**candidate["baseline"])
    baseline_scores = cross_val_score(
        baseline, training_features, training_prices,
        scoring="neg_root_mean_squared_error",
        cv=cv, n_jobs=1, error_score="raise",
    )
    search = GridSearchCV(
        estimator=_pipeline(name),
        param_grid=candidate["grid"],
        scoring="neg_root_mean_squared_error",
        cv=cv,
        refit=True,
        n_jobs=1,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(training_features, training_prices)
    best_cv_rmse = float(-search.best_score_)
    baseline_cv_rmse = float(-baseline_scores.mean())
    best_fold_rmse = [
        float(-search.cv_results_[f"split{fold}_test_score"][search.best_index_])
        for fold in range(search.n_splits_)
    ]
    return {
        "search": search,
        "best_params": search.best_params_,
        "best_cv_rmse_lakhs": best_cv_rmse,
        "baseline_cv_rmse_lakhs": baseline_cv_rmse,
        "cv_rmse_improvement_lakhs": baseline_cv_rmse - best_cv_rmse,
        "tuning_improved_over_baseline": bool(best_cv_rmse < baseline_cv_rmse - 1e-12),
        "best_fold_rmse_lakhs": best_fold_rmse,
    }


def _tune_candidates(training_features, training_prices, cv):
    """Tune each candidate on the supplied training partition only."""
    return {
        name: _tune_candidate(name, training_features, training_prices, cv)
        for name in CANDIDATES
    }


def _candidate_report(tuned, holdout_features, holdout_prices):
    """Score each tuned candidate once after all choices are complete."""
    report = {}
    for name, result in tuned.items():
        predictions = result["search"].predict(holdout_features)
        report[name] = {
            "best_parameters": result["best_params"],
            "mean_cv_rmse_lakhs": result["best_cv_rmse_lakhs"],
            "baseline_cv_rmse_lakhs": result["baseline_cv_rmse_lakhs"],
            "cv_rmse_improvement_lakhs": result["cv_rmse_improvement_lakhs"],
            "tuning_improved_over_baseline": result["tuning_improved_over_baseline"],
            "best_fold_rmse_lakhs": result["best_fold_rmse_lakhs"],
            "holdout_metrics": _metrics(holdout_prices, predictions),
        }
    return report


def _winner(tuned):
    """Choose by mean training-CV RMSE; holdout metrics are not consulted."""
    return min(tuned, key=lambda name: tuned[name]["best_cv_rmse_lakhs"])


def _print_candidate_comparison(title, candidate_results):
    """Print a compact, student-readable summary of the saved metric report."""
    print(f"\n{title}")
    print("Model | CV RMSE (L) | Holdout RMSE (L) | MAE (L) | R² | Within ±10% | Within ±20%")
    print("-" * 104)
    for model_name, result in candidate_results.items():
        metrics = result["holdout_metrics"]
        print(
            f"{model_name} | {result['mean_cv_rmse_lakhs']:.3f} | "
            f"{metrics['rmse_lakhs']:.3f} | {metrics['mae_lakhs']:.3f} | "
            f"{metrics['r2']:.3f} | {metrics['within_10_percent']:.1f}% | "
            f"{metrics['within_20_percent']:.1f}%"
        )


def _chronological_indices(features):
    listing_dates = pd.to_datetime(features["listing_date"], errors="coerce", utc=True)
    dated_rows = np.flatnonzero(listing_dates.notna().to_numpy())
    if len(dated_rows) != len(features):
        raise ValueError("Chronological evaluation requires a valid listing_date for every cleaned row.")
    date_values = listing_dates.astype("int64")
    chronological_order = dated_rows[
        np.argsort(date_values.iloc[dated_rows].to_numpy(), kind="stable")
    ]
    cutoff = int(date_values.iloc[chronological_order[int(len(chronological_order) * 0.8)]])
    training_rows = chronological_order[date_values.iloc[chronological_order].to_numpy() < cutoff]
    test_rows = chronological_order[date_values.iloc[chronological_order].to_numpy() >= cutoff]
    training_rows = training_rows[
        np.argsort(date_values.iloc[training_rows].to_numpy(), kind="stable")
    ]
    test_rows = test_rows[np.argsort(date_values.iloc[test_rows].to_numpy(), kind="stable")]
    return listing_dates, training_rows, test_rows


def main(argv=None):
    # ── Read and prepare the property listings ────────────────────────────
    parser = argparse.ArgumentParser(
        description="Compare regression models using curated Bengaluru property listings"
    )
    parser.add_argument("--data", required=True, help="Path to the curated property listings CSV")
    parser.add_argument("--artifact-dir", default=str(ARTIFACT_DIR))
    args = parser.parse_args(argv)

    source = Path(args.data)
    raw_listings = pd.read_csv(source)
    features, target_prices, quality_report = prepare_property_data(raw_listings)
    # ── Primary random holdout and training-only model selection ───────────
    # Split first. Baseline CV and GridSearchCV only receive the 80% partition.
    train_indices, test_indices = train_test_split(
        np.arange(len(features)), test_size=TEST_SIZE, random_state=RANDOM_SEED,
    )
    training_features = features.iloc[train_indices]
    training_prices = target_prices.iloc[train_indices]
    holdout_features = features.iloc[test_indices]
    holdout_prices = target_prices.iloc[test_indices]
    primary_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    primary_tuned = _tune_candidates(training_features, training_prices, primary_cv)
    selected_name = _winner(primary_tuned)

    # ── Separate chronological evaluation ─────────────────────────────────
    # Search only earlier listings; later listings remain a supplementary test.
    dates, chrono_train_indices, chrono_test_indices = _chronological_indices(features)
    chrono_training_features = features.iloc[chrono_train_indices]
    chrono_training_prices = target_prices.iloc[chrono_train_indices]
    chrono_holdout_features = features.iloc[chrono_test_indices]
    chrono_holdout_prices = target_prices.iloc[chrono_test_indices]
    chrono_cv = TimeSeriesSplit(n_splits=5)
    chrono_tuned = _tune_candidates(chrono_training_features, chrono_training_prices, chrono_cv)
    chrono_selected_name = _winner(chrono_tuned)

    # ── Final scoring, artifact metadata, and saved pipeline ────────────────
    # Finish all choices before scoring the primary holdout, once per candidate.
    primary_candidate_report = _candidate_report(
        primary_tuned, holdout_features, holdout_prices,
    )
    chrono_candidate_report = _candidate_report(
        chrono_tuned, chrono_holdout_features, chrono_holdout_prices,
    )
    selected_model = primary_tuned[selected_name]["search"].best_estimator_

    metadata = {
        "label": "curated listing evaluation",
        "dataset_file": "user-provided curated listings CSV",
        "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "schema_version": SCHEMA_VERSION,
        "target": TARGET,
        "target_unit": "lakhs of INR (1 lakh = INR 100,000)",
        "area_unit": "square feet",
        "feature_schema": FEATURE_SCHEMA,
        "property_id_excluded": True,
        "target_derived_features_excluded": True,
        "candidate_algorithms": list(CANDIDATES),
        "primary_split": {
            "type": "train_test_split",
            "test_size": TEST_SIZE,
            "random_state": RANDOM_SEED,
            "train_rows": int(len(train_indices)),
            "test_rows": int(len(test_indices)),
        },
        "cv_strategy": "GridSearchCV, five-fold KFold(shuffle=True, random_state=42), scoring=neg_root_mean_squared_error, refit=True; only the 80% training partition is provided to the search.",
        "preprocessing": "Each candidate uses a scikit-learn Pipeline with numeric median imputation, categorical Unknown imputation, one-hot encoding (min_frequency=5, handle_unknown=ignore), and date featurization; preprocessing is fitted within each CV fold. Numeric features are scaled only for Linear Regression and KNN.",
        "engineered_features": ["bhk_x_balconies = BHK * balconies"],
        "locality_growth": estimate_locality_growth(training_features, training_prices),
        "training_property_age_range": {
            "minimum": float(training_features["property_age_years"].min()),
            "maximum": float(training_features["property_age_years"].max()),
            "median": float(training_features["property_age_years"].median()),
        },
        "parameter_grids": {
            name: {
                key.removeprefix("model__"): value
                for key, value in spec["grid"].items()
            }
            for name, spec in CANDIDATES.items()
        },
        "fixed_estimator_parameters": {
            name: {key.removeprefix("model__"): value for key, value in spec.get("fixed", {}).items()}
            for name, spec in CANDIDATES.items()
        },
        "primary_winner_by_mean_training_cv_rmse": selected_name,
        "primary_candidate_results": primary_candidate_report,
        "primary_selected_holdout_metrics": primary_candidate_report[selected_name]["holdout_metrics"],
        "chronological_supplementary_evaluation": {
            "strategy": "80% earlier-date training / 20% later-date test; separate five-fold TimeSeriesSplit GridSearchCV on training dates only.",
            "train_rows": int(len(chrono_train_indices)),
            "test_rows": int(len(chrono_test_indices)),
            "train_end": str(dates.iloc[chrono_train_indices].max().date()),
            "test_start": str(dates.iloc[chrono_test_indices].min().date()),
            "winner_by_mean_training_cv_rmse": chrono_selected_name,
            "candidate_results": chrono_candidate_report,
            "selected_holdout_metrics": chrono_candidate_report[chrono_selected_name]["holdout_metrics"],
        },
        "quality_audit": quality_report,
        "limitations": [
            "The dataset contains user-curated listing information; asking prices may differ from completed sale prices.",
            "Evaluation metrics measure performance on this dataset and are not guarantees for future listings.",
            "Holdout metrics are evaluation results, not prediction intervals or guarantees.",
        ],
        "scikit_learn_version": sklearn.__version__,
        "model_name": selected_name,
        "model_parameters": primary_tuned[selected_name]["best_params"],
        "training_rows": int(len(train_indices)),
        "known_categories_from_random_training_partition": {
            column: sorted(training_features[column].dropna().astype(str).unique().tolist())
            for column in ("location", "condition", "property_type", "furnishing_status")
        },
    }
    save_property_artifacts(selected_model, metadata, args.artifact_dir)
    _print_candidate_comparison(
        "Primary holdout comparison (prices in INR lakhs)",
        primary_candidate_report,
    )
    _print_candidate_comparison(
        "Chronological holdout comparison (prices in INR lakhs)",
        chrono_candidate_report,
    )
    print(json.dumps(metadata, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
