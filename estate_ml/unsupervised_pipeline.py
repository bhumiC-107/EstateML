"""End-to-end unsupervised training, selection, persistence, and assignment.

This module orchestrates the existing clustering estimators and result helpers in
``estate_ml.clustering``. It does not replace the supervised price model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Callable, Iterable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, pairwise_distances_argmin_min
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from estate_ml.clustering import (
    BINARY_FEATURES,
    CLUSTERING_ALGORITHMS,
    DEFAULT_CATEGORICAL_FEATURES,
    DEFAULT_NUMERIC_FEATURES,
    apply_experiment_configuration,
    build_cluster_profile,
    build_experiment_configurations,
    interpret_clusters,
    load_dataset,
    _is_price_derived_feature,
    _fit_labels,
    _make_cluster_estimator,
    run_clustering_experiments,
)

UNSUPERVISED_SCHEMA_VERSION = "estate-unsupervised-v2"
UNSUPERVISED_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "model" / "unsupervised" / "v2"
AMENITY_FEATURES = tuple(BINARY_FEATURES)
DEFAULT_SELECTION_THRESHOLDS = {
    "minimum_cluster_count": 2,
    "minimum_cluster_size": 5,
    "minimum_non_tiny_share": 0.50,
    "maximum_dbscan_noise_percent": 60.0,
    "minimum_stability_ari": 0.20,
    "minimum_stability_repeats": 2,
    "minimum_valid_metrics": 2,
}


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    """Feature options with conservative defaults for locality and redundancy."""

    include_location: bool = False
    location_min_frequency: int = 25
    location_weight: float = 0.25
    coordinate_weight: float = 0.20
    include_amenity_count: bool = False
    derive_floor_ratio: bool = True
    include_listing_date: bool = False

    def __post_init__(self) -> None:
        if self.location_min_frequency < 1:
            raise ValueError("location_min_frequency must be at least 1.")
        for name, weight in (("location_weight", self.location_weight), ("coordinate_weight", self.coordinate_weight)):
            if not math.isfinite(float(weight)) or not 0 <= float(weight) <= 1:
                raise ValueError(f"{name} must be a finite value between 0 and 1.")


@dataclass
class PipelineRunResult:
    validation_data: pd.DataFrame
    validation_report: dict[str, object]
    validation_issues: pd.DataFrame
    engineered_feature_names: list[str]
    feature_documentation: list[dict[str, str]]
    preprocessor: "PropertyFeaturePreprocessor"
    candidate_results: pd.DataFrame
    selected_configuration: dict[str, object] | None
    selected_algorithm: str | None
    selection_rationale: list[str]
    selection_thresholds: dict[str, float | int]
    selected_model: object | None
    selected_labels: pd.DataFrame
    cluster_profiles: pd.DataFrame
    cluster_interpretations: list[dict[str, object]]
    anomaly_report: pd.DataFrame
    visualization_data: pd.DataFrame
    metadata: dict[str, object]
    no_satisfactory_model: bool
    fitted_estimators: dict[int, object] = field(default_factory=dict, repr=False)
    assignment_state: dict[str, object] = field(default_factory=dict, repr=False)


def _safe_numeric(series: pd.Series) -> tuple[pd.Series, int]:
    parsed = pd.to_numeric(series, errors="coerce").astype(float)
    malformed = int((series.notna() & parsed.isna()).sum())
    nonfinite = parsed.notna() & ~np.isfinite(parsed)
    malformed += int(nonfinite.sum())
    parsed.loc[nonfinite] = np.nan
    return parsed, malformed


def validate_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Validate records, preserve unusual observations, and report every correction/exclusion."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("validate_dataset expects a pandas DataFrame.")
    if frame.empty:
        raise ValueError("The dataset contains no rows.")
    if "property_id" not in frame.columns:
        raise ValueError("The dataset must include property_id so labels can be traced to listings.")
    original_rows, original_columns = len(frame), list(frame.columns)
    data = frame.copy().reset_index(drop=True)
    issues: list[dict[str, object]] = []
    malformed_counts: dict[str, int] = {}
    known_numeric = set(DEFAULT_NUMERIC_FEATURES) | {"latitude", "longitude"}
    for column in known_numeric & set(data.columns):
        before = data[column].copy()
        converted, count = _safe_numeric(data[column])
        if count:
            malformed_counts[column] = count
            malformed = before.notna() & converted.isna()
            for idx in np.flatnonzero(malformed.to_numpy()):
                issues.append({"row": int(idx), "property_id": data.at[idx, "property_id"], "column": column, "value": before.at[idx], "classification": "corrected to missing", "reason": "Malformed, infinite, or non-numeric measurement."})
        data[column] = converted

    hard_ranges = {
        "area_sqft": (0, 100_000), "BHK": (0, 20), "bathrooms": (0, 40),
        "balconies": (0, 30), "property_age_years": (0, 200),
        "floor_number": (0, 300), "total_floors": (1, 300),
        "parking_spaces": (0, 100), "latitude": (-90, 90), "longitude": (-180, 180),
    }
    corrected_values = 0
    excluded_mask = data["property_id"].isna() | data["property_id"].astype("string").str.strip().eq("")
    for idx in np.flatnonzero(excluded_mask.to_numpy()):
        issues.append({"row": int(idx), "property_id": None, "column": "property_id", "classification": "excluded", "reason": "Missing identifier; cannot preserve row-to-label identity."})
    for column, (lower, upper) in hard_ranges.items():
        if column not in data:
            continue
        vals = data[column]
        # Values past soft bounds but within these broad physical limits are kept.
        lower_invalid = vals <= lower if column in {"area_sqft", "BHK"} else vals < lower
        bad = vals.notna() & (lower_invalid | (vals > upper))
        for idx in np.flatnonzero(bad.to_numpy()):
            issues.append({"row": int(idx), "property_id": data.at[idx, "property_id"], "column": column, "value": vals.at[idx], "classification": "corrected to missing", "reason": f"Outside broad valid range [{lower}, {upper}]."})
        corrected_values += int(bad.sum())
        data.loc[bad, column] = np.nan

    for column in AMENITY_FEATURES:
        if column not in data:
            continue
        values = data[column]
        invalid = values.notna() & ~values.isin([0, 1])
        for idx in np.flatnonzero(invalid.to_numpy()):
            issues.append({"row": int(idx), "property_id": data.at[idx, "property_id"], "column": column, "value": values.at[idx], "classification": "corrected to missing", "reason": "Binary amenity field must be 0 or 1."})
        corrected_values += int(invalid.sum())
        data.loc[invalid, column] = np.nan

    if {"floor_number", "total_floors"}.issubset(data.columns):
        inconsistent = data["floor_number"].notna() & data["total_floors"].notna() & (data["floor_number"] > data["total_floors"])
        for idx in np.flatnonzero(inconsistent.to_numpy()):
            issues.append({"row": int(idx), "property_id": data.at[idx, "property_id"], "column": "floor_number/total_floors", "value": f"{data.at[idx, 'floor_number']}/{data.at[idx, 'total_floors']}", "classification": "corrected to missing", "reason": "Floor number exceeds the stated number of floors."})
        corrected_values += int(inconsistent.sum()) * 2
        data.loc[inconsistent, ["floor_number", "total_floors"]] = np.nan

    suspicious_ranges = {
        "area_sqft": (10_000, None), "BHK": (8, None), "bathrooms": (12, None),
        "property_age_years": (100, None), "floor_number": (80, None), "total_floors": (100, None),
    }
    flagged_counts: dict[str, int] = {}
    for column, (minimum, maximum) in suspicious_ranges.items():
        if column not in data:
            continue
        values = data[column]
        suspicious = values.notna() & (values >= minimum)
        if maximum is not None:
            suspicious &= values <= maximum
        flagged_counts[column] = int(suspicious.sum())
        for idx in np.flatnonzero(suspicious.to_numpy()):
            issues.append({"row": int(idx), "property_id": data.at[idx, "property_id"], "column": column, "value": values.at[idx], "classification": "flagged unusual; retained", "reason": "Unusual value retained for review; it may still be valid."})

    duplicate_mask = frame.reset_index(drop=True).duplicated(keep="first")
    for idx in np.flatnonzero(duplicate_mask.to_numpy()):
        issues.append({"row": int(idx), "property_id": data.at[idx, "property_id"], "column": "*", "classification": "excluded duplicate", "reason": "Exact duplicate record; first copy retained."})
    id_keys = data["property_id"].astype("string").str.strip()
    duplicate_id_mask = id_keys.notna() & id_keys.ne("") & id_keys.duplicated(keep="first") & ~duplicate_mask
    for idx in np.flatnonzero(duplicate_id_mask.to_numpy()):
        issues.append({"row": int(idx), "property_id": data.at[idx, "property_id"], "column": "property_id", "classification": "excluded duplicate ID", "reason": "Property IDs must be unique for unambiguous lookup; first occurrence retained."})
    excluded_mask |= duplicate_mask | duplicate_id_mask
    excluded_rows = int(excluded_mask.sum())
    retained = data.loc[~excluded_mask].reset_index(drop=True)
    if len(retained) < 10:
        raise ValueError(f"Only {len(retained)} usable rows remain after validation; at least 10 are needed.")
    if len(set(retained["property_id"].astype(str))) < 2:
        raise ValueError("At least two distinct property IDs are required.")

    missing_counts = {column: int(retained[column].isna().sum()) for column in retained.columns}
    numeric_summary = retained.select_dtypes(include=[np.number]).describe().replace({np.nan: None}).to_dict()
    categorical_summary = {}
    for column in retained.select_dtypes(exclude=[np.number]).columns:
        categorical_summary[column] = retained[column].astype("string").fillna("<missing>").value_counts().head(20).to_dict()
    report: dict[str, object] = {
        "input_shape": [original_rows, len(original_columns)],
        "retained_shape": [len(retained), len(retained.columns)],
        "columns": original_columns,
        "data_types": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_values_by_column": missing_counts,
        "malformed_or_nonfinite_numeric_values": malformed_counts,
        "corrected_values_to_missing": corrected_values + sum(malformed_counts.values()),
        "rows_excluded": excluded_rows,
        "rows_retained": len(retained),
        "exact_duplicate_rows": int(duplicate_mask.sum()),
        "duplicate_property_ids": int(duplicate_id_mask.sum()),
        "suspicious_values_flagged_and_retained": flagged_counts,
        "numerical_summary": numeric_summary,
        "categorical_frequencies_top_20": categorical_summary,
        "usable_feature_columns": [column for column in retained.columns if column not in {"property_id", "listing_date"}],
    }
    return retained, report, pd.DataFrame(issues)


def default_feature_selection(frame: pd.DataFrame, config: FeatureEngineeringConfig | None = None) -> list[str]:
    config = config or FeatureEngineeringConfig()
    selected = [column for column in DEFAULT_NUMERIC_FEATURES if column in frame.columns]
    selected.extend(column for column in DEFAULT_CATEGORICAL_FEATURES if column in frame.columns and (column != "location" or config.include_location))
    selected.extend(column for column in ("latitude", "longitude") if column in frame.columns)
    return list(dict.fromkeys(selected))


def engineer_property_features(
    frame: pd.DataFrame,
    selected_features: Iterable[str],
    config: FeatureEngineeringConfig,
    *,
    frequent_locations: set[str] | None = None,
) -> pd.DataFrame:
    """Derive documented features without modifying the source property frame."""
    selected = list(dict.fromkeys(selected_features))
    forbidden = [
        column for column in selected
        if column == "property_id" or column not in frame.columns or _is_price_derived_feature(column)
    ]
    if forbidden:
        raise ValueError(
            "Feature selection contains identifiers, price-derived features, or missing columns: "
            f"{forbidden}"
        )
    features = frame[selected].copy()
    if "listing_date" in features:
        if not config.include_listing_date:
            raise ValueError("Select the listing-date feature option before using listing_date.")
        dates = pd.to_datetime(features.pop("listing_date"), errors="coerce", utc=True)
        features["listing_year"] = dates.dt.year.astype(float)
        features["listing_month"] = dates.dt.month.astype(float)
    if config.derive_floor_ratio and {"floor_number", "total_floors"}.issubset(features.columns):
        floors = pd.to_numeric(features["floor_number"], errors="coerce")
        totals = pd.to_numeric(features["total_floors"], errors="coerce")
        ratio = floors.div(totals.where(totals > 0))
        ratio = ratio.where((floors >= 0) & (floors <= totals))
        features = features.drop(columns=["floor_number", "total_floors"])
        features["floor_ratio"] = ratio
    available_amenities = [column for column in AMENITY_FEATURES if column in features]
    if config.include_amenity_count and available_amenities:
        features["amenity_count"] = features[available_amenities].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    if "location" in features:
        if not config.include_location:
            raise ValueError("Location is excluded by default. Enable include_location to cluster on locality.")
        values = features["location"].astype("string").str.strip().fillna("Unknown")
        if frequent_locations is not None:
            values = values.where(values.isin(frequent_locations), "Other")
        features["location"] = values
    for column in features.columns:
        if column in BINARY_FEATURES:
            features[column] = pd.to_numeric(features[column], errors="coerce")
        elif pd.api.types.is_numeric_dtype(features[column]):
            features[column] = pd.to_numeric(features[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        else:
            features[column] = features[column].astype("string").str.strip().replace("", pd.NA).fillna("Unknown")
    return features


class PropertyFeaturePreprocessor:
    """Fitted, reusable feature engineering and preprocessing state."""

    def __init__(self, selected_features: Iterable[str], config: FeatureEngineeringConfig | None = None):
        self.selected_features = tuple(selected_features)
        self.config = config or FeatureEngineeringConfig()

    def fit(self, frame: pd.DataFrame):
        selected = list(self.selected_features)
        self.frequent_locations_: set[str] | None = None
        if "location" in selected:
            if not self.config.include_location:
                raise ValueError("Location is excluded unless explicitly enabled in the feature configuration.")
            values = frame["location"].astype("string").str.strip().fillna("Unknown")
            counts = values.value_counts()
            self.frequent_locations_ = set(counts[counts >= self.config.location_min_frequency].index.astype(str))
        engineered = engineer_property_features(frame, selected, self.config, frequent_locations=self.frequent_locations_)
        self.engineered_columns_ = list(engineered.columns)
        self.binary_columns_ = [column for column in engineered if column in BINARY_FEATURES]
        self.categorical_columns_ = [column for column in engineered if not pd.api.types.is_numeric_dtype(engineered[column])]
        self.numeric_columns_ = [column for column in engineered if column not in self.binary_columns_ + self.categorical_columns_]
        transformers = []
        if self.numeric_columns_:
            transformers.append(("continuous", Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]), self.numeric_columns_))
        if self.binary_columns_:
            transformers.append(("binary", SimpleImputer(strategy="most_frequent", keep_empty_features=True), self.binary_columns_))
        if self.categorical_columns_:
            transformers.append(("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), self.categorical_columns_))
        if not transformers:
            raise ValueError("No usable numeric, binary, or categorical features were selected.")
        self.transformer_ = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)
        self.transformer_.fit(engineered)
        self.feature_names_out_ = list(self.transformer_.get_feature_names_out())
        self.input_schema_ = list(frame.columns)
        self._fit_engineered_ = engineered
        return self

    def _transform_engineered(self, engineered: pd.DataFrame) -> np.ndarray:
        matrix = np.asarray(self.transformer_.transform(engineered), dtype=float)
        for index, name in enumerate(self.feature_names_out_):
            if name.startswith("continuous__latitude") or name.startswith("continuous__longitude"):
                matrix[:, index] *= self.config.coordinate_weight
            if name.startswith("categorical__location_"):
                matrix[:, index] *= self.config.location_weight
        return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [column for column in self.selected_features if column not in frame.columns]
        if missing:
            raise ValueError(f"Incoming property data is missing required feature columns: {missing}")
        engineered = engineer_property_features(
            frame, self.selected_features, self.config, frequent_locations=self.frequent_locations_,
        )
        # A one-row input can infer an all-missing numeric column as object/string
        # (for example, when the user leaves property age unknown). Restore the
        # fitted training dtypes before passing data to sklearn imputers.
        for column in self.numeric_columns_ + self.binary_columns_:
            if column in engineered:
                numeric = pd.to_numeric(engineered[column], errors="coerce").astype(float)
                engineered[column] = numeric.replace([np.inf, -np.inf], np.nan)
        for column in self.categorical_columns_:
            if column in engineered:
                engineered[column] = (
                    engineered[column].astype("string").str.strip()
                    .replace("", pd.NA).fillna("Unknown")
                )
        return self._transform_engineered(engineered)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        self.fit(frame)
        engineered, self._fit_engineered_ = self._fit_engineered_, None
        return self._transform_engineered(engineered)

    def get_feature_names_out(self) -> list[str]:
        if not hasattr(self, "feature_names_out_"):
            raise ValueError("Preprocessor has not been fitted.")
        return list(self.feature_names_out_)


def feature_documentation(config: FeatureEngineeringConfig, selected_features: Iterable[str]) -> list[dict[str, str]]:
    docs = []
    if config.derive_floor_ratio and {"floor_number", "total_floors"}.issubset(selected_features):
        docs.append({"Feature": "floor_ratio", "Reason": "Floor number divided by total floors; replaces the two source columns and becomes missing for impossible or unavailable combinations."})
    if config.include_amenity_count and any(column in selected_features for column in AMENITY_FEATURES):
        docs.append({"Feature": "amenity_count", "Reason": "Count of available binary amenities; optional because it overlaps with the individual amenity indicators."})
    if "listing_date" in selected_features and config.include_listing_date:
        docs.extend([
            {"Feature": "listing_year", "Reason": "Calendar year derived from listing_date; raw date text is never passed to the model."},
            {"Feature": "listing_month", "Reason": "Calendar month derived from listing_date; raw date text is never passed to the model."},
        ])
    if "location" in selected_features:
        docs.append({"Feature": "location", "Reason": f"Rare localities below {config.location_min_frequency} rows are grouped as Other and one-hot columns are weighted by {config.location_weight:.2f}; location is opt-in."})
    if not docs:
        docs.append({"Feature": "No derived features", "Reason": "Only selected source characteristics are transformed; property_id and raw dates are excluded."})
    return docs


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    """Stable SHA-256 fingerprint for detecting when a saved run belongs to another dataset."""
    stable = frame.copy().reindex(sorted(frame.columns), axis=1)
    hashed = pd.util.hash_pandas_object(stable, index=True).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


def pipeline_settings_fingerprint(
    selected_features: Iterable[str],
    feature_config: FeatureEngineeringConfig,
    configurations: Iterable[dict[str, object]],
    *,
    max_rows: int | None,
    random_seed: int,
) -> str:
    settings = {
        "selected_features": list(selected_features),
        "feature_config": asdict(feature_config),
        "configurations": list(configurations),
        "max_rows": max_rows,
        "random_seed": int(random_seed),
    }
    return hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def default_pipeline_configurations(max_configurations: int = 24) -> list[dict[str, object]]:
    """Small reproducible grid covering every existing algorithm."""
    return build_experiment_configurations(
        CLUSTERING_ALGORITHMS,
        [2, 3, 4],
        [1.8, 2.5, 3.2],
        [5],
        max_configurations=max_configurations,
    )


def _stability_score(
    matrix: np.ndarray,
    baseline_labels: np.ndarray,
    algorithm: str,
    parameters: dict[str, object],
    *,
    random_seed: int,
    repeats: int,
    max_rows: int,
) -> tuple[float, list[float], str]:
    """Compare subsample assignments with the full evaluation fit using adjusted Rand index."""
    from sklearn.metrics import adjusted_rand_score

    n_rows = len(matrix)
    if repeats < 2 or n_rows < 20:
        return np.nan, [], "At least two repeated subsamples and 20 rows are required."
    rng = np.random.default_rng(random_seed)
    reference_pool = np.arange(n_rows)
    if len(reference_pool) > max_rows:
        reference_pool = np.sort(rng.choice(reference_pool, size=max_rows, replace=False))
    subset_size = max(10, int(len(reference_pool) * 0.8))
    if subset_size >= n_rows:
        subset_size = max(2, n_rows - 1)
        reference_pool = np.arange(n_rows)
    scores: list[float] = []
    errors = []
    for repeat in range(repeats):
        repeat_rng = np.random.default_rng(random_seed + 7919 * (repeat + 1))
        selected = np.sort(repeat_rng.choice(reference_pool, size=min(subset_size, len(reference_pool)), replace=False))
        try:
            estimator = _make_cluster_estimator(algorithm, parameters, random_seed + repeat + 1)
            sample_labels = np.asarray(_fit_labels(estimator, matrix[selected], algorithm), dtype=int)
            score = float(adjusted_rand_score(baseline_labels[selected], sample_labels))
            if math.isfinite(score):
                scores.append(score)
        except Exception as exc:
            errors.append(f"repeat {repeat + 1}: {type(exc).__name__}: {exc}")
    if not scores:
        return np.nan, [], "; ".join(errors) or "Stability could not be calculated."
    return float(np.mean(scores)), scores, "; ".join(errors)


def _cluster_size_evidence(labels: np.ndarray, minimum_size: int) -> dict[str, object]:
    labels = np.asarray(labels, dtype=int)
    regular = labels[labels != -1]
    counts = pd.Series(regular).value_counts().sort_index()
    tiny = counts[counts < minimum_size]
    return {
        "cluster_sizes": {str(int(cluster)): int(size) for cluster, size in counts.items()},
        "smallest_cluster": int(counts.min()) if len(counts) else 0,
        "tiny_cluster_count": int(len(tiny)),
        "tiny_property_percent": float(tiny.sum() / len(regular) * 100) if len(regular) else 100.0,
        "noise_percent": float(np.mean(labels == -1) * 100) if len(labels) else 100.0,
    }


def _add_quality_gates(
    candidate_results: pd.DataFrame,
    labels_by_id: dict[int, np.ndarray],
    thresholds: dict[str, float | int],
) -> pd.DataFrame:
    results = candidate_results.copy()
    results["Selection Eligible"] = False
    results["Eligibility Reasons"] = ""
    results["Cluster Size Distribution"] = "{}"
    results["Smallest Cluster"] = 0
    results["Tiny Cluster Count"] = 0
    results["Tiny Property (%)"] = np.nan
    results["Pareto Front"] = False
    results["Selected"] = False
    results["Metric Rank"] = np.nan
    for index, row in results.iterrows():
        reasons: list[str] = []
        if row["Status"] != "Success" or index not in labels_by_id:
            reasons.append("Candidate fit failed.")
            results.at[index, "Eligibility Reasons"] = "; ".join(reasons)
            continue
        labels = labels_by_id[int(row["Configuration ID"])]
        min_size = max(int(thresholds["minimum_cluster_size"]), int(math.ceil(len(labels) * 0.005)))
        evidence = _cluster_size_evidence(labels, min_size)
        metric_values = [row.get("Silhouette Score"), row.get("Davies-Bouldin Score"), row.get("Calinski-Harabasz Score")]
        valid_metrics = sum(pd.notna(value) and math.isfinite(float(value)) for value in metric_values)
        if int(row["Clusters Found"]) < int(thresholds["minimum_cluster_count"]):
            reasons.append(f"Fewer than {thresholds['minimum_cluster_count']} non-noise clusters.")
        if evidence["smallest_cluster"] < 2 and row["Clusters Found"] > 0:
            reasons.append("At least one cluster is a singleton.")
        if evidence["tiny_property_percent"] > (1 - float(thresholds["minimum_non_tiny_share"])) * 100:
            reasons.append(f"More than {(1 - float(thresholds['minimum_non_tiny_share'])) * 100:.0f}% of non-noise properties are in tiny clusters.")
        if row["Algorithm"] == "DBSCAN" and evidence["noise_percent"] > float(thresholds["maximum_dbscan_noise_percent"]):
            reasons.append(f"DBSCAN noise exceeds {thresholds['maximum_dbscan_noise_percent']}%.")
        if valid_metrics < int(thresholds["minimum_valid_metrics"]):
            reasons.append(f"Fewer than {thresholds['minimum_valid_metrics']} internal metrics are valid.")
        stability = row.get("Stability ARI", np.nan)
        raw_repeats = row.get("Stability Repeats", 0)
        repeats = int(raw_repeats) if pd.notna(raw_repeats) and math.isfinite(float(raw_repeats)) else 0
        if repeats < int(thresholds.get("minimum_stability_repeats", 2)):
            reasons.append(f"Fewer than {thresholds.get('minimum_stability_repeats', 2)} stability refits succeeded.")
        if pd.isna(stability) or not math.isfinite(float(stability)):
            reasons.append("Stability was unavailable.")
        elif float(stability) < float(thresholds["minimum_stability_ari"]):
            reasons.append(f"Stability ARI is below {thresholds['minimum_stability_ari']:.2f}.")
        results.at[index, "Cluster Size Distribution"] = json.dumps(evidence["cluster_sizes"], sort_keys=True)
        results.at[index, "Smallest Cluster"] = evidence["smallest_cluster"]
        results.at[index, "Tiny Cluster Count"] = evidence["tiny_cluster_count"]
        results.at[index, "Tiny Property (%)"] = evidence["tiny_property_percent"]
        results.at[index, "Noise (%)"] = evidence["noise_percent"]
        results.at[index, "Selection Eligible"] = not reasons
        results.at[index, "Eligibility Reasons"] = "; ".join(reasons) if reasons else "Passed all documented quality gates."

    valid = results["Selection Eligible"]
    # Percentile ranks normalize incomparable units; lower rank is better.
    metric_rank_columns = []
    for column, ascending in (("Silhouette Score", False), ("Davies-Bouldin Score", True), ("Calinski-Harabasz Score", False)):
        values = pd.to_numeric(results[column], errors="coerce")
        usable = valid & np.isfinite(values)
        ranks = pd.Series(np.nan, index=results.index, dtype=float)
        if usable.any():
            ranks.loc[usable] = values.loc[usable].rank(pct=True, ascending=ascending)
        metric_rank_columns.append(ranks)
    metric_ranks = pd.concat(metric_rank_columns, axis=1)
    # Missing metric dimensions receive the worst percentile rank so a candidate
    # cannot gain an advantage merely because one score was undefined.
    results["Metric Rank"] = metric_ranks.fillna(1.0).mean(axis=1).where(valid)
    if valid.any():
        stability = pd.to_numeric(results["Stability ARI"], errors="coerce")
        runtime = pd.to_numeric(results["Total Runtime (s)"], errors="coerce")
        noise = pd.to_numeric(results["Noise (%)"], errors="coerce")
        objective_ranks = pd.DataFrame(index=results.index)
        objective_ranks["metric"] = results["Metric Rank"]
        objective_ranks["stability"] = (1 - stability.rank(pct=True, ascending=True)).where(valid)
        objective_ranks["runtime"] = runtime.rank(pct=True, ascending=True).where(valid)
        objective_ranks["noise"] = noise.rank(pct=True, ascending=True).where(valid)
        objective_ranks = objective_ranks.where(valid)
        objectives = objective_ranks.fillna(1.0)
        candidate_indices = results.index[valid].tolist()
        for current in candidate_indices:
            current_values = objectives.loc[current].to_numpy(dtype=float)
            dominated = False
            for other in candidate_indices:
                if current == other:
                    continue
                other_values = objectives.loc[other].to_numpy(dtype=float)
                if np.all(other_values <= current_values) and np.any(other_values < current_values):
                    dominated = True
                    break
            results.at[current, "Pareto Front"] = not dominated
    return results


def _select_configuration(results: pd.DataFrame) -> tuple[int | None, list[str]]:
    eligible = results.loc[results["Selection Eligible"]]
    if eligible.empty:
        return None, ["No candidate passed every minimum quality gate; no model was selected."]
    front = eligible.loc[eligible["Pareto Front"]]
    if front.empty:
        front = eligible
    # Lexicographic tie-break: stability, multi-metric rank, runtime, then fewer clusters.
    chosen = front.sort_values(
        ["Stability ARI", "Metric Rank", "Total Runtime (s)", "Clusters Found"],
        ascending=[False, True, True, True], kind="stable",
    ).iloc[0]
    reasons = [
        "Passed minimum cluster-count, cluster-size, valid-metric, DBSCAN-noise, and stability gates.",
        "Lies on the non-dominated Pareto front across normalized metric ranks, stability, runtime, and noise where available.",
        f"Tie-break favored stability ARI {chosen['Stability ARI']:.3f}, then balanced metric rank, runtime, and the simpler cluster count.",
        "This is an automated recommendation based on internal evidence, not ground-truth accuracy.",
    ]
    return int(chosen["Configuration ID"]), reasons


def _make_anomaly_report(labelled: pd.DataFrame, matrix: np.ndarray, algorithm: str, model: object) -> pd.DataFrame:
    label_column = f"{algorithm}_cluster"
    labels = labelled[label_column].to_numpy(dtype=int)
    if algorithm == "DBSCAN":
        anomalies = labelled.loc[labels == -1].copy()
        if not anomalies.empty:
            anomalies["Potential anomaly reason"] = "DBSCAN density noise; this is not evidence of fraud or a property defect."
            anomalies["Anomaly distance"] = np.nan
        return anomalies
    regular_labels = sorted(set(labels.tolist()) - {-1})
    centers = np.vstack([matrix[labels == cluster].mean(axis=0) for cluster in regular_labels])
    cluster_to_center = {cluster: index for index, cluster in enumerate(regular_labels)}
    assigned_centers = np.vstack([centers[cluster_to_center[label]] for label in labels])
    distances = np.linalg.norm(matrix - assigned_centers, axis=1)
    threshold_by_cluster = {
        cluster: float(np.quantile(distances[labels == cluster], 0.99))
        for cluster in regular_labels
    }
    flagged = np.array([distances[i] > threshold_by_cluster[labels[i]] for i in range(len(labels))])
    anomalies = labelled.loc[flagged].copy()
    anomalies["Potential anomaly reason"] = "Feature-space distance is in the top 1% within its assigned cluster; review only."
    anomalies["Anomaly distance"] = distances[flagged]
    return anomalies


def train_unsupervised_pipeline(
    frame: pd.DataFrame,
    *,
    selected_features: Iterable[str] | None = None,
    feature_config: FeatureEngineeringConfig | None = None,
    configurations: Iterable[dict[str, object]] | None = None,
    max_rows: int | None = 1500,
    random_seed: int = 42,
    max_configurations: int = 24,
    stability_repeats: int = 2,
    stability_sample_rows: int = 800,
    selection_thresholds: dict[str, float | int] | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
) -> PipelineRunResult:
    """Run validation through automatic model selection and profiling in one call."""
    if progress_callback:
        progress_callback("Validating dataset", 0.05)
    validated, validation_report, issues = validate_dataset(frame)
    feature_config = feature_config or FeatureEngineeringConfig()
    selected = list(selected_features) if selected_features is not None else default_feature_selection(validated, feature_config)
    if not selected:
        raise ValueError("Select at least one usable property feature.")
    if max_rows is not None and max_rows < 20:
        raise ValueError("Use at least 20 rows for the automated training pipeline.")
    if stability_repeats < 2:
        raise ValueError("Use at least two stability subsamples so stability is actually assessed.")
    if stability_sample_rows < 20:
        raise ValueError("Stability sample cap must be at least 20 rows.")
    evaluation_data = validated.copy()
    if max_rows is not None and max_rows < len(evaluation_data):
        evaluation_data = evaluation_data.sample(n=max_rows, random_state=random_seed).sort_index().reset_index(drop=True)
    if progress_callback:
        progress_callback("Engineering and preprocessing shared features", 0.20)
    preprocessor = PropertyFeaturePreprocessor(selected, feature_config)
    matrix = preprocessor.fit_transform(evaluation_data)
    names = preprocessor.get_feature_names_out()
    configurations = list(configurations) if configurations is not None else default_pipeline_configurations(max_configurations)
    if not configurations:
        raise ValueError("The automatic pipeline requires at least one candidate configuration.")
    if progress_callback:
        progress_callback(f"Evaluating {len(configurations)} candidate configurations", 0.35)
    raw_results, labels_by_id, evaluation_rows, fitted_estimators = run_clustering_experiments(
        evaluation_data,
        selected,
        configurations,
        max_rows=None,
        random_seed=random_seed,
        max_configurations=max_configurations,
        prepared_matrix=matrix,
        retain_estimators=True,
        preserve_rows=True,
    )
    configuration_by_id = {idx: config for idx, config in enumerate(configurations)}
    for config_id, labels in labels_by_id.items():
        if progress_callback:
            progress_callback(f"Measuring stability for configuration {config_id + 1}", 0.55 + 0.25 * (config_id + 1) / max(1, len(configurations)))
        config = configuration_by_id[config_id]
        stability_start = perf_counter()
        stability, repeated_scores, stability_error = _stability_score(
            matrix, labels, str(config["Algorithm"]), dict(config.get("Parameters", {})),
            random_seed=random_seed + config_id * 101, repeats=stability_repeats,
            max_rows=stability_sample_rows,
        )
        raw_results.loc[raw_results["Configuration ID"] == config_id, "Stability ARI"] = stability
        raw_results.loc[raw_results["Configuration ID"] == config_id, "Stability Repeats"] = len(repeated_scores)
        raw_results.loc[raw_results["Configuration ID"] == config_id, "Stability Scores"] = json.dumps(repeated_scores)
        raw_results.loc[raw_results["Configuration ID"] == config_id, "Stability Error"] = stability_error
        original_runtime = float(raw_results.loc[raw_results["Configuration ID"] == config_id, "Execution Time (s)"].iloc[0])
        stability_runtime = perf_counter() - stability_start
        raw_results.loc[raw_results["Configuration ID"] == config_id, "Total Runtime (s)"] = original_runtime + stability_runtime
    # Stability time includes extra fits. Attribute it via timing each config's stability section.
    # The runner's Execution Time remains the initial fit plus metric calculation.
    results = _add_quality_gates(raw_results, labels_by_id, selection_thresholds or DEFAULT_SELECTION_THRESHOLDS)
    selected_id, rationale = _select_configuration(results)
    results["Selection Outcome"] = results["Eligibility Reasons"]
    if selected_id is not None:
        results.loc[results["Configuration ID"] == selected_id, "Selected"] = True
        for idx, candidate in results.iterrows():
            if int(candidate["Configuration ID"]) == selected_id:
                results.at[idx, "Selection Outcome"] = "Selected by automated Pareto and tie-break strategy."
            elif bool(candidate["Selection Eligible"]):
                result_reason = "Passed quality gates but was dominated on the Pareto comparison." if not bool(candidate["Pareto Front"]) else "Passed quality gates and is Pareto-viable, but lost the stability/metric/runtime/cluster-count tie-break."
                results.at[idx, "Selection Outcome"] = result_reason
    selected_config: dict[str, object] | None = None
    selected_model = None
    selected_labels = pd.DataFrame(columns=["property_id", "cluster_label"])
    cluster_profiles = pd.DataFrame()
    interpretations: list[dict[str, object]] = []
    anomaly_report = pd.DataFrame()
    visualization_data = evaluation_rows.copy()
    selected_estimators: dict[int, object] = {}
    assignment_state: dict[str, object] = {}
    if selected_id is not None:
        selected_config = configuration_by_id[selected_id]
        selected_model = fitted_estimators[selected_id]
        selected_algorithm = str(selected_config["Algorithm"])
        visualization_data, _ = apply_experiment_configuration(evaluation_rows, selected_id, labels_by_id, raw_results)
        label_column = f"{selected_algorithm}_cluster"
        profile, interpretations = interpret_clusters(visualization_data, label_column)
        cluster_profiles = profile
        selected_labels = visualization_data[["property_id", label_column]].rename(columns={label_column: "cluster_label"}).copy()
        anomaly_report = _make_anomaly_report(visualization_data, matrix, selected_algorithm, selected_model)
        selected_estimators[selected_id] = selected_model
        regular_labels = sorted(set(labels_by_id[selected_id].tolist()) - {-1})
        assignment_state["cluster_representatives"] = {
            int(cluster): matrix[labels_by_id[selected_id] == cluster].mean(axis=0)
            for cluster in regular_labels
        }
        if selected_algorithm == "DBSCAN" and hasattr(selected_model, "components_"):
            core_indices = np.asarray(selected_model.core_sample_indices_, dtype=int)
            assignment_state["dbscan_core_vectors"] = np.asarray(selected_model.components_)
            assignment_state["dbscan_core_labels"] = labels_by_id[selected_id][core_indices]
        rationale.extend([
            f"Selected {selected_algorithm} with parameters {selected_config['Parameters']}.",
            f"Metrics: Silhouette={results.loc[results['Configuration ID'] == selected_id, 'Silhouette Score'].iloc[0]}, "
            f"Davies–Bouldin={results.loc[results['Configuration ID'] == selected_id, 'Davies-Bouldin Score'].iloc[0]}, "
            f"Calinski–Harabasz={results.loc[results['Configuration ID'] == selected_id, 'Calinski-Harabasz Score'].iloc[0]}.",
        ])
    else:
        selected_algorithm = None
    thresholds = dict(DEFAULT_SELECTION_THRESHOLDS)
    if selection_thresholds:
        thresholds.update(selection_thresholds)
    metadata: dict[str, object] = {
        "schema_version": UNSUPERVISED_SCHEMA_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": dataset_fingerprint(frame),
        "source_rows": len(frame),
        "retained_rows": len(validated),
        "evaluated_rows": len(evaluation_data),
        "evaluation_set": "Sample" if len(evaluation_data) < len(validated) else "Full dataset",
        "random_seed": random_seed,
        "settings_sha256": pipeline_settings_fingerprint(selected, feature_config, configurations, max_rows=max_rows, random_seed=random_seed),
        "selected_features": selected,
        "selected_algorithm": selected_algorithm,
        "selected_parameters": selected_config.get("Parameters") if selected_config else None,
        "selection_rationale": rationale,
        "selection_candidate_count": len(configurations),
        "engineered_feature_names": names,
        "feature_engineering_config": asdict(feature_config),
        "selection_thresholds": thresholds,
        "stability_method": f"{stability_repeats} reproducible 80% subsamples, capped at {stability_sample_rows} reference rows; adjusted Rand index versus full evaluation fit.",
        "versions": {"python": __import__("platform").python_version(), "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "location_policy": "Excluded from default features; when enabled, rare localities are grouped and locality one-hot columns are downweighted.",
        "price_feature_used": False,
    }
    if progress_callback:
        progress_callback("Model selection and interpretation complete", 1.0)
    return PipelineRunResult(
        validation_data=validated,
        validation_report=validation_report,
        validation_issues=issues,
        engineered_feature_names=names,
        feature_documentation=feature_documentation(feature_config, selected),
        preprocessor=preprocessor,
        candidate_results=results,
        selected_configuration=selected_config,
        selected_algorithm=selected_algorithm,
        selection_rationale=rationale,
        selection_thresholds=thresholds,
        selected_model=selected_model,
        selected_labels=selected_labels,
        cluster_profiles=cluster_profiles,
        cluster_interpretations=interpretations,
        anomaly_report=anomaly_report,
        visualization_data=visualization_data,
        metadata=metadata,
        no_satisfactory_model=selected_id is None,
        fitted_estimators=selected_estimators,
        assignment_state=assignment_state,
    )


def assign_new_property(result: PipelineRunResult, property_record: dict[str, object] | pd.DataFrame) -> dict[str, object]:
    """Assign one new listing with the saved preprocessing state and no refitting."""
    if result.selected_model is None or result.selected_algorithm is None:
        raise ValueError("No satisfactory selected model is available for assignment.")
    row = property_record if isinstance(property_record, pd.DataFrame) else pd.DataFrame([property_record])
    if len(row) != 1:
        raise ValueError("Provide exactly one new property row.")
    matrix = result.preprocessor.transform(row)
    algorithm = result.selected_algorithm
    model = result.selected_model
    if algorithm == "K-Means":
        label = int(model.predict(matrix)[0])
        distance = float(np.linalg.norm(matrix[0] - model.cluster_centers_[label]))
        return {"cluster_label": label, "assignment_method": "K-Means predict", "distance_to_centroid": distance, "confidence": None}
    if algorithm == "Gaussian Mixture":
        probabilities = model.predict_proba(matrix)[0]
        label = int(model.predict(matrix)[0])
        return {"cluster_label": label, "assignment_method": "GMM posterior assignment", "confidence": float(np.max(probabilities)), "posterior_probabilities": probabilities.tolist()}
    if algorithm == "Agglomerative":
        representatives = result.assignment_state.get("cluster_representatives", {})
        cluster_ids = np.asarray(list(representatives.keys()), dtype=int)
        if len(cluster_ids) == 0:
            raise ValueError("No Agglomerative cluster representatives are available.")
        centers = np.vstack([representatives[int(cluster)] for cluster in cluster_ids])
        nearest, distances = pairwise_distances_argmin_min(matrix, centers)
        return {"cluster_label": int(cluster_ids[nearest[0]]), "assignment_method": "Approximation: nearest training-cluster centroid", "distance_to_centroid": float(distances[0]), "confidence": None}
    if algorithm == "DBSCAN":
        core_vectors = result.assignment_state.get("dbscan_core_vectors")
        core_labels = result.assignment_state.get("dbscan_core_labels")
        if core_vectors is None or len(core_vectors) == 0:
            return {"cluster_label": -1, "assignment_method": "DBSCAN has no stored core samples; assigned as noise", "distance_to_nearest_core": None, "confidence": None}
        nearest, distances = pairwise_distances_argmin_min(matrix, np.asarray(core_vectors))
        distance = float(distances[0])
        label = int(np.asarray(core_labels)[nearest[0]]) if distance <= float(model.eps) else -1
        return {"cluster_label": label, "assignment_method": "Approximation: nearest DBSCAN core sample within eps; otherwise noise", "distance_to_nearest_core": distance, "eps": float(model.eps), "confidence": None}
    raise ValueError(f"Unsupported selected algorithm: {algorithm}")


def find_similar_with_pipeline(
    result: PipelineRunResult,
    property_id: object,
    *,
    n_neighbors: int = 5,
) -> tuple[pd.DataFrame, list[str]]:
    """Find similar rows using the exact fitted feature transformation from the selected pipeline."""
    if n_neighbors < 1:
        raise ValueError("Choose at least one similar property.")
    rows = result.validation_data.reset_index(drop=True)
    matches = np.flatnonzero(rows["property_id"].astype(str).to_numpy() == str(property_id))
    if len(matches) != 1:
        raise ValueError(f"Property ID {property_id!r} must match exactly one validated listing.")
    matrix = result.preprocessor.transform(rows)
    estimator = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(rows)), metric="euclidean")
    estimator.fit(matrix)
    distances, indices = estimator.kneighbors(matrix[matches[0]:matches[0] + 1])
    pairs = [(int(i), float(d)) for i, d in zip(indices[0], distances[0]) if int(i) != int(matches[0])][:n_neighbors]
    similar = rows.iloc[[index for index, _ in pairs]].copy()
    similar.insert(0, "Similarity rank", range(1, len(similar) + 1))
    similar["Feature distance"] = [distance for _, distance in pairs]
    return similar, result.engineered_feature_names


def save_pipeline_artifact(
    result: PipelineRunResult,
    directory: str | Path = UNSUPERVISED_ARTIFACT_DIR,
) -> Path:
    """Persist the trusted local selected pipeline and a human-readable metadata sidecar."""
    if result.selected_model is None:
        raise ValueError("Cannot save a selected model because no candidate passed the quality gates.")
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    model_path = target / "estate_unsupervised_pipeline.joblib"
    metadata_path = target / "metadata.json"
    with tempfile.NamedTemporaryFile(dir=target, prefix=".estate_unsupervised_", suffix=".tmp", delete=False) as handle:
        temp_model = Path(handle.name)
    with tempfile.NamedTemporaryFile(dir=target, prefix=".estate_unsupervised_metadata_", suffix=".tmp", delete=False, mode="w", encoding="utf-8") as handle:
        temp_metadata = Path(handle.name)
    try:
        joblib.dump(result, temp_model)
        digest = _sha256_file(temp_model)
        sidecar = {**result.metadata, "artifact_sha256": digest}
        temp_metadata.write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")
        # Either interrupted replacement order is detected by the sidecar digest.
        os.replace(temp_model, model_path)
        os.replace(temp_metadata, metadata_path)
    finally:
        temp_model.unlink(missing_ok=True)
        temp_metadata.unlink(missing_ok=True)
    return model_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pipeline_artifact(directory: str | Path = UNSUPERVISED_ARTIFACT_DIR) -> PipelineRunResult:
    """Load the project's own local artifact; never pass arbitrary uploaded paths here."""
    target = Path(directory)
    model_path = target / "estate_unsupervised_pipeline.joblib"
    metadata_path = target / "metadata.json"
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("No saved unsupervised model was found. Train the automatic pipeline first.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != UNSUPERVISED_SCHEMA_VERSION:
        raise ValueError("The saved unsupervised artifact uses an incompatible schema version.")
    if metadata.get("artifact_sha256") != _sha256_file(model_path):
        raise ValueError("Saved unsupervised artifact integrity check failed; the model and metadata do not match.")
    # joblib/pickle artifacts are executable. This path is deliberately fixed to
    # this application's locally generated artifact directory, not user input.
    result = joblib.load(model_path)
    if not isinstance(result, PipelineRunResult) or result.metadata.get("schema_version") != UNSUPERVISED_SCHEMA_VERSION:
        raise ValueError("Saved unsupervised model artifact is incompatible or invalid.")
    sidecar_metadata = {key: value for key, value in metadata.items() if key != "artifact_sha256"}
    if sidecar_metadata != result.metadata:
        raise ValueError("Saved unsupervised artifact metadata does not match the fitted pipeline.")
    saved_sklearn = metadata.get("versions", {}).get("scikit_learn")
    if saved_sklearn != sklearn.__version__:
        raise ValueError(f"Saved unsupervised model requires scikit-learn {saved_sklearn}; this environment has {sklearn.__version__}.")
    if metadata.get("selected_features") != list(result.preprocessor.selected_features):
        raise ValueError("Saved feature schema does not match the fitted preprocessing object.")
    return result
