"""Training and inference contract for user-curated Bengaluru listings.

This module owns the listing-data cleaning and model contract. Target prices are in
Indian lakhs (1 lakh = INR 100,000); area is square feet.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "price_lakhs"
ID_COLUMN = "property_id"
# ── Model input contract (target and identifier are intentionally separate) ──
NUMERIC_FEATURES = [
    "latitude", "longitude", "area_sqft", "BHK", "bathrooms", "balconies",
    "property_age_years", "floor_number", "total_floors", "parking_spaces",
    "lift", "security", "gym", "swimming_pool", "power_backup", "gated_community",
]
ENGINEERED_NUMERIC_FEATURES = ["bhk_x_balconies"]
CATEGORICAL_FEATURES = ["location", "condition", "property_type", "furnishing_status"]
FEATURE_SCHEMA = [
    "location", "latitude", "longitude", "area_sqft", "BHK", "bathrooms",
    "balconies", "listing_date", "property_age_years", "condition", "floor_number",
    "total_floors", "property_type", "furnishing_status", "parking_spaces", "lift",
    "security", "gym", "swimming_pool", "power_backup", "gated_community",
]
DATE_FEATURES = ["listing_date_ordinal", "listing_month_sin", "listing_month_cos"]
SCHEMA_VERSION = "bengaluru-listings-v3"
ARTIFACT_DIR = Path("artifacts") / "model"
REQUIRED_PREDICTION_FEATURES = {"location", "area_sqft", "BHK", "bathrooms", "balconies"}


def normalize_property_location(value: Any) -> str:
    """Normalize locality whitespace without guessing spelling variants."""
    cleaned = " ".join(str(value or "").split())
    if not cleaned:
        return "Unknown"
    return cleaned


def prepare_property_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Validate supplied rows; preserve valid listings and explain exclusions.

    `property_id` is retained only for audit identity and is never returned as
    a model input. Numeric feature corruption is set to missing for in-pipeline
    median imputation; malformed categories/dates become explicit unknowns.
    Target values are never imputed. Clearly impossible floors and invalid
    target/area/bedroom rows are excluded with reason counts.
    """
    # Step 1: select the documented columns and parse numeric measurements.
    listings = frame.copy()
    missing_columns = sorted(set([*FEATURE_SCHEMA, TARGET]) - set(listings.columns))
    if missing_columns:
        raise ValueError(f"Property CSV is missing required columns: {missing_columns}")
    listings = listings.loc[:, [*FEATURE_SCHEMA, TARGET, *([ID_COLUMN] if ID_COLUMN in listings.columns else [])]].copy()
    numeric_malformed = {}
    numeric_nonfinite = {}
    for col in [*NUMERIC_FEATURES, TARGET]:
        parsed = pd.to_numeric(listings[col], errors="coerce").astype(float)
        numeric_malformed[col] = int((listings[col].notna() & parsed.isna()).sum())
        numeric_nonfinite[col] = int((parsed.notna() & ~np.isfinite(parsed)).sum())
        listings[col] = parsed
        listings.loc[~np.isfinite(listings[col]), col] = np.nan
    parsed_dates = pd.to_datetime(listings["listing_date"], errors="coerce", utc=True)
    malformed_dates = int((listings["listing_date"].notna() & parsed_dates.isna()).sum())

    listings["listing_date"] = parsed_dates

    # Step 2: exclude rows only for invalid targets, area, BHK, or impossible floors.
    reason_masks: dict[str, pd.Series] = {
        "invalid_or_nonpositive_target": listings[TARGET].isna() | (listings[TARGET] <= 0),
        "invalid_or_nonpositive_area": listings["area_sqft"].isna() | (listings["area_sqft"] <= 0),
        "invalid_bhk": listings["BHK"].isna() | (listings["BHK"] < 1) | (listings["BHK"] > 10) | (listings["BHK"] % 1 != 0),
        "invalid_floor_combination": listings["floor_number"].notna() & listings["total_floors"].notna() &
            ((listings["floor_number"] < 0) | (listings["total_floors"] < 1) | (listings["floor_number"] > listings["total_floors"])),
    }
    excluded = pd.Series(False, index=listings.index)
    for mask in reason_masks.values():
        excluded |= mask
    exclusion_counts = {reason: int(mask.sum()) for reason, mask in reason_masks.items()}
    kept_listings = listings.loc[~excluded].copy()

    # Step 3: mark invalid optional features missing for fold-fitted imputation.
    # Invalid optional numeric measurements are imputed inside each training
    # fold. Values beyond conservative physical bounds are treated as missing.
    bounds = {
        "latitude": (-90, 90), "longitude": (-180, 180), "area_sqft": (1, 100_000),
        "BHK": (1, 10), "bathrooms": (1, 30), "balconies": (0, 20),
        "property_age_years": (0, 200), "floor_number": (0, 200),
        "total_floors": (1, 200), "parking_spaces": (0, 50),
        "lift": (0, 1), "security": (0, 1), "gym": (0, 1),
        "swimming_pool": (0, 1), "power_backup": (0, 1), "gated_community": (0, 1),
    }
    corruption_counts = {}
    for col, (low, high) in bounds.items():
        invalid_values = kept_listings[col].notna() & ~kept_listings[col].between(low, high)
        if col in {"BHK", "bathrooms", "balconies", "floor_number", "total_floors", "parking_spaces", "lift", "security", "gym", "swimming_pool", "power_backup", "gated_community"}:
            invalid_values |= kept_listings[col].notna() & (kept_listings[col] % 1 != 0)
        # Impossible floor relationships are excluded above; other invalid
        # optional numeric entries are missing and will be median-imputed.
        corruption_counts[col] = int(invalid_values.sum()) + numeric_malformed.get(col, 0) + numeric_nonfinite.get(col, 0)
        kept_listings.loc[invalid_values, col] = np.nan
    for col in CATEGORICAL_FEATURES:
        values = kept_listings[col].astype("string").str.strip()
        values = values.mask(values.isna() | values.eq(""), "Unknown")
        kept_listings[col] = values.str.replace(r"\s+", " ", regex=True)
    # A malformed date is explicit Unknown; valid timestamps are later mapped
    # to ordinal and month features by the inference transformer.
    kept_listings.loc[kept_listings["listing_date"].isna(), "listing_date"] = pd.NaT
    features = kept_listings[FEATURE_SCHEMA].copy()
    target_prices = kept_listings[TARGET].astype(float).copy()
    report = {
        "input_rows": int(len(listings)), "kept_rows": int(len(kept_listings)),
        "excluded_rows": int(excluded.sum()), "exclusion_reasons": exclusion_counts,
        "numeric_values_marked_missing_for_imputation": corruption_counts,
        "missing_by_column_before": {c: int(frame[c].isna().sum()) for c in frame.columns},
        "malformed_numeric_values_before": numeric_malformed,
        "nonfinite_numeric_values_before": numeric_nonfinite,
        "malformed_listing_dates_before": malformed_dates,
        "duplicate_full_rows": int(frame.duplicated().sum()),
        "duplicate_property_ids": int(frame[ID_COLUMN].duplicated().sum()) if ID_COLUMN in frame else None,
        "duplicate_feature_signatures": int(frame.drop(columns=[ID_COLUMN, TARGET], errors="ignore").duplicated().sum()),
        "price_per_sqft_lakhs": (target_prices / features["area_sqft"]).describe().to_dict(),
    }
    return features, target_prices, report


def _date_to_features(values: pd.Series) -> pd.DataFrame:
    dates = pd.to_datetime(values, errors="coerce", utc=True)
    ordinal = dates.map(lambda x: float(x.toordinal()) if pd.notna(x) else np.nan)
    month = dates.dt.month.astype(float)
    return pd.DataFrame({
        "listing_date_ordinal": ordinal,
        "listing_month_sin": np.sin(2 * np.pi * month / 12),
        "listing_month_cos": np.cos(2 * np.pi * month / 12),
    }, index=values.index)


class PropertyFrameTransformer(BaseEstimator, TransformerMixin):
    """Pickle-friendly minimal date featurizer; wraps fitted sklearn pipeline."""
    def fit(self, X, y=None):
        columns = list(X.columns)
        if columns != FEATURE_SCHEMA:
            raise ValueError(f"Property training schema mismatch: expected {FEATURE_SCHEMA}, got {columns}.")
        self.feature_names_in_ = np.asarray(columns, dtype=object)
        return self

    def transform(self, X):
        columns = list(X.columns)
        if columns != FEATURE_SCHEMA:
            raise ValueError(f"Property inference schema mismatch: expected {FEATURE_SCHEMA}, got {columns}.")
        frame = X.copy()
        balconies = pd.to_numeric(frame["balconies"], errors="coerce")
        frame["bhk_x_balconies"] = pd.to_numeric(frame["BHK"], errors="coerce") * balconies
        date_features = _date_to_features(frame["listing_date"])
        frame = frame.drop(columns=["listing_date"])
        return pd.concat([frame, date_features], axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(
            [c for c in FEATURE_SCHEMA if c != "listing_date"]
            + ENGINEERED_NUMERIC_FEATURES + DATE_FEATURES,
            dtype=object,
        )


def make_pipeline(
    estimator,
    params: dict[str, Any] | None = None,
    *,
    scale_numeric: bool = True,
) -> Pipeline:
    """Build preprocessing and regression as one fold-fitted sklearn pipeline."""
    model = estimator(**(params or {}))
    numeric_features = [
        feature for feature in NUMERIC_FEATURES if feature != "area_sqft"
    ] + ["area_sqft", *ENGINEERED_NUMERIC_FEATURES, *DATE_FEATURES]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        # with_mean=False remains compatible with sparse one-hot output.
        numeric_steps.append(("scale", StandardScaler(with_mean=False)))
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore", min_frequency=5, sparse_output=True,
        )),
    ])
    preprocessing = ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric_features),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("date_features", PropertyFrameTransformer()),
        ("preprocess", preprocessing),
        ("model", model),
    ])


def build_property_input(input_values: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
    """Return one validated row in the exact raw model schema order."""
    input_record = (
        input_values.iloc[0].to_dict()
        if isinstance(input_values, pd.DataFrame)
        else dict(input_values)
    )
    missing_features = [name for name in FEATURE_SCHEMA if name not in input_record]
    unexpected_features = [name for name in input_record if name not in FEATURE_SCHEMA]
    if missing_features or unexpected_features:
        raise ValueError(
            "Property prediction schema mismatch; "
            f"missing={missing_features}, unexpected={unexpected_features}."
        )
    prediction_row = pd.DataFrame(
        [{name: input_record[name] for name in FEATURE_SCHEMA}],
        columns=FEATURE_SCHEMA,
    )
    for feature_name in NUMERIC_FEATURES:
        value = prediction_row.at[0, feature_name]
        if value is None or (isinstance(value, str) and not value.strip()):
            if feature_name in REQUIRED_PREDICTION_FEATURES:
                raise ValueError(f"{feature_name} is required.")
            prediction_row.at[0, feature_name] = np.nan
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{feature_name} must be numeric.") from None
        if not np.isfinite(numeric_value):
            raise ValueError(f"{feature_name} must be finite.")
        prediction_row.at[0, feature_name] = numeric_value
    if prediction_row.at[0, "area_sqft"] <= 0:
        raise ValueError("area_sqft must be positive square feet.")
    bhk = float(prediction_row.at[0, "BHK"])
    if not bhk.is_integer() or not 1 <= bhk <= 10:
        raise ValueError("BHK must be a whole number from 1 through 10.")
    feature_limits = {
        "latitude": (-90, 90), "longitude": (-180, 180), "area_sqft": (1, 100_000),
        "bathrooms": (1, 30), "balconies": (0, 20), "property_age_years": (0, 200),
        "floor_number": (0, 200), "total_floors": (1, 200), "parking_spaces": (0, 50),
        "lift": (0, 1), "security": (0, 1), "gym": (0, 1),
        "swimming_pool": (0, 1), "power_backup": (0, 1), "gated_community": (0, 1),
    }
    integer_features = {
        "BHK", "bathrooms", "balconies", "property_age_years", "floor_number",
        "total_floors", "parking_spaces", "lift", "security", "gym",
        "swimming_pool", "power_backup", "gated_community",
    }
    for feature_name, (lower, upper) in feature_limits.items():
        numeric_value = float(prediction_row.at[0, feature_name])
        if np.isnan(numeric_value):
            if feature_name in REQUIRED_PREDICTION_FEATURES:
                raise ValueError(f"{feature_name} is required.")
            continue
        if not lower <= numeric_value <= upper:
            raise ValueError(f"{feature_name} must be between {lower} and {upper}.")
        if feature_name in integer_features and not numeric_value.is_integer():
            raise ValueError(f"{feature_name} must be a whole number.")
    floor_number = float(prediction_row.at[0, "floor_number"])
    total_floors = float(prediction_row.at[0, "total_floors"])
    if np.isfinite(floor_number) and np.isfinite(total_floors) and floor_number > total_floors:
        raise ValueError("floor_number cannot exceed total_floors.")
    for feature_name in CATEGORICAL_FEATURES:
        if feature_name == "location":
            prediction_row.at[0, feature_name] = normalize_property_location(
                prediction_row.at[0, feature_name]
            )
        else:
            prediction_row.at[0, feature_name] = (
                str(prediction_row.at[0, feature_name] or "Unknown").strip() or "Unknown"
            )
    prediction_row.at[0, "listing_date"] = pd.to_datetime(
        prediction_row.at[0, "listing_date"], errors="coerce", utc=True,
    )
    return prediction_row


def predict_property(model, values: dict[str, Any] | pd.DataFrame) -> float:
    row = build_property_input(values)
    prediction = float(model.predict(row)[0])
    if not np.isfinite(prediction) or prediction <= 0:
        raise ValueError("Property model returned an invalid estimate for these inputs.")
    return prediction


def predict_property_batch(model, values: list[dict[str, Any]]) -> list[float]:
    """Validate and predict many property rows in one model call."""
    if not values:
        return []
    rows = pd.concat([build_property_input(record) for record in values], ignore_index=True)
    predictions = np.asarray(model.predict(rows), dtype=float).reshape(-1)
    if len(predictions) != len(values) or not np.isfinite(predictions).all() or (predictions <= 0).any():
        raise ValueError("Property model returned an invalid estimate for one or more candidates.")
    return predictions.tolist()


def save_property_artifacts(model, metadata: dict[str, Any], directory: str | Path = ARTIFACT_DIR) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path / "property_price_model.joblib")
    (path / "model_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")


def load_property_artifacts(directory: str | Path = ARTIFACT_DIR):
    path = Path(directory)
    model_path = path / "property_price_model.joblib"
    metadata_path = path / "model_metadata.json"
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Model files are missing. Train them with `python -m estate_ml.train_model --data <CSV>`.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if metadata.get("scikit_learn_version") != sklearn.__version__:
        raise ValueError(
            "Model library version mismatch: this model was trained with scikit-learn "
            f"{metadata.get('scikit_learn_version')}, but this app uses {sklearn.__version__}."
        )
    model = joblib.load(model_path)
    if list(getattr(model, "feature_names_in_", [])) != FEATURE_SCHEMA:
        raise ValueError("The saved model input fields do not match its metadata.")
    return model, metadata
