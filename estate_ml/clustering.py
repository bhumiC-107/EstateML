"""Reusable preparation, clustering, and evaluation for property listings."""
from __future__ import annotations

from time import perf_counter
from pathlib import Path
from typing import Iterable
import json

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

ID_COLUMN = "property_id"
DATE_COLUMN = "listing_date"
DEFAULT_NUMERIC_FEATURES = [
    "area_sqft", "BHK", "bathrooms", "balconies", "property_age_years",
    "floor_number", "total_floors", "parking_spaces", "lift", "security",
    "gym", "swimming_pool", "power_backup", "gated_community",
]
DEFAULT_CATEGORICAL_FEATURES = ["location", "condition", "property_type", "furnishing_status"]
COORDINATE_FEATURES = ["latitude", "longitude"]
BINARY_FEATURES = ["lift", "security", "gym", "swimming_pool", "power_backup", "gated_community"]
CLUSTERING_ALGORITHMS = ("K-Means", "Agglomerative", "DBSCAN", "Gaussian Mixture")
MAX_EXPERIMENT_CONFIGURATIONS = 24
MAX_EXPERIMENT_SILHOUETTE_ROWS = 2000


def load_dataset(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load a CSV or copy a frame, validate the basic property schema, and clean it."""
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    else:
        frame = pd.read_csv(source)
    if frame.empty:
        raise ValueError("The dataset contains no rows.")
    if ID_COLUMN not in frame.columns:
        raise ValueError(f"The dataset must contain the identifier column {ID_COLUMN!r}.")
    frame = frame.drop_duplicates().reset_index(drop=True)
    for column in frame.columns:
        if column in (ID_COLUMN, DATE_COLUMN):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            frame[column] = values
    valid_ranges = {
        "latitude": (-90, 90), "longitude": (-180, 180), "area_sqft": (1, 100_000),
        "BHK": (1, 10), "bathrooms": (0, 30), "balconies": (0, 20),
        "property_age_years": (0, 200), "floor_number": (0, 200),
        "total_floors": (1, 200), "parking_spaces": (0, 50),
        **{column: (0, 1) for column in BINARY_FEATURES},
    }
    for column, (low, high) in valid_ranges.items():
        if column in frame:
            invalid = frame[column].notna() & ~frame[column].between(low, high)
            frame.loc[invalid, column] = np.nan
    return frame


def available_features(frame: pd.DataFrame) -> list[str]:
    """Return selectable property features, excluding identifiers, dates and price leakage."""
    return [
        c for c in frame.columns
        if c not in (ID_COLUMN, DATE_COLUMN) and not _is_price_derived_feature(c)
    ]


def _is_price_derived_feature(column: str) -> bool:
    """Keep targets and price-derived measurements out of unsupervised features."""
    name = "_".join(str(column).strip().lower().replace("/", " ").split())
    tokens = set(name.split("_"))
    return "price" in name or bool(tokens & {
        "amount", "cost", "rent", "sale", "selling", "valuation", "valued", "value",
    })


def _feature_frame(frame: pd.DataFrame, selected_features: Iterable[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    selected = list(dict.fromkeys(selected_features))
    price_columns = [c for c in selected if _is_price_derived_feature(c)]
    if price_columns:
        raise ValueError(f"price-derived columns are not allowed as clustering features: {price_columns}")
    invalid = [
        c for c in selected
        if c not in frame.columns or c == ID_COLUMN
    ]
    if invalid:
        raise ValueError(f"Invalid clustering feature(s): {invalid}")
    if not selected:
        raise ValueError("Select at least one clustering feature.")
    features = frame[selected].copy()
    if DATE_COLUMN in selected:
        dates = pd.to_datetime(features.pop(DATE_COLUMN), errors="coerce", utc=True)
        features["listing_year"] = dates.dt.year.astype(float)
        features["listing_month"] = dates.dt.month.astype(float)
    numeric, categorical = [], []
    for column in features.columns:
        if column in COORDINATE_FEATURES or column in BINARY_FEATURES or pd.api.types.is_numeric_dtype(features[column]):
            numeric.append(column)
            features[column] = pd.to_numeric(features[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        else:
            categorical.append(column)
            features[column] = features[column].astype("string").str.strip().replace("", pd.NA).fillna("Unknown")
    return features, numeric, categorical


def prepare_features(frame: pd.DataFrame, selected_features: Iterable[str]) -> tuple[np.ndarray, list[str]]:
    """Fit imputers, encoding, and scaling on this run's data and return a dense matrix."""
    features, numeric, categorical = _feature_frame(frame, selected_features)
    if not numeric and not categorical:
        raise ValueError("Selected features do not contain usable values.")
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    matrix = preprocessor.fit_transform(features)
    names = numeric + (list(preprocessor.named_transformers_["categorical"].named_steps["onehot"].get_feature_names_out(categorical)) if categorical else [])
    # Latitude/longitude contribute at one fifth the scale of other measurements.
    for index, name in enumerate(names):
        if name in COORDINATE_FEATURES:
            matrix[:, index] *= 0.2
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0), names


def _scores(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    ignore_noise: bool,
    silhouette_indices: np.ndarray | None = None,
) -> tuple[float, float, float]:
    mask = labels != -1 if ignore_noise else np.ones(len(labels), dtype=bool)
    x, y = matrix[mask], labels[mask]
    clusters = np.unique(y)
    if len(x) < 3 or len(clusters) < 2 or len(clusters) >= len(x):
        return np.nan, np.nan, np.nan
    if silhouette_indices is None:
        silhouette_x, silhouette_y = x, y
    else:
        selected = np.asarray(silhouette_indices, dtype=int)
        if ignore_noise:
            selected = selected[labels[selected] != -1]
        silhouette_x, silhouette_y = matrix[selected], labels[selected]
        if len(silhouette_y) < 3 or len(np.unique(silhouette_y)) < 2 or len(np.unique(silhouette_y)) >= len(silhouette_y):
            silhouette_x, silhouette_y = np.empty((0, matrix.shape[1])), np.empty((0,), dtype=int)
    results = []
    for metric_index, metric in enumerate((silhouette_score, davies_bouldin_score, calinski_harabasz_score)):
        try:
            metric_x, metric_y = (silhouette_x, silhouette_y) if metric_index == 0 else (x, y)
            if len(metric_y) < 3 or len(np.unique(metric_y)) < 2 or len(np.unique(metric_y)) >= len(metric_y):
                results.append(np.nan)
            else:
                results.append(float(metric(metric_x, metric_y)))
        except (ValueError, FloatingPointError):
            results.append(np.nan)
    return tuple(results)


def _make_cluster_estimator(algorithm: str, parameters: dict[str, object], random_seed: int):
    """Create one of the project's supported estimators from a shared config shape."""
    if algorithm == "K-Means":
        return KMeans(n_clusters=int(parameters["n_clusters"]), random_state=random_seed, n_init=10)
    if algorithm == "Agglomerative":
        return AgglomerativeClustering(n_clusters=int(parameters["n_clusters"]))
    if algorithm == "DBSCAN":
        return DBSCAN(eps=float(parameters["eps"]), min_samples=int(parameters["min_samples"]))
    if algorithm == "Gaussian Mixture":
        return GaussianMixture(n_components=int(parameters["n_clusters"]), random_state=random_seed, n_init=1)
    raise ValueError(f"Unsupported clustering algorithm: {algorithm!r}")


def _fit_labels(estimator, matrix: np.ndarray, algorithm: str) -> np.ndarray:
    if algorithm == "Gaussian Mixture":
        estimator.fit(matrix)
        return estimator.predict(matrix)
    return estimator.fit_predict(matrix)


def build_experiment_configurations(
    algorithms: Iterable[str],
    k_values: Iterable[int],
    dbscan_eps_values: Iterable[float],
    dbscan_min_samples_values: Iterable[int],
    *,
    max_configurations: int = MAX_EXPERIMENT_CONFIGURATIONS,
) -> list[dict[str, object]]:
    """Build a deterministic grid and reject it if it exceeds the explicit search limit."""
    if not 1 <= max_configurations <= MAX_EXPERIMENT_CONFIGURATIONS:
        raise ValueError(f"Maximum configurations must be between 1 and {MAX_EXPERIMENT_CONFIGURATIONS}.")
    selected = list(dict.fromkeys(algorithms))
    unknown = [algorithm for algorithm in selected if algorithm not in CLUSTERING_ALGORITHMS]
    if unknown:
        raise ValueError(f"Unsupported algorithm(s): {unknown}")
    if not selected:
        raise ValueError("Select at least one clustering algorithm.")
    ks = list(dict.fromkeys(int(k) for k in k_values))
    eps_values = list(dict.fromkeys(float(eps) for eps in dbscan_eps_values))
    min_samples_values = list(dict.fromkeys(int(value) for value in dbscan_min_samples_values))
    configurations: list[dict[str, object]] = []
    for algorithm in selected:
        if algorithm == "DBSCAN":
            if not eps_values or not min_samples_values:
                raise ValueError("DBSCAN requires at least one eps and min_samples value.")
            configurations.extend(
                {"Algorithm": algorithm, "Parameters": {"eps": eps, "min_samples": minimum}}
                for eps in eps_values for minimum in min_samples_values
            )
        else:
            if not ks:
                raise ValueError(f"{algorithm} requires at least one cluster-count value.")
            configurations.extend(
                {"Algorithm": algorithm, "Parameters": {"n_clusters": k}}
                for k in ks
            )
    if len(configurations) > max_configurations:
        raise ValueError(
            f"This search contains {len(configurations)} configurations, above the selected limit of "
            f"{max_configurations}. Narrow the ranges or increase the limit (up to {MAX_EXPERIMENT_CONFIGURATIONS})."
        )
    return configurations


def run_clustering_experiments(
    frame: pd.DataFrame,
    selected_features: Iterable[str],
    configurations: Iterable[dict[str, object]],
    *,
    max_rows: int | None = None,
    random_seed: int = 42,
    max_configurations: int = MAX_EXPERIMENT_CONFIGURATIONS,
    prepared_matrix: np.ndarray | None = None,
    retain_estimators: bool = False,
    preserve_rows: bool = False,
):
    """Evaluate configurations on the same sampled rows and one fitted feature matrix.

    Returns result rows, label arrays keyed by Configuration ID (successful fits only),
    and the exact evaluation rows in the positional order used by those arrays.
    Failed configurations remain in the results with an explanatory Error field.
    """
    configs = list(configurations)
    if not 1 <= max_configurations <= MAX_EXPERIMENT_CONFIGURATIONS:
        raise ValueError(f"Maximum configurations must be between 1 and {MAX_EXPERIMENT_CONFIGURATIONS}.")
    if not configs:
        raise ValueError("No clustering configurations were supplied.")
    if len(configs) > max_configurations:
        raise ValueError(f"Received {len(configs)} configurations, exceeding the limit of {max_configurations}.")
    clean = frame.copy().reset_index(drop=True) if preserve_rows else load_dataset(frame)
    if "property_id" not in clean:
        raise ValueError("Evaluation rows must contain property_id.")
    source_rows = len(clean)
    if max_rows is not None:
        if max_rows < 2:
            raise ValueError("The exploratory sample must contain at least 2 rows.")
        if max_rows < source_rows:
            clean = clean.sample(n=max_rows, random_state=random_seed).sort_index().reset_index(drop=True)
    matrix = prepare_features(clean, selected_features)[0] if prepared_matrix is None else np.asarray(prepared_matrix)
    if matrix.ndim != 2 or matrix.shape[0] != len(clean):
        raise ValueError("Prepared feature matrix must have one row per evaluation property.")
    labels_by_configuration: dict[int, np.ndarray] = {}
    estimators_by_configuration = {}
    if len(clean) > MAX_EXPERIMENT_SILHOUETTE_ROWS:
        silhouette_indices = np.sort(np.random.default_rng(random_seed).choice(
            len(clean), MAX_EXPERIMENT_SILHOUETTE_ROWS, replace=False,
        ))
    else:
        silhouette_indices = None
    result_rows: list[dict[str, object]] = []
    for config_id, config in enumerate(configs):
        config = config if isinstance(config, dict) else {}
        algorithm = str(config.get("Algorithm", ""))
        raw_parameters = config.get("Parameters", {})
        parameters = dict(raw_parameters) if isinstance(raw_parameters, dict) else {}
        base = {
            "Configuration ID": config_id,
            "Algorithm": algorithm,
            "Parameter Configuration": json.dumps(parameters, sort_keys=True, default=str),
            "Clusters Found": 0,
            "Noise Points": 0,
            "Noise (%)": 0.0,
            "Metric Population": "Non-noise points only" if algorithm == "DBSCAN" else "All points",
            "Silhouette Score": np.nan,
            "Davies-Bouldin Score": np.nan,
            "Calinski-Harabasz Score": np.nan,
            "Execution Time (s)": np.nan,
            "Rows Evaluated": len(clean),
            "Evaluation Set": "Sample" if len(clean) < source_rows else "Full dataset",
            "Status": "Failed",
            "Error": "",
        }
        start = perf_counter()
        try:
            estimator = _make_cluster_estimator(algorithm, parameters, random_seed)
            labels = np.asarray(_fit_labels(estimator, matrix, algorithm), dtype=int)
            silhouette, davies_bouldin, calinski_harabasz = _scores(
                matrix, labels, ignore_noise=algorithm == "DBSCAN",
                silhouette_indices=silhouette_indices,
            )
            elapsed = perf_counter() - start
            noise_points = int(np.sum(labels == -1))
            base.update({
                "Clusters Found": len(set(labels.tolist()) - {-1}),
                "Noise Points": noise_points,
                "Noise (%)": noise_points / len(labels) * 100,
                "Silhouette Score": silhouette,
                "Davies-Bouldin Score": davies_bouldin,
                "Calinski-Harabasz Score": calinski_harabasz,
                "Execution Time (s)": elapsed,
                "Status": "Success",
            })
            labels_by_configuration[config_id] = labels
            estimators_by_configuration[config_id] = estimator
        except Exception as exc:  # retain bad parameter sets for transparent review
            base["Execution Time (s)"] = perf_counter() - start
            base["Error"] = f"{type(exc).__name__}: {exc}"
        result_rows.append(base)
    evaluation_rows = clean.copy()
    try:
        coordinates = PCA(n_components=2, random_state=random_seed).fit_transform(matrix)
        evaluation_rows["PCA_1"] = coordinates[:, 0]
        evaluation_rows["PCA_2"] = coordinates[:, 1]
    except (ValueError, np.linalg.LinAlgError):
        evaluation_rows["PCA_1"] = np.nan
        evaluation_rows["PCA_2"] = np.nan
    base_result = (pd.DataFrame(result_rows), labels_by_configuration, evaluation_rows)
    return (*base_result, estimators_by_configuration) if retain_estimators else base_result


def rank_experiment_results(results: pd.DataFrame) -> pd.DataFrame:
    """Rank valid runs using several metrics plus runtime and DBSCAN noise trade-offs."""
    ranked = results.copy()
    metric_columns = ["Silhouette Score", "Davies-Bouldin Score", "Calinski-Harabasz Score"]
    for column in metric_columns:
        if column not in ranked:
            ranked[column] = np.nan
    ranked["Metric Rank"] = np.nan
    valid = ranked.get("Status", pd.Series("Success", index=ranked.index)).eq("Success")
    metric_ranks = []
    for column, ascending in ((metric_columns[0], False), (metric_columns[1], True), (metric_columns[2], False)):
        values = pd.to_numeric(ranked[column], errors="coerce")
        usable = valid & np.isfinite(values)
        rank = pd.Series(np.nan, index=ranked.index, dtype=float)
        if usable.any():
            rank.loc[usable] = values.loc[usable].rank(method="average", pct=True, ascending=ascending)
        metric_ranks.append(rank)
    metric_frame = pd.concat(metric_ranks, axis=1)
    counts = metric_frame.notna().sum(axis=1)
    ranked["Metric Rank"] = metric_frame.mean(axis=1)
    ranked["Metric Count"] = counts
    runtime = pd.to_numeric(ranked.get("Execution Time (s)", pd.Series(np.nan, index=ranked.index)), errors="coerce")
    runtime_rank = pd.Series(np.nan, index=ranked.index, dtype=float)
    runtime_usable = valid & np.isfinite(runtime)
    if runtime_usable.any():
        runtime_rank.loc[runtime_usable] = runtime.loc[runtime_usable].rank(method="average", pct=True, ascending=True)
    noise = pd.to_numeric(ranked.get("Noise (%)", pd.Series(0.0, index=ranked.index)), errors="coerce")
    noise_rank = pd.Series(np.nan, index=ranked.index, dtype=float)
    noise_usable = valid & np.isfinite(noise)
    if noise_usable.any():
        noise_rank.loc[noise_usable] = noise.loc[noise_usable].rank(method="average", pct=True, ascending=True)
    ranked["Balanced Suggestion Score"] = (
        0.70 * ranked["Metric Rank"] + 0.15 * runtime_rank + 0.15 * noise_rank
    )
    eligible = valid & counts.ge(2) & np.isfinite(ranked["Balanced Suggestion Score"])
    ranked["Recommendation"] = np.where(eligible, "Automated shortlist candidate", "Not ranked — insufficient valid metrics or failed run")
    ranked["Recommendation Rank"] = pd.Series(pd.NA, index=ranked.index, dtype="Int64")
    if eligible.any():
        ranks = ranked.loc[eligible, "Balanced Suggestion Score"].rank(method="min", ascending=True).astype(int)
        ranked.loc[eligible, "Recommendation Rank"] = ranks
    ranked["Recommendation Basis"] = np.where(
        eligible,
        "Combined rank across valid clustering metrics, runtime, and noise percentage; review cluster count and usefulness too.",
        "Not enough valid information to suggest this configuration.",
    )
    ranked = ranked.sort_values(
        ["Recommendation Rank", "Balanced Suggestion Score", "Configuration ID"],
        na_position="last", kind="stable",
    ).reset_index(drop=True)
    return ranked


def apply_experiment_configuration(
    evaluation_rows: pd.DataFrame,
    configuration_id: int,
    labels_by_configuration: dict[int, np.ndarray],
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply stored experiment labels to their exact evaluation rows for existing UI views."""
    match = results.loc[results["Configuration ID"] == configuration_id]
    if match.empty:
        raise ValueError(f"Configuration ID {configuration_id} was not found.")
    result = match.iloc[0]
    if result["Status"] != "Success" or configuration_id not in labels_by_configuration:
        raise ValueError("The selected configuration did not complete successfully, so it has no labels to apply.")
    labels = np.asarray(labels_by_configuration[configuration_id])
    if len(labels) != len(evaluation_rows):
        raise ValueError("Configuration labels do not match the saved evaluation rows.")
    algorithm = str(result["Algorithm"])
    labelled = evaluation_rows.copy()
    labelled[f"{algorithm}_cluster"] = labels.astype(int)
    return labelled, result.copy()


def run_clustering(
    frame: pd.DataFrame,
    selected_features: Iterable[str],
    *,
    n_clusters: int = 4,
    eps: float = 0.8,
    min_samples: int = 8,
    random_seed: int = 42,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all four algorithms on one prepared feature matrix.

    DBSCAN's quality scores use non-noise points only; its noise count is
    reported separately. The returned labelled frame preserves every source column.
    """
    clean = load_dataset(frame)
    source_rows = len(clean)
    if max_rows is not None:
        if max_rows < 2:
            raise ValueError("The exploratory sample must contain at least 2 rows.")
        if max_rows < source_rows:
            clean = clean.sample(n=max_rows, random_state=random_seed).sort_index().reset_index(drop=True)
    if n_clusters < 2 or n_clusters >= len(clean):
        raise ValueError("Number of clusters must be at least 2 and smaller than the row count.")
    matrix, _ = prepare_features(clean, selected_features)
    algorithms = [
        ("K-Means", {"n_clusters": n_clusters}),
        ("Agglomerative", {"n_clusters": n_clusters}),
        ("DBSCAN", {"eps": eps, "min_samples": min_samples}),
        ("Gaussian Mixture", {"n_clusters": n_clusters}),
    ]
    labelled = clean.copy()
    rows = []
    for name, parameters in algorithms:
        start = perf_counter()
        estimator = _make_cluster_estimator(name, parameters, random_seed)
        labels = _fit_labels(estimator, matrix, name)
        elapsed = perf_counter() - start
        labelled[f"{name}_cluster"] = labels.astype(int)
        silhouette, davies_bouldin, calinski_harabasz = _scores(
            matrix, labels, ignore_noise=name == "DBSCAN"
        )
        regular = set(labels) - {-1}
        rows.append({
            "Algorithm": name,
            "Silhouette Score": silhouette,
            "Silhouette Rows": len(clean) - int(np.sum(labels == -1)) if name == "DBSCAN" else len(clean),
            "Silhouette Method": "All eligible rows",
            "Davies-Bouldin Score": davies_bouldin,
            "Calinski-Harabasz Score": calinski_harabasz,
            "Execution Time (s)": elapsed,
            "Clusters Found": len(regular),
            "Noise Points": int(np.sum(labels == -1)),
            "Metric Population": "Non-noise points only" if name == "DBSCAN" else "All points",
            "Rows Used": len(clean),
            "Evaluation Set": "Sample" if len(clean) < source_rows else "Full dataset",
        })
    # PCA is only a display projection. The algorithms and metrics above use
    # the original prepared feature matrix, never these two visualization axes.
    try:
        coordinates = PCA(n_components=2, random_state=random_seed).fit_transform(matrix)
        labelled["PCA_1"] = coordinates[:, 0]
        labelled["PCA_2"] = coordinates[:, 1]
    except (ValueError, np.linalg.LinAlgError):
        labelled["PCA_1"] = np.nan
        labelled["PCA_2"] = np.nan
    return labelled, pd.DataFrame(rows)


def build_cluster_profile(labelled: pd.DataFrame, label_column: str, *, top_locations: int = 5) -> pd.DataFrame:
    """Summarize regular clusters; return DBSCAN noise count separately."""
    if label_column not in labelled:
        raise ValueError(f"Cluster label column {label_column!r} is missing.")
    data = labelled.loc[labelled[label_column] != -1].copy()

    def mode_text(values: pd.Series) -> str:
        modes = values.dropna().astype(str).mode()
        return ", ".join(modes.head(3).tolist()) if not modes.empty else "N/A"

    def locations_text(values: pd.Series) -> str:
        counts = values.dropna().astype(str).value_counts().head(top_locations)
        return ", ".join(f"{name} ({count})" for name, count in counts.items()) if not counts.empty else "N/A"

    profile = pd.DataFrame({"Cluster": [], "Properties": []})
    if data.empty:
        profile.attrs["noise_points"] = int((labelled[label_column] == -1).sum())
        return profile
    grouped = data.groupby(label_column, sort=True, dropna=False)
    rows = []
    for cluster, group in grouped:
        row = {
            "Cluster": int(cluster),
            "Properties": len(group),
            "Share of properties (%)": len(group) / len(labelled) * 100,
        }
        if "area_sqft" in group:
            area = pd.to_numeric(group["area_sqft"], errors="coerce")
            row["Mean area (sq ft)"] = area.mean()
            row["Median area (sq ft)"] = area.median()
        if "BHK" in group:
            row["Most common BHK"] = mode_text(group["BHK"])
        if "property_age_years" in group:
            row["Mean property age (years)"] = pd.to_numeric(group["property_age_years"], errors="coerce").mean()
        for column, label in (("property_type", "Most common property type"), ("furnishing_status", "Most common furnishing")):
            if column in group:
                row[label] = mode_text(group[column])
        if "location" in group:
            row["Most common locations"] = locations_text(group["location"])
        for amenity in BINARY_FEATURES:
            if amenity in group:
                values = pd.to_numeric(group[amenity], errors="coerce")
                row[f"{amenity} available (%)"] = values.mean() * 100
        rows.append(row)
    profile = pd.DataFrame(rows)
    profile.attrs["noise_points"] = int((labelled[label_column] == -1).sum())
    return profile


def interpret_clusters(labelled: pd.DataFrame, label_column: str) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Compare each regular cluster with the overall evaluated data and describe measured differences."""
    profile = build_cluster_profile(labelled, label_column)
    descriptions: list[dict[str, object]] = []
    if profile.empty:
        return profile, descriptions
    overall_area = pd.to_numeric(labelled["area_sqft"], errors="coerce").mean() if "area_sqft" in labelled else np.nan
    overall_bhk = pd.to_numeric(labelled["BHK"], errors="coerce").mean() if "BHK" in labelled else np.nan
    overall_age = pd.to_numeric(labelled["property_age_years"], errors="coerce").mean() if "property_age_years" in labelled else np.nan
    overall_modes = {
        column: (labelled[column].dropna().astype(str).mode().iloc[0]
                 if column in labelled and not labelled[column].dropna().empty else None)
        for column in ("property_type", "furnishing_status")
    }
    overall_amenities = {
        column: pd.to_numeric(labelled[column], errors="coerce").mean() * 100
        for column in BINARY_FEATURES if column in labelled
    }
    for _, row in profile.iterrows():
        cluster = int(row["Cluster"])
        group = labelled.loc[labelled[label_column] == cluster]
        statements: list[str] = []
        area_mean = row.get("Mean area (sq ft)", np.nan)
        if pd.notna(area_mean) and pd.notna(overall_area) and overall_area > 0:
            difference = (area_mean / overall_area - 1) * 100
            if difference <= -10:
                statements.append(f"Mean area is {abs(difference):.0f}% below the overall mean")
            elif difference >= 10:
                statements.append(f"Mean area is {difference:.0f}% above the overall mean")
        bhk_mean = pd.to_numeric(group["BHK"], errors="coerce").mean() if "BHK" in group else np.nan
        if pd.notna(bhk_mean) and pd.notna(overall_bhk):
            difference = bhk_mean - overall_bhk
            if difference <= -0.5:
                statements.append(f"Average BHK is lower than the overall average ({bhk_mean:.1f} vs {overall_bhk:.1f})")
            elif difference >= 0.5:
                statements.append(f"Average BHK is higher than the overall average ({bhk_mean:.1f} vs {overall_bhk:.1f})")
        age_mean = row.get("Mean property age (years)", np.nan)
        if pd.notna(age_mean) and pd.notna(overall_age):
            difference = age_mean - overall_age
            if difference <= -2:
                statements.append(f"Mean property age is {abs(difference):.1f} years below the overall mean")
            elif difference >= 2:
                statements.append(f"Mean property age is {difference:.1f} years above the overall mean")
        for column, profile_key, label in (
            ("property_type", "Most common property type", "property type"),
            ("furnishing_status", "Most common furnishing", "furnishing status"),
        ):
            common = row.get(profile_key)
            baseline = overall_modes[column]
            if common and baseline and common != baseline:
                statements.append(f"Most common {label} is {common}; overall most common is {baseline}")
        if "location" in group and not group["location"].dropna().empty:
            local_modes = group["location"].dropna().astype(str).value_counts()
            location = local_modes.index[0]
            cluster_share = local_modes.iloc[0] / len(group) * 100
            overall_share = (labelled["location"].astype(str) == location).mean() * 100
            if cluster_share - overall_share >= 10:
                statements.append(f"{location} is more concentrated here ({cluster_share:.0f}% vs {overall_share:.0f}% overall)")
        for amenity, baseline in overall_amenities.items():
            cluster_pct = row.get(f"{amenity} available (%)", np.nan)
            if pd.notna(cluster_pct) and pd.notna(baseline):
                difference = cluster_pct - baseline
                if abs(difference) >= 15:
                    direction = "higher" if difference > 0 else "lower"
                    statements.append(f"{amenity} availability is {direction} ({cluster_pct:.0f}% vs {baseline:.0f}% overall)")
        descriptions.append({
            "Cluster": cluster,
            "Properties": int(row["Properties"]),
            "Share of properties (%)": float(row["Share of properties (%)"]),
            "Measured differences": statements,
            "Interpretation": "; ".join(statements) if statements else "No large difference from overall averages was detected in the available columns.",
        })
    return profile, descriptions


def filter_cluster_properties(
    labelled: pd.DataFrame,
    label_column: str,
    cluster: int,
    *,
    location: str = "",
    bhk: Iterable[object] | None = None,
    area_min: float | None = None,
    area_max: float | None = None,
    property_types: Iterable[object] | None = None,
) -> pd.DataFrame:
    """Filter already-labelled rows without changing or refitting their labels."""
    if label_column not in labelled:
        raise ValueError(f"Cluster label column {label_column!r} is missing.")
    mask = labelled[label_column] == cluster
    if location and "location" in labelled:
        mask &= labelled["location"].astype("string").str.contains(location, case=False, na=False, regex=False)
    if bhk and "BHK" in labelled:
        mask &= labelled["BHK"].isin(list(bhk))
    if "area_sqft" in labelled:
        area = pd.to_numeric(labelled["area_sqft"], errors="coerce")
        if area_min is not None:
            mask &= area.ge(area_min)
        if area_max is not None:
            mask &= area.le(area_max)
    if property_types and "property_type" in labelled:
        mask &= labelled["property_type"].isin(list(property_types))
    return labelled.loc[mask].copy()


def find_similar_properties(
    frame: pd.DataFrame,
    property_id: object,
    selected_features: Iterable[str],
    *,
    n_neighbors: int = 5,
) -> tuple[pd.DataFrame, list[str]]:
    """Find similar listings using the same scaled/encoded feature preparation as clustering."""
    if n_neighbors < 1:
        raise ValueError("Choose at least one similar property.")
    clean = load_dataset(frame)
    matches = np.flatnonzero(clean[ID_COLUMN].astype(str).to_numpy() == str(property_id))
    if len(matches) != 1:
        if len(matches) == 0:
            raise ValueError(f"Property ID {property_id!r} was not found in the dataset.")
        raise ValueError(f"Property ID {property_id!r} is not unique in the dataset.")
    matrix, feature_names = prepare_features(clean, selected_features)
    count = min(n_neighbors + 1, len(clean))
    neighbors = NearestNeighbors(n_neighbors=count, metric="euclidean")
    neighbors.fit(matrix)
    distances, indices = neighbors.kneighbors(matrix[matches[0]:matches[0] + 1])
    results = [(int(index), float(distance)) for index, distance in zip(indices[0], distances[0]) if int(index) != int(matches[0])]
    results = results[:n_neighbors]
    similar = clean.iloc[[index for index, _ in results]].copy()
    similar.insert(0, "Similarity rank", range(1, len(similar) + 1))
    similar["Feature distance"] = [distance for _, distance in results]
    return similar, feature_names
