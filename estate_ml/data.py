"""Shared, leakage-aware data preparation for EstateML experiments."""
from __future__ import annotations

import pandas as pd


def _find_column(df: pd.DataFrame, names: list[str]) -> str | None:
    lookup = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    for name in names:
        if name in lookup:
            return lookup[name]
    return None


def _parse_area(value):
    """Convert numeric areas and ranges such as '1000-1200' to numeric sqft."""
    if pd.isna(value):
        return None
    text = str(value).strip().replace(",", "")
    if "-" in text:
        parts = text.split("-")
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except (ValueError, IndexError):
            return None
    try:
        return float(text)
    except ValueError:
        return None


def load_training_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load the repository's Bengaluru housing CSV into safe numeric features.

    Deliberately excludes price_per_sqft because it is derived from the target
    price and would leak target information into training.
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    area_col = _find_column(df, ["total_sqft", "total_area", "area"])
    price_col = _find_column(df, ["price", "price_lakhs", "target"])
    bath_col = _find_column(df, ["bath", "bathrooms", "baths"])
    balcony_col = _find_column(df, ["balcony", "balconies"])
    size_col = _find_column(df, ["size", "bhk", "bedrooms"])

    if area_col is None or price_col is None:
        raise ValueError(
            "CSV must contain an area column (e.g. total_sqft) and a target "
            "price column (e.g. price). Found: " + ", ".join(df.columns)
        )

    out = pd.DataFrame(index=df.index)
    out["total_sqft"] = df[area_col].map(_parse_area)
    out["price"] = pd.to_numeric(
        df[price_col].astype(str).str.replace(",", "", regex=False).str.replace("₹", "", regex=False),
        errors="coerce",
    )

    if size_col is not None:
        if str(size_col).lower() == "bhk":
            out["bhk"] = pd.to_numeric(df[size_col], errors="coerce")
        else:
            out["bhk"] = pd.to_numeric(
                df[size_col].astype(str).str.extract(r"(\d+(?:\.\d+)?)", expand=False),
                errors="coerce",
            )
    else:
        out["bhk"] = float("nan")

    out["bath"] = (
        pd.to_numeric(df[bath_col], errors="coerce")
        if bath_col is not None else float("nan")
    )
    out["balcony"] = (
        pd.to_numeric(df[balcony_col], errors="coerce")
        if balcony_col is not None else float("nan")
    )

    # Basic plausibility filters; avoid fabricating values for the target.
    out = out.replace([float("inf"), float("-inf")], pd.NA)
    out = out.dropna(subset=["total_sqft", "price"])
    out = out[(out["total_sqft"] > 100) & (out["price"] > 0)]
    if out.empty:
        raise ValueError("No usable rows remain after parsing area and price.")

    # Fit imputation values on training split in a more advanced pipeline if
    # missingness is material. Here, median imputation is applied within each
    # split by the caller's sklearn Pipeline.
    features = out[["total_sqft", "bhk", "bath", "balcony"]].copy()
    target = out["price"].astype(float).copy()
    return features, target
