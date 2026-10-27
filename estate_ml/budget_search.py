"""Bounded, model-based search over hypothetical property configurations."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from itertools import product
import math
from typing import Any, Callable

from estate_ml.property_model import (
    CATEGORICAL_FEATURES, FEATURE_SCHEMA, predict_property, predict_property_batch,
)

AMENITIES = ("lift", "security", "gym", "swimming_pool", "power_backup", "gated_community")
AREA_MIN, AREA_MAX = 400, 4719  # documented curated-listing support range
MAX_CANDIDATES = 3000


@dataclass(frozen=True)
class SearchRequest:
    budget_lakhs: float
    localities: tuple[str, ...]
    bhk: int | None = None
    min_area_sqft: int = AREA_MIN
    min_bathrooms: int = 1
    balcony_required: bool = False
    property_type: str = "Unknown"
    furnishing_status: str = "Unknown"
    condition: str = "Unknown"
    amenities: tuple[str, ...] = ()
    priority: str = "overall fit"
    include_alternatives: bool = False
    result_limit: int = 12


@dataclass
class SearchResult:
    candidates: list[dict[str, Any]]
    evaluated: int
    supported_localities: tuple[str, ...]


def get_supported_categories(model: Any, metadata: dict[str, Any]) -> dict[str, list[str]]:
    """Use saved category metadata, falling back to fitted encoder categories."""
    raw = metadata.get("known_categories_from_random_training_partition") or {}
    categories = {name: list(values) for name, values in raw.items() if values}
    if all(name in categories for name in CATEGORICAL_FEATURES):
        return categories
    try:
        encoder = model.named_steps["preprocess"].named_transformers_["categorical"].named_steps["onehot"]
        names = list(encoder.feature_names_in_)
        for name, values in zip(names, encoder.categories_):
            if name not in categories:
                categories[name] = sorted(str(value) for value in values if str(value) != "nan")
    except (AttributeError, KeyError, TypeError):
        pass
    return categories


def budget_to_lakhs(amount: float, unit: str) -> float:
    value = float(amount)
    if value <= 0:
        raise ValueError("Budget must be greater than zero.")
    if unit == "INR":
        return value / 100_000
    if unit == "lakhs":
        return value
    raise ValueError("Budget unit must be INR or lakhs.")


def _validate_request(request: SearchRequest, categories: dict[str, list[str]]) -> tuple[str, ...]:
    if not math.isfinite(float(request.budget_lakhs)) or request.budget_lakhs <= 0:
        raise ValueError("Budget must be greater than zero and finite.")
    if not AREA_MIN <= request.min_area_sqft <= AREA_MAX:
        raise ValueError(f"Minimum area must be between {AREA_MIN} and {AREA_MAX} sq ft.")
    if request.bhk is not None and request.bhk not in range(1, 6):
        raise ValueError("BHK must be between 1 and 5, or flexible.")
    if request.min_bathrooms not in range(1, 7):
        raise ValueError("Minimum bathrooms must be between 1 and 6.")
    if request.bhk is not None and request.min_bathrooms > min(6, request.bhk + 1):
        max_bathrooms = min(6, request.bhk + 1)
        raise ValueError(f"A {request.bhk} BHK configuration can have at most {max_bathrooms} bathrooms in this search.")
    if not isinstance(request.result_limit, int) or not 1 <= request.result_limit <= 50:
        raise ValueError("Result limit must be between 1 and 50.")
    valid_locs = set(categories.get("location", []))
    if not request.localities or any(loc not in valid_locs for loc in request.localities):
        raise ValueError("Choose one or more supported localities.")
    for field_name in ("property_type", "condition", "furnishing_status"):
        value = getattr(request, field_name)
        if value != "Unknown" and value not in set(categories.get(field_name, [])):
            raise ValueError(f"Unsupported {field_name.replace('_', ' ')}: {value}.")
    if any(a not in AMENITIES for a in request.amenities):
        raise ValueError("An unsupported amenity was selected.")
    return tuple(dict.fromkeys((*request.localities, *(categories.get("location", []) if request.include_alternatives else []))))


def search_properties(
    model: Any,
    metadata: dict[str, Any],
    request: SearchRequest,
    *,
    predictor: Callable[[Any, dict[str, Any]], float] = predict_property,
    batch_predictor: Callable[[Any, list[dict[str, Any]]], list[float]] = predict_property_batch,
    max_candidates: int = MAX_CANDIDATES,
) -> SearchResult:
    """Enumerate a deterministic bounded candidate grid and rank feasible rows."""
    if not isinstance(max_candidates, int) or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer.")
    candidate_limit = min(max_candidates, MAX_CANDIDATES)
    categories = get_supported_categories(model, metadata)
    locations = _validate_request(request, categories)
    bhks = [request.bhk] if request.bhk is not None else [1, 2, 3, 4, 5]
    areas = sorted({request.min_area_sqft, *range(((request.min_area_sqft + 99)//100)*100, AREA_MAX + 1, 100), AREA_MAX})
    bathrooms = range(max(1, request.min_bathrooms), 7)
    balconies = (1, 2) if request.balcony_required else (0, 1, 2)
    # Only explicitly preferred amenities are varied; unspecified values remain missing.
    amenity_options = list(product((0, 1), repeat=len(request.amenities))) if request.amenities else [()]
    def category_choices(name: str, preferred: str) -> list[str]:
        if preferred == "Unknown":
            return ["Unknown"]
        return list(dict.fromkeys([preferred, *categories.get(name, []), "Unknown"]))

    property_types = category_choices("property_type", request.property_type)
    conditions = category_choices("condition", request.condition)
    furnishing_options = category_choices("furnishing_status", request.furnishing_status)
    dimensions = [locations, bhks, areas, list(bathrooms), list(balconies), amenity_options,
                  property_types, conditions, furnishing_options]
    space_size = math.prod(len(values) for values in dimensions)
    # Walk a reproducible affine permutation of candidate indexes. Sampling
    # every Nth row from itertools.product can alias with inner dimensions and
    # accidentally omit a category or amenity variant.
    step = max(1, space_size // candidate_limit + 1)
    while math.gcd(step, space_size) != 1:
        step += 1
    offset = space_size // 3
    sample_count = min(space_size, candidate_limit)
    candidates: list[dict[str, Any]] = []
    candidate_inputs: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    evaluated = 0
    for sample_index in range(sample_count):
        encoded_index = (offset + sample_index * step) % space_size
        selected = []
        for values_for_dimension in reversed(dimensions):
            encoded_index, remainder = divmod(encoded_index, len(values_for_dimension))
            selected.append(values_for_dimension[remainder])
        (location, bhk, area, bath, balcony, amenity_values,
         property_type, condition, furnishing) = reversed(selected)
        values: dict[str, Any] = {feature: None for feature in FEATURE_SCHEMA}
        values.update({
            "location": location, "area_sqft": area, "BHK": bhk,
            "bathrooms": bath, "balconies": balcony,
            "listing_date": date.today().isoformat(),
            "condition": condition, "property_type": property_type,
            "furnishing_status": furnishing,
        })
        for amenity, value in zip(request.amenities, amenity_values):
            values[amenity] = value
        signature = tuple(values.get(key) for key in FEATURE_SCHEMA)
        if signature in seen:
            continue
        seen.add(signature)
        # The model dataset supports at most one more bathroom than bedrooms,
        # capped at its documented six-bathroom upper range.
        if bath > min(6, bhk + 1):
            continue
        if area < request.min_area_sqft or (request.balcony_required and balcony == 0):
            continue

        candidate_inputs.append(values)

    if predictor is predict_property:
        predictions = batch_predictor(model, candidate_inputs)
        if len(predictions) != len(candidate_inputs):
            raise ValueError("Batch predictor returned a different number of prices than candidates.")
    else:
        predictions = []
        for values in candidate_inputs:
            try:
                predictions.append(float(predictor(model, values)))
            except (TypeError, ValueError):
                predictions.append(float("nan"))
    evaluated = len(candidate_inputs)

    ranges = metadata.get("training_feature_ranges") or {}
    if not isinstance(ranges, dict):
        ranges = {}
    for values, prediction in zip(candidate_inputs, predictions):
        price = float(prediction)
        if not math.isfinite(price) or price <= 0 or price > request.budget_lakhs:
            continue
        location, bhk, area, bath = values["location"], values["BHK"], values["area_sqft"], values["bathrooms"]
        amenities_met = sum(values.get(a) == 1 for a in request.amenities)
        preference_matches = sum((preferred == "Unknown" or values[feature] == preferred)
            for feature, preferred in (("property_type", request.property_type),
                                       ("condition", request.condition),
                                       ("furnishing_status", request.furnishing_status)))
        if request.priority == "maximize area":
            score = (area, preference_matches, amenities_met, -abs(price-request.budget_lakhs))
        elif request.priority == "bedroom count":
            score = (bhk, preference_matches, area, amenities_met, -price)
        elif request.priority == "amenities":
            score = (amenities_met, preference_matches, area, -price)
        else:
            score = (location in request.localities, preference_matches, amenities_met, area, -abs(price-request.budget_lakhs))
        out_of_range = []
        for feature, bounds in ranges.items():
            value = values.get(feature)
            if value is None or not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                continue
            if not float(bounds[0]) <= float(value) <= float(bounds[1]):
                out_of_range.append(feature)
        support_note = (
            "Outside recorded training range: " + ", ".join(out_of_range)
            if out_of_range else "Feature-combination support cannot be verified from bundled data."
        )
        candidates.append({**values, "predicted_price_lakhs": price, "budget_remaining_lakhs": request.budget_lakhs-price,
                           "budget_utilization_pct": price/request.budget_lakhs*100, "amenities_met": amenities_met,
                           "hard_constraints_met": True, "out_of_training_range": bool(out_of_range),
                           "support_note": support_note, "_score": score})
    candidates.sort(key=lambda row: (row["_score"], row["predicted_price_lakhs"], row["location"]), reverse=True)
    # Diversity: one result per locality/BHK/area group before returning near variants.
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in candidates:
        amenity_signature = tuple(row.get(name) for name in request.amenities)
        soft_signature = (amenity_signature, row["property_type"], row["condition"], row["furnishing_status"])
        grouped.setdefault(soft_signature, []).append(row)
    diverse: list[dict[str, Any]] = []
    signatures = list(grouped)
    offsets = {signature: 0 for signature in signatures}
    seen_result_keys: set[tuple[Any, ...]] = set()
    # Rotate through soft-preference variants so a preference influences rank
    # without hiding all nonpreferred options from the recommendation set.
    while len(diverse) < request.result_limit:
        added = False
        for signature in signatures:
            rows = grouped[signature]
            while offsets[signature] < len(rows):
                row = rows[offsets[signature]]
                offsets[signature] += 1
                row_key = (row["location"], row["BHK"], int(row["area_sqft"] // 300),
                           *signature)
                if row_key not in seen_result_keys:
                    diverse.append(row)
                    seen_result_keys.add(row_key)
                    added = True
                    break
            if len(diverse) >= request.result_limit:
                break
        if not added:
            break
    for row in diverse:
        row.pop("_score", None)
    return SearchResult(diverse, evaluated, locations)


def tradeoff(baseline: dict[str, Any], alternative: dict[str, Any], budget_lakhs: float) -> dict[str, Any]:
    base_price = float(baseline["predicted_price_lakhs"])
    alt_price = float(alternative["predicted_price_lakhs"])
    dims = [key for key in ("location", "BHK", "area_sqft", "bathrooms", "balconies",
                            "property_type", "furnishing_status", "condition", *AMENITIES)
            if baseline.get(key) != alternative.get(key)]
    return {"baseline_price_lakhs": base_price, "alternative_price_lakhs": alt_price,
            "difference_lakhs": alt_price-base_price,
            "difference_pct": ((alt_price-base_price)/base_price*100 if base_price else None),
            "dimensions": dims, "baseline_remaining_lakhs": budget_lakhs-base_price,
            "alternative_remaining_lakhs": budget_lakhs-alt_price}


def preference_summary(candidate: dict[str, Any], request: SearchRequest) -> tuple[list[str], list[str]]:
    """Describe user-selected soft preferences met and missed by one result."""
    met: list[str] = []
    missed: list[str] = []
    for feature, preferred, label in (
        ("property_type", request.property_type, "Property type"),
        ("condition", request.condition, "Condition"),
        ("furnishing_status", request.furnishing_status, "Furnishing"),
    ):
        if preferred != "Unknown":
            (met if candidate.get(feature) == preferred else missed).append(f"{label}: {preferred}")
    for amenity in request.amenities:
        label = amenity.replace("_", " ").title()
        (met if candidate.get(amenity) == 1 else missed).append(f"{label}")
    (met if candidate.get("location") in request.localities else missed).append("Preferred locality")
    return met, missed


def tradeoff_display(baseline: dict[str, Any], alternative: dict[str, Any], budget_lakhs: float) -> dict[str, Any]:
    """Create consistent user-facing comparison values without implying equivalence."""
    result = tradeoff(baseline, alternative, budget_lakhs)

    def number_delta(key: str, unit: str) -> str:
        left, right = baseline.get(key), alternative.get(key)
        if left is None or right is None:
            return "Unavailable"
        difference = float(right) - float(left)
        if difference == 0:
            return "No change"
        sign = "+" if difference > 0 else "−"
        magnitude = abs(difference)
        value = f"{magnitude:g}" if magnitude.is_integer() else f"{magnitude:.1f}"
        return f"{sign}{value} {unit}"

    def value_or_unknown(candidate: dict[str, Any], key: str) -> str:
        value = candidate.get(key)
        return "Unspecified" if value in (None, "", "Unknown") else str(value)

    result["price_difference_display"] = "No change" if result["difference_lakhs"] == 0 else f"{result['difference_lakhs']:+.2f} lakh"
    result["price_difference_pct_display"] = (
        "Unavailable" if result["difference_pct"] is None
        else ("No change" if result["difference_pct"] == 0 else f"{result['difference_pct']:+.1f}%")
    )
    result["area_difference_display"] = number_delta("area_sqft", "sq ft")
    result["bhk_difference_display"] = number_delta("BHK", "BHK")
    result["budget_remaining_display"] = (
        f"₹{result['baseline_remaining_lakhs']:,.2f} L → ₹{result['alternative_remaining_lakhs']:,.2f} L"
    )
    result["category_differences"] = {
        label: f"{value_or_unknown(baseline, key)} → {value_or_unknown(alternative, key)}"
        for key, label in (("location", "Locality"), ("property_type", "Property type"),
                           ("furnishing_status", "Furnishing"), ("condition", "Condition"))
        if baseline.get(key) != alternative.get(key)
    }
    result["features_gained"] = [name.replace("_", " ").title() for name in AMENITIES
        if baseline.get(name) == 0 and alternative.get(name) == 1]
    result["features_given_up"] = [name.replace("_", " ").title() for name in AMENITIES
        if baseline.get(name) == 1 and alternative.get(name) == 0]
    return result
