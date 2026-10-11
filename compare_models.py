
import joblib
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from estate_ml.data import load_training_data


# 1. Load dataset and create the same test split
X, y = load_training_data("bengaluru_house_prices.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# 2. Load AutoML model
automl = joblib.load("artifacts/automl_model.joblib")
automl_predictions = automl.predict(X_test)


# 3. Evaluate AutoML
def evaluate_model(name, actual, predictions):
    rmse = mean_squared_error(actual, predictions) ** 0.5
    mae = mean_absolute_error(actual, predictions)
    r2 = r2_score(actual, predictions)

    print(f"\n{name}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")


evaluate_model("AUTOML MODEL", y_test, automl_predictions)


# 4. Load original model and scaler
original_model = joblib.load("final_model.joblib")
scaler = joblib.load("scaler.joblib")

# Re-create the seven features expected by the original model.
X_original = X_test.copy()

X_original["price_per_sqft"] = 0
X_original["sqft_per_bhk"] = (
    X_original["total_sqft"]
    / X_original["bhk"].replace(0, 1)
)
X_original["bath_per_bhk"] = (
    X_original["bath"]
    / X_original["bhk"].replace(0, 1)
)

# Preserve the scaler's expected feature order.
X_original = X_original.reindex(
    columns=scaler.feature_names_in_
)

X_scaled = pd.DataFrame(
    scaler.transform(X_original),
    columns=scaler.feature_names_in_,
    index=X_original.index,
)
original_predictions = original_model.predict(X_scaled)

evaluate_model("ORIGINAL RANDOM FOREST", y_test, original_predictions)

print("\nTest rows:", len(X_test))
print("AutoML features:", list(X.columns))
print("Original model features:", list(scaler.feature_names_in_))

print(
    "\nIMPORTANT: Both models use the same test rows, but this is not "
    "necessarily a fair performance comparison. The original model's "
    "training data, feature engineering, and target processing have "
    "not yet been verified."
)
