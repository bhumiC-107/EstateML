# Bengaluru property listing estimator

This model uses the user's curated listing data and powers the app's estimate and Property Compare pages. The source CSV is not bundled in this repository; the saved model artifact is used at runtime. Listing prices may differ from completed sales.

## Reproducible training

```powershell
python -m estate_ml.train_model --data "C:\path\to\your-curated-listings.csv"
```

The command saves `artifacts/model/property_price_model.joblib` and `artifacts/model/model_metadata.json`. The target `price_lakhs` is lakhs of INR (1 lakh = ₹100,000); `area_sqft` is square feet. `property_id` and target-derived inputs are excluded. The raw schema is unchanged, while the pipeline derives `bhk_x_balconies = BHK * balconies` for an interaction in the Linear Regression model. Metadata stores and inference validates the ordered raw schema:

`location`, `latitude`, `longitude`, `area_sqft`, `BHK`, `bathrooms`, `balconies`, `listing_date`, `property_age_years`, `condition`, `floor_number`, `total_floors`, `property_type`, `furnishing_status`, `parking_spaces`, `lift`, `security`, `gym`, `swimming_pool`, `power_backup`, `gated_community`.

Only Linear Regression, Decision Tree Regressor, Random Forest Regressor, and K-Nearest Neighbors Regressor are compared. The primary seed-42 split is made before fitting preprocessing: 80% training and 20% untouched test. Five-fold `GridSearchCV` scores by negative RMSE and receives only the primary training partition. Candidate pipelines fit imputers, category encoders, and scalers inside each fold. Numeric features are scaled for Linear Regression and KNN; tree pipelines do not scale. Each grid contains the standard estimator defaults, allowing a CV comparison to its untuned baseline. The primary winner is chosen only by lowest mean training-CV RMSE. The chronological supplement uses a separate earlier-date training subset and TimeSeriesSplit grid searches; it cannot affect the primary winner.

## Data checks

The source has 5,000 rows, no missing cells, exact duplicate rows, duplicate IDs, or duplicate feature signatures. All targets are finite and positive (₹17.23–₹1,397.48 lakh); area is 400–4,719 sq ft; BHK is 1–5; bathrooms 1–6; balconies 0–4; and there are no invalid floor relationships. No rows were excluded. There are 337 listings with fewer bathrooms than bedrooms; this was flagged but retained because it is not inherently impossible. All amenity flags are binary. The Tukey 1.5×IQR upper fence flags 242 prices above ₹396.72 lakh; those valid rows were retained because the fence alone does not show corruption.

## Primary evaluation on curated listings

The primary split is `train_test_split(test_size=0.2, random_state=42)`: 4,000 training rows and 1,000 test rows. The baseline uses each estimator's ordinary default hyperparameters, with deterministic random states for tree estimators. “Tuning improved” is based only on best mean CV RMSE versus default CV RMSE. Each tuned candidate is evaluated once on the untouched primary holdout after all primary choices. RMSE and MAE are lakhs of INR; percentages are the share of holdout predictions within the indicated relative error.

| Candidate | Best parameters | Default CV RMSE | Tuned CV RMSE | Tuning improved? | Holdout RMSE | MAE | R² | Within ±10% | Within ±20% |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| Linear Regression | `fit_intercept=True` | 51.47 | 51.47 | No (tie) | 47.36 | 34.45 | 0.807 | 29.7% | 55.2% |
| Decision Tree Regressor | `max_depth=10, min_samples_split=2, min_samples_leaf=5` | 74.82 | 62.04 | Yes | 57.67 | 40.13 | 0.713 | 26.6% | 50.1% |
| Random Forest Regressor | `n_estimators=200, max_depth=None, min_samples_split=5, min_samples_leaf=2` | 54.64 | 53.91 | Yes | 47.57 | 34.01 | 0.805 | 29.9% | 56.5% |
| K-Nearest Neighbors Regressor | `n_neighbors=15, weights=distance, p=1` | 76.48 | 71.09 | Yes | 66.41 | 47.81 | 0.620 | 22.1% | 41.3% |

Linear Regression is selected: its mean training-CV RMSE is 51.47 lakhs, the lowest of the four. Random Forest's primary holdout RMSE is slightly lower than Linear Regression's, but holdout scores were not used to choose the winner. The saved property pipeline is the selected GridSearchCV estimator refit on the primary 4,000-row training partition.

## Chronological supplementary evaluation on curated listings

Chronological training uses dates through 2026-01-04 (3,999 rows); testing uses dates from 2026-01-05 (1,001 rows). Each candidate's separate five-fold `TimeSeriesSplit` search and baseline comparison use only earlier-date training rows. The later-date holdout is evaluated after those supplementary choices; this search does not select or alter the primary model.

| Candidate | Best parameters | Default CV RMSE | Tuned CV RMSE | Tuning improved? | Holdout RMSE | MAE | R² | Within ±10% | Within ±20% |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| Linear Regression | `fit_intercept=False` | 64.61 | 51.05 | Yes | 50.83 | 37.92 | 0.760 | 28.4% | 51.8% |
| Decision Tree Regressor | `max_depth=10, min_samples_split=2, min_samples_leaf=5` | 86.36 | 67.11 | Yes | 65.18 | 42.76 | 0.605 | 26.3% | 50.0% |
| Random Forest Regressor | `n_estimators=200, max_depth=None, min_samples_split=5, min_samples_leaf=2` | 56.76 | 55.49 | Yes | 50.93 | 36.72 | 0.759 | 29.0% | 54.1% |
| K-Nearest Neighbors Regressor | `n_neighbors=15, weights=distance, p=1` | 80.45 | 73.12 | Yes | 65.85 | 48.52 | 0.596 | 22.2% | 42.3% |

The primary selected model's random holdout R² is 0.806; the separate chronological CV winner's holdout R² is 0.760. R² is a regression metric, not “accuracy.” These scores describe evaluation against this curated dataset; they do not guarantee performance on future listings or completed sale prices.

## Future value scenarios

The source listing dates span January 2023 through October 2026. For projection, the trainer estimates monthly median asking price per square foot using only the primary training partition. It fits log-price trends for the city and each locality with at least 18 observed months; locality trends are shrunk toward the city trend and bounded to annual rates from −10% to +15%. The conservative and optimistic scenarios are the fitted locality/city trend minus or plus two percentage points, bounded by the same limits. These are scenario ranges, not confidence intervals.

For each year, the model predicts the same property at an older building age while holding its listing date fixed, then multiplies by the scenario's market trend. If age is unknown, the model's training median is used. The age-price relationship is an association in the listings, not proof of causal depreciation; projections beyond the training age range are flagged in the UI. The “year to review selling” marks the first projected year when value growth falls below a user-entered alternative-return assumption. It is not a recommendation.

The Finance page's buy-versus-rent illustration uses the base forecast when opened from an estimate, an amortizing loan, renter investment of the down payment, and monthly investment or drawdown of the ownership-cost/rent difference. It excludes taxes, purchase and sale costs, insurance, rent deposits, and personal circumstances.

## Interpreting bedroom and balcony differences

The model does not impose a rule that more bedrooms must mean a higher estimate at the same total area. In the supplied data, the overall median asking price per square foot is about ₹11,655 for 2-BHK listings and ₹11,335 for 3-BHK listings. These are unadjusted group medians, not matched properties, and do not establish a causal bedroom effect. Because the dataset has few exact like-for-like listings, changing only BHK can produce a lower model estimate. The app does not show a separate BHK comparison. Balcony count is a relatively weak signal; the retained interaction lets its learned relationship vary by BHK without rewarding balconies-per-bedroom.
