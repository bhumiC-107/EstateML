# EstateML

EstateML estimates Bengaluru property prices using user-curated live listings and research. The main estimate and Property Compare page use the same saved model. Finance tools do not use the price model.

> **About these estimates:** Listings reflect asking prices and research at collection time. They may differ from completed sale prices. Evaluation scores describe performance on this dataset; they do not guarantee future prices.

## Install and run

Use Python 3.12 from the repository root:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app opens with the estimate page. The sidebar also contains Property Compare and Finance.

## Train the model

Provide the property CSV explicitly. The source CSV is not bundled in this repository; the app uses the saved model artifact and does not need the CSV at runtime:

```powershell
python -m estate_ml.train_model --data "C:\path\to\your-curated-listings.csv"
```

Training evaluates Linear Regression, Decision Tree, Random Forest, and K-Nearest Neighbors with five-fold `GridSearchCV`. The primary 80/20 split uses seed 42; hyperparameter and model selection use only the training partition's cross-validation results. A separate chronological evaluation trains on earlier listing dates and tests on later dates. The saved pipeline performs preprocessing fitted on training folds.

The current supplied CSV has 5,000 rows. Its target `price_lakhs` is in lakhs of INR; `area_sqft` is square feet. Read [MODEL_REPORT.md](MODEL_REPORT.md) for the schema, evaluation, and limitations.

## Future value and buy-versus-rent scenarios

The estimate page can project property value for up to ten years. It combines the model's property-age effect with locality-specific annual trends in monthly median asking price per square foot. Trend estimates are fitted from the training partition, shrunk toward the city trend when locality data is sparse, and shown as a conservative/base/optimistic range. The listing date is held fixed while aging is projected to avoid counting time effects twice. The growth range and alternative-return comparison are scenarios, not guarantees or financial advice.

The Finance page includes a buy-versus-rent illustration. It compares projected owner equity with a renter's modeled investment of the down payment and monthly cost difference. Taxes, transaction costs, rent deposits, and personal circumstances are not modeled.

The preprocessing pipeline derives `BHK × balconies` so the selected Linear Regression model can represent a different balcony relationship by bedroom configuration. This interaction is built consistently during training and prediction; the raw property input schema is unchanged.

## Verify

```powershell
python -m unittest discover -s tests -v
python e2e_test.py
```

## Active pages

- `app.py` — property price estimates and recent estimates.
- `pages/Compare.py` — side-by-side estimates from the same property model.
- `pages/finance.py` — finance calculators.

This checkout contains the curated-listing estimator, its training code, the property comparison page, and finance tools. The former dataset and model pipeline have been removed.

Prediction history is stored in local SQLite. Hosted environments with ephemeral storage may lose it after restart. Set `ESTATEML_DATA_DIR` to select a writable history directory.
