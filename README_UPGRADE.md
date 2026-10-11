# EstateML Integrated Upgrade

This is a review bundle for the existing EstateML repository. Copy the files into
the repository while preserving the current `app.py`, `pages/Compare.py`,
`pages/EDA.py`, `pages/finance.py`, `final_model.joblib`, and `scaler.joblib`.

## What is added

- MLflow experiment tracking for repeatable model comparisons.
- A lightweight AutoML-style benchmark using Random Forest, Extra Trees, and
  HistGradientBoostingRegressor.
- A Model Lab page at `pages/Model_Lab.py`.
- Docker packaging for the existing Streamlit app.
- DVC pipeline configuration for tracking the new experiment and artifacts.
- A leakage-aware training-data parser in `estate_ml/data.py`.

## Important model-safety decision

The current app's saved model and scaler expect a particular feature schema. The
original training script is not present in the repository, so the precise training
semantics cannot be reconstructed with confidence from inference code alone.
Therefore, the new benchmark uses only `total_sqft`, `bhk`, `bath`, and `balcony`
and deliberately excludes `price_per_sqft`, which is derived from the target.
Its artifacts are saved under `artifacts/automl_model.joblib` and are NOT used
by the live prediction app. Do not replace the current production artifacts until
the candidate has been evaluated and the app's input mapping is deliberately
migrated.

## Local setup

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Run MLflow + AutoML benchmark

```bash
python -m estate_ml.train_automl --data bengaluru_house_prices.csv
mlflow ui --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000` to inspect the experiment runs. The Streamlit Model
Lab page shows the latest summary JSON if the training script has completed.

## DVC workflow

DVC configuration is supplied in `dvc.yaml` and `params.yaml`.

```bash
dvc init
dvc repro
dvc status
```

To track data with DVC, first initialize DVC, then add the dataset:

```bash
dvc add bengaluru_house_prices.csv
git add bengaluru_house_prices.csv.dvc .gitignore dvc.yaml params.yaml
git commit -m "Track housing data and AutoML pipeline with DVC"
```

Configure a remote storage location before expecting teammates or CI to retrieve
DVC-managed data/artifacts:

```bash
dvc remote add -d storage <your-storage-url>
dvc push
```

Do not commit credentials or private storage URLs. DVC remote setup depends on
the storage provider you choose.

## Docker

The Docker image needs the existing model and scaler files because the production
app loads them at startup. Keep the files in the build context, or use a private
artifact download step in your own deployment.

```bash
docker compose up --build
```

Open `http://localhost:8501`.

## Important follow-up checks

1. The existing README says MySQL, but current `app.py` creates `sqlite:///predictions.db`.
   Update one or the other so docs and implementation agree.
2. Current displayed `predicted price ± RMSE` is an error-based range, not a calibrated
   statistical prediction interval.
3. Run the app and test prediction, comparison, EDA, finance, prediction history,
   CSV downloads, and Model Lab after copying files.
4. Verify the CSV's target unit is lakhs. If prices are in another unit, normalize the
   target before comparing metrics.
5. Add the new files to Git and test in a branch before merging.
