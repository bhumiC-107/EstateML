# EstateML — Bengaluru Real Estate Price Prediction

EstateML is an end-to-end machine learning web application designed to deliver data-driven residential property valuations across the Bengaluru housing market. Combining interactive consumer tools with an enterprise MLOps workflow, EstateML integrates **Streamlit**, **scikit-learn**, **AutoML**, **MLflow**, **DVC**, and **SQLite**.

Try our website here : https://estateml-dre4q7qeekrjfk95v2sjzk.streamlit.app/

---

## 1. Problem Statement & Objectives

Estimating residential real estate values in fast-growing metropolitan markets like Bengaluru is challenging due to high variance across built-up area, room configurations, amenities, and micro-market pricing tiers. Traditional valuations often rely on subjective estimates or opaque heuristics.

### Objectives
1. **Accurate & Transparent Valuation**: Provide instant, objective price estimates in Lakhs (INR) based on physical property configurations (`total_sqft`, `bhk`, `bath`, `balcony`).
2. **Financial Decision Support**: Assist prospective home buyers and owners with comparative property evaluation, mortgage EMI calculations, buyer affordability assessments, and gross rental yield benchmarking.
3. **Reproducible Machine Learning**: Implement a rigorous MLOps lifecycle featuring automated model selection (AutoML), experiment tracking with MLflow, version-controlled data pipelines via DVC, and audit-ready test suites.

---

## 2. Key Features

* **Interactive Property Valuation (`app.py`)**:
  * Clean, responsive two-column interface for specifying total area, bedrooms, bathrooms, and balconies.
  * Instant market value prediction in Lakhs with estimated price per square foot.
  * Production inference pipeline using scikit-learn models and StandardScaler transforms.
* **Side-by-Side Property Comparator (`pages/Compare.py`)**:
  * Direct comparison of two properties (Property A vs. Property B).
  * Evaluates relative price-to-area value with radar charts and feature breakdown tables.
  * One-click CSV export of comparative analysis.
* **Market Insights & Exploratory Data Analysis (`pages/EDA.py`)**:
  * Visual exploration of over 13,000 listings from the Bengaluru House Price dataset.
  * Key performance indicators, price distribution histograms, bedroom-tier analysis, and an interactive feature correlation heatmap.
* **Financial Tools & Mortgage Calculators (`pages/finance.py`)**:
  * **Loan EMI Calculator**: Monthly EMI, total interest, and complete multi-year principal vs. interest amortization breakdown.
  * **Affordability Estimator**: Maximum property budget derived from monthly income, existing loan obligations, and FOIR (Fixed Obligation to Income Ratio) limits.
  * **Rental Yield Calculator**: Gross annual rental yields benchmarked against major Indian metropolitan averages.
* **Prediction History & Persistence**:
  * All user valuation requests are automatically persisted to a local SQLite database (`predictions.db`).
  * Real-time query of recent valuations alongside interactive historical price trend charts.
* **Model Performance & AutoML Benchmarking (`pages/Model_Lab.py`)**:
  * Live display of cross-validation and holdout test metrics across ensemble algorithms.
  * Interactive error metrics comparison chart (RMSE & MAE).
  * Target-leakage disclosures and collapsible technical MLflow architecture details.
* **Data & Experiment Pipelines**:
  * **MLflow Tracking**: Logs model runs, hyperparameters, metrics, and serialized artifacts.
  * **DVC Pipeline (`dvc.yaml`)**: Manages end-to-end data dependencies and pipeline stages.

---

## 3. Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ / 3.13 | Core programming language |
| **User Interface** | Streamlit | Reactive multi-page web application framework |
| **Data Processing** | pandas, NumPy | Data ingestion, cleaning, and feature manipulation |
| **Machine Learning** | scikit-learn | Ensembles (Extra Trees, Random Forest, HistGradientBoosting) |
| **Visualizations** | Plotly Express & Graph Objects | Interactive radar charts, heatmaps, and error comparisons |
| **Model Persistence** | joblib | Serialized model pipelines and scalers |
| **Database** | SQLite via SQLAlchemy | Local database logging for valuation history |
| **Experiment Tracking** | MLflow | Run tracking, parameter logging, and model registry |
| **Data Versioning** | DVC (Data Version Control) | Reproducible data pipelines and stages |
| **Containerization** | Docker & Docker Compose | Dockerfile and compose definitions for container deployment |

---

## 4. Architecture & Repository Structure

```
EstateML-main/
│
├── app.py                     # Main property valuation application & history dashboard
├── compare_models.py          # Standalone evaluation comparing AutoML vs baseline model
├── e2e_test.py                # 9-point end-to-end integration test suite
├── requirements.txt           # Python application dependencies
├── bengaluru_house_prices.csv # Primary Bengaluru housing dataset (13,000+ listings)
├── final_model.joblib         # Production model artifact
├── scaler.joblib              # Production StandardScaler artifact
├── predictions.db             # SQLite database for valuation history (generated)
│
├── estate_ml/                 # Core machine learning package
│   ├── __init__.py
│   ├── data.py                # Leakage-aware data cleaning & feature loading
│   └── train_automl.py        # AutoML training script with 5-fold CV & MLflow logging
│
├── pages/                     # Streamlit multi-page interface
│   ├── Compare.py             # Side-by-side property comparison tool
│   ├── EDA.py                 # Market insights, distributions & correlation heatmap
│   ├── finance.py             # EMI, affordability, and rental yield calculators
│   └── Model_Lab.py           # Model performance, benchmarks & MLflow details
│
├── artifacts/                 # AutoML outputs & evaluation metadata
│   ├── automl_model.joblib    # Best candidate model (Extra Trees Regressor)
│   ├── automl_metrics.json    # Holdout and cross-validation benchmark metrics
│   └── automl_feature_columns.json # Feature schema contract
│
├── .streamlit/
│   └── config.toml            # Global light theme and aesthetic design tokens
│
├── dvc.yaml                   # DVC pipeline stages and dependencies
├── params.yaml                # Hyperparameters for training and model evaluation
├── Dockerfile                 # Application container build definition
└── docker-compose.yml         # Container orchestration configuration
```

---

## 5. How the AutoML Engine Works

EstateML features an automated machine learning benchmark module located in `estate_ml/train_automl.py`.

### 1. Data Ingestion & Leakage Prevention
Data is loaded through `estate_ml/data.py`, which strictly isolates raw property attributes:
* Input Features: `total_sqft`, `bhk`, `bath`, `balcony`.
* Target: `price` (in Lakhs).
* Missing values are imputed using median values fitted exclusively on training splits.
* **Target leakage features (such as `price_per_sqft`) are explicitly excluded**.

### 2. Candidate Algorithm Evaluation
The AutoML pipeline benchmarks three distinct regression architectures:
1. **Random Forest Regressor**: Standard bagging ensemble averaging predictions across decision trees.
2. **Extra Trees Regressor (Extremely Randomized Trees)**: Introduces random split thresholds to reduce variance.
3. **HistGradientBoosting Regressor**: Histogram-based gradient boosting optimized for numeric tabular data.

### 3. Evaluation & Model Selection
* Candidate models undergo **5-fold cross-validation** followed by evaluation on a separate 20% holdout test set (`random_state=42`).
* The engine evaluates Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Coefficient of Determination ($R^2$).
* The top-performing model on holdout test RMSE is automatically exported to `artifacts/automl_model.joblib`, while metrics are logged to MLflow and serialized to `artifacts/automl_metrics.json`.

---

## 6. Model Evaluation Results

The table below reflects the actual benchmark results logged in `artifacts/automl_metrics.json`:

| Model Architecture | 5-Fold CV RMSE (₹ L) | Test RMSE (₹ L) | Test MAE (₹ L) | Test $R^2$ Score | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Extra Trees Regressor** | **105.12** | **95.51** | **38.66** | **0.5592** | ★ **Selected Benchmark** |
| **HistGradientBoosting** | 102.58 | 95.75 | 38.66 | 0.5570 | Evaluated Candidate |
| **Random Forest Regressor** | 108.25 | 98.31 | 39.16 | 0.5330 | Evaluated Candidate |

### Understanding the Metrics
* **RMSE (Root Mean Squared Error)**: Measures the square root of the average squared errors. Because errors are squared, RMSE heavily penalizes large valuation misses. Extra Trees achieved the lowest test error at **₹95.51 Lakhs**.
* **MAE (Mean Absolute Error)**: The average absolute difference between predicted and actual property prices. Both Extra Trees and HistGradientBoosting achieved an identical MAE of **₹38.66 Lakhs**, indicating an average valuation spread of approximately ₹38.6 Lakhs across all property price segments.
* **$R^2$ (Coefficient of Determination)**: Indicates the proportion of price variance explained by the model using only the four basic property features. The benchmark explains **~56% ($R^2 = 0.5592$)** of the price variation without relying on micro-market coordinates or builder tiers.

---


## 8. Running Comparisons, Tests & MLflow

### Run the End-to-End Integration Test Suite
To verify the entire environment (model artifacts, SQLite database, AutoML pipeline, DVC config, and MLflow API compatibility):
```powershell
py e2e_test.py
```
Expected output:
```
=== END-TO-END TEST ===
[PASS] 3 model artifacts loaded
[PASS] Original model: 32.94 Lakhs
[PASS] AutoML model: 86.58 Lakhs
[PASS] SQLite DB: records verified
[PASS] data.py: 13267 rows, features: ['total_sqft', 'bhk', 'bath', 'balcony']
[PASS] MLflow: experiments and runs verified
[PASS] DVC initialized: True
[PASS] requirements.txt: mlflow+dvc present, no mysql dep
[PASS] train_automl.py: uses artifact_path= (MLflow 3.x API)

=== ALL 9 TESTS PASSED ===
```

### Run Candidate Model Training (AutoML + MLflow)
To re-run the 5-fold cross-validation training cycle:
```powershell
py -m estate_ml.train_automl --data bengaluru_house_prices.csv
```

### Run Model Evaluation Script
Compare the baseline model and the AutoML pipeline on an identical test split:
```powershell
py compare_models.py
```

### Launch the MLflow Experiment Tracking UI
To inspect runs, parameters, loss curves, and artifact trees:
```powershell
py -m mlflow ui --host 127.0.0.1 --port 5000
```
Open `http://localhost:5000` in your web browser.

---

## 9. Visual Interface & Navigation

The web application is styled with a custom fintech design system featuring a warm ivory page background (`#f5f5f0`), deep navy cards (`#1a1a2e`), restrained gold accents (`#c9922a`), and high-contrast typography.

* **Home (`/`)**: Main property valuation form, instant estimated market value card, configuration summary, and recent valuation audit table.
* **Compare (`/Compare`)**: Two-column side-by-side property comparison with radar visualization and CSV export.
* **Market Insights (`/EDA`)**: Listings breakdown, BHK pricing distributions, and correlation heatmap.
* **Financial Tools (`/finance`)**: Tabbed interface for mortgage loan EMI calculators, buyer budget affordability, and city-level rental yield benchmarks.
* **Model Lab (`/Model_Lab`)**: Live candidate model comparison cards, error analysis charts, and technical experiment documentation.

*(Visual screenshots can be captured directly from the live application running at `http://localhost:8501`.)*

---

## 10. Honest Disclosures & Known Limitations

To maintain academic and professional integrity, the following project characteristics are explicitly noted:

1. **Target Leakage in Original Baseline Model**:
   * The original baseline model (`final_model.joblib`) reported an unusually low RMSE (~₹37.58 Lakhs). This occurred because its original training pipeline incorporated `price_per_sqft`, a feature derived directly from the sale price target (`price * 100,000 / total_sqft`).
   * In a realistic valuation scenario, price per square foot is unknown prior to valuation.
   * The new AutoML benchmark models (`artifacts/automl_model.joblib`) strictly eliminate this leakage by training solely on raw structural attributes (`total_sqft`, `bhk`, `bath`, `balcony`). Their higher RMSE (~₹95.51 Lakhs) represents true out-of-sample generalization.
2. **Confidence Range vs. Statistical Intervals**:
   * The illustrative price range shown on valuation cards is calculated from the model's RMSE ($Estimated \pm RMSE$), rather than formal conformal prediction or Bayesian credible intervals.
3. **Docker Verification**:
   * A valid `Dockerfile` and `docker-compose.yml` are supplied in the repository for containerized execution. However, live container execution was not verified locally because Docker Desktop was not installed on this environment.
4. **Local Tracking Server**:
   * MLflow tracking is configured using a local SQLite registry (`sqlite:///mlflow.db`) and local artifact store (`./mlruns`). No external cloud tracking server (e.g., AWS S3 or Databricks) is assumed.

---

## 11. Future Roadmap

- [ ] **Geospatial & Locality Encoding**: Integrate latitude, longitude, and locality clusters (e.g., Whitefield, Indiranagar, Electronic City) to improve $R^2$ accuracy.
- [ ] **Conformal Prediction Intervals**: Implement non-parametric conformal prediction intervals to guarantee statistical coverage rates for property buyers.
- [ ] **Automated CI/CD Pipeline**: Integrate GitHub Actions to run `e2e_test.py` and automated linting on every pull request.
- [ ] **Cloud DVC Remote**: Configure an S3 or Google Cloud Storage remote for team-wide DVC artifact synchronization.

---

## 12. License

This project is developed for educational and demonstration purposes under the MIT License.

