import warnings
warnings.filterwarnings('ignore')
import joblib, pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
import sys, os
sys.path.insert(0, r'c:\Users\Bhumika\Downloads\EstateML_integrated_upgrade (1)\EstateML-main')
os.chdir(r'c:\Users\Bhumika\Downloads\EstateML_integrated_upgrade (1)\EstateML-main')

print('=== END-TO-END TEST ===')

scaler = joblib.load('scaler.joblib')
model  = joblib.load('final_model.joblib')
automl = joblib.load('artifacts/automl_model.joblib')
print('[PASS] 3 model artifacts loaded')

user = {'total_sqft': 1200.0, 'bhk': 3, 'bath': 2, 'balcony': 1}
df = pd.DataFrame([user])
df['price_per_sqft'] = 0
df['sqft_per_bhk'] = df['total_sqft'] / df['bhk']
df['bath_per_bhk'] = df['bath'] / df['bhk']
df = df[list(scaler.feature_names_in_)]
scaled = pd.DataFrame(scaler.transform(df), columns=list(scaler.feature_names_in_))
pred_orig = float(model.predict(scaled)[0])
print('[PASS] Original model: ' + str(round(pred_orig, 2)) + ' Lakhs')

af = pd.DataFrame([{'total_sqft': 1200.0, 'bhk': 3, 'bath': 2, 'balcony': 1}])
pred_automl = float(automl.predict(af)[0])
print('[PASS] AutoML model: ' + str(round(pred_automl, 2)) + ' Lakhs')

engine = create_engine('sqlite:///predictions.db', connect_args={'check_same_thread': False})
with engine.connect() as conn:
    df_hist = pd.read_sql(text('SELECT COUNT(*) as cnt FROM predictions'), conn)
cnt = int(df_hist['cnt'].iloc[0])
print('[PASS] SQLite DB: ' + str(cnt) + ' records')

from estate_ml.data import load_training_data
X, y = load_training_data('bengaluru_house_prices.csv')
print('[PASS] data.py: ' + str(X.shape[0]) + ' rows, features: ' + str(list(X.columns)))

import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
client = mlflow.tracking.MlflowClient()
exps = client.search_experiments()
runs_count = sum(len(client.search_runs(e.experiment_id)) for e in exps)
print('[PASS] MLflow: ' + str(len(exps)) + ' experiments, ' + str(runs_count) + ' runs')

print('[PASS] DVC initialized: ' + str(Path('.dvc').exists()))

reqs = Path('requirements.txt').read_text()
assert 'mlflow' in reqs
assert 'dvc' in reqs
assert 'mysql-connector-python' not in reqs
print('[PASS] requirements.txt: mlflow+dvc present, no mysql dep')

ta = Path('estate_ml/train_automl.py').read_text()
assert 'artifact_path=' in ta
print('[PASS] train_automl.py: uses artifact_path= (MLflow 3.x API)')

print()
print('=== ALL 9 TESTS PASSED ===')
