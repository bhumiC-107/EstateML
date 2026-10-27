from pathlib import Path
import json
import tempfile
import unittest

import joblib
from sqlalchemy import text

from estate_ml.history import initialize_history_database
from estate_ml.property_model import load_property_artifacts


ROOT = Path(__file__).resolve().parents[1]


class DeploymentSetupTests(unittest.TestCase):
    def test_runtime_dependencies_are_pinned_without_tracking_or_search_packages(self):
        runtime = [line.strip() for line in (ROOT / "requirements.txt").read_text().splitlines() if line.strip()]
        self.assertTrue(all("==" in line for line in runtime))
        self.assertNotIn("openpyxl==3.1.5", runtime)
        self.assertFalse(any(name in " ".join(runtime).lower() for name in ("mlflow", "dvc", "optuna")))
        trainer = (ROOT / "estate_ml" / "train_model.py").read_text(encoding="utf-8")
        for name in ("LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor", "KNeighborsRegressor"):
            self.assertIn(name, trainer)
        self.assertIn("GridSearchCV", trainer)
        for forbidden in ("GradientBoostingRegressor", "XGBRegressor", "ExtraTreesRegressor", "HistGradientBoostingRegressor"):
            self.assertNotIn(forbidden, trainer)

    def test_artifact_loader_reports_missing_distributed_model_and_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "Model files are missing"):
                load_property_artifacts(tmp)

    def test_artifact_loader_rejects_incompatible_schema_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            joblib.dump({}, Path(tmp) / "property_price_model.joblib")
            (Path(tmp) / "model_metadata.json").write_text(
                '{"schema_version":"bengaluru-listings-v1","feature_schema":["area_sqft"]}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "feature schema is incompatible"):
                load_property_artifacts(tmp)

    def test_artifact_loader_reports_schema_version_mismatch_separately(self):
        from estate_ml.property_model import FEATURE_SCHEMA
        with tempfile.TemporaryDirectory() as tmp:
            joblib.dump({}, Path(tmp) / "property_price_model.joblib")
            (Path(tmp) / "model_metadata.json").write_text(
                json.dumps({"schema_version": "old-version", "feature_schema": FEATURE_SCHEMA}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "schema version is incompatible"):
                load_property_artifacts(tmp)

    def test_new_history_database_and_legacy_schema_migration(self):
        with tempfile.TemporaryDirectory() as tmp:
            new_path = Path(tmp) / "nested" / "new.db"
            engine = initialize_history_database(new_path)
            with engine.begin() as connection:
                names = {row[1] for row in connection.execute(text("PRAGMA table_info(predictions)"))}
                self.assertEqual(names, {"id", "location", "total_sqft", "bhk", "bath", "balcony", "predicted_price", "timestamp", "model_source"})
            engine.dispose()

            legacy_path = Path(tmp) / "legacy.db"
            legacy_engine = initialize_history_database(legacy_path)
            with legacy_engine.begin() as connection:
                connection.execute(text("DROP TABLE predictions"))
                connection.execute(text("""
                    CREATE TABLE predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        total_sqft REAL, bhk INTEGER, bath INTEGER,
                        predicted_price REAL, timestamp TEXT
                    )
                """))
                connection.execute(text("INSERT INTO predictions (total_sqft, bhk, bath, predicted_price, timestamp) VALUES (900, 2, 1, 42, 'old')"))
            legacy_engine.dispose()

            migrated = initialize_history_database(legacy_path)
            with migrated.begin() as connection:
                columns = {row[1] for row in connection.execute(text("PRAGMA table_info(predictions)"))}
                self.assertIn("location", columns)
                self.assertIn("balcony", columns)
                self.assertIn("model_source", columns)
                old_rows = connection.execute(text("SELECT total_sqft, location FROM predictions")).all()
                connection.execute(text("""
                    INSERT INTO predictions (location, total_sqft, bhk, bath, balcony, predicted_price, timestamp)
                    VALUES ('whitefield', 1200, 2, 2, 1, 65, 'new')
                """))
                new_row = connection.execute(text("SELECT location FROM predictions WHERE timestamp='new'")).scalar_one()
            self.assertEqual(old_rows, [(900.0, None)])
            self.assertEqual(new_row, "whitefield")
            migrated.dispose()


if __name__ == "__main__":
    unittest.main()
