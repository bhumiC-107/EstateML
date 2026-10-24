"""SQLite prediction-history setup and additive schema migrations."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


PREDICTION_COLUMNS = {
    "id": "INTEGER",
    "location": "TEXT",
    "total_sqft": "REAL",
    "bhk": "INTEGER",
    "bath": "INTEGER",
    "balcony": "INTEGER",
    "predicted_price": "REAL",
    "timestamp": "TEXT",
    "model_source": "TEXT",
}


def initialize_history_database(db_path: str | Path):
    """Create a new history DB or add columns absent from an older SQLite table."""
    path = Path(db_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        URL.create("sqlite", database=str(path)),
        connect_args={"check_same_thread": False},
    )
    with engine.begin() as connection:
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                location TEXT,
                total_sqft REAL,
                bhk INTEGER,
                bath INTEGER,
                balcony INTEGER,
                predicted_price REAL,
                timestamp TEXT,
                model_source TEXT
            )
        """))
        existing = {row[1] for row in connection.execute(text("PRAGMA table_info(predictions)"))}
        for column, sql_type in PREDICTION_COLUMNS.items():
            if column not in existing:
                connection.execute(text(f"ALTER TABLE predictions ADD COLUMN {column} {sql_type}"))
    return engine
