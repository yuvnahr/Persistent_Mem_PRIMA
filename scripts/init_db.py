"""
init_db.py

Initializes the SQLite database for PRIMA by executing the schema.sql file.
This script is idempotent and safe to run multiple times.
"""

import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "memory.db"
SCHEMA_PATH = PROJECT_ROOT / "src" / "prima_memory" / "persistence" / "schema.sql"


def init_db():
    # Ensure data directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Connect to SQLite database
    conn = sqlite3.connect(DB_PATH)
    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        conn.executescript(schema_sql)
        conn.commit()

        print(f"[PRIMA] Database initialized at {DB_PATH}")

    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
