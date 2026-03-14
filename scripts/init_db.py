"""
init_db.py

Initializes the SQLite database for PRIMA by executing the schema.sql file.
Safe to run multiple times (idempotent).
"""

from pathlib import Path
import sqlite3

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_PATH = PROJECT_ROOT / "data" / "memory.db"
SCHEMA_PATH = PROJECT_ROOT / "src" / "prima_memory" / "persistence" / "schema.sql"


def init_db() -> None:
    """Initialize the PRIMA SQLite database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_PATH}")

    conn = sqlite3.connect(DB_PATH)

    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")

        with SCHEMA_PATH.open("r", encoding="utf-8") as f:
            schema_sql = f.read()

        conn.executescript(schema_sql)
        conn.commit()

        print(f"[PRIMA] Database initialized at: {DB_PATH}")

    finally:
        conn.close()


if __name__ == "__main__":
    init_db()