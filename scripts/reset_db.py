"""
reset_db.py

Completely clears all PRIMA database tables while preserving schema.
Safe for development and testing.
"""

from pathlib import Path
import sqlite3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "memory.db"


def reset_db() -> None:
    conn = sqlite3.connect(DB_PATH)

    try:
        conn.execute("PRAGMA foreign_keys = OFF;")

        tables = [
            "memory_links",
            "memory_evolution",
            "memory_notes",
        ]

        for table in tables:
            conn.execute(f"DELETE FROM {table};")

        # reset autoincrement counters
        conn.execute("DELETE FROM sqlite_sequence;")

        conn.commit()

        print("[PRIMA] Database reset successfully.")

    finally:
        conn.close()


if __name__ == "__main__":
    reset_db()