from __future__ import annotations

import datetime
import hashlib
import sqlite3
from pathlib import Path

import pytest

from prima_memory.core.memory_store import MemoryStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = PROJECT_ROOT / "src" / "prima_memory" / "persistence" / "schema.sql"


def fake_embedding(text: str) -> bytes:
    """Generate deterministic embedding bytes."""
    return hashlib.sha256(text.encode("utf-8")).digest()


@pytest.fixture()
def temp_db(tmp_path: Path) -> Path:
    """Create temporary SQLite database with schema."""
    db_path = tmp_path / "test_memory.db"

    conn = sqlite3.connect(db_path)

    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        conn.executescript(f.read())

    conn.commit()
    conn.close()

    return db_path


def test_insert_and_get_memory(temp_db: Path) -> None:
    store = MemoryStore(db_path=temp_db)

    now = datetime.datetime.utcnow().isoformat()

    store.insert_memory(
        memory_id="test1",
        content="Test memory content",
        created_at=now,
        context="Testing insertion",
        keywords=["test"],
        tags=["unit"],
        embedding=fake_embedding("Test memory content"),
    )

    memory = store.get_memory("test1")

    assert memory is not None
    assert memory["content"] == "Test memory content"


def test_get_all_embeddings(temp_db: Path) -> None:
    store = MemoryStore(db_path=temp_db)

    now = datetime.datetime.utcnow().isoformat()

    store.insert_memory(
        memory_id="test2",
        content="Embedding test",
        created_at=now,
        context=None,
        keywords=[],
        tags=[],
        embedding=fake_embedding("Embedding test"),
    )

    embeddings = store.get_all_embeddings()

    assert len(embeddings) == 1
    assert embeddings[0][0] == "test2"


def test_insert_and_get_links(temp_db: Path) -> None:
    store = MemoryStore(db_path=temp_db)

    now = datetime.datetime.utcnow().isoformat()

    store.insert_memory(
        memory_id="m1",
        content="Memory 1",
        created_at=now,
        context=None,
        keywords=[],
        tags=[],
        embedding=b"1",
    )

    store.insert_memory(
        memory_id="m2",
        content="Memory 2",
        created_at=now,
        context=None,
        keywords=[],
        tags=[],
        embedding=b"2",
    )

    store.insert_link("m1", "m2", "related", 0.9)

    links = store.get_links("m1")

    assert len(links) == 1
    assert links[0]["relation_type"] == "related"


def test_update_memory(temp_db: Path) -> None:
    store = MemoryStore(db_path=temp_db)

    now = datetime.datetime.utcnow().isoformat()

    store.insert_memory(
        memory_id="m3",
        content="Memory before update",
        created_at=now,
        context=None,
        keywords=[],
        tags=[],
        embedding=b"3",
    )

    store.update_memory(
        memory_id="m3",
        context="Updated context",
        keywords=["update"],
        tags=["edited"],
    )

    memory = store.get_memory("m3")

    assert memory["context"] == "Updated context"


def test_log_evolution(temp_db: Path) -> None:
    store = MemoryStore(db_path=temp_db)

    now = datetime.datetime.utcnow().isoformat()

    store.insert_memory(
        memory_id="m4",
        content="Memory for evolution",
        created_at=now,
        context=None,
        keywords=[],
        tags=[],
        embedding=b"4",
    )

    store.log_evolution(
        memory_id="m4",
        timestamp=now,
        action="context_update",
        details={"reason": "unit test"},
    )

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM memory_evolution")
    count = cursor.fetchone()[0]

    conn.close()

    assert count == 1