-- -----------------------------
-- Core memory notes table
-- -----------------------------
CREATE TABLE IF NOT EXISTS memory_notes (
    id TEXT PRIMARY KEY,
    
    -- Immutable core
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,

    -- Access tracking
    last_accessed TEXT NOT NULL,
    retrieval_count INTEGER NOT NULL DEFAULT 0,

    -- Semantic metadata (LLM-generated)
    context TEXT,
    keywords TEXT,     -- JSON-encoded list
    tags TEXT,         -- JSON-encoded list

    -- Vector representation
    embedding BLOB     -- Serialized list of floats (optional)
);

-- -----------------------------
-- Memory links (graph edges)
-- -----------------------------
CREATE TABLE IF NOT EXISTS memory_links (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,

    -- Relationship metadata
    relation_type TEXT DEFAULT 'related',
    strength REAL,

    PRIMARY KEY (source_id, target_id),

    FOREIGN KEY (source_id) REFERENCES memory_notes(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES memory_notes(id) ON DELETE CASCADE
);

-- -----------------------------
-- Memory evolution history
-- -----------------------------
CREATE TABLE IF NOT EXISTS memory_evolution (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    memory_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,

    action TEXT NOT NULL,
    details TEXT,     -- JSON blob describing the evolution

    FOREIGN KEY (memory_id) REFERENCES memory_notes(id) ON DELETE CASCADE
);

-- -----------------------------
-- Useful indexes
-- -----------------------------
CREATE INDEX IF NOT EXISTS idx_memory_created_at
ON memory_notes(created_at);

CREATE INDEX IF NOT EXISTS idx_memory_last_accessed
ON memory_notes(last_accessed);

CREATE INDEX IF NOT EXISTS idx_links_source
ON memory_links(source_id);

CREATE INDEX IF NOT EXISTS idx_links_target
ON memory_links(target_id);

CREATE INDEX IF NOT EXISTS idx_evolution_memory
ON memory_evolution(memory_id);
