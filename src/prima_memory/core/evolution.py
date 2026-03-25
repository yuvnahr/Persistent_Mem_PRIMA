"""
evolution.py

Memory evolution module for PRIMA.

Responsible for refining existing MemoryNotes based on
LLM analysis of new information and relationships.

Implements the A-MEM evolution stage (Ps3).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from prima_memory.core.memory_store import MemoryStore
from prima_memory.core.note import MemoryNote
from prima_memory.llm.llm_service import LLMService


class MemoryEvolver:
    """
    Evolves memory notes by refining semantic metadata using LLM.
    """

    def __init__(
        self,
        store: MemoryStore,
        llm_service: Optional[LLMService] = None,
        min_retrievals: int = 2,
    ) -> None:
        self.store = store
        self.llm_service = llm_service
        self.min_retrievals = min_retrievals

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def evolve(
        self,
        source: MemoryNote,
        related: List[MemoryNote],
    ) -> None:
        """
        Evolve related memories based on new information using LLM.

        Implements A-MEM Ps3: Use LLM to analyze how existing memories
        should evolve based on the new memory and relationships.
        """

        if self.llm_service is None:
            # Basic fallback evolution: copy tags from source to related memories
            for target in related:
                if target is None or target.id == source.id:
                    continue
                changed_tags = list(set(target.tags) | set(source.tags))
                if changed_tags != target.tags:
                    target.tags = changed_tags
                    target.context = (
                        f"{target.context}. Refined with evolution context."
                    )
                update_memory = getattr(self.store, "update_memory", None)
                if callable(update_memory):
                    update_memory(
                        memory_id=target.id,
                        context=target.context,
                        keywords=target.keywords,
                        tags=target.tags,
                    )
                else:
                    # Fallback for SQLiteMemoryStore / custom stores.
                    connect = getattr(self.store, "_connect", None)
                    if callable(connect):
                        conn = connect()
                        try:
                            conn.execute(
                                """
                                UPDATE memory_notes
                                SET context = ?, keywords = ?, tags = ?
                                WHERE id = ?
                                """,
                                (
                                    target.context,
                                    json.dumps(target.keywords),
                                    json.dumps(target.tags),
                                    target.id,
                                ),
                            )
                            conn.commit()
                        finally:
                            conn.close()

                    # Record fallback evolution event
                    record_ev = getattr(self.store, "record_evolution", None)
                    if callable(record_ev):
                        record_ev(
                            target,
                            action="llm_evolution",
                            details={
                                "reason": "fallback tag/context merge",
                                "added_tags": [
                                    tag for tag in target.tags if tag in source.tags
                                ],
                            },
                        )
            return

        for target in related:
            if target is None or target.id == source.id:
                continue

            # Use LLM to decide evolution
            evolution_decision = self.llm_service.evolve_memory(
                new_memory=source,
                target_memory=target,
                neighborhood=related,
            )

            if evolution_decision:
                self._apply_evolution(target, evolution_decision)

    # --------------------------------------------------
    # Evolution application
    # --------------------------------------------------

    def _apply_evolution(
        self,
        target: MemoryNote,
        evolution_decision: Dict[str, Any],
    ) -> None:
        """
        Apply the LLM-suggested evolution to a target memory.
        """
        changes: Dict[str, Dict[str, Any]] = {}

        # Update context
        new_context = evolution_decision.get("updated_context")
        if new_context and new_context != target.context:
            changes["context"] = {"old": target.context, "new": new_context}
            target.context = new_context

        # Update keywords
        new_keywords = evolution_decision.get("updated_keywords", [])
        if new_keywords != target.keywords:
            changes["keywords"] = {"old": target.keywords, "new": new_keywords}
            target.keywords = new_keywords

        # Update tags
        new_tags = evolution_decision.get("updated_tags", [])
        if new_tags != target.tags:
            changes["tags"] = {"old": target.tags, "new": new_tags}
            target.tags = new_tags

        if not changes:
            return

        # --------------------------------------------------
        # Persist changes
        # --------------------------------------------------

        update_memory = getattr(self.store, "update_memory", None)

        if callable(update_memory):
            update_memory(
                memory_id=target.id,
                context=target.context,
                keywords=target.keywords,
                tags=target.tags,
            )

        # --------------------------------------------------
        # SQLiteMemoryStore fallback
        # --------------------------------------------------

        else:
            update_note = getattr(self.store, "update_note", None)

            if callable(update_note):
                update_note(target)

            # Force persistence of semantic fields
            import sqlite3

            conn = sqlite3.connect(self.store.db_path)

            try:
                conn.execute(
                    """
                    UPDATE memory_notes
                    SET context = ?, tags = ?, keywords = ?
                    WHERE id = ?
                    """,
                    (
                        target.context,
                        json.dumps(target.tags),
                        json.dumps(target.keywords),
                        target.id,
                    ),
                )
                conn.commit()

            finally:
                conn.close()

        # --------------------------------------------------
        # Log evolution event
        # --------------------------------------------------

        timestamp = getattr(target, "timestamp", None)

        log_evolution = getattr(self.store, "log_evolution", None)
        record_evolution = getattr(self.store, "record_evolution", None)

        if callable(log_evolution):
            log_evolution(
                memory_id=target.id,
                timestamp=timestamp,
                action="llm_evolution",
                details={
                    **changes,
                    "reason": evolution_decision.get(
                        "evolution_reason", "LLM suggested evolution"
                    ),
                },
            )

        elif callable(record_evolution):
            record_evolution(
                note=target,
                action="llm_evolution",
                details={
                    **changes,
                    "reason": evolution_decision.get(
                        "evolution_reason", "LLM suggested evolution"
                    ),
                },
            )
