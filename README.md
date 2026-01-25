# PRIMA Implementation of Persistent Agentic Memory

## Core Principles
- Atomic memory notes
- Semantic linking between memories
- Memory evolution over time
- Task-agnostic and model-agnostic design

## Project Overview

This repository implements a **persistent memory system for LLM agents** inspired by the **A-MEM (Agentic Memory)** framework and backed by **PRIMA** for long-term storage.

The objective is to move beyond traditional retrieval-augmented generation (RAG) and toward a **self-evolving, agentic memory system** that:
1. Stores structured memories
2. Forms semantic links between memories
3. Evolves historical memories over time

---

## Current Status (Phase 1 Evaluation)

### High-Level Verdict

| Component | Status |
|--------|--------|
| Understanding of A-MEM theory | Correct |
| System architecture | Correct direction |
| Persistent memory (PRIMA) | Implemented |
| Retrieval mechanism | Implemented (basic) |
| Link generation | Partial |
| Memory evolution | Missing |
| Agentic behavior | Not yet achieved |

**Conclusion:**  
The current system functions as a *persistent memory + retrieval wrapper*, not yet as a full **agentic memory system**.

---

## A-MEM Requirements vs Current Implementation

### Core A-MEM Operations

| Operation | A-MEM Definition | Current State |
|---------|------------------|---------------|
| Note Construction | LLM-generated structured notes (context, keywords, tags) | Partial |
| Link Generation | LLM-decided semantic links (Zettelkasten boxes) | Weak |
| Memory Evolution | Old memories rewritten based on new ones | Missing |
| Contextual Retrieval | Retrieval + linked memory expansion | Partial |

Only **Note Construction** is partially implemented.  
**Memory Evolution**, the defining feature of A-MEM, is currently absent.

---

## Phase 1 Empirical Findings

### Baseline vs PRIMA Output Behavior

| Metric | Baseline | PRIMA |
|------|----------|-------|
| Memory usage | 0 | 0 |
| Cross-session dependency | None | None |
| Historical grounding | None | None |
| Concept refinement | No | No |
| Answer consistency over time | Low | Low |

Observed behavior:
- Frequent fallback to general knowledge
- Frequent “no relevant memories found”
- No improvement over baseline answers

**Root Cause:**  
Stored memories lack semantic depth and never evolve.

---

## Root Cause Analysis

### Issue 1: Shallow Memory Notes
- Memories mostly store raw text
- Lack contextual abstraction
- Low embedding separability

**Effect:** Poor retrieval and ineffective linking

---

### Issue 2: Non-agentic Link Generation
- Links not explicitly stored
- No semantic reasoning persisted
- No multi-cluster memory membership

**Effect:** Flat memory structure (no Zettelkasten boxes)

---

### Issue 3: Missing Memory Evolution (Critical)
- Old memories are immutable
- No refinement or abstraction
- No long-term learning behavior

**Effect:** Long-term interactions behave like first-time conversations

---

## Phase 2 Roadmap

### Phase 2 Goal

Transform the system into a **self-evolving agentic memory system** consistent with A-MEM theory.

---

### 1. Enhanced Note Construction

Each memory must include:

| Field | Description |
|-----|-------------|
| content | Raw interaction text |
| timestamp | Time of interaction |
| context | LLM-generated significance |
| keywords | Conceptual abstractions |
| tags | Task- and reasoning-level labels |
| embedding | Content + context embedding |
| links | Semantic memory connections |

---

### 2. Explicit Semantic Link Generation

Process:
1. Retrieve top-k candidate memories
2. Use LLM to decide semantic relationships
3. Persist:
   - Linked memory IDs
   - Natural-language justification

Expected outcome:
- Emergence of Zettelkasten-style memory clusters
- Multi-box memory membership

---

### 3. Memory Evolution Module (Core Phase 2 Feature)

For each new memory:
1. Identify related historical memories
2. Prompt LLM:
   - Does this new memory refine or correct prior ones?
3. Update old memories:
   - context
   - tags
   - keywords
4. Replace old versions in PRIMA

Expected outcome:
- Progressive abstraction
- Improved cross-session reasoning
- Stable conceptual representations

---

### 4. Controlled Re-Evaluation

Re-run **exact same evaluation questions**.

Success indicators:

| Indicator | Expected Result |
|--------|----------------|
| Memory usage count | > 0 |
| Cross-session references | Present |
| Answer precision | Improved |
| Concept drift | Reduced |
| PRIMA vs baseline gap | Clearly visible |

---

## Project Value Assessment

| Dimension | Evaluation |
|--------|------------|
| Research relevance | High |
| Technical foundation | Solid |
| Current alignment with A-MEM | Partial |
| Publishability (Phase 1) | Low |
| Publishability (Post Phase 2) | High |

---

## Final Note

This project has successfully implemented the **infrastructure** of A-MEM but not yet its **agentic behavior**.

Phase 2 is focused on:
- Autonomous memory structuring
- Semantic linking
- Memory evolution over time

Once these are implemented, the system should begin to reproduce the qualitative gains reported in the A-MEM paper.