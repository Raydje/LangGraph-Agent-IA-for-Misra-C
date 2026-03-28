# Project: Multi-Agent Compliance Validator

## Goal
Production-quality multi-agent system that parses technical standards (DO-178B, MISRA-C 2023) and validates code or requirements against them. GitHub CV portfolio project.

**Constraints:** Free or minimal-cost services only.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| LLM | Google Gemini 2.0 Flash (`langchain-google-genai`) |
| Embeddings | `gemini-embedding-001` (768 dims) |
| Vector DB | Pinecone (free tier) |
| Document DB | MongoDB Atlas M0 (free) via Motor (async) |
| Agent framework | LangGraph + LangChain Core |
| Config | Pydantic Settings (`python-dotenv`) |
| Language | Python 3.11+ |

---

## Architecture

```
POST /api/v1/query
  └── LangGraph (graph/builder.py)
        ├── orchestrator_node   temp=0.0  → classifies intent: search | validate | explain
        │                                    overrides to "validate" if code_snippet present
        ├── rag_node                       → hybrid retrieval
        │     ├── sparse: MongoDB regex match on rule IDs in query text
        │     └── dense:  Pinecone top_k=5 semantic search → fetch ground truth from MongoDB
        ├── [validate path only]
        │     ├── validation_node  temp=0.1 → structured JSON verdict with cited rules
        │     └── critique_node    temp=0.0 → 5-criteria hallucination review, loops back if rejected
        └── assemble_node                  → formats final_response by intent
```

### Graph Routing (`graph/edges.py`)
- `route_after_rag`: `"validate"` → `validation_node`, else → `assemble_node`
- `should_loop_or_finish`: `critique_approved=True` or `iteration_count >= max_iterations` → `assemble_node`, else → `validation_node`

**Loop cap:** `max_iterations=3` (set in route handler initial state).

---

## Key Files

| Path | Role |
|---|---|
| `main.py` | FastAPI app factory, mounts router, startup logging |
| `app/config.py` | `get_settings()` — cached Pydantic Settings loader |
| `app/utils.py` | `parse_json_response()` — strips LLM markdown fences before JSON parsing |
| `app/api/routes.py` | Route handlers: `GET /health`, `POST /query`, `GET /rules`, `POST /seed` |
| `app/api/dependencies.py` | `get_compiled_graph()` — cached compiled LangGraph instance |
| `app/graph/builder.py` | `build_graph()` — StateGraph wiring + `assemble_node` defined inline |
| `app/graph/edges.py` | `route_after_rag`, `should_loop_or_finish` — conditional edge functions |
| `app/graph/nodes/orchestrator.py` | Intent classification node |
| `app/graph/nodes/rag.py` | Hybrid retrieval node (Pinecone + MongoDB) |
| `app/graph/nodes/validation.py` | DO-178B compliance checker node |
| `app/graph/nodes/critique.py` | Hallucination reviewer node |
| `app/models/state.py` | `ComplianceState` TypedDict — shared graph state |
| `app/models/requests.py` | `ComplianceQueryRequest`, `IngestRuleRequest` |
| `app/models/responses.py` | `ComplianceQueryResponse`, `HealthResponse`, `IngestResponse` |
| `app/services/llm_service.py` | `get_llm(temperature)` — Gemini client factory |
| `app/services/embedding_service.py` | `get_embedding(text)` → 768-dim vector |
| `app/services/pinecone_service.py` | `query_pinecone(vector, top_k)`, `upsert_vectors()` |
| `app/services/mongodb_service.py` | `get_rules_by_ids()`, `get_rules_by_metadata()`, `upsert_rule()` |
| `app/data/seed_rules.py` | 10 static DO-178B bootstrap rules |
| `app/data/ingest.py` | Embeds rules → upserts to Pinecone + MongoDB |
| `data/misra_c_2023__headlines_for_cppcheck.txt` | Raw MISRA-C 2023 rule headlines |
| `architecture-app.md` | Living design document |

---

## ComplianceState Schema (`app/models/state.py`)

```python
# Input
query: str
code_snippet: str
standard: str                          # e.g. "DO-178B", "MISRA-C"

# Orchestrator output
intent: Literal["search", "validate", "explain"]
orchestrator_reasoning: str

# RAG output
retrieved_rules: list[RetrievedRule]   # {rule_id, standard, section, dal_level, title, full_text, relevance_score}
rag_query_used: str
metadata_filters_applied: dict

# Validation output
validation_result: str
is_compliant: bool
confidence_score: float
cited_rules: list[str]

# Critique loop
critique_feedback: str
critique_approved: bool
iteration_count: int
max_iterations: int
critique_history: Annotated[list[CritiqueEntry], operator.add]  # append-only

# Final
final_response: str
error: str
```

> **Note (Misra-C_Compliance branch):** `rag_node` currently returns `{"documents": ..., "context": ...}` — keys not in `ComplianceState`. This state schema drift is the active in-progress work: aligning the MISRA-C RAG output to populate `retrieved_rules`.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Pings MongoDB and Pinecone; returns `status: healthy\|degraded` |
| `POST` | `/api/v1/query` | Main inference endpoint — runs the full LangGraph pipeline |
| `GET` | `/api/v1/rules` | Lists rules from MongoDB filtered by `standard` and optional `dal_level` |
| `POST` | `/api/v1/seed` | Ingests the 10 static DO-178B seed rules into Pinecone + MongoDB |

### `/query` request shape
```json
{
  "query": "Does this code handle memory allocation safely?",
  "code_snippet": "char *p = malloc(n);",
  "standard": "MISRA-C"
}
```

---

## Environment Variables (`.env`)

```
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.0-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
MONGODB_URI=
MONGODB_DB=
MONGODB_COLLECTION=
```

---

## Development Conventions

- **Async everywhere:** all service calls and route handlers must be `async`.
- **Structured LLM output:** always use `parse_json_response()` from `app/utils.py` to strip markdown fences before `json.loads()`.
- **Service singletons:** use module-level getters (`get_llm()`, `get_embedding()`, `_get_db()`, `_get_index()`). Never instantiate services inside node functions.
- **Pydantic for I/O:** request/response schemas in `app/models/requests.py` and `app/models/responses.py` only.
- **One node per file** under `app/graph/nodes/`. `assemble_node` is an exception — defined inline in `builder.py`.
- **State keys must match `ComplianceState`:** returning unknown keys from a node is silently ignored by LangGraph; always verify against the TypedDict.
- **Keep costs free:** Gemini free tier, Pinecone free tier (1 index), MongoDB Atlas M0.

---

## Running the App

```bash
uvicorn main:app --reload
```

Seed the knowledge base (first run only):
```bash
curl -X POST http://localhost:8000/api/v1/seed
```

Health check:
```bash
curl http://localhost:8000/api/v1/health
```

---

## Current Active Branch
`Misra-C_Compliance` — integrating MISRA-C 2023 rule ingestion and validation.

**In progress:**
- Align `rag_node` output keys (`documents`, `context`) → `retrieved_rules` to match `ComplianceState`
- Ingest MISRA-C 2023 rules from `data/misra_c_2023__headlines_for_cppcheck.txt` into Pinecone + MongoDB
- Update `validation_node` prompt to handle MISRA-C rule format (section.rule_number, category) alongside DO-178B