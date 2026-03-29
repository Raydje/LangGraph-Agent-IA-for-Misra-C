# Project: Multi-Agent Compliance Validator

## Goal
Production-quality multi-agent system that parses technical standards (MISRA C:2023) and validates C code against them. GitHub CV portfolio project.

**Constraints:** Free or minimal-cost services only.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| LLM | Google Gemini 2.5 Flash (`langchain-google-genai`) |
| Embeddings | `gemini-embedding-001` (768 dims) |
| Vector DB | Pinecone (free tier, serverless, cosine, auto-created if absent) |
| Document DB | MongoDB Atlas M0 (free) via Motor (async) |
| Agent framework | LangGraph + LangChain Core |
| Config | Pydantic Settings (`python-dotenv`) |
| Language | Python 3.11+ |

---

## Architecture

```
POST /api/v1/query
  └── LangGraph (graph/builder.py)
        ├── orchestrator_node   temp=0.0  → structured output (OrchestratorOutput Pydantic schema)
        │                                    classifies intent: search | validate | explain
        │                                    overrides to "validate" if code_snippet present
        ├── rag_node                       → hybrid retrieval (MISRA C:2023 only)
        │     ├── sparse: MongoDB regex match on rule IDs in query text
        │     └── dense:  Pinecone top_k=5 → fetch full rules via get_misra_rules_by_pinecone_ids()
        ├── [validate path only]
        │     ├── validation_node  temp=0.1 → structured JSON verdict with cited rules
        │     └── critique_node    temp=0.0 → 5-criteria hallucination review, loops back if rejected
        └── assemble_node                  → formats final_response by intent (defined inline in builder.py)
```

### Graph Routing (`graph/edges.py`)
- `route_after_rag`: `"validate"` → `validation_node`, else → `assemble_node`
- `should_loop_or_finish`: `critique_approved=True` → `assemble_node`; `iteration_count < max_iterations` → `validation_node`; else → `assemble_node`

**Loop cap:** `max_iterations` is read from `settings.max_critique_iterations` (default `4`) and set in the route handler initial state.

---

## Key Files

| Path | Role |
|---|---|
| `main.py` | FastAPI app factory, mounts router at `/api/v1`, startup logging, root `/` redirects to `/docs` |
| `app/config.py` | `get_settings()` — `lru_cache`-wrapped Pydantic Settings; includes `max_critique_iterations=4`, `confidence_threshold=0.85` |
| `app/utils.py` | `parse_json_response()` — strips LLM markdown fences before `json.loads()` |
| `app/api/routes.py` | Route handlers: `GET /health`, `POST /query`, `POST /seed` |
| `app/api/dependencies.py` | `get_compiled_graph()` — `lru_cache`-wrapped compiled LangGraph instance |
| `app/graph/builder.py` | `build_graph()` — StateGraph wiring + `assemble_node` (inline) |
| `app/graph/edges.py` | `route_after_rag`, `should_loop_or_finish` — conditional edge functions |
| `app/graph/nodes/orchestrator.py` | Intent classification via `get_structured_llm(OrchestratorOutput)`; hardcodes `standard="MISRA C:2023"` in output |
| `app/graph/nodes/rag.py` | Hybrid retrieval; returns `retrieved_rules`, `rag_query_used`, `metadata_filters_applied`; filters by `{"scope": "MISRA C:2023"}` |
| `app/graph/nodes/validation.py` | MISRA C:2023 compliance checker; synchronous; increments `iteration_count`; handles critique feedback on re-runs |
| `app/graph/nodes/critique.py` | 5-criteria hallucination reviewer; synchronous; returns `critique_approved`, `critique_feedback` |
| `app/models/state.py` | `ComplianceState` TypedDict, `RetrievedRule` TypedDict, `CritiqueEntry` TypedDict |
| `app/models/requests.py` | `ComplianceQueryRequest` (default `standard="MISRA C:2023"`), `IngestRuleRequest` |
| `app/models/responses.py` | `ComplianceQueryResponse`, `HealthResponse`, `IngestResponse`, `CritiqueDetail` |
| `app/services/llm_service.py` | `get_llm(temperature)`, `get_structured_llm(schema, temperature)` — no singleton caching, new instance per call |
| `app/services/embedding_service.py` | Module-level singleton; `get_embedding(text)` → 768-dim vector; `embed_and_store(rules)` builds Pinecone vectors with ID `MISRA_{section}.{rule_number}` |
| `app/services/pinecone_service.py` | Module-level singleton; `query_pinecone(vector, top_k, filter)`, `upsert_vectors(vectors)` (batches of 100); auto-creates index on first use |
| `app/services/mongodb_service.py` | `get_misra_rules_by_pinecone_ids(rule_ids)` (primary), `get_rules_by_ids()`, `get_rules_by_metadata(filters)`, `insert_rules()`, `create_indexes()` |
| `app/data/seed_rules.py` | 10 static DO-178B rules (legacy, not used by current MISRA pipeline) |
| `app/data/ingest.py` | `parse_misra_file()` → `upload_to_mongodb()` → Pinecone embedding; called by `POST /seed` |
| `data/misra_c_2023__headlines_for_cppcheck.txt` | Raw MISRA C:2023 rule headlines (source of truth for ingestion) |
| `architecture-app.md` | Intended directory tree (reference only; some test files listed are not yet created) |
| `tests/unit/graph/nodes/test_rag.py` | Only existing test file |

---

## ComplianceState Schema (`app/models/state.py`)

```python
# Input
query: str
code_snippet: str
standard: str                          # hardcoded to "MISRA C:2023" by orchestrator_node

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
critique_history: Annotated[list[CritiqueEntry], operator.add]  # append-only (never written to currently)

# Final
final_response: str
error: str
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Pings MongoDB and Pinecone; returns `status: healthy\|degraded` |
| `POST` | `/api/v1/query` | Main inference endpoint — runs the full LangGraph pipeline |
| `POST` | `/api/v1/seed` | Parses `data/misra_c_2023__headlines_for_cppcheck.txt` → ingests into MongoDB + Pinecone |

### `/query` request shape
```json
{
  "query": "Does this code handle memory allocation safely?",
  "code_snippet": "char *p = malloc(n);",
  "standard": "MISRA C:2023"
}
```

---

## Environment Variables (`.env`)

Only three variables need to be set manually; all others fall back to `Settings` defaults:

```
# Required
GEMINI_API_KEY=
PINECONE_API_KEY=
MONGODB_URI=

# Optional (defaults shown)
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
PINECONE_INDEX_NAME=compliance-rules
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
MONGODB_DATABASE=compliance_db
MONGODB_COLLECTION=rules
```

> **Note:** `.env` must NOT be committed. Add it to `.gitignore`. Create a `.env.example` with empty values as a reference.

---

## Development Conventions

- **Async everywhere:** all service calls and route handlers must be `async`. Exception: `validation_node` and `critique_node` are currently synchronous.
- **Structured LLM output:** use `get_structured_llm(schema)` for Pydantic-validated output (orchestrator). Use `parse_json_response()` + `json.loads()` for freeform JSON from other nodes.
- **Service singletons:** module-level getters (`_get_db()`, `_get_index()`, `_service_instance`). Never instantiate services inside node functions. Exception: `get_llm()` creates a new instance each call.
- **Pydantic for I/O:** request/response schemas in `app/models/requests.py` and `app/models/responses.py` only.
- **One node per file** under `app/graph/nodes/`. `assemble_node` is an exception — defined inline in `builder.py`.
- **State keys must match `ComplianceState`:** nodes returning unknown keys are silently ignored by LangGraph.
- **MISRA ID format:** Pinecone vector IDs use `MISRA_{section}.{rule_number}`. MongoDB stores `section` (int) and `rule_number` (int) as separate fields with a compound unique index.
- **Keep costs free:** Gemini free tier, Pinecone free tier (1 index), MongoDB Atlas M0.

---

## Known Gaps / Open Issues

| Issue | Location | Detail |
|---|---|---|
| `Dir X.Y` directives skipped | `app/data/ingest.py` | Regex only matches `Rule X.Y`; all MISRA Directives are silently dropped during ingestion |
| `critique_history` never written | `critique.py`, `validation.py` | `ComplianceState` field and `ComplianceQueryResponse.critique_history` exist but no node populates them |
| `GET /rules` endpoint missing | `app/api/routes.py` | Listed in early docs, never implemented |
| Test suite incomplete | `tests/unit/` | Only `test_rag.py` exists; `pytest`/`pytest-asyncio` absent from `requirements.txt`; `test_rag.py` patches the wrong MongoDB function |
| `IngestRuleRequest.dal_level` | `app/models/requests.py` | DO-178B-specific field (`pattern="^[A-E]$"`); unused by current MISRA ingestion |

---

## Running the App

```bash
uvicorn main:app --reload
```

Seed the knowledge base (run once after first setup):
```bash
curl -X POST http://localhost:8000/api/v1/seed
```

Health check:
```bash
curl http://localhost:8000/api/v1/health
```
