MyProjectCv/
├── .env                          # GEMINI_API_KEY + PINECONE_API_KEY
├── .gitignore
├── requirements.txt
├── main.py                       # FastAPI entry → uvicorn main:app --reload
└── app/
    ├── config.py                 # Pydantic Settings (.env loader)
    ├── utils.py                  # JSON response parser (strips markdown fences)
    ├── models/
    │   ├── state.py              # ComplianceState TypedDict (LangGraph state)
    │   ├── requests.py           # ComplianceQueryRequest, IngestRuleRequest
    │   └── responses.py          # ComplianceQueryResponse, HealthResponse
    ├── graph/
    │   ├── builder.py            # StateGraph: 5 nodes, conditional edges, compile
    │   ├── edges.py              # route_after_rag(), should_loop_or_finish()
    │   └── nodes/
    │       ├── orchestrator.py   # Intent classifier (search/validate/explain)
    │       ├── rag.py            # Hybrid search: Pinecone + MongoDB
    │       ├── validation.py     # LLM compliance check (temp=0.1)
    │       └── critique.py       # Hallucination detector (temp=0.0, 5 criteria)
    ├── services/
    │   ├── llm_service.py        # Gemini 2.0 Flash wrapper
    │   ├── embedding_service.py  # Google embedding-001 (768 dims)
    │   ├── pinecone_service.py   # Auto-creates index, query, upsert
    │   └── mongodb_service.py    # Async motor CRUD + indexes
    ├── api/
    │   ├── routes.py             # /health, /query, /rules, /seed
    │   └── dependencies.py       # Graph DI (cached)
    └── data/
        ├── seed_rules.py         # 10 DO-178B compliance rules
        └── ingest.py             # Seeds Pinecone + MongoDB
tests/
└── unit/
    ├── services/
    │   ├── test_mongodb_service.py   # Unit tests for MongoDB CRUD functions
    │   ├── test_embedding_service.py # Unit tests for get_embedding()
    │   └── test_pinecone_service.py  # Unit tests for query/upsert
    ├── graph/
    │   ├── nodes/
    │   │   ├── test_orchestrator.py  # Intent classification tests
    │   │   ├── test_rag.py           # Hybrid retrieval tests
    │   │   ├── test_validation.py    # Compliance checker tests
    │   │   └── test_critique.py      # Hallucination reviewer tests
    │   └── test_edges.py             # Route logic tests
    └── conftest.py                   # Shared fixtures (mock DB, mock LLM, etc.)