MyProjectCv/
├── main.py                       # FastAPI entry → uvicorn main:app --reload
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
│
├── app/
│   ├── config.py                 # Pydantic Settings (.env loader)
│   ├── utils.py                  # structlog logger initialization
│   ├── models_pricing.py         # Gemini model pricing table
│   │
│   ├── auth/                     # Authentication sub-package
│   │   ├── models.py             # User, Token, API Key domain models
│   │   ├── service.py            # Crypto primitives, JWT & API Key logic
│   │   ├── dependencies.py       # Dual-token JWT + API key resolver
│   │   └── router.py             # Auth endpoints (/api/v1/auth)
│   │
│   ├── models/
│   │   └── state.py              # ComplianceState TypedDict (LangGraph state)
│   │
│   ├── graph/
│   │   ├── builder.py            # StateGraph wiring + assemble_node (inline)
│   │   ├── edges.py              # Conditional routing logic
│   │   └── nodes/
│   │       ├── orchestrator.py   # Intent classifier (search/validate/explain)
│   │       ├── rag.py            # Hybrid search: Pinecone + MongoDB
│   │       ├── validation.py     # LLM compliance check (temp=0.1)
│   │       ├── critique.py       # Hallucination reviewer (temp=0.0)
│   │       └── remedier.py       # Remediation suggester (temp=0.2)
│   │
│   ├── services/
│   │   ├── llm_service.py        # Gemini / LangChain wrappers
│   │   ├── embedding_service.py  # gemini-embedding-001
│   │   ├── pinecone_service.py   # Vector DB operations
│   │   ├── mongodb_service.py    # Rule storage + Checkpointing support
│   │   ├── usage_service.py      # Token/Cost tracking service
│   │   └── service_container.py  # Singleton service registry
│   │
│   ├── api/
│   │   ├── dependencies.py       # Graph & DB dependency injection
│   │   ├── rate_limit.py         # Redis-backed rate limiter
│   │   └── v1/
│   │       ├── routes.py         # Main API endpoints (/query, /replay, /history)
│   │       ├── requests.py       # API Pydantic request models
│   │       └── responses.py      # API Pydantic response models
│   │
│   └── data/
│       └── ingest.py             # MISRA ingestion pipeline
│
├── data/
│   ├── misra_c_2023__headlines_for_cppcheck.txt
│   └── golden_dataset.json
│
├── deploy/
│   └── k8s/                      # Kubernetes resources
│
└── tests/
    ├── unit/                     # Isolated component tests
    ├── integration/              # Contract tests for external services
    └── non_regression/           # Golden dataset E2E tests


---

## Auth System

### Strategy: Dual-Token (API Key + JWT)

All endpoints except `GET /health` and the `/auth/*` registration/login group require authentication via:

```
Authorization: Bearer <token>
```

The dependency `get_current_principal` (in `app/auth/dependencies.py`) detects the token type at runtime:
- Token starts with `ak_` → **API key path**: DB lookup by `key_id` + bcrypt verify of secret
- Anything else → **JWT path**: stateless HS256 signature + expiry verify

### Scopes (RBAC)

| Scope | Grants access to |
|---|---|
| `query:read` | `POST /query`, `GET /history/{thread_id}` |
| `admin:seed` | `POST /seed` |
| `admin:replay` | `POST /replay/{thread_id}/{checkpoint_id}` |
| `admin:all` | All of the above (wildcard — satisfies any scope check) |

### Auth Endpoints (`/api/v1/auth`)

| Method | Path | Auth required | Description |
|---|---|---|---|
| `POST` | `/auth/register` | No | Create account; add `admin_token` to get admin scopes |
| `POST` | `/auth/token` | No | OAuth2 password flow → access token (15 min) + refresh token (30 days) |
| `POST` | `/auth/refresh` | No | Rotate refresh token — old token revoked, new pair issued |
| `POST` | `/auth/api-keys` | Yes | Generate an API key (secret shown once) |
| `GET` | `/auth/api-keys` | Yes | List caller's active API keys |
| `DELETE` | `/auth/api-keys/{key_id}` | Yes | Soft-revoke an API key |

### MongoDB Collections

**`users`**
```json
{
  "_id": "uuid",
  "email": "unique",
  "hashed_password": "bcrypt",
  "scopes": ["query:read"],
  "is_active": true,
  "refresh_tokens": [{"token": "...", "issued_at": "ISO8601"}],
  "created_at": "ISO8601"
}
```
Index: `email` (unique)

**`api_keys`**
```json
{
  "key_id": "8 hex chars",
  "name": "human-readable label",
  "key_hash": "bcrypt hash of secret portion",
  "user_id": "ref → users._id",
  "scopes": ["query:read"],
  "expires_at": "ISO8601 | null",
  "is_active": true,
  "last_used_at": "ISO8601 | null",
  "created_at": "ISO8601"
}
```
Indexes: `key_id`, `user_id`

### API Key Format

```
ak_<key_id>_<secret>
   └──8 hex──┘ └──43 url-safe base64 chars──┘
```

`key_id` is stored plaintext — used for O(1) DB lookup before the expensive bcrypt verification.
`secret` is bcrypt-hashed — never stored in plaintext, shown to the caller only once.

### Privilege Escalation Prevention

When creating an API key, requested scopes are intersected with the caller's own scopes.
A `query:read` user cannot create a key with `admin:seed` scope.

### Admin Registration

Set `ADMIN_REGISTRATION_TOKEN` in `.env` to a strong random value.
Include `{"admin_token": "<value>"}` in the `POST /auth/register` body to receive scopes:
`["query:read", "admin:seed", "admin:replay", "admin:all"]`
