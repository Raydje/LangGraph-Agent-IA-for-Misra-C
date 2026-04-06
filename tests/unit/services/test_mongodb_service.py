import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from pymongo.errors import PyMongoError
from app.services.mongodb_service import MongoDBService, MongoDBCheckpointService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_collection(docs: list[dict]) -> MagicMock:
    """Return a Motor collection mock whose find().to_list() returns docs."""
    coll = MagicMock()
    cursor = MagicMock()
    cursor.to_list = AsyncMock(return_value=docs)
    coll.find.return_value = cursor
    return coll


def _make_service(docs: list[dict]) -> MongoDBService:
    """Create a MongoDBService with a mocked collection (bypasses __init__)."""
    svc = object.__new__(MongoDBService)
    svc.collection = _make_mock_collection(docs)
    return svc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_valid_ids_return_annotated_docs():
    docs = [
        {"rule_type": "RULE", "section": 15, "rule_number": 1, "title": "No recursion"},
        {"rule_type": "RULE", "section": 1, "rule_number": 3, "title": "Undefined behaviour"},
    ]
    svc = _make_service(docs)
    result = await svc.get_misra_rules_by_pinecone_ids(["MISRA_RULE_15.1", "MISRA_RULE_1.3"])

    assert len(result) == 2
    rule_ids = {doc["rule_id"] for doc in result}
    assert "MISRA_RULE_15.1" in rule_ids
    assert "MISRA_RULE_1.3" in rule_ids


@pytest.mark.asyncio
async def test_dir_ids_return_annotated_docs():
    docs = [
        {"rule_type": "DIR", "section": 4, "rule_number": 1, "title": "Run-time failures shall be minimized"},
    ]
    svc = _make_service(docs)
    result = await svc.get_misra_rules_by_pinecone_ids(["MISRA_DIR_4.1"])

    assert len(result) == 1
    assert result[0]["rule_id"] == "MISRA_DIR_4.1"


@pytest.mark.asyncio
async def test_empty_ids_returns_empty_without_db_call():
    svc = _make_service([])
    result = await svc.get_misra_rules_by_pinecone_ids([])

    svc.collection.find.assert_not_called()
    assert result == []


@pytest.mark.asyncio
async def test_all_malformed_ids_skips_db():
    svc = _make_service([])
    result = await svc.get_misra_rules_by_pinecone_ids(["badformat", "MISRA_abc.def", "MISRA_"])

    svc.collection.find.assert_not_called()
    assert result == []


@pytest.mark.asyncio
async def test_mixed_valid_and_malformed_ids_only_queries_valid():
    docs = [{"rule_type": "RULE", "section": 1, "rule_number": 1, "title": "T1"}]
    svc = _make_service(docs)
    result = await svc.get_misra_rules_by_pinecone_ids(["MISRA_RULE_1.1", "bad_id", "MISRA_abc.1"])

    assert len(result) == 1
    assert result[0]["rule_id"] == "MISRA_RULE_1.1"


@pytest.mark.asyncio
async def test_rule_id_annotated_on_returned_doc():
    doc = {"rule_type": "RULE", "section": 15, "rule_number": 5, "title": "No goto"}
    svc = _make_service([doc])
    result = await svc.get_misra_rules_by_pinecone_ids(["MISRA_RULE_15.5"])

    assert result[0]["rule_id"] == "MISRA_RULE_15.5"


@pytest.mark.asyncio
async def test_find_called_with_correct_or_conditions():
    doc = {"rule_type": "RULE", "section": 15, "rule_number": 1, "title": "T"}
    svc = _make_service([doc])
    await svc.get_misra_rules_by_pinecone_ids(["MISRA_RULE_15.1"])

    call_filter = svc.collection.find.call_args[0][0]
    assert "$or" in call_filter
    assert {"rule_type": "RULE", "section": 15, "rule_number": 1} in call_filter["$or"]


@pytest.mark.asyncio
async def test_multiple_ids_build_multiple_or_conditions():
    docs = [
        {"rule_type": "RULE", "section": 1, "rule_number": 1},
        {"rule_type": "RULE", "section": 2, "rule_number": 3},
    ]
    svc = _make_service(docs)
    await svc.get_misra_rules_by_pinecone_ids(["MISRA_RULE_1.1", "MISRA_RULE_2.3"])

    call_filter = svc.collection.find.call_args[0][0]
    assert len(call_filter["$or"]) == 2


@pytest.mark.asyncio
async def test_excludes_id_field_from_projection():
    doc = {"rule_type": "RULE", "section": 1, "rule_number": 1}
    svc = _make_service([doc])
    await svc.get_misra_rules_by_pinecone_ids(["MISRA_RULE_1.1"])

    projection = svc.collection.find.call_args[0][1]
    assert projection == {"_id": 0}


# ---------------------------------------------------------------------------
# MongoDBCheckpointService.__init__ and close
# ---------------------------------------------------------------------------

def test_checkpoint_service_init_stores_client_db_collection():
    mock_settings = MagicMock()
    mock_settings.mongodb_uri = "mongodb://localhost:27017"
    mock_settings.mongodb_database = "testdb"
    mock_settings.mongodb_checkpoints_collection = "checkpoints"

    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_client.__getitem__ = MagicMock(return_value=mock_db)
    mock_db.__getitem__ = MagicMock(return_value=mock_collection)

    with patch("app.services.mongodb_service.get_settings", return_value=mock_settings), \
         patch("app.services.mongodb_service.MongoClient", return_value=mock_client):
        svc = MongoDBCheckpointService()

    assert svc.client is mock_client
    assert svc.db is mock_client[mock_settings.mongodb_database]
    assert svc.collection is mock_db[mock_settings.mongodb_checkpoints_collection]


def test_checkpoint_service_close_calls_client_close():
    svc = object.__new__(MongoDBCheckpointService)
    svc.client = MagicMock()
    svc.close()
    svc.client.close.assert_called_once()


# ---------------------------------------------------------------------------
# MongoDBService.__init__ and close
# ---------------------------------------------------------------------------

def test_mongodb_service_init_stores_client_db_collection():
    mock_settings = MagicMock()
    mock_settings.mongodb_uri = "mongodb://localhost:27017"
    mock_settings.mongodb_database = "testdb"
    mock_settings.mongodb_collection = "rules"
    mock_settings.mongodb_timeout = 5

    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_client.__getitem__ = MagicMock(return_value=mock_db)
    mock_db.__getitem__ = MagicMock(return_value=mock_collection)

    with patch("app.services.mongodb_service.get_settings", return_value=mock_settings), \
         patch("app.services.mongodb_service.AsyncIOMotorClient", return_value=mock_client) as mock_motor:
        svc = MongoDBService()

    mock_motor.assert_called_once_with(
        mock_settings.mongodb_uri,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        socketTimeoutMS=5000,
    )
    assert svc.client is mock_client


def test_mongodb_service_close_calls_client_close():
    svc = object.__new__(MongoDBService)
    svc.client = MagicMock()
    svc.close()
    svc.client.close.assert_called_once()


# ---------------------------------------------------------------------------
# MongoDBService.get_rules_by_ids
# ---------------------------------------------------------------------------

async def test_get_rules_by_ids_happy_path():
    docs = [{"rule_id": "MISRA_RULE_1.1", "title": "No dead code"}]
    svc = object.__new__(MongoDBService)
    svc.collection = _make_mock_collection(docs)

    result = await svc.get_rules_by_ids(["MISRA_RULE_1.1"])

    assert result == docs
    svc.collection.find.assert_called_once_with(
        {"rule_id": {"$in": ["MISRA_RULE_1.1"]}}, {"_id": 0}
    )


async def test_get_rules_by_ids_pymongo_error_returns_empty():
    svc = object.__new__(MongoDBService)
    coll = MagicMock()
    cursor = MagicMock()
    cursor.to_list = AsyncMock(side_effect=PyMongoError("connection refused"))
    coll.find.return_value = cursor
    svc.collection = coll

    result = await svc.get_rules_by_ids(["MISRA_RULE_1.1"])

    assert result == []


# ---------------------------------------------------------------------------
# MongoDBService.get_misra_rules_by_pinecone_ids — PyMongoError branch
# ---------------------------------------------------------------------------

async def test_get_misra_rules_by_pinecone_ids_pymongo_error_returns_empty():
    svc = object.__new__(MongoDBService)
    coll = MagicMock()
    cursor = MagicMock()
    cursor.to_list = AsyncMock(side_effect=PyMongoError("timeout"))
    coll.find.return_value = cursor
    svc.collection = coll

    result = await svc.get_misra_rules_by_pinecone_ids(["MISRA_RULE_1.1"])

    assert result == []


# ---------------------------------------------------------------------------
# MongoDBService.get_rules_by_metadata
# ---------------------------------------------------------------------------

async def test_get_rules_by_metadata_happy_path():
    docs = [{"section": 3, "rule_number": 4, "title": "Rule 3.4"}]
    svc = object.__new__(MongoDBService)
    svc.collection = _make_mock_collection(docs)

    result = await svc.get_rules_by_metadata({"section": 3, "rule_number": 4})

    assert result == docs
    svc.collection.find.assert_called_once_with({"section": 3, "rule_number": 4}, {"_id": 0})


async def test_get_rules_by_metadata_pymongo_error_returns_empty():
    svc = object.__new__(MongoDBService)
    coll = MagicMock()
    cursor = MagicMock()
    cursor.to_list = AsyncMock(side_effect=PyMongoError("network error"))
    coll.find.return_value = cursor
    svc.collection = coll

    result = await svc.get_rules_by_metadata({"section": 1})

    assert result == []


# ---------------------------------------------------------------------------
# MongoDBService.insert_rules
# ---------------------------------------------------------------------------

async def test_insert_rules_calls_insert_many_when_non_empty():
    svc = object.__new__(MongoDBService)
    svc.collection = MagicMock()
    svc.collection.insert_many = AsyncMock()

    rules = [{"rule_id": "MISRA_RULE_1.1"}]
    await svc.insert_rules(rules)

    svc.collection.insert_many.assert_called_once_with(rules)


async def test_insert_rules_skips_insert_when_empty():
    svc = object.__new__(MongoDBService)
    svc.collection = MagicMock()
    svc.collection.insert_many = AsyncMock()

    await svc.insert_rules([])

    svc.collection.insert_many.assert_not_called()


# ---------------------------------------------------------------------------
# MongoDBService.create_indexes
# ---------------------------------------------------------------------------

async def test_create_indexes_drops_old_index_then_creates_new():
    svc = object.__new__(MongoDBService)
    svc.collection = MagicMock()
    svc.collection.drop_index = AsyncMock()
    svc.collection.create_index = AsyncMock()

    await svc.create_indexes()

    svc.collection.drop_index.assert_called_once_with("section_1_rule_number_1")
    svc.collection.create_index.assert_called_once()


async def test_create_indexes_continues_if_drop_raises():
    """drop_index raising should be silently swallowed (index may not exist)."""
    svc = object.__new__(MongoDBService)
    svc.collection = MagicMock()
    svc.collection.drop_index = AsyncMock(side_effect=Exception("index not found"))
    svc.collection.create_index = AsyncMock()

    await svc.create_indexes()  # must not raise

    svc.collection.create_index.assert_called_once()
