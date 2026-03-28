from motor.motor_asyncio import AsyncIOMotorClient
from app.config import get_settings

_client = None
_db = None

_INDEX_FIELDS = [("section", 1), ("rule_number", 1)]


def _get_db():
    global _client, _db
    if _db is None:
        settings = get_settings()
        _client = AsyncIOMotorClient(settings.mongodb_uri)
        _db = _client[settings.mongodb_database]
    return _db


async def get_rules_collection():
    db = _get_db()
    return db[get_settings().mongodb_collection]


async def get_rules_by_ids(rule_ids: list[str]) -> list[dict]:
    coll = await get_rules_collection()
    cursor = coll.find({"rule_id": {"$in": rule_ids}}, {"_id": 0})
    return await cursor.to_list(length=100)


async def get_rules_by_metadata(filters: dict) -> list[dict]:
    coll = await get_rules_collection()
    cursor = coll.find(filters, {"_id": 0}).hint(_INDEX_FIELDS)
    return await cursor.to_list(length=100)


async def insert_rules(rules: list[dict]) -> None:
    coll = await get_rules_collection()
    if rules:
        await coll.insert_many(rules)


async def create_indexes() -> None:
    coll = await get_rules_collection()
    await coll.create_index(_INDEX_FIELDS, unique=True)
