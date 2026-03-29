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


async def get_misra_rules_by_pinecone_ids(rule_ids: list[str]) -> list[dict]:
    """
    Resolves Pinecone IDs (e.g. 'MISRA_15.1') to MongoDB documents.
    MongoDB stores rules with separate 'section' and 'rule_number' int fields.
    Returns each doc annotated with a 'rule_id' key matching the original Pinecone ID.
    """
    or_conditions = []
    id_map: dict[tuple, str] = {}  # (section, rule_number) -> original Pinecone ID

    for rid in rule_ids:
        # Expected format: MISRA_{section}.{rule_number}
        parts = rid.removeprefix("MISRA_").split(".")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            section, rule_number = int(parts[0]), int(parts[1])
            or_conditions.append({"section": section, "rule_number": rule_number})
            id_map[(section, rule_number)] = rid

    if not or_conditions:
        return []

    coll = await get_rules_collection()
    cursor = coll.find({"$or": or_conditions}, {"_id": 0})
    docs = await cursor.to_list(length=100)

    for doc in docs:
        key = (doc.get("section"), doc.get("rule_number"))
        doc["rule_id"] = id_map.get(key, "")

    return docs


async def get_rules_by_metadata(filters: dict) -> list[dict]:
    """ find misra rules by metadata fields like section and rule_number 
        example filters: {"section": "3", "rule_number": "4"} for MISRA 3.4
    """
    coll = await get_rules_collection()
    cursor = coll.find(filters, {"_id": 0})
    return await cursor.to_list(length=100)


async def insert_rules(rules: list[dict]) -> None:
    coll = await get_rules_collection()
    if rules:
        await coll.insert_many(rules)


async def create_indexes() -> None:
    coll = await get_rules_collection()
    await coll.create_index(_INDEX_FIELDS, unique=True)
