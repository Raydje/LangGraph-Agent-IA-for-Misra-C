from motor.motor_asyncio import AsyncIOMotorClient
import re
from app.config import get_settings

_client = None
_db = None

_INDEX_FIELDS = [("rule_type", 1), ("section", 1), ("rule_number", 1)]


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
    Resolves Pinecone IDs (e.g. 'MISRA_RULE_15.1', 'MISRA_DIR_4.1') to MongoDB documents.
    MongoDB stores rules with separate 'rule_type', 'section' and 'rule_number' int fields.
    Returns each doc annotated with a 'rule_id' key matching the original Pinecone ID.
    """
    _ID_RE = re.compile(r'^MISRA_(RULE|DIR)_(\d+)\.(\d+)$')
    or_conditions = []
    id_map: dict[tuple, str] = {}  # (rule_type, section, rule_number) -> original Pinecone ID

    for rid in rule_ids:
        m = _ID_RE.match(rid)
        if m:
            rule_type, section, rule_number = m.group(1), int(m.group(2)), int(m.group(3))
            or_conditions.append({"rule_type": rule_type, "section": section, "rule_number": rule_number})
            id_map[(rule_type, section, rule_number)] = rid

    if not or_conditions:
        return []

    coll = await get_rules_collection()
    cursor = coll.find({"$or": or_conditions}, {"_id": 0})
    docs = await cursor.to_list(length=100)

    for doc in docs:
        key = (doc.get("rule_type"), doc.get("section"), doc.get("rule_number"))
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
    try:
        await coll.drop_index("section_1_rule_number_1")
    except Exception:
        pass  # Index doesn't exist — that's fine
    await coll.create_index(_INDEX_FIELDS, unique=True)
