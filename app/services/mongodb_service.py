import contextlib
import re

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from app.config import get_settings
from app.utils import logger

_INDEX_FIELDS = [("rule_type", 1), ("section", 1), ("rule_number", 1)]
_ID_RE = re.compile(r"^MISRA_(RULE|DIR)_(\d+)\.(\d+)$")


# Sync pymongo client — MongoDBSaver (langgraph-checkpoint-mongodb) requires pymongo, not Motor.
class MongoDBCheckpointService:
    def __init__(self) -> None:
        settings = get_settings()
        self.client: MongoClient = MongoClient(settings.mongodb_uri)
        self.db = self.client[settings.mongodb_database]
        self.collection: Collection = self.db[settings.mongodb_checkpoints_collection]

    def close(self) -> None:
        self.client.close()


# MongoDB service for MISRA rules storage and retrieval
class MongoDBService:
    def __init__(self) -> None:
        settings = get_settings()
        timeout_ms = settings.mongodb_timeout * 1000
        # Driver-level timeouts ensure the Motor client respects connection and
        # operation deadlines natively, rather than relying on asyncio.wait_for
        # which cancels the coroutine mid-flight and can leave the connection
        # pool in an inconsistent state.
        self.client: AsyncIOMotorClient = AsyncIOMotorClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=timeout_ms,
            connectTimeoutMS=timeout_ms,
            socketTimeoutMS=timeout_ms,
        )
        self.db = self.client[settings.mongodb_database]
        self.collection: AsyncIOMotorCollection = self.db[settings.mongodb_collection]

    def close(self) -> None:
        self.client.close()

    async def get_rules_by_ids(self, rule_ids: list[str]) -> list[dict]:
        try:
            cursor = self.collection.find({"rule_id": {"$in": rule_ids}}, {"_id": 0})
            return await cursor.to_list(length=100)
        except PyMongoError as exc:
            logger.error(
                "MongoDB query failed in get_rules_by_ids",
                error=str(exc),
            )
            return []

    async def get_misra_rules_by_pinecone_ids(self, rule_ids: list[str]) -> list[dict]:
        """
        Resolves Pinecone IDs (e.g. 'MISRA_RULE_15.1', 'MISRA_DIR_4.1') to MongoDB documents.
        MongoDB stores rules with separate 'rule_type', 'section' and 'rule_number' int fields.
        Returns each doc annotated with a 'rule_id' key matching the original Pinecone ID.
        """
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

        try:
            cursor = self.collection.find({"$or": or_conditions}, {"_id": 0})
            docs = await cursor.to_list(length=100)

            for doc in docs:
                key = (doc.get("rule_type"), doc.get("section"), doc.get("rule_number"))
                doc["rule_id"] = id_map.get(key, "")

            return docs
        except PyMongoError as exc:
            logger.error(
                "MongoDB query failed in get_misra_rules_by_pinecone_ids",
                error=str(exc),
            )
            return []

    async def get_rules_by_metadata(self, filters: dict) -> list[dict]:
        """Find MISRA rules by metadata fields.
        Example: {"section": 3, "rule_number": 4} for MISRA 3.4 (ints, not strings).
        """
        try:
            cursor = self.collection.find(filters, {"_id": 0})
            return await cursor.to_list(length=100)
        except PyMongoError as exc:
            logger.error("MongoDB query failed in get_rules_by_metadata", error=str(exc))
            return []

    async def insert_rules(self, rules: list[dict]) -> None:
        if rules:
            await self.collection.insert_many(rules)

    async def create_indexes(self) -> None:
        with contextlib.suppress(Exception):
            await self.collection.drop_index("section_1_rule_number_1")
        await self.collection.create_index(_INDEX_FIELDS, unique=True)
