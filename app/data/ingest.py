"""
Seed data ingestion script.
Run with: python -m app.data.ingest
"""
import re
import sys
import os
import json
import asyncio
from typing import Any
from pathlib import Path
from pymongo import ReplaceOne

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services.embedding_service import EmbeddingService
from app.services.mongodb_service import MongoDBService
from app.services.pinecone_service import PineconeService
from app.services.service_container import create_service_container
from app.utils import logger


def parse_misra_file(filepath: str) -> list[dict]:
    """
    Parses the MISRA C:2023 text file and extracts structured metadata.
    """
    rules = []
    
    # Resolve the path relative to this script
    # app/data/ingest.py -> project root is 3 levels up
    base_dir = Path(__file__).resolve().parent.parent.parent
    file_path = base_dir / filepath
    
    logger.info("📂 Attempting to read file", file_path=file_path)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        logger.error("Error: Could not find file", file_path=file_path)
        return []

    current_rule: dict[str, Any] | None = None

    # Regex to match lines like: "Rule 1.1    Required", "Dir 4.1\tRequired", or "Rule 22.15\tMandatory"
    header_pattern = re.compile(r'^(Rule|Dir)\s+(\d+)\.(\d+)\s+(.+)$')

    for line in lines:
        line = line.strip()

        # Skip comments and empty lines
        if not line or line.startswith('#'):
            continue

        header_match = header_pattern.match(line)

        if header_match:
            # If we were already building a rule, save it before starting a new one
            if current_rule and current_rule.get('full_text'):
                rules.append(current_rule)

            # Extract metadata using regex groups
            rule_type = header_match.group(1).upper()   # "RULE" or "DIR"
            section = int(header_match.group(2))
            rule_number = int(header_match.group(3))
            category = header_match.group(4).strip()

            # Initialize the new rule object
            current_rule = {
                "scope": "MISRA C:2023",
                "rule_type": rule_type,
                "section": section,
                "rule_number": rule_number,
                "category": category,
                "full_text": ""
            }
        elif current_rule:
            # If it's not a header, it must be the rule text.
            # Append it to the current rule's full_text.
            if current_rule["full_text"]: #multiple lines of text for a single rule
                current_rule["full_text"] += " " + line
            else:
                current_rule["full_text"] = line
                
    # Don't forget to add the very last rule in the file!
    if current_rule and current_rule.get('full_text'):
        rules.append(current_rule)
        
    return rules


async def upload_to_mongodb(rules: list[dict], svc: MongoDBService):
    """Uploads parsed rules to MongoDB asynchronously."""
    if not rules:
        logger.warning("No rules to upload.")
        return

    logger.info("Connecting to MongoDB Atlas...")
    try:
        await svc.create_indexes()
    except Exception as e:
        logger.error("Error creating indexes in MongoDB")
        return

    logger.info("Preparing to insert/update rules...", number_of_rules=len(rules))

    operations = []
    for rule in rules:
        query = {"rule_type": rule["rule_type"], "section": rule["section"], "rule_number": rule["rule_number"]}
        operations.append(ReplaceOne(query, rule, upsert=True))

    if operations:
        result = await svc.collection.bulk_write(operations)
        logger.info("✅ Successfully processed rules in MongoDB!", number_of_rules=len(rules))
        logger.info("   - Inserted:", number_inserted=result.upserted_count)
        logger.info("   - Modified:", number_modified=result.modified_count)

async def main(mongodb: MongoDBService, pinecone: PineconeService, embedder: EmbeddingService) -> dict:
    # Test the parser
    target_file = "data/misra_c_2023__headlines_for_cppcheck.txt"
    parsed_rules = parse_misra_file(target_file)

    if parsed_rules:
        logger.info("✅ Successfully parsed rules!", number_of_rules=len(parsed_rules))
        logger.info("Sample of the first parsed rule:")
        logger.info(json.dumps(parsed_rules[0], indent=2))

        # You can also verify the last rule to ensure the loop finished correctly
        logger.info("\nSample of the last parsed rule:")
        logger.info(json.dumps(parsed_rules[-1], indent=2))

        # Now upload to MongoDB
        logger.info("2. Uploading to MongoDB...")
        await upload_to_mongodb(parsed_rules, mongodb)

        logger.info("3. Uploading to Pinecone...")
        # Add 'await' here!
        vectors_upserted = await embedder.embed_and_store(parsed_rules, pinecone)

        return {"rules_ingested": len(parsed_rules), "vectors_upserted": vectors_upserted}

    return {"rules_ingested": 0, "vectors_upserted": 0}


async def run_ingest_cli() -> None:
    """CLI entry point — manages service lifecycle via the centralised container."""
    async with create_service_container() as container:
        result = await main(
            mongodb=container.mongodb,
            pinecone=container.pinecone,
            embedder=container.embedding,
        )
        logger.info("Ingestion complete", **result)

if __name__ == "__main__":  # pragma: no cover
    asyncio.run(run_ingest_cli())