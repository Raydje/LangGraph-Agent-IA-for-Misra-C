"""
Seed data ingestion script.
Run with: python -m app.data.ingest
"""
import re
import sys
import os
import json
import asyncio
from pathlib import Path
from pymongo import ReplaceOne

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.data.seed_rules import SEED_RULES
from app.services.pinecone_service import upsert_vectors
from app.services.embedding_service import EmbeddingService
from app.services.mongodb_service import insert_rules, create_indexes, get_rules_collection


def parse_misra_file(filepath: str) -> list[dict]:
    """
    Parses the MISRA-C text file and extracts structured metadata.
    """
    rules = []
    
    # Resolve the path relative to this script
    # app/data/ingest.py -> project root is 3 levels up
    base_dir = Path(__file__).resolve().parent.parent.parent
    file_path = base_dir / filepath
    
    print(f"📂 Attempting to read file: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")
        return []

    current_rule = None
    
    # Regex to match lines like: "Rule 1.1    Required" or "Rule 22.15\tMandatory"
    header_pattern = re.compile(r'^Rule\s+(\d+)\.(\d+)\s+(.+)$')

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
            section = int(header_match.group(1))
            rule_number = int(header_match.group(2))
            category = header_match.group(3).strip()
            
            # Initialize the new rule object
            current_rule = {
                "scope": "MISRA C:2023",
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


async def upload_to_mongodb(rules: list[dict]):
    """Uploads parsed rules to MongoDB asynchronously."""
    if not rules:
        print("No rules to upload.")
        return

    print("Connecting to MongoDB Atlas...")
    collection = await get_rules_collection()
    await create_indexes("section", "rule_number")

    print(f"Preparing to insert/update {len(rules)} rules...")

    operations = []
    for rule in rules:
        query = {"section": rule["section"], "rule_number": rule["rule_number"]}
        operations.append(ReplaceOne(query, rule, upsert=True))

    if operations:
        result = await collection.bulk_write(operations)
        print(f"✅ Successfully processed {len(rules)} rules in MongoDB!")
        print(f"   - Inserted: {result.upserted_count}")
        print(f"   - Modified: {result.modified_count}")

async def main():
    # Test the parser
    target_file = "data/misra_c_2023__headlines_for_cppcheck.txt"
    parsed_rules = parse_misra_file(target_file)
    
    if parsed_rules:
        print(f"✅ Successfully parsed {len(parsed_rules)} rules!\n")
        print("Sample of the first parsed rule:")
        print(json.dumps(parsed_rules[0], indent=2))
        
        # You can also verify the last rule to ensure the loop finished correctly
        print("\nSample of the last parsed rule:")
        print(json.dumps(parsed_rules[-1], indent=2))
        
        # Now upload to MongoDB
        print("2. Uploading to MongoDB...")
        await upload_to_mongodb(parsed_rules)
        
        print("3. Uploading to Pinecone...")
        embedder = EmbeddingService()
        # Add 'await' here!
        await embedder.embed_and_store(parsed_rules)
        
if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())        