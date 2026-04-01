# app/graph/nodes/rag.py

from typing import Any
from app.models.state import ComplianceState, RetrievedRule
from app.services.embedding_service import get_embedding
from app.services.pinecone_service import query_pinecone
from app.services.mongodb_service import get_misra_rules_by_pinecone_ids
from app.utils import logger

async def rag_node(state: ComplianceState) -> dict[str, Any]:
    """
    LangGraph node responsible for hybrid retrieval.
    Currently focused strictly on fetching MISRA C:2023 rules using Pinecone (vector search)
    and MongoDB (document retrieval).
    """
    logger.info("RAG_node invoked", query=state.get("query", ""))
    query = state.get("query", "")
    
    # 1. Embed the user's query
    vector = await get_embedding(query)
    logger.info("RAG_node - query embedded")
    # 2. Build metadata filters strictly for MISRA C:2023
    # Based on your ingest script, we lock the scope to MISRA C:2023 
    # so we don't accidentally retrieve any other standards later.
    metadata_filters = {"scope": "MISRA C:2023"}

    # 3. Query Pinecone for top K matches (semantic search)
    # Fetching the top 5 most relevant MISRA C:2023 rules
    pinecone_results = await query_pinecone(
        vector=vector,
        top_k=5,
        filter=metadata_filters
    )

    matches = pinecone_results.get("matches", [])
    # Extract the IDs and keep track of their relevance scores
    rule_ids = [match["id"] for match in matches]
    scores_map = {match["id"]: match.get("score", 0.0) for match in matches}

    retrieved_rules: list[RetrievedRule] = []
    logger.info(f"RAG_node - retrieved {len(rule_ids)} matching IDs from Pinecone", rule_ids=rule_ids)
    if rule_ids:
        # 4. Fetch the full MISRA C:2023 documents from MongoDB
        mongo_docs = await get_misra_rules_by_pinecone_ids(rule_ids)
        logger.info(f"RAG_node - retrieved {len(mongo_docs)} documents from MongoDB based on Pinecone IDs")
        # 5. Format the documents into the TypedDict expected by LangGraph
        for doc in mongo_docs:
            r_id = doc.get("rule_id", "")
            
            rule_entry: RetrievedRule = {
                "rule_id": r_id,
                "standard": "MISRA C:2023", # Hardcoded for now
                "section": doc.get("section", ""),
                "category": doc.get("category", "N/A"),
                "title": doc.get("title", f"Rule {r_id}"),
                "full_text": doc.get("full_text", doc.get("text", "")),
                "relevance_score": scores_map.get(r_id, 0.0)
            }
            retrieved_rules.append(rule_entry)

        # Ensure the final list is sorted by relevance score descending 
        retrieved_rules.sort(key=lambda x: x["relevance_score"], reverse=True)
    logger.info(f"RAG_node - formatted {len(retrieved_rules)} retrieved rules for state update")
    # 6. Return the state update dictionary.
    return {
        "retrieved_rules": retrieved_rules,
        "rag_query_used": query,
        "metadata_filters_applied": metadata_filters
    }