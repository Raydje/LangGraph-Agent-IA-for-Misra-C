# app/graph/nodes/rag.py

from typing import Any

from langchain_core.runnables import RunnableConfig

from app.models.state import ComplianceState, RetrievedRule
from app.utils import logger


async def rag_node(state: ComplianceState, config: RunnableConfig) -> dict[str, Any]:
    """
    LangGraph node responsible for hybrid retrieval.
    Currently focused strictly on fetching MISRA C:2023 rules using Pinecone (vector search)
    and MongoDB (document retrieval).
    """

    # Get configurable dependencies from the config (passed via the route handler)
    logger.info("RAG_node - extracting dependencies from config")
    mongo_db = config["configurable"].get("mongo_db")
    pinecone_service = config["configurable"].get("pinecone_service")
    embedding_service = config["configurable"].get("embedding_service")

    if embedding_service is None:
        raise ValueError("embedding_service is not configured")
    if pinecone_service is None:
        raise ValueError("pinecone_service is not configured")
    if mongo_db is None:
        raise ValueError("mongo_db is not configured")

    logger.info("RAG_node invoked", query=state.get("query", ""))
    query = state.get("query", "")

    # 1. Embed the user's query
    vector = await embedding_service.get_embedding(query)
    logger.info("RAG_node - query embedded")

    # 2. Build metadata filters strictly for MISRA C:2023
    metadata_filters = {"scope": state.get("standard", "MISRA C:2023")}
    logger.info("RAG_NODE scope", scope=metadata_filters["scope"])

    # 3. Query Pinecone for top K matches (semantic search)
    # Fetching the top 5 most relevant rules
    pinecone_results = await pinecone_service.query(vector=vector, top_k=5, filter=metadata_filters)

    matches = pinecone_results.get("matches", [])
    # Extract the IDs and keep track of their relevance scores
    rule_ids = [match["id"] for match in matches]
    scores_map = {match["id"]: match.get("score", 0.0) for match in matches}

    retrieved_rules: list[RetrievedRule] = []
    logger.info("RAG_node - Pinecone query completed", number_of_matches=len(matches))
    logger.info("RAG_node - retrieved  matching IDs from Pinecone", rule_ids=rule_ids)
    if rule_ids:
        # 4. Fetch the full documents from MongoDB
        mongo_docs = await mongo_db.get_misra_rules_by_pinecone_ids(rule_ids)
        logger.info(
            "RAG_node - retrieved  documents from MongoDB based on Pinecone IDs", number_of_documents=len(mongo_docs)
        )
        # 5. Format the documents into the TypedDict expected by LangGraph
        for doc in mongo_docs:
            r_id = doc.get("rule_id", "")

            rule_entry: RetrievedRule = {
                "rule_id": r_id,
                "standard": "MISRA C:2023",  # Hardcoded for now
                "section": doc.get("section", ""),
                "category": doc.get("category", "N/A"),
                "title": doc.get("title", f"Rule {r_id}"),
                "full_text": doc.get("full_text", doc.get("text", "")),
                "relevance_score": scores_map.get(r_id, 0.0),
            }
            retrieved_rules.append(rule_entry)

        # Ensure the final list is sorted by relevance score descending
        retrieved_rules.sort(key=lambda x: x["relevance_score"], reverse=True)
    logger.info("RAG_node - formatted retrieved rules for state update", number_of_rules=len(retrieved_rules))
    # 6. Return the state update dictionary.
    return {"retrieved_rules": retrieved_rules, "rag_query_used": query, "metadata_filters_applied": metadata_filters}
