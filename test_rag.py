# test_rag.py

import asyncio
from pprint import pprint
from dotenv import load_dotenv

# Load environment variables (API keys for Gemini, Pinecone, Mongo)
load_dotenv()

from app.graph.nodes.rag import rag_node
from app.models.state import ComplianceState

async def test_rag_execution():
    print("Initializing dummy state...")
    
    # Create a mock LangGraph state
    # We only fill in the fields that rag.py actually uses for now.
    dummy_state: ComplianceState = {
        "query": "Are goto statements allowed in MISRA-C?",
        "standard": "MISRA-C",
        "code_snippet": "",
        "intent": "search",
        "orchestrator_reasoning": "",
        "retrieved_rules": [],
        "rag_query_used": "",
        "metadata_filters_applied": {},
        "validation_result": "",
        "is_compliant": False,
        "confidence_score": 0.0,
        "cited_rules": [],
        "critique_feedback": "",
        "critique_approved": False,
        "iteration_count": 0,
        "max_iterations": 3,
        "critique_history": [],
        "final_response": "",
        "error": ""
    }

    print(f"Executing RAG node for query: '{dummy_state['query']}'\n")
    
    # Run the node
    result_update = await rag_node(dummy_state)

    print("--- RAG NODE OUTPUT ---")
    print(f"Query Used: {result_update.get('rag_query_used')}")
    print(f"Filters Applied: {result_update.get('metadata_filters_applied')}")
    print(f"\nRetrieved {len(result_update.get('retrieved_rules', []))} rules:\n")

    # Print the retrieved rules elegantly
    for i, rule in enumerate(result_update.get("retrieved_rules", [])):
        print(f"[{i+1}] Rule ID: {rule['rule_id']} (Score: {rule['relevance_score']:.4f})")
        print(f"    Title: {rule['title']}")
        # Print a snippet of the full text to verify MongoDB retrieval worked
        text_snippet = rule['full_text'][:150].replace('\n', ' ') + "..."
        print(f"    Text : {text_snippet}\n")

if __name__ == "__main__":
    # Run the async test function
    asyncio.run(test_rag_execution())