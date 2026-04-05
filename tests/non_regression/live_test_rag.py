import asyncio
from dotenv import load_dotenv

# Ensure environment variables are loaded before importing app modules
load_dotenv()

# Import your actual code
from app.graph.nodes.rag import rag_node
from app.models.state import ComplianceState
from app.services.mongodb_service import MongoDBService
from app.services.pinecone_service import PineconeService
from app.services.embedding_service import EmbeddingService # Internal helper for singleton

async def test_rag_node_live():
    print("🚀 Initializing Services for Live RAG Test...")
    
    # 1. Initialize real services (Ensure your .env is loaded)
    mongo_db = MongoDBService()
    pinecone_svc = PineconeService()
    embedding_svc = EmbeddingService()

    # 2. Prepare the Initial State
    # This mimics the state passed by the Orchestrator node
    sample_state: ComplianceState = {
        "query": "Is it allowed to use the goto statement in C?",
        "standard": "MISRA C:2023",
        "code_snippet": "void func() { goto label; label: return; }",
        "retrieved_rules": [],
        "rag_query_used": "",
        "metadata_filters_applied": {}
    }

    # 3. Mock the LangGraph Config
    # In production, this is usually populated in your FastAPI routes/dependencies
    config = {
        "configurable": {
            "mongo_db": mongo_db,
            "pinecone_service": pinecone_svc,
            "embedding_service": embedding_svc
        }
    }

    print(f"🔍 Testing Query: '{sample_state['query']}'")
    
    try:
        # 4. Execute the Node
        result = await rag_node(sample_state, config)

        # 5. Analyze Results
        print("\n✅ RAG Node Execution Successful!")
        print("-" * 50)
        print(f"Used Query: {result.get('rag_query_used')}")
        print(f"Filters Applied: {result.get('metadata_filters_applied')}")
        print(f"Rules Retrieved: {len(result.get('retrieved_rules', []))}")
        
        for i, rule in enumerate(result['retrieved_rules']):
            print(f"\n[Match #{i+1}] Score: {rule['relevance_score']:.4f}")
            print(f"ID: {rule['rule_id']} | Category: {rule['category']}")
            print(f"Title: {rule['title']}")
            # Print first 100 chars of text
            print(f"Text: {rule['full_text'][:100]}...")

    except Exception as e:
        print(f"❌ Error during RAG node test: {e}")
    finally:
        # Close connections
        mongo_db.close()
        pinecone_svc.index.close()
if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_rag_node_live())