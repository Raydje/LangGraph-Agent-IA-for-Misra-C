import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import get_settings
from app.services.pinecone_service import upsert_vectors

_service_instance = None


def _get_service() -> "EmbeddingService":
    global _service_instance
    if _service_instance is None:
        _service_instance = EmbeddingService()
    return _service_instance


async def get_embedding(text: str) -> list[float]:
    return await _get_service().embeddings.aembed_query(text)


class EmbeddingService:
    def __init__(self):
        settings = get_settings()
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=settings.gemini_api_key,
            output_dimensionality=settings.embedding_dimensions,
        )

    async def embed_and_store(self, rules: list[dict]) -> None:
        if not rules:
            return

        print(f"Generating embeddings for {len(rules)} rules...")
        
        texts = [rule["full_text"] for rule in rules]
        all_embeddings = []
        
        # We process 90 at a time to stay safely under the 100/minute limit
        batch_size = 90 
        for i in range(0, len(texts), batch_size):
            print(f"  -> Embedding rules {i} to {min(i+batch_size, len(texts))}...")
            batch_texts = texts[i : i + batch_size]
            
            batch_embeddings = self.embeddings.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            # If there are still more rules to process, we MUST wait for the minute to reset
            if i + batch_size < len(texts):
                print("  ⏳ Approaching Free Tier API limit. Sleeping for 60 seconds to reset quota...")
                time.sleep(60) 

        print("Packaging vectors...")
        
        vectors = []
        for rule, embedding_vector in zip(rules, all_embeddings):
            vector_id = f"MISRA_{rule['section']}.{rule['rule_number']}"
            
            metadata = {
                "scope": rule["scope"],
                "section": rule["section"],
                "rule_number": rule["rule_number"],
                "category": rule["category"],
                "text": rule["full_text"]
            }
            
            vectors.append({
                "id": vector_id,
                "values": embedding_vector,
                "metadata": metadata
            })

        print("Delegating upload to pinecone_service...")
        await upsert_vectors(vectors)
        print(f"✅ Successfully passed {len(vectors)} embeddings to Pinecone!")