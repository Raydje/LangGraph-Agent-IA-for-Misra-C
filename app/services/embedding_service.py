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

    async def embed_and_store(self, rules: list[dict]) -> int:
        if not rules:
            return 0

        print(f"Generating embeddings for {len(rules)} rules...")
        
        texts = [rule["full_text"] for rule in rules]

        all_embeddings = self.embeddings.embed_documents(texts)

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
        upserted = await upsert_vectors(vectors)
        print(f"✅ Successfully passed {len(vectors)} embeddings to Pinecone!")
        return upserted