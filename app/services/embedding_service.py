from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import get_settings
from app.utils import logger
from app.services.pinecone_service import PineconeService


class EmbeddingService:
    def __init__(self):
        settings = get_settings()
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=settings.gemini_api_key,
            output_dimensionality=settings.embedding_dimensions,
        )
        
    async def get_embedding(self, text: str) -> list[float]:
        return await self.embeddings.aembed_query(text)

    async def embed_and_store(self, rules: list[dict], pinecone_service: PineconeService) -> int:
        if not rules:
            return 0

        logger.info("Generating embeddings for {len(rules)} rules...", number_of_rules=len(rules))
        
        texts = [rule["full_text"] for rule in rules]

        all_embeddings = await self.embeddings.aembed_documents(texts)

        logger.info("Packaging vectors...")
        
        vectors = []
        for rule, embedding_vector in zip(rules, all_embeddings):
            rule_type = rule.get("rule_type", "Rule")
            vector_id = f"MISRA_{rule_type.upper()}_{rule['section']}.{rule['rule_number']}"

            metadata = {
                "scope": rule["scope"],
                "rule_type": rule_type,
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

        logger.info("Delegating upload to pinecone_service...")
        upserted = await pinecone_service.upsert_vectors(vectors)
        logger.info("✅ Successfully passed embeddings to Pinecone!", number_of_embeddings=len(vectors))
        return upserted