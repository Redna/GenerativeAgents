
from typing import List, Optional, Type, TypeVar
import uuid
from pydantic import BaseModel, Field

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

from datetime import datetime


class BaseSchema(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str

class TimeAndImportanceBaseSchema(BaseSchema):
    created_at: datetime = datetime.now()
    last_accessed_at: datetime = datetime.now()
    expiration_date: datetime = datetime.now()
    importance: float = 0.5


class QdrantCollection:
    _rerank_limit: int = 200
    
    def __init__(self, client: QdrantClient, 
                collection_name: str,
                embedding_model: SentenceTransformer,
                data_schema: Type[BaseSchema],
                decay_rate: float = Field(default=0.01)):
        self.client = client
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.data_schema = data_schema
        self.decay_rate = decay_rate
        
        if collection_name not in self.client.get_collections():
            vectors_config = models.VectorParams(size=embedding_model.get_sentence_embedding_dimension(), 
                                                 distance=models.Distance.COSINE)
            self.client.create_collection(collection_name, vectors_config=vectors_config)
            self.client.create_payload_index(
                self.collection_name, field_name="memory_type", field_schema="keyword")
            self.client.create_payload_index(
                self.collection_name, field_name="created", field_schema="integer")    
    
    def get_relevant_entries(self, query, filter=None, limit=5) -> BaseSchema:
        points = self.client.search(collection_name=self.collection_name, filter=filter, limit=limit query=query)
        return [self.data_schema(**point['payload']) for point in points]
        
    def add(self, entries: List[T]):
        ids = [entry.id for entry in entries]
        payloads = [entry.model_dump(exclude=["id"]) for entry in entries]
        vectors = self.embedding_model.encode([entry.content for entry in entries])
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=models.Batch(
                ids=ids,
                payloads=payloads,
                vectors=vectors
            )
        )
        
class TimeAndImportanceWrapper(QdrantCollection):
    rerank_limit = 200
    def __init__(self, client: QdrantClient, collection_name: str, embedding_model: SentenceTransformer, data_schema: Type[TimeAndImportanceBaseSchema], decay_rate: float = 0.01):
        self.collection = super().__init__(client, collection_name, embedding_model, data_schema, decay_rate=decay_rate)
        
    def get_relevant_entries(self, query, filter=None, limit=5) -> TimeAndImportanceBaseSchema:
        candiates = super().get_relevant_entries(query, filter, self.rerank_limit)
        return candiates
        
        
    def _get_entry_date(self, field: str, entry: TimeAndImportanceBaseSchema) -> datetime.datetime:
        """Return the value of the date field of a document."""
        if field in entry:
            if type(entry[field]) == float:
                return datetime.datetime.fromtimestamp(entry[field])
            return entry[field]
        return datetime.datetime.now()
    
    def _get_combined_score(
        self,
        entry: TimeAndImportanceBaseSchema,
        vector_relevance: float,
        current_time: datetime.datetime,
    ) -> float:
        """Return the combined score for a document."""
        hours_passed = self._get_hours_passed(
            current_time,
            self._get_entry_date("last_accessed_at", entry),
        )
        score = (1.0 - self.decay_rate) ** hours_passed
        score += entry.importance
        score += vector_relevance
        return score
    
    @staticmethod        
    def _get_hours_passed(time: datetime.datetime, ref_time: datetime.datetime) -> float:
        """Get the hours passed between two datetimes."""
        return (time - ref_time).total_seconds() / 3600