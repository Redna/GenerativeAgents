
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar
import uuid
from pydantic import BaseModel, Field

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

from datetime import datetime

from generative_agents import global_state
from langchain_openai.embeddings import OpenAIEmbeddings

DIMENSION = 4096
embedder = OpenAIEmbeddings(openai_api_base="http://localhost:8080", deployment="model", api_key="na")

class BaseSchema(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    vector: List[float] = None


class TimeAndImportanceBaseSchema(BaseSchema):
    created_at: datetime = global_state.time.time
    last_accessed_at: datetime = global_state.time.time
    expiration_date: Optional[datetime] = None
    importance: float = 0.5


T = TypeVar("T", bound=BaseSchema)
K = TypeVar("K", bound=TimeAndImportanceBaseSchema)


class QdrantCollection:
    _rerank_limit: int = 200

    def __init__(self, client: QdrantClient,
                 collection_name: str,
                 data_schema: Type[T],
                 decay_rate: float = Field(default=0.01)):
        self.client = client
        self.collection_name = collection_name
        self.data_schema = data_schema
        self.decay_rate = decay_rate

        if collection_name not in self.client.get_collections():
            vectors_config = models.VectorParams(size=DIMENSION,
                                                 distance=models.Distance.COSINE)
            self.client.create_collection(
                collection_name, vectors_config=vectors_config)
            self.client.create_payload_index(
                self.collection_name, field_name="memory_type", field_schema="keyword")
            self.client.create_payload_index(
                self.collection_name, field_name="created", field_schema="integer")

    def _get_relevant_entries_with_scores(self, query, filter=None, limit=5) -> List[Tuple[T, float]]:
        
        if type(query) == list:
            query = ", ".join(query)

        query_vector = embedder.embed_query(query)
        try:
            points = self.client.search(collection_name=self.collection_name,
                                    query_filter=filter,
                                    limit=limit,
                                    query_vector=query_vector, 
                                    with_vectors=True)
        except Exception as e:
            raise Exception(f"Error raised by Qdrant: {e}")              

        result = []
        for point in points:
            _id = point.id
            score = point.score
            payload = point.payload
            payload["vector"] = point.vector
            result.append((self.data_schema(id=_id, **payload), score))

        return result

    def get_relevant_entries(self, query, filter=None, limit=5) -> List[T]:
        result = self._get_relevant_entries_with_scores(query, filter, limit)
        result = [entry for entry, _ in result]
        return self.add(result, new_vectors=False)

    def get_by_id(self, id: str) -> Optional[T]:
        points = self.client.retrieve(collection_name=self.collection_name,
                                        ids=[id])
        return points[0] if points else None

    def add(self, entries: List[T], new_vectors=True) -> List[T]:
        if any([not isinstance(entry, self.data_schema) for entry in entries]):
            raise Exception("Entries must be of type {}".format(
                self.data_schema.__name__))

        ids = [entry.id for entry in entries]
        payloads = [entry.model_dump(exclude=["id"]) for entry in entries]

        if new_vectors or not all([entry.vector for entry in entries]):
            vectors = embedder.embed_query([entry.content for entry in entries])
        else:
            vectors = [entry.vector for entry in entries]

        self.client.upsert(
            collection_name=self.collection_name,
            points=models.Batch(
                ids=ids,
                payloads=payloads,
                vectors=vectors
            )
        )

        return entries


class TimeAndImportanceWrapper(QdrantCollection):
    rerank_limit = 200

    def __init__(self, client: QdrantClient, collection_name: str, data_schema: Type[K], decay_rate: float = 0.01):
        self.collection = super().__init__(client, collection_name, data_schema, decay_rate=decay_rate)

    def add(self, entries: List[K], new_vectors=True) -> List[K]:
        current_time = global_state.time.time

        for entry in entries:
            entry.last_accessed_at = current_time

        return super().add(entries, new_vectors)

    def get_relevant_entries(self, query, filter=None, limit=5) -> List[K]:
        candiates = super()._get_relevant_entries_with_scores(
            query, filter, self.rerank_limit)
        current_time = global_state.time.time

        combined_scores = [(entry, self._get_combined_score(
            entry, score, current_time)) for entry, score in candiates]
        combined_scores.sort(key=lambda x: x[1], reverse=True)

        result = []

        for entry, _ in combined_scores[: limit]:
            entry.last_accessed_at = current_time
            result += [entry]

        return self.add(result, new_vectors=False)

    def _get_entry_date(self, field: str, entry: K) -> datetime:
        """Return the value of the date field of a document."""
        return getattr(entry, field)

    def _get_combined_score(
        self,
        entry: K,
        vector_relevance: float,
        current_time: datetime,
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
    def _get_hours_passed(time: datetime, ref_time: datetime) -> float:
        """Get the hours passed between two datetimes."""
        return (time - ref_time).total_seconds() / 3600
