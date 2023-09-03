import datetime
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
import uuid

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.pydantic_v1 import Field
from langchain.schema import BaseRetriever, Document
from langchain.vectorstores.base import VectorStore


def _get_hours_passed(time: datetime.datetime, ref_time: datetime.datetime) -> float:
    """Get the hours passed between two datetimes."""
    return (time - ref_time).total_seconds() / 3600


class StatelessTimeWeightedVectorStoreRetriever(BaseRetriever):
    """Retriever that combines embedding similarity with
    recency in retrieving values."""

    vectorstore: VectorStore
    """The vectorstore to store documents and determine salience."""

    search_kwargs: dict = Field(default_factory=lambda: dict(k=200))
    """Keyword arguments to pass to the vectorstore similarity search."""

    decay_rate: float = Field(default=0.01)
    """The exponential decay factor used as (1.0-decay_rate)**(hrs_passed)."""

    k: int = 15
    """The maximum number of documents to retrieve in a given call."""

    other_score_keys: List[str] = []
    """Other keys in the metadata to factor into the score, e.g. 'importance'."""

    default_salience: Optional[float] = None
    """The salience to assign memories not retrieved from the vector store.

    None assigns no salience to documents not fetched from the vector store.
    """

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def _document_get_date(self, field: str, document: Document) -> datetime.datetime:
        """Return the value of the date field of a document."""
        if field in document.metadata:
            if type(document.metadata[field]) == float:
                return datetime.datetime.fromtimestamp(document.metadata[field])
            return document.metadata[field]
        return datetime.datetime.now()

    def _get_combined_score(
        self,
        document: Document,
        vector_relevance: Optional[float],
        current_time: datetime.datetime,
    ) -> float:
        """Return the combined score for a document."""
        hours_passed = _get_hours_passed(
            current_time,
            self._document_get_date("last_accessed_at", document),
        )
        score = (1.0 - self.decay_rate) ** hours_passed
        for key in self.other_score_keys:
            if key in document.metadata:
                score += document.metadata[key]
        if vector_relevance is not None:
            score += vector_relevance
        return score

    def get_salient_docs(self, query: str) -> Dict[int, Tuple[Document, float]]:
        """Return documents that are salient to the query."""
        docs_and_scores: List[Tuple[Document, float]]
        docs_and_scores = self.vectorstore.similarity_search_with_relevance_scores(
            query, **self.search_kwargs
        )
        
        return docs_and_scores

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Return documents that are relevant to the query."""
        current_time = datetime.datetime.now()
        
        docs_and_scores = self.get_salient_docs(query)

        rescored_docs = [
            (doc, self._get_combined_score(doc, relevance, current_time))
            for doc, relevance in docs_and_scores
        ]
        rescored_docs.sort(key=lambda x: x[1], reverse=True)

        result = []
        # Ensure frequently accessed memories aren't forgotten
        for doc, _ in rescored_docs[: self.k]:
            doc.metadata["last_accessed_at"] = current_time
            result.append(doc)

        self.vectorstore.add_documents(result)
        return result

    @staticmethod
    def _prepare_documents(self, documents: List[Document], **kwargs) -> Tuple[List[Document], Dict[str, any]]:
        current_time = kwargs.get("current_time")
        if current_time is None:
            current_time = datetime.datetime.now()
        # Avoid mutating input documents
        dup_docs = [deepcopy(d) for d in documents]
        
        ids = []
        for i, doc in enumerate(dup_docs):
            if "last_accessed_at" not in doc.metadata:
                doc.metadata["last_accessed_at"] = current_time
            if "created_at" not in doc.metadata:
                doc.metadata["created_at"] = current_time
            
            if "id" not in doc.metadata:
                doc.metadata["id"] = str(uuid.uuid4())
            ids += [doc.metadata["id"]]

        kwargs["ids"] = ids
        return dup_docs, kwargs


    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add documents to vectorstore."""
        documents, kwargs = self._prepare_documents(documents, **kwargs)
        return self.vectorstore.add_documents(documents, **kwargs)

    async def aadd_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        """Add documents to vectorstore."""
        documents, kwargs = self._prepare_documents(documents, **kwargs)
        return await self.vectorstore.aadd_documents(documents, **kwargs)
