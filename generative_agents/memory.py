import numpy as np

import dataclasses
from enum import Enum, auto
from typing import List, Optional, Union

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

class MemoryType(Enum):
    """ This enum is used to define the memory_type of memory. """
    OBSERVATION = auto()
    SHORT_TERM_PLAN = auto()
    LONG_TERM_PLAN = auto()
    LOCATION = auto()

    def __str__(self):
        return self.name.capitalize()
    
class MemoryEntry:

    def __init__(self, data: str, embedding: any, importance: int = 5, recency: float = 1.0, memory_type: MemoryType = MemoryType.OBSERVATION):
        self.data = data
        self.embedding = embedding
        self.importance = importance
        self.recency = recency
        self.memory_type = memory_type

    def __str__(self):
        return f"{self.memory_type}: {self.data} [importance: {self.importance}, recency: {self.recency}]"
    

class MemoryManager:
    def __init__(self, prune_factor=0.01):
        self.entries = []
        self.prune_factor = prune_factor

    def add(self, data: Union[str, List[str]], importance: Union[int, List[int]], memory_type: MemoryType = MemoryType.OBSERVATION):
        if type(data) == str:
            data = [data]

        if type(importance) == int:
            importance = [importance]

        if memory_type == MemoryType.LONG_TERM_PLAN or memory_type == MemoryType.SHORT_TERM_PLAN:
            self._convert_plan_to_observation(memory_type)

        for entry, entry_importance in zip(data, importance):
            embedding = model.encode(entry)
            self.entries.append(MemoryEntry(data=entry, embedding=embedding, importance=entry_importance, memory_type=memory_type))


    def retrieve(self, context: str, top_k=10) -> List[str]:
        embedding = model.encode(context)
        return [entry.data for entry in self._filter(embedding, top_k)]

    def _filter(self, embedding: any, top_k=10):
        """ This method is used to find the most similar entries to the query. """
        relevent_memories = sorted(self.entries, key=lambda entry: self.score(embedding, entry), reverse=True)
        relevent_memories = relevent_memories[:top_k]

        for entry in relevent_memories:
            entry.recency = 1.0
        
        return relevent_memories
    
    def retrieve_most_recent_and_important_memories(self, top_k = 10):
        """ This method is used to find the most recent and most important memories. """
        return sorted(self.entries, key=lambda entry: entry.recency * entry.importance, reverse=True)[:top_k]

    def score(self, embedding: any, entry: MemoryEntry):
        """ This method is used to calculate the score of the entry. """
        relevance = util.dot_score(embedding, entry.embedding)
        return float(relevance.squeeze()) * entry.recency * entry.importance

    def update(self):
        for entry in self.entries:
            entry.recency = 0.95 * entry.recency
            if entry.recency <= self.prune_factor:
                self.entries.remove(entry)

    def _convert_plan_to_observation(self, plan_type: MemoryType):
        for entry in self.entries:
            if entry.memory_type == plan_type:
                entry.memory_type == MemoryType.OBSERVATION

    def retrieve_by(self, memory_type: MemoryType, top_k=None):
        result = [entry for entry in self.entries if entry.memory_type == memory_type]
        return result[:top_k] if top_k else result
    
    def as_stream(self):
        for entry in self.entries:
            yield f"{entry.memory_type}: {entry.data} [importance: {entry.importance}, recency: {entry.recency}]"
                