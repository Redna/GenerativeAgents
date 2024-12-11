from enum import Enum
import os
from typing import Dict, List, Optional
from pydantic import BaseModel

from qdrant_client import QdrantClient
from qdrant_client import models

from generative_agents import global_state
from generative_agents.persistence.qdrant_wrapper import TimeAndImportanceWrapper, TimeAndImportanceBaseSchema

_client = QdrantClient(":memory:")


class MemoryType(Enum):
    CHAT = "chat"
    OBSERVATION = "observation"
    EVENT = "event"
    THOUGHT = "thought"

class ConversationFilling(BaseModel):
    name: str
    utterance: str
    end: bool

class MemoryEntry(TimeAndImportanceBaseSchema):
    memory_type: str
    depth: int

    subject: str
    predicate: str
    object_: str

    poignancy: float = .5
    keywords: List[str] = []
    filling: List[str | ConversationFilling] = []

    hash_key: str = None


class AgentCollection:
    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        self.last_checkpoint = -1
        self._collection = TimeAndImportanceWrapper(
        client=_client, collection_name=agent_name, data_schema=MemoryEntry)

    def add(self, memory_entry: MemoryEntry) -> MemoryEntry:
        return self.add([memory_entry])[0]

    def get(self, context: str, limit=50) -> MemoryEntry:
        return self.collection.get_relevant_entries(context, limit=limit)

    def get_by_hash(self, hash_key:str):
        filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="hash_key",
                    match=models.MatchText(text=hash_key),
                )
            ]
        )

        result_set = self.collection.get_relevant_entries(query="", filter=filter)
        return result_set

    def get_by_type(self, context: str, memory_type: MemoryType, limit=50):
        filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="memory_type",
                    match=models.MatchText(text=memory_type.value),
                )
            ]
        )

        result_set = self.collection.get_relevant_entries(context, filter=filter, limit=limit)
        return result_set

    def get_last_chat(self, with_agent_name) -> Optional[MemoryEntry]:
        filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="memory_type",
                    match=models.MatchText(text=MemoryType.CHAT.value),
                ),
                models.FieldCondition(
                    key="object_",
                    match=models.MatchText(text=with_agent_name),
                )
            ]
        )

        result_set = self.collection.get_last_points(filter=filter, limit=1)

        return result_set[0] if result_set else None

    def get_last_entries(self, limit=50):
        return self.collection.get_last_points(limit=limit)

    @property
    def collection(self):
        current_tick = global_state.tick
        if current_tick == self.last_checkpoint:
            return self._collection

        self.last_checkpoint = current_tick
        return self._create_or_restore_snapshot(current_tick)

    def _create_or_restore_snapshot(self, snapshopt_tick: int):
        snapshot_folder = "./storage/{self._collection.collection_name}/snapshots"
        snapshot_path = f"{snapshot_folder}/{snapshopt_tick}"
        os.makedirs(snapshot_folder, exist_ok=True)

        if os.path.exists(snapshot_path):
            _client.recover_snapshot(collection_name=self.agent_name,
                                     location=f"file://{snapshot_path}",
                                     wait=True)
        else:
            snapshot_info = _client.create_snapshot(collection_name=self.agent_name,
                                                           location=f"file://{snapshot_path}")
            with open(snapshot_info, "wb") as fh:
                fh.write(snapshot_info)

        return self._collection